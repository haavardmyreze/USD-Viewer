"""OpenEXR texture decoding.

glTF only carries 8-bit PNG/JPEG images, so EXR maps are read as float,
converted to the encoding glTF expects for their role and quantised:

- colour maps (base colour, emission) must be sRGB-encoded. Scene-linear data is
  encoded with the sRGB transfer curve (ACEScg / ACES2065-1 are converted to
  Rec.709 primaries first); data already tagged as sRGB is kept as is.
- data maps (roughness, metalness, normals, occlusion, opacity) are linear in
  glTF. They are quantised directly unless tagged sRGB, which is decoded.
- normal maps stored in [-1, 1] are remapped to [0, 1].

Needs the `OpenEXR` package (pip install OpenEXR).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

MAGIC = b"v/1\x01"

# linear AP1 / AP0 -> linear Rec.709 (Bradford-adapted D60 -> D65)
_ACESCG_TO_709 = np.array([[1.70505, -0.62179, -0.08326],
                           [-0.13026, 1.14080, -0.01055],
                           [-0.02400, -0.12897, 1.15297]], dtype=np.float32)
_ACES_TO_709 = np.array([[2.52169, -1.13413, -0.38756],
                         [-0.27648, 1.37272, -0.09624],
                         [-0.01538, -0.15298, 1.16835]], dtype=np.float32)


def is_exr(data: bytes | None) -> bool:
    return bool(data) and data[:4] == MAGIC


def available() -> bool:
    try:
        import OpenEXR  # noqa: F401
    except ImportError:
        return False
    return True


def classify_colorspace(name: str | None) -> str | None:
    """Map an OCIO / MaterialX / USD colour space name to raw|linear|acescg|aces|srgb|None."""
    if not name:
        return None
    n = str(name).strip().lower()
    if n in ("auto", ""):
        return None
    if "raw" in n or "non-color" in n or "non_color" in n or n in ("data", "utility - raw"):
        return "raw"
    if "acescg" in n or "ap1" in n:
        return "acescg"
    if "aces2065" in n or "ap0" in n:
        return "aces"
    if n.startswith("lin") or "linear" in n:
        return "linear"
    if "srgb" in n or "gamma" in n or "rec709_display" in n:
        return "srgb"
    return None


def _pick_channels(channels: dict) -> tuple[list[str], str]:
    """Choose which channels to read: returns (names, kind) with kind rgb|rgba|y|ya."""
    layers: dict[str, dict[str, str]] = {}
    for full in channels:
        layer, _, short = full.rpartition(".")
        layers.setdefault(layer, {})[short.upper()] = full

    def usable(chs):
        return {"R", "G", "B"} <= chs.keys() or "Y" in chs or "R" in chs

    layer = "" if "" in layers and usable(layers[""]) else next(
        (name for name, chs in layers.items() if usable(chs)), next(iter(layers)))
    chs = layers[layer]
    if {"R", "G", "B"} <= chs.keys():
        names = [chs["R"], chs["G"], chs["B"]]
        return (names + [chs["A"]], "rgba") if "A" in chs else (names, "rgb")
    for gray in ("Y", "R"):
        if gray in chs:
            return ([chs[gray], chs["A"]], "ya") if "A" in chs else ([chs[gray]], "y")
    first = next(iter(chs.values()))
    return [first], "y"


def read(data: bytes, path: str | None = None) -> tuple[np.ndarray, str]:
    """Decode to a float32 (H, W, C) array; returns (pixels, kind)."""
    import OpenEXR

    tmp = None
    if not path or not os.path.isfile(path):  # e.g. inside a usdz package
        fd, tmp = tempfile.mkstemp(suffix=".exr")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        path = tmp
    try:
        with OpenEXR.File(path, separate_channels=True) as f:
            channels = f.channels()
            names, kind = _pick_channels(channels)
            planes = [np.asarray(channels[n].pixels, dtype=np.float32) for n in names]
    finally:
        if tmp:
            os.unlink(tmp)
    shape = planes[0].shape
    if any(p.shape != shape for p in planes):
        raise ValueError("subsampled EXR channels are not supported")
    return np.stack(planes, axis=-1), kind


def _box_reduce(px: np.ndarray, max_size: int) -> np.ndarray:
    """Average whole pixel blocks down towards max_size (in linear float, before encoding)."""
    if not max_size:
        return px
    f = max(px.shape[0], px.shape[1]) // max_size
    if f < 2:
        return px
    h, w = px.shape[0] // f, px.shape[1] // f
    return px[:h * f, :w * f].reshape(h, f, w, f, px.shape[2]).mean(axis=(1, 3))


def _srgb_encode(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def _srgb_decode(x):
    return np.where(x <= 0.04045, x / 12.92, np.power((np.maximum(x, 0.04045) + 0.055) / 1.055, 2.4))


def to_image(px: np.ndarray, kind: str, *, color: bool, normal: bool = False,
             colorspace: str | None = None, max_size: int = 0):
    """Convert decoded EXR pixels to an 8-bit PIL image (RGB, RGBA, L or LA)."""
    from PIL import Image

    px = np.nan_to_num(_box_reduce(px, max_size), nan=0.0, posinf=1.0, neginf=0.0)
    alpha = px[..., -1] if kind in ("rgba", "ya") else None
    rgb = px[..., :3] if kind.startswith("rgb") else px[..., :1]
    cs = classify_colorspace(colorspace)

    if color:
        if kind.startswith("rgb") and cs in ("acescg", "aces"):
            rgb = rgb @ (_ACESCG_TO_709 if cs == "acescg" else _ACES_TO_709).T
        if cs != "srgb":
            rgb = _srgb_encode(np.clip(rgb, 0.0, 1.0))
    elif cs == "srgb":
        rgb = _srgb_decode(np.clip(rgb, 0.0, 1.0))
    if normal and rgb.size and float(rgb.min()) < -0.01:
        rgb = rgb * 0.5 + 0.5

    planes = rgb if alpha is None else np.concatenate([rgb, alpha[..., None]], axis=-1)
    out = (np.clip(planes, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out[..., 0] if out.shape[-1] == 1 else out)  # L / LA / RGB / RGBA
