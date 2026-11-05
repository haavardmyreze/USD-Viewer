"""The pipeline's bookkeeping, read the way usd-refgraph reads it.

Two things say who made a file and what state it is in, and both are in the
files themselves (usd-refgraph: "the files are the database"):

* every published layer carries a record in `customLayerData`, written by the
  USD Publisher HDA: artist, status, comment, hip_file, rop_path and
  export_datetime_unix;
* the naming convention (pipeline guide, sections 15.2-15.10) says which entity
  a file belongs to and which block of it: `char-bob.usda` is an assembly,
  `char-bob_lookdev.usda` its lookdev block, `char-bob_bc_4k.exr` a texture.

This mirrors usd-refgraph's `pipeline.py` and `naming.py`, including the older
wip / ready / final status spellings; change one and change the other.
"""

from __future__ import annotations

import os
import re
from typing import Any

# -- the publish record (customLayerData) ------------------------------------

#: Weakest first: an entity is only as finished as its least finished part.
STATUS_ORDER = ["placeholder", "production_ready", "locked"]

STATUS_ALIASES = {
    "placeholder": "placeholder",
    "production_ready": "production_ready",
    "productionready": "production_ready",
    "locked": "locked",
    "wip": "placeholder",
    "in_progress": "placeholder",
    "not_started": "placeholder",
    "ready": "production_ready",
    "final": "locked",
    "published": "locked",
    "approved": "locked",
}

KNOWN_KEYS = {"artist", "status", "comment", "hip_file", "rop_path", "export_datetime_unix"}


def read_record(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """A layer's `customLayerData` as a publish record, or None when it has none."""
    if not data:
        return None
    text = {str(k): str(v).strip() for k, v in dict(data).items() if v is not None}
    raw_status = text.get("status") or None
    status = STATUS_ALIASES.get(raw_status.lower().replace("-", "_"), "unknown") if raw_status else "unknown"
    record: dict[str, Any] = {"status": status}
    if status == "unknown" and raw_status:
        record["statusRaw"] = raw_status
    for key, field in (("artist", "artist"), ("comment", "comment"), ("hip_file", "hipFile"),
                       ("rop_path", "ropPath")):
        if text.get(key):
            record[field] = text[key]
    exported = _epoch_ms(text.get("export_datetime_unix"))
    if exported:
        record["exportedAt"] = exported
    extra = {k: v for k, v in text.items() if k not in KNOWN_KEYS and v}
    if extra:
        record["extra"] = extra
    if len(record) == 1 and status == "unknown":
        return None
    return record


def rollup_status(statuses: list[str]) -> str:
    known = [s for s in statuses if s in STATUS_ORDER]
    return min(known, key=STATUS_ORDER.index) if known else "unknown"


def _epoch_ms(value: str | None) -> int | None:
    """`export_datetime_unix` is authored as a string of whole seconds."""
    try:
        seconds = float(value) if value else 0.0
    except ValueError:
        return None
    return int(seconds * 1000) if seconds > 0 else None


# -- the naming convention ----------------------------------------------------

#: One token: lowercase words joined by hyphens, never underscores (15.10).
TOKEN = r"[a-z0-9]+(?:-[a-z0-9]+)*"
#: `<sequence>-<shot>`, e.g. `kilo-0010` (15.3).
SHOT = r"[a-z]{3,5}-[0-9]{4}"

SHOT_ROOT_RE = re.compile(rf"^{SHOT}\.usda$")
SHOT_BLOCK_RE = re.compile(rf"^{SHOT}_{TOKEN}\.(usda|usdc)$")
ASSEMBLY_RE = re.compile(rf"^{TOKEN}\.usda$")
BLOCK_RE = re.compile(rf"^{TOKEN}_{TOKEN}\.(usda|usdc)$")

# Textures (15.9): <entity>[_<descriptor>]_<channel>_<resolution>[.<udim>].<ext>
CHANNEL = r"(?P<channel>bc|n|aormt|m|disp)"
RESOLUTION = r"(?P<res>1k|2k|4k|8k)"
TEXTURE_RE = re.compile(
    rf"^{TOKEN}(?:_{TOKEN})?_{CHANNEL}_{RESOLUTION}(?:\.(?:[0-9]{{4}}|<UDIM>))?\.(?:exr|png|tif|jpg)$"
)

CHANNEL_LABELS = {
    "bc": "base colour", "n": "normal", "aormt": "AO / rough / metal", "m": "mask", "disp": "displacement",
}

CATEGORIES = {
    "char": "character", "prop": "prop", "env": "environment", "veh": "vehicle",
    "fx": "fx", "set": "set", "mtl": "material",
}


def classify(path: str) -> tuple[str, str | None]:
    """The role (assembly, block, texture, other) and tier (asset, set, shot) a name declares."""
    name = os.path.basename(path)
    tier = "set" if name.startswith("set-") else "asset"
    if SHOT_ROOT_RE.match(name):
        return "assembly", "shot"
    if SHOT_BLOCK_RE.match(name):
        return "block", "shot"
    if ASSEMBLY_RE.match(name):
        return "assembly", tier
    if BLOCK_RE.match(name):
        return "block", tier
    if TEXTURE_RE.match(name):
        return "texture", tier
    return "other", None


def split_entity(path: str) -> tuple[str, str | None]:
    """`char-bob_lookdev.usda` -> ("char-bob", "lookdev"); an assembly has no block."""
    stem = os.path.basename(path).split(".", 1)[0]
    entity, _, block = stem.partition("_")
    if classify(path)[0] == "texture":
        return entity, None
    return entity, block or None


def category_of(entity: str) -> str | None:
    return CATEGORIES.get(entity.split("-", 1)[0])


def texture_info(path: str) -> dict[str, str]:
    """What a texture's name says: its channel and resolution (15.9)."""
    match = TEXTURE_RE.match(os.path.basename(path))
    if not match:
        return {}
    return {"channel": CHANNEL_LABELS.get(match["channel"], match["channel"]), "resolution": match["res"]}
