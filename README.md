# USD Viewer

A browser viewer for multi-file OpenUSD scenes with first-person fly navigation.

Drop a folder (or `.zip` / `.usdz`) onto the page. A small Python service uses the
real OpenUSD library (`usd-core`) to compose the stage, resolving sublayers,
references, payloads, variants and instancing exactly as USD defines them. It then
converts the result to a single GLB, which the three.js viewer loads.

```
Browser                          Conversion service (Python + usd-core)
───────                          ──────────────────────────────────────
Drop folder / .zip / .usdz ───▶  store upload (content-hashed)
                                 find root layer (not referenced by others)
                                 open stage → compose everything
                                 report missing assets + warnings
three.js + fly controls   ◀───   convert → GLB (cached)
```

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
uvicorn server.app:app --port 8000
```

Open <http://localhost:8000> and drop a USD folder onto the page.

Or with Docker:

```bash
docker build -t usd-viewer .
```

```bash
docker run -p 8000:8000 -v usd-viewer-data:/data usd-viewer
```

To generate a small multi-file test scene (Z-up, centimetres, references, payload,
variants, instancing, a texture and a deliberately missing reference):

```bash
python -m tests.make_scene sample_scene
```

## Controls

| Input | Action |
| --- | --- |
| Click | Capture the mouse to look around (Esc releases) |
| Right-drag | Look without capturing the mouse |
| W A S D / arrows | Fly in the view direction |
| E / Q (or Space / C) | Up / down |
| Shift / Ctrl | 4× faster / 4× slower |
| Mouse wheel | Change base fly speed |
| F | Frame the whole scene |
| G | Toggle grid |
| B | Force double-sided materials (handy for flipped normals) |
| I | Hide / show the side panel |

The URL keeps the upload id and root layer (`#upload=…&root=…`), so a reload
restores the scene for as long as the upload is kept on the server (48 h by default).
A `.glb` file dropped on the page is loaded directly, without the server.

## What the converter supports (first pass)

- `Mesh` with any polygon counts, holes and left-handed orientation; `Cube`, `Sphere`,
  `Cylinder`, `Cone` and `Plane` gprims
- `normals`, `st` and `displayColor` primvars at every interpolation, indexed or not
- `UsdPreviewSurface` materials: diffuse/opacity, metallic/roughness, normal,
  occlusion and emissive, as values or `UsdUVTexture` textures (PNG/JPEG passed
  through, other Pillow-readable formats re-encoded), plus `GeomSubset` material bindings
- Native instancing: instances share one glTF mesh
- `upAxis` and `metersPerUnit`: output is always Y-up metres
- Visibility and purpose (`guide` and `proxy` are hidden)
- Reports unresolved asset paths, composition errors, unsupported prims and
  materials, layers and variant sets

**Not yet supported:** `PointInstancer`, MDL / MaterialX materials (rendered grey),
`UsdTransform2d` texture transforms, `Capsule`, curves, points, subdivision
(meshes are shown as their control cage with smooth normals), animation, USD
cameras and lights. These are listed in the viewer's warnings when present.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `USDV_DATA_DIR` | `./data` | Where uploads and converted scenes are stored |
| `USDV_MAX_UPLOAD_MB` | `4096` | Maximum upload size (also applies to unzipped archives) |
| `USDV_MAX_AGE_HOURS` | `48` | Uploads and scenes older than this are pruned |

## API

- `POST /api/uploads`: multipart with `files` and a matching `paths` field (relative
  path of each file). Returns `uploadId`, `usdFiles`, `rootCandidates` and the suggested `root`.
- `POST /api/uploads/{uploadId}/convert` with `{"root": "path/to/root.usda"}` (optional).
  Returns the report, including `glbUrl`.
- `GET /api/scenes/{sceneId}/scene.glb`
- `GET /api/health`

## Development

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

three.js is vendored in `web/vendor/three`, so no CDN is needed at runtime. To
update it, run `scripts/vendor-three.sh <version>`.
