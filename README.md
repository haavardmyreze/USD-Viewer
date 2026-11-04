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

Open <http://localhost:8000> and either drop a USD folder onto the page, or paste
a path into **Open a path on this machine** (e.g.
`V:\project\sets\set-alhambra\set-alhambra_layout.usda`).

Opening by path reads the files in place, so references that climb out of the
file's folder (`@../../assets/column.usda@`) resolve. A browser drop can only send the
files you drop: drop a single layout file and its `../../assets` references are
missing, so drop the project root instead. Path opening is offered only to a browser on
the same machine as the server (see `USDV_LOCAL_PATHS`). Scenes opened by path are
re-converted automatically when any layer or texture changes on disk.

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
| P | Toggle proxy geometry (proxy-purpose prims instead of render-purpose ones) |

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
- `PointInstancer`, including nested instancers and `inactiveIds` / `invisibleIds`,
  written as GPU instancing (`EXT_mesh_gpu_instancing`)
- MaterialX `standard_surface` / `open_pbr_surface` (preferred over an auto-generated
  `UsdPreviewSurface` fallback): base colour, roughness, metalness, normal, emission,
  through image, convert and extract nodes. Triplanar texture nodes (e.g. Karma's
  `kma_*triplanar*`) are rendered as triplanar projection in the viewer; hex tiling is
  approximated as plain triplanar
- Textures larger than 2048 px are downscaled (opaque ones re-encoded as JPEG)
- `upAxis` and `metersPerUnit`: output is always Y-up metres
- Visibility and purpose: `render` geometry by default, `proxy` geometry with the
  Proxy toggle; `guide` is always hidden
- Reports unresolved asset paths, composition errors, unsupported prims and
  materials, layers and variant sets

**Not yet supported:** MDL materials (rendered grey), arbitrary MaterialX node graphs
beyond the nodes above, `UsdTransform2d` texture transforms, `Capsule`, curves, points, subdivision
(meshes are shown as their control cage with smooth normals), animation, USD
cameras and lights. These are listed in the viewer's warnings when present.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `USDV_DATA_DIR` | `./data` | Where uploads and converted scenes are stored |
| `USDV_MAX_UPLOAD_MB` | `4096` | Maximum upload size (also applies to unzipped archives) |
| `USDV_MAX_AGE_HOURS` | `48` | Uploads and scenes older than this are pruned |
| `USDV_MAX_TEXTURE_SIZE` | `2048` | Longest texture side in the GLB; `0` keeps originals |
| `USDV_LOCAL_PATHS` | `auto` | Open-by-path: `auto` = only for browsers on the server machine, `1` = always, `0` = never |

## API

- `POST /api/uploads`: multipart with `files` and a matching `paths` field (relative
  path of each file). Returns `uploadId`, `usdFiles`, `rootCandidates` and the suggested `root`.
- `POST /api/uploads/{uploadId}/convert` with `{"root": "path/to/root.usda", "purpose": "render"}`
  (both optional). Returns the report, including `glbUrl`.
- `POST /api/open` with `{"path": "/abs/path/to/root.usda" or folder, "root": …, "purpose": …}`:
  convert files in place on the server machine (see `USDV_LOCAL_PATHS`).
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
