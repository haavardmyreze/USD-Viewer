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
                                 find root layer (pipeline assemblies first)
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
| Left-drag | Orbit around the point under the cursor |
| Shift + left-drag / middle-drag | Pan |
| Scroll | Zoom towards the point under the cursor |
| Right-drag | Look around; W A S D fly while it is held |
| Right-drag + scroll | Change fly speed |
| Double-click | Fly to the point under the cursor |
| W A S D / arrows, E / Q | Fly (also without the mouse), up / down |
| Shift / Ctrl | 4x faster / 4x slower |
| Click | Select the prim under the cursor (an instance picks its PointInstancer) |
| F | Frame the selection, or the whole scene |
| Z | Shading: material, no material, matcap (also the three buttons in the top bar) |
| T / I / Esc | Scene graph, scene info, clear the selection |
| [ / ] | Previous / next variant of the default prim |
| G / L / B / P | Grid, shadows, double-sided, proxy geometry |
| I / O / ? | Inspector, open another scene, controls help |

On touch screens one finger orbits and two fingers pan and pinch-zoom.

## Performance

Heavy scenes stay interactive because:

- The converter writes simplified levels of detail (each about a quarter of the
  previous) for meshes with 20k+ triangles. The viewer culls every instance
  against the view and picks a level by its size on screen.
- Frames are only drawn when something changes. While the camera moves, detail and
  resolution adapt to keep it smooth; when it stops, one sharper frame is drawn.
- Shadows are rendered once per scene, not every frame.
- Picking (orbit pivot, zoom target, double-click) reads depth from the GPU, so it
  costs the same for any amount of geometry.

The Detail control in the inspector trades quality for speed (Fast, Balanced, High,
Full = always full resolution).

The URL keeps the upload id and root layer (`#upload=…&root=…`), so a reload
restores the scene for as long as the upload is kept on the server (48 h by default).
A `.glb` file dropped on the page is loaded directly, without the server.

## Scene graph

**T** (or the tree button in the top bar) opens the composed prim hierarchy. It
is usd-refgraph's scene tree, in the same design: Houdini's Scene Graph Tree icons
for each prim type (a kind's stars on Xforms and Scopes), connector lines,
`over`s in italics, deactivated prims struck through, and reference, payload,
variant and instance flags on the rows. It opens collapsed (Expand all opens it,
materials stay closed) and is
read a level at a time, so a heavy stage is never walked whole; arrow keys walk
it, Shift-click or Shift+arrow expands or collapses a whole branch.

Selecting works both ways. A prim picked in the tree is highlighted in the view
(everything under it, or the instances of a prototype); double-click or **F**
frames it. Clicking geometry opens the tree onto its prim; a PointInstancer
instance selects the instancer and names the instance (`…/PointInstancer[4]`).
The detail panel shows the prim's type, kind, schemas and how much of it is drawn,
and every layer with an opinion on it, strongest first.

The tree is the stage as converted: same payloads, same variant selections.

The Houdini icons in `web/vendor/houdini/` are SideFX's own SVGs, copied unchanged
from usd-refgraph; check your Houdini licence before shipping this outside the
studio.

## Shading

As in Blender's viewport, the three buttons at the top right (or **Z**) switch
between the scene's own **materials**, **no material** and a **matcap**. No material
shows everything in one neutral grey, still lit by the sun and environment and
casting shadows, to judge form and lighting without textures (Blender's Solid
mode). Matcap puts one lit-sphere look on everything, independent of the lighting;
pick Clay, Studio, Ceramic, Metal, Red wax or Normals in the Display tab.

## Variants

When the default prim (or the only root prim) has variant sets, the top bar shows
one switcher per set: pick a variant, step with ‹ ›, or press **[** / **]**. Any
prim's variant sets can be switched from its detail panel in the scene graph.

Switching re-composes the stage with the selection in an anonymous session
layer; the files on disk are never changed. The camera, the scene graph and the
selection stay where they were. Each combination is converted once and cached,
and textures are cached across conversions, so stepping back and forth is quick.
The selections are kept in the URL (`#…&variants=…`).

## Root layer

A folder (dropped or opened by path) opens its pipeline **assembly**, as
usd-refgraph names them: `<name>.usda` with lowercase, hyphenated tokens and no
`_`. A shot root (`<seq>-<shot>.usda`) comes first, then sets (`set-…`), then assets;
blocks (`<name>_<block>.usda|usdc`) are not offered. A folder with no conventionally
named files falls back to the layers nothing else references.

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
  through image, convert, extract and separate nodes. Triplanar texture nodes (e.g. Karma's
  `kma_*triplanar*`) are rendered as triplanar projection in the viewer; hex tiling is
  approximated as plain triplanar
- Textures larger than 2048 px are downscaled (opaque ones re-encoded as JPEG)
- OpenEXR textures (via the `OpenEXR` package), converted to 8 bit using the authored
  colour space: linear (and ACEScg / ACES2065-1) colour maps are sRGB-encoded, maps
  tagged `srgb_texture` are kept as is, data maps (`Raw`) stay linear, and normal maps
  stored in -1..1 are remapped
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
| `USDV_LODS` | `1` | Generate levels of detail (`0` to disable) |
| `USDV_LOD_MIN_TRIANGLES` | `20000` | Meshes with fewer triangles get no LODs |
| `USDV_LOCAL_PATHS` | `auto` | Open-by-path: `auto` = only for browsers on the server machine, `1` = always, `0` = never |

## API

- `POST /api/uploads`: multipart with `files` and a matching `paths` field (relative
  path of each file). Returns `uploadId`, `usdFiles`, `rootCandidates` and the suggested `root`.
- `POST /api/uploads/{uploadId}/convert` with `{"root": "path/to/root.usda", "purpose": "render",
  "variants": {"/Prim": {"set": "variant"}}}` (all optional). Returns the report, including
  `glbUrl` and the default prim's variant sets (`defaultVariants`).
- `POST /api/open` with `{"path": "/abs/path/to/root.usda" or folder, "root": …, "purpose": …,
  "variants": …}`: convert files in place on the server machine (see `USDV_LOCAL_PATHS`).
- `GET /api/scenes/{sceneId}/scene.glb`. Nodes carry the prim they belong to in
  `extras.usdPath` (and `usdProto`, `usdInstanceIds` for PointInstancer instances).
- `GET /api/scenes/{sceneId}/prims?path=/World`: one level of the prim hierarchy;
  `/subtree?path=…` a whole branch (up to 5,000 prims); `/reveal?path=…` the levels
  down to a prim; `/prim?path=…` one prim's detail and opinions.
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
