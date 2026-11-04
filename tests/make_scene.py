"""Build a small multi-file USD scene that exercises the converter.

    python -m tests.make_scene OUT_DIR

Layout:
    shot.usda                  root: Z-up, centimetres, sublayers layout + lights
    layers/layout.usda         references assets, instancing, variants, payload
    layers/lights.usda         a distant light
    assets/crate/crate.usda    textured UsdPreviewSurface, GeomSubsets
    assets/crate/tex/wood.png
    assets/floor.usdc          binary crate file, a big plane with displayColor
    assets/tower.usda          loaded through a payload; colour variant set
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdShade


def write_png(path: Path, w: int = 64, h: int = 64) -> None:
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            check = ((x // 8) + (y // 8)) % 2
            row += bytes((180, 120, 60) if check else (120, 70, 30))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", raw) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _stage(path: Path, default_prim: str | None = None) -> Usd.Stage:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    if default_prim:
        prim = UsdGeom.Xform.Define(stage, f"/{default_prim}").GetPrim()
        stage.SetDefaultPrim(prim)
        Usd.ModelAPI(prim).SetKind(Kind.Tokens.component)
    return stage


def make_crate(out: Path) -> None:
    write_png(out / "assets/crate/tex/wood.png")
    stage = _stage(out / "assets/crate/crate.usda", "Crate")
    mesh = UsdGeom.Mesh.Define(stage, "/Crate/geo")
    s = 50.0  # 1 m crate in cm, sitting on z=0
    pts = [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0), (-s, -s, 2 * s), (s, -s, 2 * s), (s, s, 2 * s), (-s, s, 2 * s)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([(-s, -s, 0), (s, s, 2 * s)])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)] * 6)

    # wood material (textured) on all faces, metal material on the top via a GeomSubset
    wood = UsdShade.Material.Define(stage, "/Crate/mtl/wood")
    surf = UsdShade.Shader.Define(stage, "/Crate/mtl/wood/surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    reader = UsdShade.Shader.Define(stage, "/Crate/mtl/wood/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, "/Crate/mtl/wood/diffuse_tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./tex/wood.png")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    wood.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")

    metal = UsdShade.Material.Define(stage, "/Crate/mtl/metal")
    msurf = UsdShade.Shader.Define(stage, "/Crate/mtl/metal/surface")
    msurf.CreateIdAttr("UsdPreviewSurface")
    msurf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.6, 0.65))
    msurf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    msurf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    metal.CreateSurfaceOutput().ConnectToSource(msurf.ConnectableAPI(), "surface")

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(wood)
    top = UsdShade.MaterialBindingAPI(mesh).CreateMaterialBindSubset("top", [1])
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(metal)
    stage.Save()


def make_floor(out: Path) -> None:
    stage = _stage(out / "assets/floor.usdc", "Floor")
    plane = UsdGeom.Mesh.Define(stage, "/Floor/geo")
    s = 1000.0
    plane.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        [(0.3, 0.35, 0.3), (0.35, 0.3, 0.3), (0.3, 0.3, 0.35), (0.33, 0.33, 0.33)])
    stage.Save()


def make_tower(out: Path) -> None:
    stage = _stage(out / "assets/tower.usda", "Tower")
    root = stage.GetDefaultPrim()
    UsdGeom.Cylinder.Define(stage, "/Tower/body").CreateHeightAttr(600.0)
    body = UsdGeom.Cylinder(stage.GetPrimAtPath("/Tower/body"))
    body.CreateRadiusAttr(80.0)
    UsdGeom.XformCommonAPI(body).SetTranslate((0, 0, 300))
    roof = UsdGeom.Cone.Define(stage, "/Tower/roof")
    roof.CreateRadiusAttr(120.0)
    roof.CreateHeightAttr(200.0)
    UsdGeom.XformCommonAPI(roof).SetTranslate((0, 0, 700))
    roof.CreateDisplayColorPrimvar().Set([(0.6, 0.1, 0.1)])

    vset = root.GetVariantSets().AddVariantSet("color")
    for name, color in (("stone", (0.6, 0.6, 0.55)), ("sand", (0.85, 0.75, 0.5))):
        vset.AddVariant(name)
        vset.SetVariantSelection(name)
        with vset.GetVariantEditContext():
            body.CreateDisplayColorPrimvar().Set([color])
    vset.SetVariantSelection("stone")
    stage.Save()


def make_layers(out: Path) -> None:
    lights = _stage(out / "layers/lights.usda")
    UsdLux.DistantLight.Define(lights, "/World/Sun").CreateIntensityAttr(3.0)
    lights.Save()

    layout = _stage(out / "layers/layout.usda")
    world = UsdGeom.Xform.Define(layout, "/World")
    floor = layout.DefinePrim("/World/Floor")
    floor.GetReferences().AddReference("../assets/floor.usdc")

    # 3 x 3 instanced crates
    for i in range(3):
        for j in range(3):
            p = layout.DefinePrim(f"/World/Crates/crate_{i}_{j}")
            p.GetReferences().AddReference("../assets/crate/crate.usda")
            p.SetInstanceable(True)
            UsdGeom.XformCommonAPI(UsdGeom.Xform(p)).SetTranslate((i * 150.0 - 150, j * 150.0 + 300, 0))

    tower = layout.DefinePrim("/World/Tower")
    tower.GetPayloads().AddPayload("../assets/tower.usda")
    UsdGeom.XformCommonAPI(UsdGeom.Xform(tower)).SetTranslate((-500, -400, 0))
    tower.GetVariantSets().GetVariantSet("color").SetVariantSelection("sand")

    ball = UsdGeom.Sphere.Define(layout, "/World/Ball")
    ball.CreateRadiusAttr(40.0)
    ball.CreateDisplayColorPrimvar().Set([(0.1, 0.3, 0.8)])
    UsdGeom.XformCommonAPI(ball).SetTranslate((300, -200, 40))

    hidden = UsdGeom.Cube.Define(layout, "/World/HiddenCube")
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    missing = layout.DefinePrim("/World/Missing")
    missing.GetReferences().AddReference("../assets/does_not_exist.usda")
    layout.Save()


def make_root(out: Path) -> Path:
    root = _stage(out / "shot.usda")
    root.GetRootLayer().subLayerPaths.append("./layers/lights.usda")
    root.GetRootLayer().subLayerPaths.append("./layers/layout.usda")
    root.SetDefaultPrim(root.GetPrimAtPath("/World"))
    cam = UsdGeom.Camera.Define(root, "/World/Camera")
    UsdGeom.XformCommonAPI(cam).SetTranslate((0, -1500, 300))
    root.Save()
    return out / "shot.usda"


def build(out: Path) -> Path:
    out = Path(out)
    make_crate(out)
    make_floor(out)
    make_tower(out)
    make_layers(out)
    return make_root(out)


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "sample_scene")))
