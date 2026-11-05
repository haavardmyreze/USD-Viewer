"""Test scenes for the viewer's compiled shading networks (MaterialX / UsdPreviewSurface).

    python -m tests.make_shader_scenes <out_dir>

Writes:
  spaces.usda     Z-up, centimetres: unlit quads whose colour is fract(P * 0.01) in
                  world and object space, and the raw UVs, to check the spaces the
                  network sees against USD's.
  triplanar.usda  Z-up, centimetres: a rotated, scaled cube with a world-space
                  MaterialX triplanarprojection of a four-colour image.
  nodes.usda      many materials, each using a different set of library nodes
                  (noise, ramps, mixes, colour transforms, normal maps, UsdUVTexture,
                  UsdTransform2d, open_pbr_surface, gltf_pbr, ...): they must compile.
  quads.png       2x2 pixel colours: red top-left, green top-right, blue bottom-left,
                  white bottom-right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

T = Sdf.ValueTypeNames


def quads_png(path: Path) -> None:
    from PIL import Image
    img = Image.new("RGB", (64, 64))
    px = img.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = [[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 255)]][y // 32][x // 32]
    img.save(path)


class Net:
    def __init__(self, stage, mat_path):
        self.stage = stage
        self.mat = UsdShade.Material.Define(stage, mat_path)
        self.path = mat_path

    def node(self, name, sid, **inputs):
        s = UsdShade.Shader.Define(self.stage, f"{self.path}/{name}")
        s.CreateIdAttr(sid)
        for k, v in inputs.items():
            self.set(s, k, v)
        return s

    @staticmethod
    def set(s, k, v):
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], UsdShade.Shader):
            src, out = v
            o = src.GetOutput(out) or src.CreateOutput(out, T.Float)
            s.CreateInput(k, o.GetTypeName()).ConnectToSource(o)
        elif isinstance(v, UsdShade.Shader):
            o = v.GetOutput("out") or v.CreateOutput("out", T.Color3f)
            s.CreateInput(k, o.GetTypeName()).ConnectToSource(o)
        elif isinstance(v, Sdf.AssetPath):
            s.CreateInput(k, T.Asset).Set(v)
        elif isinstance(v, str):
            s.CreateInput(k, T.String).Set(v)
        elif isinstance(v, bool):
            s.CreateInput(k, T.Bool).Set(v)
        elif isinstance(v, int):
            s.CreateInput(k, T.Int).Set(v)
        elif isinstance(v, float):
            s.CreateInput(k, T.Float).Set(v)
        elif isinstance(v, Gf.Vec2f):
            s.CreateInput(k, T.Float2).Set(v)
        elif isinstance(v, Gf.Vec3f):
            s.CreateInput(k, T.Color3f).Set(v)
        else:
            raise TypeError(v)

    def out(self, s, name="out", type_=T.Color3f):
        if not s.GetOutput(name):
            s.CreateOutput(name, type_)
        return s

    def surface(self, s, context="mtlx"):
        o = s.GetOutput("out") or s.CreateOutput("out", T.Token)
        self.mat.CreateSurfaceOutput(context).ConnectToSource(o)
        return self.mat


def quad(stage, path, x0, y0, size, z, mat, st=True):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([(x0, y0, z), (x0 + size, y0, z), (x0 + size, y0 + size, z), (x0, y0 + size, z)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr([(x0, y0, z), (x0 + size, y0 + size, z)])
    if st:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", T.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def unlit(stage, path, color_node_fn):
    n = Net(stage, path)
    col = color_node_fn(n)
    s = n.node("surf", "ND_surface_unlit")
    Net.set(s, "emission_color", col)
    return n.surface(s)


def spaces(out: Path) -> None:
    stage = Usd.Stage.CreateNew(str(out / "spaces.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.Xform.Define(stage, "/World")

    def fract_of(space):
        def fn(n):
            pos = n.out(n.node("pos", "ND_position_vector3", space=space), type_=T.Vector3f)
            mul = n.out(n.node("mul", "ND_multiply_vector3FA", in2=0.01), type_=T.Vector3f)
            Net.set(mul, "in1", (pos, "out"))
            fr = n.out(n.node("fract", "ND_modulo_vector3FA", in2=1.0), type_=T.Vector3f)
            Net.set(fr, "in1", (mul, "out"))
            conv = n.out(n.node("conv", "ND_convert_vector3_color3"))
            Net.set(conv, "in", (fr, "out"))
            return conv
        return fn

    world = unlit(stage, "/World/Looks/World", fract_of("world"))
    obj = unlit(stage, "/World/Looks/Object", fract_of("object"))

    def uv_fn(n):
        tc = n.out(n.node("tc", "ND_texcoord_vector2"), type_=T.Float2)
        conv = n.out(n.node("conv", "ND_convert_vector2_vector3"), type_=T.Vector3f)
        Net.set(conv, "in", (tc, "out"))
        c = n.out(n.node("c", "ND_convert_vector3_color3"))
        Net.set(c, "in", (conv, "out"))
        return c

    uvm = unlit(stage, "/World/Looks/UV", uv_fn)

    def tex_fn(n):
        return n.out(n.node("img", "ND_image_color3", file=Sdf.AssetPath("./quads.png")))

    texm = unlit(stage, "/World/Looks/Tex", tex_fn)

    def udim_fn(n):
        return n.out(n.node("img", "ND_image_color3", file=Sdf.AssetPath("./tiles/paint.<UDIM>.png")))

    udimm = unlit(stage, "/World/Looks/Udim", udim_fn)

    def houdini138_fn(n):
        """As Houdini 20 writes it (MaterialX 1.38): a file shared through a dot node,
        an image read as color4 and the red channel spread to grey by a swizzle."""
        path = n.node("path", "ND_dot_filename", **{"in": Sdf.AssetPath("./quads.png")})
        path.CreateOutput("out", T.Asset)
        img = n.out(n.node("img", "ND_image_color4"), type_=T.Color4f)
        Net.set(img, "file", (path, "out"))
        rgb = n.out(n.node("rgb", "ND_swizzle_color4_color3", channels="rgb"))
        Net.set(rgb, "in", (img, "out"))
        grey = n.out(n.node("grey", "ND_swizzle_color3_color3", channels="rrr"))
        Net.set(grey, "in", rgb)
        return grey

    swz = unlit(stage, "/World/Looks/Houdini138", houdini138_fn)
    # world-space quad: placed by a transform, so world and object differ
    x = UsdGeom.Xform.Define(stage, "/World/Moved")
    x.AddTranslateOp().Set((130.0, 20.0, 5.0))
    x.AddScaleOp().Set((2.0, 2.0, 2.0))
    quad(stage, "/World/Moved/WorldQuad", 0, 0, 40, 10, world)
    x2 = UsdGeom.Xform.Define(stage, "/World/Moved2")
    x2.AddTranslateOp().Set((-20.0, 20.0, 5.0))
    x2.AddScaleOp().Set((2.0, 2.0, 2.0))
    quad(stage, "/World/Moved2/ObjectQuad", 10, 5, 40, 10, obj)
    quad(stage, "/World/UVQuad", 0, 120, 80, 25, uvm)
    quad(stage, "/World/TexQuad", 100, 120, 80, 25, texm)
    # UDIM: st spans two tiles each way (1001, 1002, 1011, 1012 = red, green, blue, missing)
    uq = quad(stage, "/World/UdimQuad", 200, 120, 80, 25, udimm)
    quad(stage, "/World/SwizzleQuad", 300, 120, 80, 25, swz)
    UsdGeom.PrimvarsAPI(uq).GetPrimvar("st").Set([(0, 0), (2, 0), (2, 2), (0, 2)])
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()


def triplanar(out: Path) -> None:
    stage = Usd.Stage.CreateNew(str(out / "triplanar.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.Xform.Define(stage, "/World")
    n = Net(stage, "/World/Looks/Tri")
    pos = n.out(n.node("pos", "ND_position_vector3", space="world"), type_=T.Vector3f)
    nrm = n.out(n.node("nrm", "ND_normal_vector3", space="world"), type_=T.Vector3f)
    mul = n.out(n.node("mul", "ND_multiply_vector3FA", in2=1.0 / 40.0), type_=T.Vector3f)
    Net.set(mul, "in1", (pos, "out"))
    tri = n.out(n.node("tri", "ND_triplanarprojection_color3", blend=0.3))
    for a in "xyz":
        Net.set(tri, f"file{a}", Sdf.AssetPath("./quads.png"))
    Net.set(tri, "position", (mul, "out"))
    Net.set(tri, "normal", (nrm, "out"))
    s = n.node("surf", "ND_surface_unlit")
    Net.set(s, "emission_color", tri)
    mat = n.surface(s)
    x = UsdGeom.Xform.Define(stage, "/World/Rock")
    x.AddTranslateOp().Set((15.0, -7.0, 30.0))
    x.AddRotateXYZOp().Set((20.0, 35.0, 10.0))
    x.AddScaleOp().Set((25.0, 20.0, 15.0))
    sph = UsdGeom.Sphere.Define(stage, "/World/Rock/Shape")
    UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mat)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()


def nodes(out: Path) -> None:
    """One sphere per material; each material exercises different nodes."""
    stage = Usd.Stage.CreateNew(str(out / "nodes.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    tex = Sdf.AssetPath("./quads.png")
    builders = {}

    def reg(fn):
        builders[fn.__name__] = fn
        return fn

    @reg
    def noise_mix(n):
        nz = n.out(n.node("nz", "ND_fractal3d_float", octaves=4, lacunarity=2.0), type_=T.Float)
        mixn = n.out(n.node("mix", "ND_mix_color3", fg=Gf.Vec3f(0.8, 0.2, 0.1), bg=Gf.Vec3f(0.1, 0.3, 0.8)))
        Net.set(mixn, "mix", (nz, "out"))
        surf = n.node("surf", "ND_standard_surface_surfaceshader")
        Net.set(surf, "base_color", mixn)
        Net.set(surf, "specular_roughness", (nz, "out"))
        return surf

    @reg
    def cell_worley_ramp(n):
        w = n.out(n.node("w", "ND_worleynoise3d_float", jitter=1.0), type_=T.Float)
        c = n.out(n.node("c", "ND_cellnoise3d_float"), type_=T.Float)
        r = n.out(n.node("ramp", "ND_ramplr_color3", valuel=Gf.Vec3f(0, 0, 0), valuer=Gf.Vec3f(1, 1, 0)))
        add = n.out(n.node("add", "ND_add_color3FA"))
        Net.set(add, "in1", r)
        Net.set(add, "in2", (w, "out"))
        hsv = n.out(n.node("hsv", "ND_hsvadjust_color3", amount=Gf.Vec3f(0.2, 1.0, 1.0)))
        Net.set(hsv, "in", add)
        surf = n.node("surf", "ND_standard_surface_surfaceshader", coat=1.0, sheen=0.5)
        Net.set(surf, "base_color", hsv)
        Net.set(surf, "metalness", (c, "out"))
        return surf

    @reg
    def normalmap_tangents(n):
        img = n.out(n.node("img", "ND_image_vector3", file=tex), type_=T.Vector3f)
        nm = n.out(n.node("nm", "ND_normalmap_float", scale=0.5), type_=T.Vector3f)
        Net.set(nm, "in", (img, "out"))
        tan = n.out(n.node("tan", "ND_tangent_vector3", space="world"), type_=T.Vector3f)
        bit = n.out(n.node("bit", "ND_bitangent_vector3", space="world"), type_=T.Vector3f)
        cr = n.out(n.node("cross", "ND_crossproduct_vector3"), type_=T.Vector3f)
        Net.set(cr, "in1", (tan, "out"))
        Net.set(cr, "in2", (bit, "out"))
        conv = n.out(n.node("conv", "ND_convert_vector3_color3"))
        Net.set(conv, "in", (cr, "out"))
        surf = n.node("surf", "ND_standard_surface_surfaceshader")
        Net.set(surf, "normal", (nm, "out"))
        Net.set(surf, "base_color", conv)
        return surf

    @reg
    def preview_transform2d(n):
        rd = n.out(n.node("st", "UsdPrimvarReader_float2", varname="st"), "result", T.Float2)
        t2 = n.out(n.node("t2d", "UsdTransform2d", rotation=30.0, scale=Gf.Vec2f(2, 3),
                          translation=Gf.Vec2f(0.1, 0.2)), "result", T.Float2)
        Net.set(t2, "in", (rd, "result"))
        uvt = n.node("tex", "UsdUVTexture", file=tex, wrapS="mirror", wrapT="clamp", sourceColorSpace="sRGB")
        n.out(uvt, "rgb", T.Float3)
        n.out(uvt, "r", T.Float)
        Net.set(uvt, "st", (t2, "result"))
        surf = n.node("surf", "UsdPreviewSurface", metallic=0.2, clearcoat=0.5)
        Net.set(surf, "diffuseColor", (uvt, "rgb"))
        Net.set(surf, "roughness", (uvt, "r"))
        surf.CreateOutput("surface", T.Token)
        n.mat.CreateSurfaceOutput().ConnectToSource(surf.GetOutput("surface"))
        return None

    @reg
    def open_pbr(n):
        g = n.out(n.node("g", "ND_noise3d_color3", amplitude=Gf.Vec3f(0.5, 0.5, 0.5), pivot=0.5))
        surf = n.node("surf", "ND_open_pbr_surface_surfaceshader", coat_weight=0.7, fuzz_weight=0.3,
                      thin_film_weight=0.5, specular_roughness_anisotropy=0.5, emission_luminance=0.2)
        Net.set(surf, "base_color", g)
        return surf

    @reg
    def gltf(n):
        img = n.out(n.node("img", "ND_image_color4", file=tex, uaddressmode="constant",
                           vaddressmode="mirror", filtertype="closest"), type_=T.Color4f)
        rgb = n.out(n.node("rgb", "ND_convert_color4_color3"))
        Net.set(rgb, "in", (img, "out"))
        surf = n.node("surf", "ND_gltf_pbr_surfaceshader", metallic=0.5, transmission=0.4, iridescence=0.5)
        Net.set(surf, "base_color", rgb)
        return surf

    @reg
    def geom_primvar(n):
        gc = n.out(n.node("gc", "ND_geompropvalue_color3", geomprop="tint", default=Gf.Vec3f(1, 0, 1)))
        tc1 = n.out(n.node("tc1", "ND_texcoord_vector2", index=1), type_=T.Float2)
        img = n.out(n.node("img", "ND_image_color3", file=tex))
        Net.set(img, "texcoord", (tc1, "out"))
        mul = n.out(n.node("mul", "ND_multiply_color3"))
        Net.set(mul, "in1", gc)
        Net.set(mul, "in2", img)
        vd = n.out(n.node("vd", "ND_viewdirection_vector3"), type_=T.Vector3f)
        tp = n.out(n.node("tp", "ND_transformvector_vector3", fromspace="world", tospace="object"), type_=T.Vector3f)
        Net.set(tp, "in", (vd, "out"))
        dp = n.out(n.node("dp", "ND_dotproduct_vector3"), type_=T.Float)
        Net.set(dp, "in1", (tp, "out"))
        Net.set(dp, "in2", (tp, "out"))
        surf = n.node("surf", "ND_standard_surface_surfaceshader")
        Net.set(surf, "base_color", mul)
        Net.set(surf, "specular_roughness", (dp, "out"))
        return surf

    @reg
    def hextiled_and_colorspace(n):
        h = n.out(n.node("h", "ND_hextiledimage_color3", file=tex))
        img = n.node("img", "ND_image_color3")
        img.CreateInput("file", T.Asset).Set(tex)
        img.GetInput("file").GetAttr().SetColorSpace("acescg")
        n.out(img)
        sw = n.out(n.node("sw", "ND_switch_color3I", which=1))
        Net.set(sw, "in1", h)
        Net.set(sw, "in2", img)
        lum = n.out(n.node("lum", "ND_luminance_color3"))
        Net.set(lum, "in", sw)
        surf = n.node("surf", "ND_standard_surface_surfaceshader", transmission=0.5, thin_film_thickness=300.0)
        Net.set(surf, "base_color", lum)
        return surf

    @reg
    def unknown_node(n):
        k = n.out(n.node("k", "kma_mystery_color", inp=Gf.Vec3f(0.2, 0.9, 0.2)))
        surf = n.node("surf", "ND_standard_surface_surfaceshader")
        Net.set(surf, "base_color", k)
        return surf

    for i, (name, fn) in enumerate(builders.items()):
        n = Net(stage, f"/World/Looks/{name}")
        surf = fn(n)
        if surf is not None:
            n.surface(surf)
        x = UsdGeom.Xform.Define(stage, f"/World/{name}")
        x.AddTranslateOp().Set(((i % 3) * 2.5, (i // 3) * 2.5, 0.0))
        if name == "geom_primvar":
            sph = quad(stage, f"/World/{name}/Ball", -1, -1, 2, 0, n.mat)
            api = UsdGeom.PrimvarsAPI(sph)
            api.CreatePrimvar("st1", T.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                [(0, 0), (2, 0), (2, 2), (0, 2)])
            api.CreatePrimvar("tint", T.Color3fArray, UsdGeom.Tokens.vertex).Set(
                [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
        else:
            sph = UsdGeom.Sphere.Define(stage, f"/World/{name}/Ball")
        UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(n.mat)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()


def geometry(out: Path) -> None:
    """Curves of every basis and wrap, points with per-point colour and width, capsules, a subdiv mesh."""
    import math
    stage = Usd.Stage.CreateNew(str(out / "geometry.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    helix = [(math.cos(a) * 0.5, a * 0.08, math.sin(a) * 0.5) for a in np.linspace(0, 6 * math.pi, 19)]
    x = 0.0
    for ctype, basis, wrap in (("linear", "bezier", "nonperiodic"), ("cubic", "bezier", "nonperiodic"),
                               ("cubic", "bspline", "nonperiodic"), ("cubic", "catmullRom", "pinned"),
                               ("cubic", "bspline", "periodic")):
        c = UsdGeom.BasisCurves.Define(stage, f"/World/Curves_{basis}_{wrap}_{ctype}")
        pts = [(px + x, py, pz) for px, py, pz in helix]
        c.CreatePointsAttr(pts)
        c.CreateCurveVertexCountsAttr([len(pts)])
        c.CreateTypeAttr(ctype)
        c.CreateBasisAttr(basis)
        c.CreateWrapAttr(wrap)
        c.CreateWidthsAttr([0.04 + 0.04 * i / len(pts) for i in range(len(pts))])
        c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        c.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.9, 0.5, 0.1)])
        x += 1.4
    lines = UsdGeom.BasisCurves.Define(stage, "/World/Lines")  # no widths: GL lines
    lines.CreatePointsAttr([(0, 2, 0), (1, 2.5, 0), (2, 2, 0), (3, 2.5, 0)])
    lines.CreateCurveVertexCountsAttr([4])
    lines.CreateTypeAttr("linear")
    pts = UsdGeom.Points.Define(stage, "/World/Points")
    grid = [(i * 0.25, 3.0 + j * 0.25, 0.0) for i in range(12) for j in range(6)]
    pts.CreatePointsAttr(grid)
    pts.CreateWidthsAttr([0.08 + 0.1 * (i % 3) / 3 for i in range(len(grid))])
    pts.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    pts.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([(i / len(grid), 1 - i / len(grid), 0.5)
                                                              for i in range(len(grid))])
    for i, axis in enumerate("XYZ"):
        cap = UsdGeom.Capsule.Define(stage, f"/World/Capsule{axis}")
        cap.CreateAxisAttr(axis)
        cap.CreateRadiusAttr(0.3)
        cap.CreateHeightAttr(0.8)
        UsdGeom.XformCommonAPI(cap).SetTranslate((i * 1.5, -1.5, 0))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.Save()


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quads_png(out / "quads.png")
    from PIL import Image
    (out / "tiles").mkdir(exist_ok=True)
    for tile, color in ((1001, (255, 0, 0)), (1002, (0, 255, 0)), (1011, (0, 0, 255))):
        Image.new("RGB", (32, 32), color).save(out / "tiles" / f"paint.{tile}.png")
    spaces(out)
    triplanar(out)
    nodes(out)
    geometry(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shader_scenes")
