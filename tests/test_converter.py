import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

from server.uploads import find_root_candidates
from server.usd_to_glb import _triangulate, convert, dependency_report
from tests.glb_utils import accessor, parse_glb


@pytest.fixture(scope="module")
def converted(scene_dir):
    data, info = convert(str(scene_dir / "shot.usda"))
    gltf, binary = parse_glb(data)
    return gltf, binary, info


def _node(gltf, name):
    return next(n for n in gltf["nodes"] if n.get("name") == name)


def test_triangulate_mixed_polygons():
    corners, faces = _triangulate(np.array([3, 4, 5]))
    assert faces.tolist() == [0, 1, 1, 2, 2, 2]
    assert corners.tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6], [7, 8, 9], [7, 9, 10], [7, 10, 11]]


def test_root_detection(scene_dir):
    # assemblies only (`<name>.usda`), the one nothing references first
    candidates = find_root_candidates(scene_dir)
    assert candidates[0] == "shot.usda"
    assert "assets/floor.usdc" not in candidates


def test_root_candidates_are_pipeline_assemblies(tmp_path):
    from server.uploads import assembly_tier

    assert assembly_tier("sets/set-alhambra/set-alhambra.usda") == "set"
    assert assembly_tier("kilo-0010.usda") == "shot"
    assert assembly_tier("assets/env-stone/env-stone.usda") == "asset"
    for block in ("set-alhambra_layout.usda", "env-stone_model.usdc", "Env-Stone.usda", "env-stone.usdc"):
        assert assembly_tier(block) is None
    for rel in ("assets/env-rock/env-rock.usda", "assets/env-rock/env-rock_model.usda",
                "sets/set-yard/set-yard.usda", "sets/set-yard/set-yard_layout.usda", "shots/kilo-0010.usda"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("#usda 1.0\n")
    assert find_root_candidates(tmp_path) == [
        "shots/kilo-0010.usda", "sets/set-yard/set-yard.usda", "assets/env-rock/env-rock.usda"]


def test_stage_units_and_axis_become_root_transform(converted):
    gltf, _, info = converted
    root = _node(gltf, "usd_stage_root")
    assert gltf["scenes"][0]["nodes"] == [gltf["nodes"].index(root)]
    # Z-up centimetres -> Y-up metres
    assert root["matrix"] == [0.01, 0, 0, 0, 0, 0, -0.01, 0, 0, 0.01, 0, 0, 0, 0, 0, 1]
    assert info["upAxis"] == "Z" and info["metersPerUnit"] == 0.01


def test_composition_is_resolved(converted):
    gltf, _, info = converted
    names = {n.get("name") for n in gltf["nodes"]}
    assert "/World/Floor/geo" in names                      # reference to a .usdc
    assert "/World/Tower/body" in names                     # payload
    assert "/World/Crates/crate_2_2/geo" in names           # instance proxy
    assert "/World/HiddenCube" not in names                 # invisible
    assert info["stats"]["meshPrims"] == 13
    assert info["defaultPrim"] == "/World"


def test_instances_share_one_mesh(converted):
    gltf, _, info = converted
    crate_meshes = {n["mesh"] for n in gltf["nodes"] if "/World/Crates/" in n.get("name", "")}
    assert len(crate_meshes) == 1
    assert info["stats"]["uniqueMeshes"] == 5
    crate = _node(gltf, "/World/Crates/crate_0_1/geo")
    assert crate["matrix"][12:15] == [-150.0, 450.0, 0.0]


def test_crate_materials_texture_and_subset(converted):
    gltf, binary, _ = converted
    mesh = gltf["meshes"][_node(gltf, "/World/Crates/crate_0_0/geo")["mesh"]]
    assert len(mesh["primitives"]) == 2  # wood + metal GeomSubset
    mats = [gltf["materials"][p["material"]] for p in mesh["primitives"]]
    wood = next(m for m in mats if m["name"].endswith("/wood"))
    metal = next(m for m in mats if m["name"].endswith("/metal"))
    assert "baseColorTexture" in wood["pbrMetallicRoughness"]
    assert metal["pbrMetallicRoughness"]["metallicFactor"] == 1.0
    assert len(gltf["images"]) == 1 and gltf["images"][0]["mimeType"] == "image/png"

    # the metal subset is exactly one quad = 2 triangles
    metal_prim = next(p for p in mesh["primitives"] if gltf["materials"][p["material"]] is metal)
    assert gltf["accessors"][metal_prim["indices"]]["count"] == 6
    uv = accessor(gltf, binary, mesh["primitives"][0]["attributes"]["TEXCOORD_0"])
    assert uv.min() >= 0.0 and uv.max() <= 1.0


def test_vertex_display_color_and_variant(converted):
    gltf, binary, _ = converted
    floor = gltf["meshes"][_node(gltf, "/World/Floor/geo")["mesh"]]
    colors = accessor(gltf, binary, floor["primitives"][0]["attributes"]["COLOR_0"])
    assert colors.shape == (4, 3)

    # layout.usda selects the "sand" variant on the tower
    body = gltf["meshes"][_node(gltf, "/World/Tower/body")["mesh"]]
    color = gltf["materials"][body["primitives"][0]["material"]]["pbrMetallicRoughness"]["baseColorFactor"]
    assert color[:3] == pytest.approx([0.85, 0.75, 0.5], abs=1e-4)


def test_normals_are_unit_length(converted):
    gltf, binary, _ = converted
    for mesh in gltf["meshes"]:
        for p in mesh["primitives"]:
            n = accessor(gltf, binary, p["attributes"]["NORMAL"])
            assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)


def test_missing_reference_is_reported(scene_dir, converted):
    _, _, info = converted
    deps = dependency_report(str(scene_dir / "shot.usda"))
    assert any(u.endswith("does_not_exist.usda") for u in deps["unresolved"])
    assert len(deps["layers"]) == 6
    assert any("does_not_exist.usda" in e for e in info["compositionErrors"])
    assert info["variantSets"][0]["selection"] == "sand"


def test_left_handed_mesh_flips_winding(tmp_path):
    def make(orientation):
        path = tmp_path / f"{orientation}.usda"
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        mesh = UsdGeom.Mesh.Define(stage, "/tri")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateOrientationAttr(orientation)
        mesh.CreateSubdivisionSchemeAttr("none")
        stage.Save()
        gltf, binary = parse_glb(convert(str(path))[0])
        prim = gltf["meshes"][0]["primitives"][0]
        pos = accessor(gltf, binary, prim["attributes"]["POSITION"])
        idx = accessor(gltf, binary, prim["indices"])
        p = pos[idx]
        return np.cross(p[1] - p[0], p[2] - p[0])[2]

    assert make("rightHanded") > 0
    assert make("leftHanded") < 0


def test_usdz_with_texture(tmp_path, scene_dir):
    from pxr import UsdUtils

    usdz = tmp_path / "crate.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(scene_dir / "assets/crate/crate.usda")), str(usdz))
    gltf, _ = parse_glb(convert(str(usdz))[0])
    assert len(gltf["images"]) == 1


def _instancer_stage(path):
    """Two PointInstancers: one with an inactive instance and a prototype that
    itself contains a nested PointInstancer (2 x 3 = 6 boxes)."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")

    pi = UsdGeom.PointInstancer.Define(stage, "/World/Scatter")
    UsdGeom.XformCommonAPI(pi).SetTranslate((0, 10, 0))
    protos = UsdGeom.Scope.Define(stage, "/World/Scatter/Prototypes")
    box = UsdGeom.Cube.Define(stage, "/World/Scatter/Prototypes/Box")
    UsdGeom.XformCommonAPI(box).SetTranslate((0, 1, 0))  # prototype root xform applies
    proxy = UsdGeom.Cube.Define(stage, "/World/Scatter/Prototypes/Box/proxyGeo")
    proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
    pi.CreatePrototypesRel().SetTargets([box.GetPath()])
    pi.CreateProtoIndicesAttr([0, 0, 0])
    pi.CreatePositionsAttr([(1, 0, 0), (2, 0, 0), (3, 0, 0)])
    pi.CreateOrientationsAttr([Gf.Quath(1, 0, 0, 0)] * 3)
    pi.CreateScalesAttr([(2, 2, 2)] * 3)
    pi.CreateIdsAttr([10, 11, 12])
    pi.DeactivateId(11)

    outer = UsdGeom.PointInstancer.Define(stage, "/World/Rows")
    row = UsdGeom.Xform.Define(stage, "/World/Rows/Prototypes/Row")
    inner = UsdGeom.PointInstancer.Define(stage, "/World/Rows/Prototypes/Row/Inner")
    leaf = UsdGeom.Sphere.Define(stage, "/World/Rows/Prototypes/Row/Inner/Prototypes/Ball")
    inner.CreatePrototypesRel().SetTargets([leaf.GetPath()])
    inner.CreateProtoIndicesAttr([0, 0, 0])
    inner.CreatePositionsAttr([(0, 0, 0), (0, 0, 5), (0, 0, 10)])
    outer.CreatePrototypesRel().SetTargets([row.GetPath()])
    outer.CreateProtoIndicesAttr([0, 0])
    outer.CreatePositionsAttr([(-20, 0, 0), (-40, 0, 0)])
    stage.Save()
    return path


def _instanced_nodes(gltf, binary):
    out = {}
    for n in gltf["nodes"]:
        ext = n.get("extensions", {}).get("EXT_mesh_gpu_instancing")
        if ext:
            out[n["name"]] = {k: accessor(gltf, binary, v) for k, v in ext["attributes"].items()}
    return out


def test_point_instancer(tmp_path):
    data, info = convert(str(_instancer_stage(tmp_path / "pi.usda")))
    gltf, binary = parse_glb(data)
    assert "EXT_mesh_gpu_instancing" in gltf["extensionsUsed"]
    nodes = _instanced_nodes(gltf, binary)

    boxes = nodes["/World/Scatter/Box"]
    # instance 11 is inactive; proto root translate (0,1,0) is scaled by the instance scale 2
    assert boxes["TRANSLATION"].tolist() == [[1, 12, 0], [3, 12, 0]]
    assert np.allclose(boxes["SCALE"], 2)
    assert np.allclose(boxes["ROTATION"], [0, 0, 0, 1])

    balls = nodes["/World/Rows/Row"]
    assert len(balls["TRANSLATION"]) == 6
    assert sorted(map(tuple, balls["TRANSLATION"].tolist())) == sorted(
        (x, 0, z) for x in (-20, -40) for z in (0, 5, 10))

    assert info["stats"]["pointInstancers"] == 3  # outer + nested inner (once per prototype) + scatter
    assert info["stats"]["meshPrims"] == 8
    assert info["warnings"] == []


def test_purpose_selection(tmp_path):
    path = _instancer_stage(tmp_path / "pi.usda")
    render_info = convert(str(path))[1]
    proxy_info = convert(str(path), "proxy")[1]
    # the proxy cube under each box is only drawn in proxy mode
    assert proxy_info["stats"]["meshPrims"] == render_info["stats"]["meshPrims"] + 2
    assert proxy_info["purpose"] == "proxy"


def test_instance_decomposition_rejects_mirroring():
    from server.usd_to_glb import _decompose_trs
    mirror = np.eye(4)[None].copy()
    mirror[0, 0, 0] = -1
    assert _decompose_trs(mirror) is None
    rot = np.eye(4)[None].copy()
    rot[0, :3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]  # row-vector 90 deg about Y
    t, r, s = _decompose_trs(rot)
    assert np.allclose(np.abs(r[0]), [0, np.sqrt(0.5), 0, np.sqrt(0.5)], atol=1e-6)
    assert np.allclose(s, 1)


def test_materialx_triplanar_material(tmp_path):
    from PIL import Image
    from pxr import UsdShade

    Image.new("RGB", (3000, 1500), (200, 100, 50)).save(tmp_path / "bc.png")
    Image.new("RGB", (64, 64), (255, 128, 0)).save(tmp_path / "orm.png")
    path = tmp_path / "mtlx.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    mat = UsdShade.Material.Define(stage, "/World/Mtl")

    def shader(name, sid):
        s = UsdShade.Shader.Define(stage, f"/World/Mtl/{name}")
        s.CreateIdAttr(sid)
        return s

    surf = shader("surface", "ND_standard_surface_surfaceshader")
    pos = shader("pos", "ND_position_vector3")
    pos.CreateInput("space", Sdf.ValueTypeNames.String).Set("world")
    mul = shader("mul", "ND_multiply_vector3FA")
    mul.CreateInput("in1", Sdf.ValueTypeNames.Vector3f).ConnectToSource(pos.ConnectableAPI(), "out")
    mul.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(0.5)
    bc = shader("bc", "kma_hextiled_triplanar_color")
    bc.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./bc.png")
    bc.CreateInput("position", Sdf.ValueTypeNames.Vector3f).ConnectToSource(mul.ConnectableAPI(), "out")
    conv = shader("conv", "ND_convert_color4_color3")
    conv.CreateInput("in", Sdf.ValueTypeNames.Color4f).ConnectToSource(bc.ConnectableAPI(), "out")
    orm = shader("orm", "ND_image_color4")
    orm.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./orm.png")
    ext = shader("ext", "ND_extract_color4")
    ext.CreateInput("in", Sdf.ValueTypeNames.Color4f).ConnectToSource(orm.ConnectableAPI(), "out")
    ext.CreateInput("index", Sdf.ValueTypeNames.Int).Set(1)
    surf.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(conv.ConnectableAPI(), "out")
    surf.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).ConnectToSource(ext.ConnectableAPI(), "out")
    mat.CreateSurfaceOutput("mtlx").ConnectToSource(surf.ConnectableAPI(), "out")
    # a flat auto-generated preview shader must not win over MaterialX
    preview = shader("preview", "UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))
    mat.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.Save()

    data, info = convert(str(path))
    gltf, binary = parse_glb(data)
    m = gltf["materials"][0]
    pbr = m["pbrMetallicRoughness"]
    # the plain glTF material (fallback) still carries the textures
    assert "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr
    assert pbr["roughnessFactor"] == 1.0 and pbr["metallicFactor"] == 0.0
    # the viewer's shader: Karma's node as MaterialX triplanarprojection, projected in
    # world space in stage units (cm), scaled by the authored 0.5
    sh = m["extras"]["usdvShader"]
    assert sh["surface"] == "standard_surface"
    assert "Pworld" in sh["needs"] and "Nobject" in sh["needs"]
    assert [0.5, 0.0, 0.0, 0.0] in sh["uniforms"]
    assert "usdv_Pworld" in sh["main"]
    assert {"diffuse", "roughness", "metalness"} <= set(sh["params"])
    assert len(sh["textures"]) == 2
    # the oversized texture was downscaled and re-encoded; the ORM map is used as-is
    images = {i["name"]: i for i in gltf["images"]}
    assert images["bc.png"]["mimeType"] == "image/jpeg"
    assert images["orm.png"]["mimeType"] == "image/png"
    assert any("hex tiling is not reproduced" in w for w in info["warnings"])


def _dense_grid_stage(path, n=170):
    """One wavy n x n quad grid (~2 n^2 triangles), instanced twice by a PointInstancer."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    xs, zs = np.meshgrid(np.linspace(0, 10, n + 1), np.linspace(0, 10, n + 1))
    pts = np.stack([xs, np.sin(xs) * np.cos(zs) * 0.5, zs], -1).reshape(-1, 3)
    i = np.arange(n)
    a = (i[:, None] * (n + 1) + i[None, :]).reshape(-1)
    quads = np.stack([a, a + 1, a + n + 2, a + n + 1], 1).reshape(-1)
    pi = UsdGeom.PointInstancer.Define(stage, "/World/Tiles")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tiles/Prototypes/Tile")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * (n * n))
    mesh.CreateFaceVertexIndicesAttr([int(x) for x in quads])
    mesh.CreateSubdivisionSchemeAttr("none")
    pi.CreatePrototypesRel().SetTargets([mesh.GetPath()])
    pi.CreateProtoIndicesAttr([0, 0])
    pi.CreatePositionsAttr([(0, 0, 0), (20, 0, 0)])
    stage.Save()
    return path


def test_levels_of_detail(tmp_path):
    data, info = convert(str(_dense_grid_stage(tmp_path / "dense.usda")))
    gltf, binary = parse_glb(data)
    assert info["stats"]["uniqueMeshes"] == 1
    assert info["stats"]["lodMeshes"] >= 2
    # LOD copies live in a second scene, so other glTF viewers show only the default scene
    assert gltf["scene"] == 0 and gltf["scenes"][1]["name"] == "usdv_lods"
    tagged = [n for n in gltf["nodes"] if "usdvLod" in n.get("extras", {})]
    levels = sorted(tagged, key=lambda n: n["extras"]["usdvLod"]["level"])
    assert [n["extras"]["usdvLod"]["level"] for n in levels] == list(range(len(levels)))
    assert len({n["extras"]["usdvLod"]["group"] for n in levels}) == 1
    # every level shares the same instance transforms and has ~1/4 the triangles of the one before
    inst = {tuple(sorted(n["extensions"]["EXT_mesh_gpu_instancing"]["attributes"].items())) for n in levels}
    assert len(inst) == 1
    tris = [sum(gltf["accessors"][p["indices"]]["count"] // 3 for p in gltf["meshes"][n["mesh"]]["primitives"])
            for n in levels]
    assert all(b < a * 0.4 for a, b in zip(tris, tris[1:]))
    for n in levels[1:]:
        normals = accessor(gltf, binary, gltf["meshes"][n["mesh"]]["primitives"][0]["attributes"]["NORMAL"])
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)


def test_lods_can_be_disabled(tmp_path, monkeypatch):
    import server.usd_to_glb as conv

    monkeypatch.setattr(conv, "LOD_ENABLED", False)
    gltf, _ = parse_glb(convert(str(_dense_grid_stage(tmp_path / "dense.usda")))[0])
    assert len(gltf["scenes"]) == 1 and len(gltf["meshes"]) == 1


def _write_exr(path, pixels):
    OpenEXR = pytest.importorskip("OpenEXR")
    channels = {"RGB" if pixels.shape[-1] == 3 else "RGBA": pixels}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))


def _image_pixels(gltf, binary, name):
    import io

    from PIL import Image

    im = next(i for i in gltf["images"] if i["name"] == name)
    bv = gltf["bufferViews"][im["bufferView"]]
    start = bv.get("byteOffset", 0)
    img = Image.open(io.BytesIO(binary[start:start + bv["byteLength"]]))
    return im["mimeType"], np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)


def test_materialx_exr_textures(tmp_path):
    from pxr import UsdShade

    size = (64, 64)
    _write_exr(tmp_path / "bc.exr", np.full(size + (3,), 0.214, np.float32))          # linear 0.214 = sRGB 128
    _write_exr(tmp_path / "bc_srgb.exr", np.full(size + (3,), 0.5, np.float32))       # already sRGB-encoded
    _write_exr(tmp_path / "aorm.exr", np.dstack([np.full(size, v, np.float16) for v in (1.0, 0.6, 0.0)]))
    _write_exr(tmp_path / "n.exr", np.dstack([np.zeros(size), np.zeros(size), np.ones(size)]).astype(np.float32))

    path = tmp_path / "exr.usda"
    stage = Usd.Stage.CreateNew(str(path))

    def material(name, bc_file, bc_space):
        mat = UsdShade.Material.Define(stage, f"/World/{name}")

        def shader(n, sid):
            s = UsdShade.Shader.Define(stage, f"/World/{name}/{n}")
            s.CreateIdAttr(sid)
            return s

        def image(n, file, space):
            s = shader(n, "ND_image_color3")
            f = s.CreateInput("file", Sdf.ValueTypeNames.Asset)
            f.Set(file)
            if space:
                f.GetAttr().SetColorSpace(space)
            return s

        surf = shader("surface", "ND_standard_surface_surfaceshader")
        surf.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            image("bc", bc_file, bc_space).ConnectableAPI(), "out")
        sep = shader("sep", "ND_separate3_vector3")
        sep.CreateInput("in", Sdf.ValueTypeNames.Vector3f).ConnectToSource(
            image("aorm", "./aorm.exr", "Raw").ConnectableAPI(), "out")
        surf.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).ConnectToSource(sep.ConnectableAPI(), "outy")
        nmap = shader("nmap", "ND_normalmap_float")
        nmap.CreateInput("in", Sdf.ValueTypeNames.Vector3f).ConnectToSource(
            image("n", "./n.exr", "Raw").ConnectableAPI(), "out")
        surf.CreateInput("normal", Sdf.ValueTypeNames.Vector3f).ConnectToSource(nmap.ConnectableAPI(), "out")
        mat.CreateSurfaceOutput("mtlx").ConnectToSource(surf.ConnectableAPI(), "out")
        return mat

    for i, (file, space) in enumerate([("./bc.exr", None), ("./bc_srgb.exr", "srgb_texture")]):
        cube = UsdGeom.Cube.Define(stage, f"/World/Cube{i}")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material(f"Mtl{i}", file, space))
    stage.Save()

    data, info = convert(str(path))
    gltf, binary = parse_glb(data)
    assert not [w for w in info["warnings"] if "EXR" in w or "texture" in w.lower()]
    mime, linear = _image_pixels(gltf, binary, "bc.exr")
    assert mime == "image/jpeg"
    assert linear == pytest.approx([128, 128, 128], abs=2)   # linear data is sRGB-encoded for glTF
    _, srgb = _image_pixels(gltf, binary, "bc_srgb.exr")
    assert srgb == pytest.approx([128, 128, 128], abs=2)     # tagged sRGB is kept as is
    _, normal = _image_pixels(gltf, binary, "n.exr")
    assert normal == pytest.approx([0, 0, 255], abs=2)       # raw data is not curve-encoded
    # separate3.outy picks roughness from G; the map is already packed glTF-style
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert pbr["roughnessFactor"] == 1.0
    _, orm = _image_pixels(gltf, binary, "aorm.exr")
    assert orm[1] == pytest.approx(0.6 * 255, abs=2)


def test_exr_signed_normals_and_colorspaces():
    from server import exr

    px = np.dstack([np.full((4, 4), -1.0), np.zeros((4, 4)), np.ones((4, 4))]).astype(np.float32)
    img = exr.to_image(px, "rgb", color=False, normal=True)
    assert np.asarray(img)[0, 0].tolist() == [0, 128, 255]
    assert exr.classify_colorspace("srgb_texture") == "srgb"
    assert exr.classify_colorspace("lin_srgb") == "linear"
    assert exr.classify_colorspace("ACES - ACEScg") == "acescg"
    assert exr.classify_colorspace("Utility - Raw") == "raw"
    assert exr.classify_colorspace("auto") is None


def test_exr_without_openexr_warns(tmp_path, monkeypatch):
    from pxr import UsdShade

    from server import exr

    _write_exr(tmp_path / "bc.exr", np.full((8, 8, 3), 0.5, np.float32))
    monkeypatch.setattr(exr, "available", lambda: False)
    path = tmp_path / "s.usda"
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    mat = UsdShade.Material.Define(stage, "/Mtl")
    tex = UsdShade.Shader.Define(stage, "/Mtl/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./bc.exr")
    surf = UsdShade.Shader.Define(stage, "/Mtl/surf")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.Save()

    _, info = convert(str(path))
    assert any("OpenEXR" in w for w in info["warnings"])


def test_nodes_name_the_prim_the_scene_graph_selects(tmp_path, converted):
    gltf, _, _ = converted
    crate = _node(gltf, "/World/Crates/crate_0_1/geo")
    assert crate["extras"]["usdPath"] == "/World/Crates/crate_0_1/geo"  # an instance proxy path

    data, _ = convert(str(_instancer_stage(tmp_path / "pi.usda")))
    gltf, binary = parse_glb(data)
    nodes = {n["name"]: n for n in gltf["nodes"]}
    boxes = nodes["/World/Scatter/Box"]["extras"]
    # picking an instance selects the instancer; ids are authored indices (11 = index 1 is inactive)
    assert boxes["usdPath"] == "/World/Scatter"
    assert boxes["usdProto"] == "/World/Scatter/Prototypes/Box"
    assert accessor(gltf, binary, boxes["usdInstanceIds"]).tolist() == [0, 2]
    balls = nodes["/World/Rows/Row"]["extras"]
    # nested instancers select the outermost one: 2 rows x 3 balls, row index per ball
    assert balls["usdPath"] == "/World/Rows"
    assert balls["usdProto"] == "/World/Rows/Prototypes/Row/Inner/Prototypes/Ball"
    assert accessor(gltf, binary, balls["usdInstanceIds"]).tolist() == [0, 1, 0, 1, 0, 1]


def test_variant_overrides(scene_dir):
    tower = str(scene_dir / "assets" / "tower.usda")
    _, info = convert(tower)
    assert info["defaultVariants"] == {
        "prim": "/Tower", "sets": [{"name": "color", "selection": "stone", "options": ["sand", "stone"]}]}
    data, info = convert(tower, variants={"/Tower": {"color": "sand"}})
    assert info["defaultVariants"]["sets"][0]["selection"] == "sand"
    assert info["variants"] == {"/Tower": {"color": "sand"}}
    gltf, binary = parse_glb(data)
    body = next(m for m in gltf["meshes"] if m["name"] == "/Tower/body")
    material = gltf["materials"][body["primitives"][0]["material"]]
    assert np.allclose(material["pbrMetallicRoughness"]["baseColorFactor"][:3], (0.85, 0.75, 0.5))
    # the files on disk are untouched
    assert Usd.Stage.Open(tower).GetDefaultPrim().GetVariantSet("color").GetVariantSelection() == "stone"
    with pytest.raises(Exception):
        convert(tower, variants={"not a path": {"color": "sand"}})


def test_materialx_triplanarprojection_node(tmp_path):
    """MaterialX's own triplanar node reads filex / filey / filez, not `file`."""
    from PIL import Image
    from pxr import UsdShade

    Image.new("RGB", (32, 32), (200, 100, 50)).save(tmp_path / "bc.png")
    path = tmp_path / "tri.usda"
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    mat = UsdShade.Material.Define(stage, "/World/Mtl")
    surf = UsdShade.Shader.Define(stage, "/World/Mtl/surface")
    surf.CreateIdAttr("ND_standard_surface_surfaceshader")
    tri = UsdShade.Shader.Define(stage, "/World/Mtl/tri")
    tri.CreateIdAttr("ND_triplanarprojection_color3")
    for axis in "xyz":
        tri.CreateInput(f"file{axis}", Sdf.ValueTypeNames.Asset).Set("./bc.png")
    surf.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(tri.ConnectableAPI(), "out")
    mat.CreateSurfaceOutput("mtlx").ConnectToSource(surf.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.Save()

    data, info = convert(str(path))
    gltf, _ = parse_glb(data)
    m = gltf["materials"][0]
    assert "baseColorTexture" in m["pbrMetallicRoughness"]
    # no position / normal input: MaterialX's defaults, object space
    sh = m["extras"]["usdvShader"]
    assert {"Pobject", "Nobject"} <= set(sh["needs"]) and "Pworld" not in sh["needs"]
    assert len(sh["textures"]) == 1  # the same image on all three axes is sampled once
    assert info["warnings"] == []

    # a different image per axis is sampled per axis
    Image.new("RGB", (32, 32), (0, 0, 255)).save(tmp_path / "other.png")
    tri.GetInput("filez").Set("./other.png")
    stage.Save()
    data, info = convert(str(path))
    gltf, _ = parse_glb(data)
    assert len(gltf["materials"][0]["extras"]["usdvShader"]["textures"]) == 2


def _subdiv_cube(path, scheme=None, crease=False):
    stage = Usd.Stage.CreateNew(str(path))
    m = UsdGeom.Mesh.Define(stage, "/Cube")
    pts = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    idx = lambda x, y, z: (x > 0) * 4 + (y > 0) * 2 + (z > 0)  # noqa: E731
    faces = [[(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
             [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)], [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
             [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]]
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([idx(*v) for f in faces for v in f])
    if scheme:
        m.CreateSubdivisionSchemeAttr(scheme)
    uv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)] * 6)
    if crease:  # all 4 edges around the +x face infinitely sharp
        ring = [idx(1, -1, -1), idx(1, 1, -1), idx(1, 1, 1), idx(1, -1, 1), idx(1, -1, -1)]
        m.CreateCreaseIndicesAttr(ring)
        m.CreateCreaseLengthsAttr([5])
        m.CreateCreaseSharpnessesAttr([10.0])
    stage.Save()


def test_subdivision_surfaces(tmp_path):
    """USD's default scheme is catmullClark: the smooth surface is shown, not the cage."""
    _subdiv_cube(tmp_path / "cage.usda", scheme="none")
    _, info = convert(str(tmp_path / "cage.usda"))
    assert info["stats"]["triangles"] == 12 and info["stats"]["subdividedMeshes"] == 0

    _subdiv_cube(tmp_path / "smooth.usda")
    data, info = convert(str(tmp_path / "smooth.usda"))
    assert info["stats"]["triangles"] == 6 * 16 * 2 and info["stats"]["subdividedMeshes"] == 1
    gltf, binary = parse_glb(data)
    prim = gltf["meshes"][0]["primitives"][0]
    pos = accessor(gltf, binary, prim["attributes"]["POSITION"])
    r = np.linalg.norm(pos, axis=1)
    assert 0.8 < r.min() and r.max() < 0.95  # rounded towards the limit surface (no corner at 1.73)
    uv = accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"])
    assert uv.min() >= 0 and uv.max() <= 1

    _subdiv_cube(tmp_path / "crease.usda", crease=True)
    data, _ = convert(str(tmp_path / "crease.usda"))
    gltf, binary = parse_glb(data)
    pos = accessor(gltf, binary, gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
    # the creased face stays flat at x = 1
    assert np.isclose(pos[:, 0].max(), 1.0)
    assert (np.isclose(pos[:, 0], 1.0)).sum() >= 9


def test_curves_points_capsules(tmp_path):
    from tests import make_shader_scenes

    make_shader_scenes.geometry(tmp_path)
    data, info = convert(str(tmp_path / "geometry.usda"))
    assert not info["warnings"]
    s = info["stats"]
    assert s["curves"] == 6 and s["points"] == 72
    gltf, binary = parse_glb(data)
    meshes = {m["name"]: m for m in gltf["meshes"]}
    # cubic curves are evaluated and swept into tubes with their widths
    tube = meshes["/World/Curves_bspline_nonperiodic_cubic"]["primitives"][0]
    assert tube["mode"] == 4
    pos = accessor(gltf, binary, tube["attributes"]["POSITION"])
    nrm = accessor(gltf, binary, tube["attributes"]["NORMAL"])
    assert len(pos) % 6 == 0  # six sides per ring
    # triangles face outwards: winding agrees with the radial normals
    tri = accessor(gltf, binary, tube["indices"]).reshape(-1, 3)
    p = pos[tri]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (np.sum(fn * nrm[tri[:, 0]], axis=1) > 0).mean() > 0.99
    # pinned catmullRom reaches its end points
    cr = meshes["/World/Curves_catmullRom_pinned_cubic"]["primitives"][0]
    cpos = accessor(gltf, binary, cr["attributes"]["POSITION"])
    assert np.abs(cpos[:, 1]).min() < 0.05  # the helix starts at y = 0
    # no widths: GL lines
    assert meshes["/World/Lines"]["primitives"][0]["mode"] == 1
    # points: one instanced sphere per point, with per-point colour
    node = next(n for n in gltf["nodes"] if n.get("name") == "/World/Points")
    inst = node["extensions"]["EXT_mesh_gpu_instancing"]["attributes"]
    assert gltf["accessors"][inst["TRANSLATION"]]["count"] == 72 and "_COLOR_0" in inst
    # capsules: hemispherical ends along their axis
    cap = next(n for n in gltf["nodes"] if n.get("name") == "/World/CapsuleY")
    cpos = accessor(gltf, binary, gltf["meshes"][cap["mesh"]]["primitives"][0]["attributes"]["POSITION"])
    assert np.isclose(cpos[:, 1].max(), 0.4 + 0.3) and np.isclose(np.abs(cpos[:, 0]).max(), 0.3, atol=1e-3)
