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
    assert find_root_candidates(scene_dir) == ["shot.usda"]


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
    assert "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr
    assert pbr["roughnessFactor"] == 1.0 and pbr["metallicFactor"] == 0.0
    tri = m["extras"]["usdvTriplanar"]
    assert tri["space"] == "world"
    assert tri["scale"] == pytest.approx(50.0)  # 0.5 per cm -> 50 per metre
    # the oversized texture was downscaled and re-encoded; the ORM map is used as-is
    images = {i["name"]: i for i in gltf["images"]}
    assert images["bc.png"]["mimeType"] == "image/jpeg"
    assert images["orm.png"]["mimeType"] == "image/png"
    assert any("hex-tiled" in w for w in info["warnings"])


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
