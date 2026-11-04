import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom

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
