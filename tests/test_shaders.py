"""Shading networks compiled to GLSL for the viewer (server/mtlx_glsl.py).

These check the generated code and what the converter writes for it. Whether the
code compiles and renders like MaterialX is checked in a browser by
scripts/check_shaders.py.
"""
import os
import time

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom

from server import mtlx_glsl
from server.usd_to_glb import convert, open_stage
from tests import make_shader_scenes
from tests.glb_utils import accessor, parse_glb

pytestmark = pytest.mark.skipif(mtlx_glsl.Library.get() is None, reason="MaterialX not installed")

SHADER_TYPES = ("surfaceshader", "BSDF", "EDF", "VDF", "displacementshader", "volumeshader",
                "lightshader", "material")


def test_every_library_pattern_node_generates():
    """Every MaterialX node that computes a value (not a closure) has GLSL."""
    lib = mtlx_glsl.Library.get()
    failed = []
    count = 0
    for nd in lib.doc.getNodeDefs():
        outs = lib.outputs(nd)
        if not outs or any(o.getType() in SHADER_TYPES + ("filename",) for o in outs):
            continue
        count += 1
        gen = mtlx_glsl.Generator(lib, texture_cb=lambda info: {"texture": 0},
                                  primvar_cb=lambda name, mtype: "_usdpv0")
        try:
            for o in outs:
                gen.output_expr(mtlx_glsl.Node(nd, {}), o.getName())
        except mtlx_glsl.GenError:
            failed.append(nd.getName())
    assert count > 600
    # only pbrlib's hair helpers need shader-generator context (closure data)
    assert all("hair" in name for name in failed), failed


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    out = tmp_path_factory.mktemp("shader_scenes")
    make_shader_scenes.main(str(out))
    return out


def _materials(path):
    data, info = convert(str(path))
    gltf, binary = parse_glb(data)
    return {m["name"].rsplit("/", 1)[-1]: m for m in gltf["materials"]}, gltf, info


def test_node_scene_materials(scenes):
    mats, gltf, info = _materials(scenes / "nodes.usda")
    assert all("usdvShader" in m["extras"] for m in mats.values())
    assert set(gltf["extras"]["usdvGlsl"]) >= {"genglsl/lib/mx_math.glsl", "genglsl/lib/mx_noise.glsl"}

    prev = mats["preview_transform2d"]["extras"]["usdvShader"]
    assert prev["surface"] == "UsdPreviewSurface"
    assert prev["features"]["clearcoat"]
    assert "0.04045" in prev["main"]  # sourceColorSpace sRGB: decoded in the shader

    opbr = mats["open_pbr"]["extras"]["usdvShader"]
    assert opbr["features"]["iridescence"] and opbr["features"]["anisotropy"]
    assert "Tworld" in opbr["needs"]  # anisotropy is oriented along the tangent

    nm = mats["normalmap_tangents"]["extras"]["usdvShader"]
    assert {"Tworld", "Bworld", "Nworld"} <= set(nm["needs"]) and "normal" in nm["params"]
    tangent_prims = [p for me in gltf["meshes"] for p in me["primitives"] if "TANGENT" in p["attributes"]]
    assert tangent_prims

    pv = mats["geom_primvar"]["extras"]["usdvShader"]
    assert sorted(pv["attributes"].values()) == ["st1", "tint"]
    assert "M" in pv["needs"] and "Vworld" in pv["needs"]

    hexm = mats["hextiled_and_colorspace"]["extras"]["usdvShader"]
    assert "mx_hextiledimage_color3" in hexm["main"]
    assert "stdlib/genglsl/mx_hextiledimage.glsl" in hexm["chunks"]

    assert any("kma_mystery_color" in w and "passed through" in w for w in info["warnings"])


def test_primvar_attributes_mark_presence(scenes):
    """Primvars narrower than vec4 are padded with w = 0 (a missing one reads w = 1)."""
    data, _ = convert(str(scenes / "nodes.usda"))
    gltf, binary = parse_glb(data)
    mats = {m["name"].rsplit("/", 1)[-1]: m for m in gltf["materials"]}
    attr_names = mats["geom_primvar"]["extras"]["usdvShader"]["attributes"]
    prims = [p for me in gltf["meshes"] for p in me["primitives"]
             if any(a.upper() in p["attributes"] for a in attr_names)]
    assert len(prims) == 1
    for a, name in attr_names.items():
        arr = accessor(gltf, binary, prims[0]["attributes"][a.upper()])
        assert arr.shape[1] == 4 and np.all(arr[:, 3] == 0)
        if name == "st1":
            assert arr[:, :2].max() == pytest.approx(2.0)


def test_equal_networks_share_one_program(tmp_path):
    """Authored values go to a uniform array, so materials differing only in values share code."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "share.usda"))
    for i, color in enumerate(((1, 0, 0), (0, 0, 1))):
        n = make_shader_scenes.Net(stage, f"/World/M{i}")
        surf = n.node("surf", "ND_standard_surface_surfaceshader", specular_roughness=0.1 * (i + 1))
        make_shader_scenes.Net.set(surf, "base_color", make_shader_scenes.Gf.Vec3f(*color))
        mat = n.surface(surf)
        cube = UsdGeom.Cube.Define(stage, f"/World/C{i}")
        make_shader_scenes.UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.Save()
    mats, _, _ = _materials(tmp_path / "share.usda")
    a, b = (m["extras"]["usdvShader"] for m in mats.values())
    assert a["key"] == b["key"] and a["main"] == b["main"]
    assert a["uniforms"] != b["uniforms"]


def test_udim_atlas(scenes):
    from PIL import Image
    import io

    data, info = convert(str(scenes / "spaces.usda"))
    gltf, binary = parse_glb(data)
    mat = next(m for m in gltf["materials"] if m["name"].endswith("Udim"))
    sh = mat["extras"]["usdvShader"]
    assert "usdv_udim(" in sh["main"] and "vec2(2.0, 2.0)" in sh["main"]
    tex = gltf["textures"][sh["textures"][0]["index"]]
    view = gltf["bufferViews"][gltf["images"][tex["source"]]["bufferView"]]
    img = Image.open(io.BytesIO(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]))
    assert img.size == (64, 64)  # 2 x 2 tiles of 32 px
    px = img.convert("RGB").load()
    # tile u, v at column u, row v from the bottom; 1012 is missing (black)
    assert px[8, 56][0] > 200 and px[40, 56][1] > 200 and px[8, 8][2] > 200 and max(px[40, 8]) < 10
    assert not info["warnings"]


def test_raw_colour_images_are_not_decoded(tmp_path):
    from PIL import Image

    Image.new("RGB", (8, 8), (128, 128, 128)).save(tmp_path / "data.png")
    stage = Usd.Stage.CreateNew(str(tmp_path / "raw.usda"))
    n = make_shader_scenes.Net(stage, "/World/M")
    img = n.node("img", "ND_image_color3")
    img.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./data.png")
    img.GetInput("file").GetAttr().SetColorSpace("raw")
    n.out(img)
    surf = n.node("surf", "ND_standard_surface_surfaceshader")
    make_shader_scenes.Net.set(surf, "base_color", img)
    mat = n.surface(surf)
    cube = UsdGeom.Cube.Define(stage, "/World/C")
    make_shader_scenes.UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.Save()
    mats, _, _ = _materials(tmp_path / "raw.usda")
    assert "0.04045" not in mats["M"]["extras"]["usdvShader"]["main"]

    img.GetInput("file").GetAttr().ClearColorSpace()  # unset: 8-bit colour is sRGB ("auto")
    stage.Save()
    mats, _, _ = _materials(tmp_path / "raw.usda")
    assert "0.04045" in mats["M"]["extras"]["usdvShader"]["main"]


def test_edited_layer_is_reconverted_while_a_stage_holds_it(tmp_path):
    """The scene graph keeps stages open; their layers must not hide edits on disk."""
    path = tmp_path / "a.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Cube.Define(stage, "/A")
    stage.Save()
    held = open_stage(str(path))  # noqa: F841 - like the scene graph's stage cache
    _, info = convert(str(path))
    assert info["stats"]["meshPrims"] == 1
    time.sleep(0.01)
    path.write_text('#usda 1.0\ndef Cube "A" {}\ndef Cube "B" {}\n')
    os.utime(path)
    _, info = convert(str(path))
    assert info["stats"]["meshPrims"] == 2
