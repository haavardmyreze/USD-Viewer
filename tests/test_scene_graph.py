from pxr import Usd, UsdGeom

from server.scene_graph import SceneGraph


def _instancing_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Asset")
    UsdGeom.Mesh.Define(stage, "/Leaf/mesh")
    for i in range(2):  # nested: every tree holds two instanced leaves
        leaf = stage.DefinePrim(f"/Asset/Leaves/l{i}")
        leaf.GetReferences().AddInternalReference("/Leaf")
        leaf.SetInstanceable(True)
    for i in range(3):
        tree = UsdGeom.Xform.Define(stage, f"/World/Tree{i}").GetPrim()
        tree.GetReferences().AddInternalReference("/Asset")
        tree.SetInstanceable(True)
    loose = stage.DefinePrim("/World/Loose")
    loose.GetReferences().AddInternalReference("/Leaf")
    loose.SetInstanceable(True)

    pivoted = UsdGeom.XformCommonAPI(UsdGeom.Xform.Define(stage, "/World/Pivoted"))
    pivoted.SetTranslate((10, 0, 0))
    pivoted.SetPivot((1, 2, 3))
    pivoted.SetRotate((0, 0, 90))

    pi = UsdGeom.PointInstancer.Define(stage, "/World/Scatter")
    pi.CreatePositionsAttr([(0, 0, 0), (5, 0, 0)])
    pi.CreateProtoIndicesAttr([0, 0])
    UsdGeom.Cube.Define(stage, "/World/Scatter/protos/cube")
    pi.CreatePrototypesRel().AddTarget("/World/Scatter/protos/cube")
    stage.Save()


def test_instance_peers_follow_nested_instancing(tmp_path):
    _instancing_stage(tmp_path / "trees.usda")
    graph = SceneGraph("trees-peers", str(tmp_path / "trees.usda"))

    trees = graph.selection("/World/Tree0")
    assert trees["instances"] == ["/World/Tree1", "/World/Tree2"] and not trees["truncated"]

    # a leaf inside a tree: the same leaf in the other trees, the other leaf slot
    # (the same prototype) and copies placed outside any tree
    leaf = graph.selection("/World/Tree0/Leaves/l1/mesh")["instances"]
    assert "/World/Tree0/Leaves/l1/mesh" not in leaf
    assert {"/World/Tree0/Leaves/l0/mesh", "/World/Tree2/Leaves/l1/mesh", "/World/Loose/mesh"} <= set(leaf)
    assert all(p.endswith("/mesh") for p in leaf) and len(leaf) == 8

    assert graph.selection("/World/Pivoted")["instances"] == []


def test_pivot_is_the_authored_pivot_in_the_prims_frame(tmp_path):
    _instancing_stage(tmp_path / "trees.usda")
    graph = SceneGraph("trees-pivot", str(tmp_path / "trees.usda"))

    pivot = graph.selection("/World/Pivoted")["pivot"]
    assert pivot["point"] == [1.0, 2.0, 3.0]
    m = pivot["matrix"]
    x, y, z = pivot["point"]
    world = [x * m[0] + y * m[4] + z * m[8] + m[12], x * m[1] + y * m[5] + z * m[9] + m[13],
             x * m[2] + y * m[6] + z * m[10] + m[14]]
    assert [round(v, 6) for v in world] == [11.0, 2.0, 3.0]  # translate + pivot, in the parent

    # a PointInstancer instance: that instance's frame
    assert graph.selection("/World/Scatter", 1)["pivot"]["matrix"][12:15] == [5.0, 0.0, 0.0]
    assert graph.selection("/World/Scatter", 7)["pivot"] is None
    assert graph.selection("/")["pivot"] is None


def _pipeline_project(root):
    """A set placing an asset (assembly + model and lookdev blocks) and a material
    from a library, each layer stamped with a publish record as the publisher writes it."""
    from pxr import Sdf, UsdShade

    def record(layer, artist, status, when, **extra):
        layer.customLayerData = {"artist": artist, "status": status, "export_datetime_unix": str(when),
                                 "hip_file": f"workfile_{artist}_v001.hip", "rop_path": "/stage/rop1", **extra}

    asset = root / "assets" / "prop-vase"
    lib = root / "assets" / "mtl-clay"
    tex = lib / "tex"
    tex.mkdir(parents=True)
    asset.mkdir(parents=True)
    (tex / "mtl-clay_bc_2k.png").write_bytes(b"")

    clay = Usd.Stage.CreateNew(str(lib / "mtl-clay.usda"))
    material = UsdShade.Material.Define(clay, "/Clay")
    shader = UsdShade.Shader.Define(clay, "/Clay/image")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./tex/mtl-clay_bc_2k.png")
    shader.CreateInput("normal", Sdf.ValueTypeNames.Asset).Set("./tex/mtl-clay_n_2k.png")  # not on disk
    clay.SetDefaultPrim(material.GetPrim())
    record(clay.GetRootLayer(), "kari", "locked", 1_700_000_000)
    clay.Save()

    model = Usd.Stage.CreateNew(str(asset / "prop-vase_model.usdc"))
    UsdGeom.Mesh.Define(model, "/Vase/geo/body")
    record(model.GetRootLayer(), "ola", "production_ready", 1_700_000_100, comment="cleaned up the rim")
    model.Save()

    lookdev = Usd.Stage.CreateNew(str(asset / "prop-vase_lookdev.usda"))
    lookdev.OverridePrim("/Vase/mtl/clay").GetReferences().AddReference("../mtl-clay/mtl-clay.usda")
    body = lookdev.OverridePrim("/Vase/geo/body")
    UsdShade.MaterialBindingAPI.Apply(body).Bind(UsdShade.Material(lookdev.GetPrimAtPath("/Vase/mtl/clay")))
    record(lookdev.GetRootLayer(), "kari", "placeholder", 1_700_000_300)
    lookdev.Save()

    assembly = Usd.Stage.CreateNew(str(asset / "prop-vase.usda"))
    assembly.GetRootLayer().subLayerPaths = ["./prop-vase_lookdev.usda", "./prop-vase_model.usdc"]
    assembly.SetDefaultPrim(assembly.DefinePrim("/Vase"))
    record(assembly.GetRootLayer(), "ola", "production_ready", 1_700_000_200)
    assembly.Save()

    set_dir = root / "sets" / "set-shop"
    set_dir.mkdir(parents=True)
    layout = Usd.Stage.CreateNew(str(set_dir / "set-shop_layout.usda"))
    for i in range(2):
        layout.DefinePrim(f"/World/Vase{i}").GetReferences().AddReference("../../assets/prop-vase/prop-vase.usda")
    UsdGeom.Mesh.Define(layout, "/World/Floor")
    record(layout.GetRootLayer(), "per", "placeholder", 1_700_000_400)
    layout.Save()

    shop = Usd.Stage.CreateNew(str(set_dir / "set-shop.usda"))
    shop.GetRootLayer().subLayerPaths = ["./set-shop_layout.usda"]
    shop.Save()  # no publish record
    return set_dir / "set-shop.usda"


def test_info_reads_the_publish_records_from_the_inside_out(tmp_path):
    graph = SceneGraph("shop-info", str(_pipeline_project(tmp_path)))
    info = graph.info("/World/Vase1/geo/body")
    asset, shop = info["entities"]

    assert asset["name"] == "prop-vase" and asset["tier"] == "asset" and asset["category"] == "prop"
    assert asset["primPath"] == "/World/Vase1" and asset["introducedBy"] == "set-shop_layout.usda"
    assert [layer["block"] or layer["role"] for layer in asset["layers"]] == ["assembly", "lookdev", "model"]
    assert asset["status"] == "placeholder"  # only as finished as its least finished layer
    assert asset["artists"] == ["kari", "ola"]  # most recent publisher first
    assert asset["lastPublished"] == 1_700_000_300_000
    model = asset["layers"][2]["record"]
    assert model == {"status": "production_ready", "artist": "ola", "comment": "cleaned up the rim",
                     "hipFile": "workfile_ola_v001.hip", "ropPath": "/stage/rop1", "exportedAt": 1_700_000_100_000}

    assert shop["name"] == "set-shop" and shop["arc"] == "root"
    assert [layer["name"] for layer in shop["layers"]] == ["set-shop.usda", "set-shop_layout.usda"]
    assert any("set-shop.usda" in issue["text"] for issue in shop["issues"])  # no record

    (material,) = info["materials"]
    assert material["entity"]["name"] == "mtl-clay" and material["entity"]["status"] == "locked"
    textures = {t["name"]: t for t in material["textures"]}
    assert textures["mtl-clay_bc_2k.png"]["exists"] and textures["mtl-clay_bc_2k.png"]["channel"] == "base colour"
    assert not textures["mtl-clay_n_2k.png"]["exists"]


def test_info_for_a_prim_authored_in_the_scene_and_an_instancer(tmp_path):
    graph = SceneGraph("shop-floor", str(_pipeline_project(tmp_path)))
    (shop,) = graph.info("/World/Floor")["entities"]
    assert shop["name"] == "set-shop"

    _instancing_stage(tmp_path / "trees.usda")
    trees = SceneGraph("trees-info", str(tmp_path / "trees.usda"))
    info = trees.info("/World/Scatter", instance=1)
    assert info["target"] == "/World/Scatter/protos/cube"
    assert info["copies"] == {"kind": "pointinstancer", "same": 2, "total": 2,
                              "instancer": "/World/Scatter", "prototype": "cube"}
    assert trees.info("/World/Tree0")["copies"]["same"] == 3
