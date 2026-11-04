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
