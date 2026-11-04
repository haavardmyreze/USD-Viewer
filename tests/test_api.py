import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from tests.glb_utils import parse_glb


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _multipart(scene_dir: Path, prefix: str = ""):
    files, paths = [], []
    for p in sorted(scene_dir.rglob("*")):
        if p.is_file():
            rel = prefix + p.relative_to(scene_dir).as_posix()
            files.append(("files", (p.name, p.read_bytes())))
            paths.append(("paths", rel))
    return files, paths


def test_upload_convert_and_download(client, scene_dir):
    files, paths = _multipart(scene_dir, prefix="my_scene/")
    res = client.post("/api/uploads", files=files, data={"paths": [p for _, p in paths]})
    assert res.status_code == 200, res.text
    summary = res.json()
    assert summary["root"] == "my_scene/shot.usda"
    assert summary["fileCount"] == 7

    res = client.post(f"/api/uploads/{summary['uploadId']}/convert", json={})
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["root"] == "my_scene/shot.usda"
    assert report["unresolved"] == ["my_scene/assets/does_not_exist.usda"]
    assert "my_scene/layers/layout.usda" in report["layers"]
    assert all("/tmp" not in s for s in report["compositionErrors"])

    glb = client.get(report["glbUrl"])
    assert glb.status_code == 200
    assert glb.headers["content-type"] == "model/gltf-binary"
    parse_glb(glb.content)

    # second request is served from the cache
    again = client.post(f"/api/uploads/{summary['uploadId']}/convert", json={"root": "my_scene/shot.usda"})
    assert again.json()["sceneId"] == report["sceneId"]

    # same content uploaded again maps to the same upload id
    res2 = client.post("/api/uploads", files=files, data={"paths": [p for _, p in paths]})
    assert res2.json()["uploadId"] == summary["uploadId"]


def test_zip_upload(client, scene_dir):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in scene_dir.rglob("*"):
            if p.is_file():
                zf.write(p, "proj/" + p.relative_to(scene_dir).as_posix())
        zf.writestr("../evil.usda", "#usda 1.0\n")          # zip-slip attempt is dropped
        zf.writestr("__MACOSX/proj/._shot.usda", "junk")    # macOS metadata is dropped
    res = client.post("/api/uploads", files=[("files", ("scene.zip", buf.getvalue()))],
                      data={"paths": ["scene.zip"]})
    assert res.status_code == 200, res.text
    summary = res.json()
    assert summary["root"] == "proj/shot.usda"
    assert all("evil" not in f and "MACOSX" not in f for f in summary["usdFiles"])


def test_choose_a_different_root(client, scene_dir):
    files, paths = _multipart(scene_dir)
    summary = client.post("/api/uploads", files=files, data={"paths": [p for _, p in paths]}).json()
    res = client.post(f"/api/uploads/{summary['uploadId']}/convert", json={"root": "assets/tower.usda"})
    assert res.status_code == 200
    assert res.json()["stats"]["meshPrims"] == 2


def test_rejects_bad_input(client):
    res = client.post("/api/uploads", files=[("files", ("notes.txt", b"hello"))], data={"paths": ["notes.txt"]})
    assert res.status_code == 400
    assert client.post("/api/uploads/doesnotexist/convert", json={}).status_code == 404
    assert client.post("/api/uploads/../../etc/convert", json={}).status_code != 200


def test_root_must_be_in_upload(client, scene_dir):
    files, paths = _multipart(scene_dir)
    summary = client.post("/api/uploads", files=files, data={"paths": [p for _, p in paths]}).json()
    res = client.post(f"/api/uploads/{summary['uploadId']}/convert", json={"root": "../../etc/passwd"})
    assert res.status_code == 400


def test_health_and_static(client):
    assert client.get("/api/health").json()["ok"] is True
    assert "USD Viewer" in client.get("/").text


def test_open_path_is_local_only(client, scene_dir):
    # TestClient is not a loopback client, so the default "auto" mode refuses it
    res = client.post("/api/open", json={"path": str(scene_dir / "shot.usda")})
    assert res.status_code == 403
    assert client.get("/api/health").json()["localPaths"] is False


def test_open_path(client, scene_dir, monkeypatch):
    import server.app as app_module

    monkeypatch.setattr(app_module, "LOCAL_PATHS", "1")
    # references outside the opened file's folder resolve, because nothing is copied
    res = client.post("/api/open", json={"path": str(scene_dir / "layers" / "layout.usda")})
    assert res.status_code == 200, res.text
    report = res.json()
    assert report["source"] == "path"
    assert report["stats"]["meshPrims"] > 0
    assert client.get(report["glbUrl"]).status_code == 200
    again = client.post("/api/open", json={"path": str(scene_dir / "layers" / "layout.usda")}).json()
    assert again["sceneId"] == report["sceneId"]

    # a folder opens its best root layer; purpose changes the scene id
    folder = client.post("/api/open", json={"path": str(scene_dir), "purpose": "proxy"}).json()
    assert folder["root"] == "shot.usda" and folder["purpose"] == "proxy"
    assert folder["rootCandidates"][0] == "shot.usda"

    assert client.post("/api/open", json={"path": "relative/file.usda"}).status_code == 400
    assert client.post("/api/open", json={"path": str(scene_dir / "nope.usda")}).status_code == 404
    assert client.post("/api/open", json={"path": str(scene_dir), "purpose": "guide"}).status_code == 400
    assert client.post("/api/open", json={"path": str(scene_dir), "root": "../../etc/passwd"}).status_code == 400


def _uploaded(client, scene_dir, **body):
    files, paths = _multipart(scene_dir, prefix="my_scene/")
    summary = client.post("/api/uploads", files=files, data={"paths": [p for _, p in paths]}).json()
    res = client.post(f"/api/uploads/{summary['uploadId']}/convert", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_scene_graph_api(client, scene_dir):
    report = _uploaded(client, scene_dir)
    base = f"/api/scenes/{report['sceneId']}"

    top = client.get(f"{base}/prims", params={"path": "/"}).json()
    assert top["defaultPrim"] == "/World"
    assert [p["name"] for p in top["children"]] == ["World"]

    world = {p["name"]: p for p in client.get(f"{base}/prims", params={"path": "/World"}).json()["children"]}
    tower = world["Tower"]
    assert tower["arcs"] == ["payload"] and tower["variants"] == {"color": "sand"}
    assert "unloaded" not in tower  # payloads are loaded, like the conversion
    assert world["Crates"]["childCount"] == 9

    subtree = client.get(f"{base}/subtree", params={"path": "/World"}).json()
    assert not subtree["truncated"] and subtree["prims"] > 10

    # opening the tree onto a prim picked in 3D: every level down to its parent
    reveal = client.get(f"{base}/reveal", params={"path": "/World/Crates/crate_2_2/geo"}).json()
    assert [lvl["primPath"] for lvl in reveal["levels"]] == ["/", "/World", "/World/Crates", "/World/Crates/crate_2_2"]
    proxy = reveal["levels"][-1]["children"][0]
    assert proxy["instanceProxy"] is True

    detail = client.get(f"{base}/prim", params={"path": "/World/Tower"}).json()
    assert detail["variantSets"] == [{"name": "color", "selection": "sand", "options": ["sand", "stone"]}]
    layers = [(o["layerPath"], o["arc"]) for o in detail["opinions"]]
    # strongest first; paths relative to the upload rather than server storage
    assert ("my_scene/layers/layout.usda", "sublayer") in layers
    assert ("my_scene/assets/tower.usda", "payload") in layers

    # find walks the whole stage, so it reaches prims nobody has expanded
    found = client.get(f"{base}/find", params={"q": "CRATE_2"}).json()
    paths = [p["path"] for p in found["results"]]
    assert paths[:3] == ["/World/Crates/crate_2_0", "/World/Crates/crate_2_1", "/World/Crates/crate_2_2"]
    assert not found["truncated"]
    assert client.get(f"{base}/find", params={"q": ""}).json()["results"] == []

    assert client.get(f"{base}/prim", params={"path": "/World/Nope"}).status_code == 404
    assert client.get(f"{base}/prim", params={"path": "not a path"}).status_code == 400
    assert client.get("/api/scenes/doesnotexist/prims").status_code == 404


def test_variants_recompose_the_scene(client, scene_dir):
    plain = _uploaded(client, scene_dir)
    sand = _uploaded(client, scene_dir, variants={"/World/Tower": {"color": "stone"}})
    assert sand["sceneId"] != plain["sceneId"]
    assert sand["variants"] == {"/World/Tower": {"color": "stone"}}
    assert plain["defaultVariants"] is None  # /World has no variant sets
    # the scene graph shows the stage as converted, with the override applied
    detail = client.get(f"/api/scenes/{sand['sceneId']}/prim", params={"path": "/World/Tower"}).json()
    assert detail["variantSets"][0]["selection"] == "stone"
    again = _uploaded(client, scene_dir, variants={"/World/Tower": {"color": "stone"}})
    assert again["sceneId"] == sand["sceneId"]  # cached per selection


def test_static_files_are_revalidated(client):
    assert client.get("/js/main.js").headers["cache-control"] == "no-cache"
