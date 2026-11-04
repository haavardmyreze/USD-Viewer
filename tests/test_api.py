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
