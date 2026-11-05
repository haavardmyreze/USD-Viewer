"""HTTP service: receives USD uploads, converts them to GLB, serves the viewer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import uploads
from .scene_graph import SceneGraph, SceneGraphError
from .usd_to_glb import CONVERTER_VERSION, PURPOSES, ConversionError, convert, dependency_report

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("USDV_DATA_DIR", ROOT_DIR / "data")).resolve()
WEB_DIR = ROOT_DIR / "web"
MAX_UPLOAD_MB = int(os.environ.get("USDV_MAX_UPLOAD_MB", "4096"))
MAX_AGE_HOURS = float(os.environ.get("USDV_MAX_AGE_HOURS", "48"))
# Opening files by path reads the server's own disk, so it is only offered to a
# browser on the same machine: "auto" (loopback clients only), "1" (always), "0" (never).
LOCAL_PATHS = os.environ.get("USDV_LOCAL_PATHS", "auto").strip().lower()
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

STAGING = DATA_DIR / "staging"
UPLOADS = DATA_DIR / "uploads"
SCENES = DATA_DIR / "scenes"
for d in (STAGING, UPLOADS, SCENES):
    d.mkdir(parents=True, exist_ok=True)

# USD is happiest with one conversion at a time, and it bounds memory use.
_convert_lock = threading.Lock()

app = FastAPI(title="USD Viewer")


Variants = dict[str, dict[str, str]]


class ConvertRequest(BaseModel):
    root: str | None = None
    purpose: str = "render"
    variants: Variants | None = None  # {prim path: {variant set: selection}}


class OpenPathRequest(BaseModel):
    path: str
    root: str | None = None  # when `path` is a folder: root layer relative to it
    purpose: str = "render"
    variants: Variants | None = None


def _purpose(value: str) -> str:
    if value not in PURPOSES:
        raise HTTPException(400, f"purpose must be one of {', '.join(PURPOSES)}")
    return value


def _local_paths_allowed(request: Request) -> bool:
    if LOCAL_PATHS in ("1", "true", "yes", "on"):
        return True
    if LOCAL_PATHS in ("0", "false", "no", "off"):
        return False
    host = request.client.host if request.client else ""
    return host in _LOOPBACK or host.startswith("127.")


def _upload_dir(upload_id: str) -> Path:
    if not upload_id.isalnum():
        raise HTTPException(400, "Invalid upload id")
    path = UPLOADS / upload_id / "files"
    if not path.is_dir():
        raise HTTPException(404, "Upload not found (it may have expired); please upload again")
    return path


def _upload_summary(upload_id: str, files_dir: Path) -> dict:
    usd_files = uploads.list_usd_files(files_dir)
    candidates = uploads.find_root_candidates(files_dir)
    file_count = sum(1 for p in files_dir.rglob("*") if p.is_file())
    return {
        "uploadId": upload_id,
        "fileCount": file_count,
        "usdFiles": usd_files,
        "rootCandidates": candidates,
        "root": candidates[0] if candidates else None,
    }


def _store_upload(staged: Path) -> dict:
    files_dir = staged / "files"
    # expand any zip archives in place
    for zip_path in [p for p in files_dir.rglob("*") if p.suffix.lower() == ".zip"]:
        try:
            uploads.extract_zip(zip_path, zip_path.parent, MAX_UPLOAD_MB * 1024 * 1024)
        except uploads.UploadError as exc:
            raise HTTPException(413, str(exc))
        except Exception as exc:
            raise HTTPException(400, f"Could not read {zip_path.name}: {exc}")
        zip_path.unlink()

    if not uploads.list_usd_files(files_dir):
        raise HTTPException(400, "No USD files (.usd, .usda, .usdc, .usdz) found in the upload")

    upload_id = uploads.hash_tree(files_dir)
    final = UPLOADS / upload_id
    if final.exists():
        shutil.rmtree(staged, ignore_errors=True)
        os.utime(final)
    else:
        staged.rename(final)
    return _upload_summary(upload_id, final / "files")


@app.post("/api/uploads")
async def create_upload(request: Request):
    uploads.prune_old(UPLOADS, MAX_AGE_HOURS)
    uploads.prune_old(SCENES, MAX_AGE_HOURS)
    uploads.prune_old(STAGING, 2)

    form = await request.form(max_files=200_000, max_fields=200_000)
    files = form.getlist("files")
    paths = form.getlist("paths")
    if not files:
        raise HTTPException(400, "No files received")
    if paths and len(paths) != len(files):
        raise HTTPException(400, "'paths' must have one entry per file")

    staged = STAGING / uuid.uuid4().hex
    files_dir = staged / "files"
    files_dir.mkdir(parents=True)
    limit = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    try:
        for i, upload in enumerate(files):
            name = paths[i] if paths else upload.filename
            rel = uploads.safe_relpath(str(name or ""))
            if rel is None:
                continue
            target = files_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                while chunk := await upload.read(1 << 20):
                    total += len(chunk)
                    if total > limit:
                        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
        return await run_in_threadpool(_store_upload, staged)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    finally:
        await form.close()


@app.get("/api/uploads/{upload_id}")
def get_upload(upload_id: str):
    return _upload_summary(upload_id, _upload_dir(upload_id))


def _prefixes(base: Path) -> list[str]:
    # USD reports Windows paths with forward slashes, Python with backslashes
    return sorted({str(base) + os.sep, base.as_posix() + "/"}, key=len, reverse=True)


def _relativize(items: list[str], base: Path) -> list[str]:
    out = []
    for item in items:
        for prefix in _prefixes(base):
            if item.startswith(prefix):
                item = item[len(prefix):]
                break
        out.append(item)
    return out


def _relpath(path: str, base: Path) -> str:
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return path


def _strip_base(text: str, base: Path) -> str:
    for prefix in _prefixes(base):
        text = text.replace(prefix, "")
    return text


def _variant_key(variants: Variants | None) -> str:
    """A stable text form of variant overrides, for scene ids ("" when none)."""
    return json.dumps(variants, sort_keys=True, separators=(",", ":")) if variants else ""


def _cached_report(scene_id: str) -> dict | None:
    scene_dir = SCENES / scene_id
    report_path = scene_dir / "report.json"
    if report_path.exists():
        os.utime(scene_dir)
        return json.loads(report_path.read_text())
    return None


def _convert_and_store(scene_id: str, root_path: str, purpose: str, base: Path | None, extra: dict,
                       deps: dict | None = None, variants: Variants | None = None) -> dict:
    """Convert root_path, write the GLB + report under SCENES/scene_id."""
    started = time.perf_counter()
    with _convert_lock:
        try:
            glb_bytes, info = convert(root_path, purpose, variants)
        except ConversionError as exc:
            raise HTTPException(422, str(exc))
        if deps is None:
            deps = dependency_report(root_path)

    report = {
        "sceneId": scene_id,
        **extra,
        "glbUrl": f"/api/scenes/{scene_id}/scene.glb",
        "glbBytes": len(glb_bytes),
        "seconds": round(time.perf_counter() - started, 3),
        "layers": deps["layers"],
        "unresolved": deps["unresolved"],
        **info,
    }
    if base is not None:
        # show paths relative to the upload rather than server storage paths
        report["layers"] = _relativize(report["layers"], base)
        for key in ("warnings", "compositionErrors"):
            report[key] = [_strip_base(s, base) for s in report[key]]
        # missing files outside the upload (e.g. ../../assets) are shown relative to it,
        # which matches how the scene authored them
        report["unresolved"] = [_relpath(u, base) for u in report["unresolved"]]
    tmp = SCENES / f".{scene_id}.{uuid.uuid4().hex}"
    tmp.mkdir()
    (tmp / "scene.glb").write_bytes(glb_bytes)
    (tmp / "report.json").write_text(json.dumps(report))
    if (SCENES / scene_id).exists():
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.rename(SCENES / scene_id)
    return report


@app.post("/api/uploads/{upload_id}/convert")
def convert_upload(upload_id: str, body: ConvertRequest):
    files_dir = _upload_dir(upload_id)
    root = body.root
    if root is None:
        candidates = uploads.find_root_candidates(files_dir)
        if not candidates:
            raise HTTPException(400, "No root layer found")
        root = candidates[0]
    rel = uploads.safe_relpath(root)
    if rel is None or rel not in uploads.list_usd_files(files_dir):
        raise HTTPException(400, f"'{root}' is not a USD file in this upload")

    purpose = _purpose(body.purpose)
    vkey = _variant_key(body.variants)
    scene_id = hashlib.sha256(f"{upload_id}:{rel}:{purpose}:{vkey}:{CONVERTER_VERSION}".encode()).hexdigest()[:24]
    cached = _cached_report(scene_id)
    if cached is not None:
        return JSONResponse(cached)
    extra = {"source": "upload", "uploadId": upload_id, "root": rel}
    return _convert_and_store(scene_id, str(files_dir / rel), purpose, files_dir, extra, variants=body.variants)


def _fingerprint(paths: list[str]) -> str:
    """Changes whenever any layer or asset of the scene is edited on disk."""
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        h.update(p.encode("utf-8", "surrogatepass") + b"\0")
        try:
            st = os.stat(p.split("[", 1)[0] if p.endswith("]") else p)
            h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


@app.post("/api/open")
def open_path(body: OpenPathRequest, request: Request):
    """Convert a USD file (or folder) straight from this machine's disk.

    References are resolved against the real file system, so a layout that
    references ../../assets works without uploading the whole project.
    """
    if not _local_paths_allowed(request):
        raise HTTPException(403, "Opening files by path is only available from the machine running the server")
    purpose = _purpose(body.purpose)
    raw = body.path.strip().strip('"').strip("'")
    if not raw:
        raise HTTPException(400, "Enter a path to a USD file or folder")
    target = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not target.is_absolute():
        raise HTTPException(400, "Use an absolute path (for example V:/project/shot.usda)")
    if target.is_dir():
        candidates = uploads.find_root_candidates(target)
        if not candidates:
            raise HTTPException(404, f"No USD files found in {target}")
        rel = body.root or candidates[0]
        if uploads.safe_relpath(rel) is None:
            raise HTTPException(400, f"Invalid root: {rel}")
        root_path = (target / rel).resolve()
    elif target.is_file():
        root_path = target.resolve()
        candidates = []
    else:
        raise HTTPException(404, f"Not found: {target}")
    if root_path.suffix.lower() not in uploads.USD_EXTS or not root_path.is_file():
        raise HTTPException(400, f"Not a USD file: {root_path}")

    deps = dependency_report(str(root_path))
    fp = _fingerprint([str(root_path), *deps["layers"], *deps["assets"]])
    vkey = _variant_key(body.variants)
    scene_id = hashlib.sha256(
        f"path:{root_path}:{purpose}:{vkey}:{fp}:{CONVERTER_VERSION}".encode()).hexdigest()[:24]
    cached = _cached_report(scene_id)
    if cached is not None:
        return JSONResponse(cached)
    extra = {
        "source": "path",
        "path": root_path.as_posix(),
        "root": root_path.relative_to(target.resolve()).as_posix() if target.is_dir() else root_path.name,
        "folder": target.as_posix() if target.is_dir() else None,
        "rootCandidates": candidates,
    }
    return _convert_and_store(scene_id, str(root_path), purpose, None, extra, deps, body.variants)


@app.get("/api/scenes/{scene_id}/scene.glb")
def get_scene(scene_id: str):
    if not scene_id.isalnum():
        raise HTTPException(400, "Invalid scene id")
    path = SCENES / scene_id / "scene.glb"
    if not path.is_file():
        raise HTTPException(404, "Scene not found")
    return FileResponse(path, media_type="model/gltf-binary")


def _scene_graph(scene_id: str) -> SceneGraph:
    """The composed stage behind a converted scene, for the scene graph panel."""
    if not scene_id.isalnum():
        raise HTTPException(400, "Invalid scene id")
    report_path = SCENES / scene_id / "report.json"
    if not report_path.is_file():
        raise HTTPException(404, "Scene not found (it may have expired); open it again")
    report = json.loads(report_path.read_text())
    display = None
    if report.get("source") == "path":
        root_path = report["path"]
    elif report.get("source") == "upload":
        base = _upload_dir(report["uploadId"])
        root_path = str(base / report["root"])
        display = lambda p: _relpath(p, base)  # noqa: E731
    else:
        raise HTTPException(404, "This scene has no USD stage")
    try:
        return SceneGraph(scene_id, root_path, display, report.get("variants") or None)
    except SceneGraphError as exc:
        raise HTTPException(exc.status, exc.message)


def _graph_call(scene_id: str, fn):
    graph = _scene_graph(scene_id)
    try:
        return fn(graph)
    except SceneGraphError as exc:
        raise HTTPException(exc.status, exc.message)


@app.get("/api/scenes/{scene_id}/prims")
def scene_children(scene_id: str, path: str = "/"):
    """One level of the prim hierarchy: the children of `path`."""
    return _graph_call(scene_id, lambda g: g.children(path))


@app.get("/api/scenes/{scene_id}/subtree")
def scene_subtree(scene_id: str, path: str = "/"):
    """Every level beneath `path`, breadth first, for expand-all."""
    return _graph_call(scene_id, lambda g: g.subtree(path))


@app.get("/api/scenes/{scene_id}/reveal")
def scene_reveal(scene_id: str, path: str):
    """The levels from the root down to `path`, to open the tree onto a picked prim."""
    return _graph_call(scene_id, lambda g: g.reveal(path))


@app.get("/api/scenes/{scene_id}/find")
def scene_find(scene_id: str, q: str = ""):
    """Prims whose name contains `q`, anywhere in the stage."""
    return _graph_call(scene_id, lambda g: g.find(q))


@app.get("/api/scenes/{scene_id}/prim")
def scene_prim(scene_id: str, path: str):
    """One prim's detail: variants, schemas, property counts and opinions."""
    return _graph_call(scene_id, lambda g: g.detail(path))


@app.get("/api/scenes/{scene_id}/selection")
def scene_selection(scene_id: str, path: str, instance: int | None = None):
    """A selection's pivot (frame and point) and the other copies of its instance."""
    return _graph_call(scene_id, lambda g: g.selection(path, instance))


@app.get("/api/scenes/{scene_id}/info")
def scene_info(scene_id: str, path: str, instance: int | None = None, proto: str | None = None):
    """The info card for a clicked prim: the asset it belongs to and what placed it,
    each with its published layers' records, and the materials and textures it uses."""
    return _graph_call(scene_id, lambda g: g.info(path, instance, proto))


@app.get("/api/health")
def health(request: Request):
    from pxr import Usd

    return {
        "ok": True,
        "usdVersion": ".".join(map(str, Usd.GetVersion())),
        "converter": CONVERTER_VERSION,
        "localPaths": _local_paths_allowed(request),
    }


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """The viewer's own files change with every update: make browsers check them
    (a cheap 304 via ETag) instead of running yesterday's JavaScript from cache."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/") and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
