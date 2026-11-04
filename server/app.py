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
from .usd_to_glb import CONVERTER_VERSION, ConversionError, convert, dependency_report

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("USDV_DATA_DIR", ROOT_DIR / "data")).resolve()
WEB_DIR = ROOT_DIR / "web"
MAX_UPLOAD_MB = int(os.environ.get("USDV_MAX_UPLOAD_MB", "4096"))
MAX_AGE_HOURS = float(os.environ.get("USDV_MAX_AGE_HOURS", "48"))

STAGING = DATA_DIR / "staging"
UPLOADS = DATA_DIR / "uploads"
SCENES = DATA_DIR / "scenes"
for d in (STAGING, UPLOADS, SCENES):
    d.mkdir(parents=True, exist_ok=True)

# USD is happiest with one conversion at a time, and it bounds memory use.
_convert_lock = threading.Lock()

app = FastAPI(title="USD Viewer")


class ConvertRequest(BaseModel):
    root: str | None = None


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


def _relativize(items: list[str], base: Path) -> list[str]:
    prefix = str(base) + os.sep
    return [i[len(prefix):] if i.startswith(prefix) else i for i in items]


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

    scene_id = hashlib.sha256(f"{upload_id}:{rel}:{CONVERTER_VERSION}".encode()).hexdigest()[:24]
    scene_dir = SCENES / scene_id
    report_path = scene_dir / "report.json"
    if report_path.exists():
        os.utime(scene_dir)
        return JSONResponse(json.loads(report_path.read_text()))

    root_path = str(files_dir / rel)
    started = time.perf_counter()
    with _convert_lock:
        try:
            glb_bytes, info = convert(root_path)
        except ConversionError as exc:
            raise HTTPException(422, str(exc))
        deps = dependency_report(root_path)

    report = {
        "sceneId": scene_id,
        "uploadId": upload_id,
        "root": rel,
        "glbUrl": f"/api/scenes/{scene_id}/scene.glb",
        "glbBytes": len(glb_bytes),
        "seconds": round(time.perf_counter() - started, 3),
        "layers": _relativize(deps["layers"], files_dir),
        "unresolved": deps["unresolved"],
        **info,
    }
    # show paths relative to the upload rather than server temp paths
    prefix = str(files_dir) + os.sep
    for key in ("warnings", "compositionErrors", "unresolved"):
        report[key] = [s.replace(prefix, "") for s in report[key]]
    tmp = SCENES / f".{scene_id}.{uuid.uuid4().hex}"
    tmp.mkdir()
    (tmp / "scene.glb").write_bytes(glb_bytes)
    (tmp / "report.json").write_text(json.dumps(report))
    if scene_dir.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.rename(scene_dir)
    return report


@app.get("/api/scenes/{scene_id}/scene.glb")
def get_scene(scene_id: str):
    if not scene_id.isalnum():
        raise HTTPException(400, "Invalid scene id")
    path = SCENES / scene_id / "scene.glb"
    if not path.is_file():
        raise HTTPException(404, "Scene not found")
    return FileResponse(path, media_type="model/gltf-binary")


@app.get("/api/health")
def health():
    from pxr import Usd

    return {"ok": True, "usdVersion": ".".join(map(str, Usd.GetVersion())), "converter": CONVERTER_VERSION}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
