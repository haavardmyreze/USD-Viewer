"""Upload storage and root-layer detection."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

from pxr import Sdf

USD_EXTS = {".usd", ".usda", ".usdc", ".usdz"}
_IGNORED_PARTS = {"__MACOSX", ".DS_Store", "Thumbs.db"}
_ROOT_NAME_HINT = re.compile(r"(^|[_\-.])(root|main|scene|shot|stage|assembly|master|world|layout|set)([_\-.]|$)", re.I)


class UploadError(Exception):
    pass


def safe_relpath(name: str) -> str | None:
    """Normalise an uploaded relative path; None if it should be ignored or is unsafe."""
    name = name.replace("\\", "/").lstrip("/")
    parts = [p for p in PurePosixPath(name).parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(p in _IGNORED_PARTS or p.startswith("._") for p in parts):
        return None
    if re.match(r"^[A-Za-z]:$", parts[0]):
        return None
    return "/".join(parts)


def extract_zip(zip_path: Path, dest: Path, max_bytes: int) -> None:
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = safe_relpath(info.filename)
            if rel is None:
                continue
            total += info.file_size
            if total > max_bytes:
                raise UploadError("Archive is larger than the upload limit when extracted")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)


def hash_tree(directory: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(directory).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\1")
    return h.hexdigest()[:24]


def list_usd_files(directory: Path) -> list[str]:
    return sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in USD_EXTS
    )


def find_root_candidates(directory: Path) -> list[str]:
    """USD files that no other uploaded layer references, best guess first."""
    files = list_usd_files(directory)
    abs_files = {os.path.realpath(directory / f): f for f in files}
    referenced: set[str] = set()
    dep_count: dict[str, int] = {}

    for rel in files:
        full = str(directory / rel)
        if rel.lower().endswith(".usdz"):
            dep_count[rel] = 0
            continue
        layer = Sdf.Layer.FindOrOpen(full)
        if layer is None:
            dep_count[rel] = -1
            continue
        deps = layer.GetCompositionAssetDependencies()
        dep_count[rel] = len(deps)
        for dep in deps:
            if not dep:
                continue
            target = Sdf.ComputeAssetPathRelativeToLayer(layer, dep)
            if target:
                real = os.path.realpath(target.split("[", 1)[0] if target.endswith("]") else target)
                if real in abs_files and real != os.path.realpath(full):
                    referenced.add(abs_files[real])

    candidates = [f for f in files if f not in referenced and dep_count.get(f, 0) >= 0]
    if not candidates:
        candidates = files

    def score(rel: str):
        name = PurePosixPath(rel).name
        return (
            -1 if _ROOT_NAME_HINT.search(PurePosixPath(rel).stem) else 0,
            rel.count("/"),
            -dep_count.get(rel, 0),
            name.lower(),
        )

    return sorted(candidates, key=score)


def prune_old(directory: Path, max_age_hours: float) -> None:
    if not directory.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for child in directory.iterdir():
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
        except OSError:
            pass
