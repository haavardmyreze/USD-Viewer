"""Convert a composed USD stage into a single GLB file.

The real OpenUSD library (usd-core) opens the root layer, so every reference,
payload, sublayer, variant selection and instance is resolved exactly as USD
defines it. We then walk the composed stage and write plain glTF that three.js
can load reliably.

Supported in this first pass:
  * UsdGeomMesh (any polygon counts, holes, left-handed orientation)
  * Cube, Sphere, Cylinder, Cone, Plane gprims
  * normals / st / displayColor primvars at any interpolation (indexed or not)
  * UsdPreviewSurface materials incl. textures, GeomSubset material bindings
  * native instancing (shared glTF meshes)
  * PointInstancer (incl. nested), written as EXT_mesh_gpu_instancing
  * render / proxy purpose selection
  * upAxis + metersPerUnit (output is Y-up, metres)
"""

from __future__ import annotations

import io
import math
import os
import threading
import zipfile
from collections import Counter

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

from . import exr
from .glb import GlbBuilder

CONVERTER_VERSION = "6"

# Level of detail: meshes with at least this many triangles get simplified
# versions (each ~1/4 of the previous) that the viewer switches to by screen size.
LOD_ENABLED = os.environ.get("USDV_LODS", "1") not in ("0", "false", "no", "off")
LOD_MIN_TRIANGLES = int(os.environ.get("USDV_LOD_MIN_TRIANGLES", "20000"))
LOD_LEVELS = 3

# textures larger than this (longest side, px) are downscaled; 0 disables
MAX_TEXTURE_SIZE = int(os.environ.get("USDV_MAX_TEXTURE_SIZE", "2048"))

PURPOSES = ("render", "proxy")

# glTF sampler wrap modes
REPEAT, CLAMP, MIRROR = 10497, 33071, 33648
_COLOR_PURPOSES = ("baseColor", "emissive")
_WRAP = {"repeat": REPEAT, "mirror": MIRROR, "clamp": CLAMP, "black": CLAMP}

_DEFAULT_MATERIAL = {
    "name": "default",
    "pbrMetallicRoughness": {
        "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.6,
    },
}

_SKIP_SUBTREE_TYPES = {"Material", "NodeGraph", "Shader", "GeomSubset"}
_GPRIM_TYPES = {"Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Plane"}
_KNOWN_NON_GEOM = {
    "Xform", "Scope", "", "Camera", "SkelRoot", "Skeleton", "SkelAnimation",
    "DistantLight", "DomeLight", "SphereLight", "RectLight", "DiskLight",
    "CylinderLight", "PortalLight", "GeometryLight", "RenderSettings",
    "RenderProduct", "RenderVar", "PhysicsScene", "Material", "Shader",
    "NodeGraph", "GeomSubset", "CollectionAPI",
}


def _display(path: str) -> str:
    """Prim path for messages: hide USD's internal instancing prototype roots."""
    import re
    return re.sub(r"^/__Prototype_\d+", "<instanced>", path)


class ConversionError(Exception):
    pass


class _Report:
    def __init__(self) -> None:
        self.warnings: Counter[str] = Counter()
        self.skipped_types: Counter[str] = Counter()

    def warn(self, msg: str) -> None:
        self.warnings[msg] += 1

    def as_list(self) -> list[str]:
        out = []
        for msg, n in self.warnings.most_common():
            out.append(msg if n == 1 else f"{msg} (x{n})")
        return out


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _triangulate(counts: np.ndarray):
    """Fan-triangulate polygons.

    Returns (corners, face_of_tri): corners is (T, 3) indices into
    faceVertexIndices, face_of_tri is (T,) the source face of each triangle.
    """
    counts = counts.astype(np.int64)
    offsets = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        offsets[1:] = np.cumsum(counts)[:-1]
    ntri = np.maximum(counts - 2, 0)
    total = int(ntri.sum())
    face_of_tri = np.repeat(np.arange(len(counts)), ntri)
    tri_start = np.repeat(np.cumsum(ntri) - ntri, ntri)
    k = np.arange(total, dtype=np.int64) - tri_start + 1
    base = offsets[face_of_tri]
    corners = np.stack([base, base + k, base + k + 1], axis=1)
    return corners, face_of_tri


def _expand(values, interp, n_points, n_corners, n_faces, point_idx, corner_idx, face_idx):
    """Expand a primvar to one value per output corner. Returns None on mismatch."""
    if values is None or len(values) == 0:
        return None
    if interp in ("vertex", "varying"):
        return values[point_idx] if len(values) >= n_points else None
    if interp == "faceVarying":
        return values[corner_idx] if len(values) >= n_corners else None
    if interp == "uniform":
        return values[face_idx] if len(values) >= n_faces else None
    # constant
    return np.broadcast_to(values[0], (len(point_idx),) + values.shape[1:])


def _smooth_normals(points, tri_points):
    p0, p1, p2 = (points[tri_points[:, i]] for i in range(3))
    fn = np.cross(p1 - p0, p2 - p0)  # area weighted
    acc = np.zeros_like(points, dtype=np.float64)
    for i in range(3):
        np.add.at(acc, tri_points[:, i], fn)
    return _normalize(acc)


def _quat_from_rotation(r):
    """(N, 3, 3) column-vector rotation matrices -> (N, 4) quaternions (x, y, z, w)."""
    m00, m01, m02 = r[:, 0, 0], r[:, 0, 1], r[:, 0, 2]
    m10, m11, m12 = r[:, 1, 0], r[:, 1, 1], r[:, 1, 2]
    m20, m21, m22 = r[:, 2, 0], r[:, 2, 1], r[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((len(r), 4))
    with np.errstate(divide="ignore", invalid="ignore"):
        c0 = tr > 0
        c1 = ~c0 & (m00 > m11) & (m00 > m22)
        c2 = ~c0 & ~c1 & (m11 > m22)
        c3 = ~c0 & ~c1 & ~c2
        s = np.sqrt(np.maximum(tr + 1.0, 0)) * 2
        q[c0] = np.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s], 1)[c0]
        s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0)) * 2
        q[c1] = np.stack([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s], 1)[c1]
        s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0)) * 2
        q[c2] = np.stack([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s], 1)[c2]
        s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0)) * 2
        q[c3] = np.stack([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s], 1)[c3]
    return _normalize(q)


def _decompose_trs(mats):
    """Row-vector USD matrices (N, 4, 4) -> glTF (T, R, S) arrays, or None if any
    matrix has shear, projection or a negative determinant (not expressible as
    EXT_mesh_gpu_instancing TRS without rendering inside out)."""
    a = mats[:, :3, :3]
    if not np.allclose(mats[:, :3, 3], 0, atol=1e-9) or not np.allclose(mats[:, 3, 3], 1, atol=1e-9):
        return None
    if (np.linalg.det(a) <= 0).any():
        return None
    scale = np.linalg.norm(a, axis=2)  # row i of A is the transformed basis vector i
    if (scale < 1e-12).any():
        return None
    rot = (a / scale[:, :, None]).transpose(0, 2, 1)  # column-vector rotation
    if not np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-4):
        return None
    return mats[:, 3, :3].copy(), _quat_from_rotation(rot), scale


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    length[length == 0] = 1.0
    return v / length


def _fix_zero_normals(normals, corner_positions):
    """Replace zero-length normals (degenerate input) with the triangle normal, else +Z."""
    normals = np.array(normals, dtype=np.float64)
    bad = np.linalg.norm(normals, axis=1) < 1e-8
    if not bad.any():
        return normals
    p = corner_positions.reshape(-1, 3, 3)
    face = np.repeat(_normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])), 3, axis=0)
    normals[bad] = face[bad]
    still = np.linalg.norm(normals, axis=1) < 1e-8
    normals[still] = (0.0, 0.0, 1.0)
    return normals


def _weld(attrs: dict[str, np.ndarray]):
    """Merge identical corners. Returns (welded attrs, index buffer)."""
    names = list(attrs)
    cols = [np.asarray(attrs[n], dtype=np.float32).reshape(len(attrs[n]), -1) for n in names]
    stacked = np.ascontiguousarray(np.concatenate(cols, axis=1))
    rows = stacked.view(np.dtype((np.void, stacked.dtype.itemsize * stacked.shape[1]))).ravel()
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    out = {n: np.asarray(attrs[n], dtype=np.float32)[first] for n in names}
    return out, inverse.astype(np.uint32)


# -- procedural gprims ------------------------------------------------------


def _axis_matrix(axis: str):
    """Rotate canonical Z-axis geometry to the requested axis."""
    if axis == "X":
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    if axis == "Y":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    return np.eye(3)


def _gen_cube(size):
    h = size / 2.0
    faces = [  # normal axis, quad corners
        [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)],
        [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)],
        [(-h, h, -h), (-h, h, h), (h, h, h), (h, h, -h)],
        [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)],
        [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],
        [(-h, -h, -h), (-h, h, -h), (h, h, -h), (h, -h, -h)],
    ]
    pts = np.array([p for f in faces for p in f], dtype=np.float64)
    return pts, np.full(6, 4), np.arange(24)


def _gen_sphere(radius, seg_u=32, seg_v=16):
    pts, counts, idx = [], [], []
    for j in range(seg_v + 1):
        v = j / seg_v
        phi = math.pi * v
        for i in range(seg_u + 1):
            theta = 2 * math.pi * i / seg_u
            pts.append((radius * math.sin(phi) * math.cos(theta),
                        radius * math.sin(phi) * math.sin(theta),
                        -radius * math.cos(phi)))
    row = seg_u + 1
    for j in range(seg_v):
        for i in range(seg_u):
            a, b = j * row + i, j * row + i + 1
            c, d = (j + 1) * row + i + 1, (j + 1) * row + i
            counts.append(4)
            idx.extend([a, b, c, d])
    return np.array(pts), np.array(counts), np.array(idx)


def _gen_cylinder(r_bottom, r_top, height, seg=32):
    h = height / 2.0
    pts, counts, idx = [], [], []
    ring = lambda r, z: [(r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), z)
                         for i in range(seg)]
    # side
    side_b = ring(r_bottom, -h)
    side_t = ring(r_top, h)
    pts.extend(side_b + side_t)
    for i in range(seg):
        n = (i + 1) % seg
        counts.append(4)
        idx.extend([i, n, seg + n, seg + i])
    # caps (separate points so normals stay hard)
    if r_bottom > 0:
        start = len(pts)
        pts.extend(ring(r_bottom, -h))
        counts.append(seg)
        idx.extend(reversed(range(start, start + seg)))
    if r_top > 0:
        start = len(pts)
        pts.extend(ring(r_top, h))
        counts.append(seg)
        idx.extend(range(start, start + seg))
    return np.array(pts), np.array(counts), np.array(idx)


def _gen_plane(width, length):
    w, l = width / 2.0, length / 2.0
    pts = np.array([(-w, -l, 0), (w, -l, 0), (w, l, 0), (-w, l, 0)], dtype=np.float64)
    return pts, np.array([4]), np.arange(4)


# ---------------------------------------------------------------------------
# converter
# ---------------------------------------------------------------------------


class _TextureCache:
    """Encoded textures kept between conversions, so switching a variant or the
    purpose does not decode the same (EXR) maps again. Keyed by the image key
    plus each source file's size and modification time."""

    def __init__(self, max_bytes=768 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.items: dict = {}
        self.size = 0
        self.lock = threading.Lock()

    @staticmethod
    def _stamp(key):
        out = []
        for part in key[1:]:
            for value in (part if isinstance(part, tuple) else (part,)):
                if isinstance(value, str) and os.path.isabs(value.split("[", 1)[0]):
                    try:
                        st = os.stat(value.split("[", 1)[0])
                        out.append((value, st.st_size, st.st_mtime_ns))
                    except OSError:
                        return None
        return (key, MAX_TEXTURE_SIZE, tuple(out))

    def get(self, key, producer):
        stamp = self._stamp(key)
        if stamp is not None:
            with self.lock:
                hit = self.items.pop(stamp, None)
                if hit is not None:
                    self.items[stamp] = hit  # most recently used last
                    return hit
        result = producer()
        if result is not None and stamp is not None:
            with self.lock:
                self.items[stamp] = result
                self.size += len(result[0])
                while self.size > self.max_bytes and self.items:
                    old = self.items.pop(next(iter(self.items)))
                    self.size -= len(old[0])
        return result


_TEXTURE_CACHE = _TextureCache()


class _Converter:
    def __init__(self, stage: Usd.Stage, purpose: str = "render") -> None:
        self.stage = stage
        self.time = Usd.TimeCode.EarliestTime()
        self.report = _Report()
        self.glb = GlbBuilder()
        self.xf_cache = UsdGeom.XformCache(self.time)
        self.material_index: dict = {}
        self.image_index: dict = {}
        self.mesh_cache: dict = {}
        self.proto_cache: dict = {}
        self.stats = Counter()
        self.purpose = purpose if purpose in PURPOSES else "render"
        # "render": what a renderer sees; "proxy": what a viewport shows by default
        hidden = {UsdGeom.Tokens.guide}
        hidden.add(UsdGeom.Tokens.proxy if self.purpose == "render" else UsdGeom.Tokens.render)
        self.hidden_purposes = hidden
        self.predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        self._active_protos: set[str] = set()
        self.lods: dict[int, list[int]] = {}  # mesh index -> LOD mesh indices
        self.lod_groups: list[str] = []
        self.lod_nodes: list[int] = []

    # -- traversal ---------------------------------------------------------

    def _collect(self, start, rel=None):
        """Renderable items under `start` (inclusive).

        Returns a list of (name, mesh_index, matrices, n_tris, source), where
        matrices is an (N, 4, 4) array of row-vector USD matrices: world space, or,
        when `rel` is given, world space multiplied by `rel` (used for prototype
        space). `source` says which prim the viewer selects for it: {"path"} for a
        gprim; for PointInstancer instances the instancer's path, the authored
        instance index of each matrix ("ids") and the gprim inside the prototype.
        """
        out = []
        it = iter(Usd.PrimRange(start, self.predicate))
        for prim in it:
            if prim.IsPseudoRoot():
                continue
            type_name = prim.GetTypeName()
            if type_name in _SKIP_SUBTREE_TYPES:
                it.PruneChildren()
                continue
            img = UsdGeom.Imageable(prim)
            if img:
                # visibility is checked per prim as we descend, so it also works inside
                # prototypes whose ancestors (e.g. a hidden Prototypes scope) are invisible
                if img.GetVisibilityAttr().Get(self.time) == UsdGeom.Tokens.invisible:
                    it.PruneChildren()
                    continue
                if img.ComputePurpose() in self.hidden_purposes:
                    it.PruneChildren()
                    continue
            if type_name == "PointInstancer":
                out.extend(self._collect_instancer(prim, rel))
                it.PruneChildren()
                continue
            if type_name in _GPRIM_TYPES:
                result = self._gprim_mesh(prim, type_name)
                if result is not None:
                    mesh_index, n_tris = result
                    m = np.array(self.xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
                    if rel is not None:
                        m = m @ rel
                    path = prim.GetPath().pathString
                    out.append((path, mesh_index, m[None], n_tris, {"path": path}))
            elif type_name == "Camera":
                self.stats["cameras"] += 1
            elif type_name not in _KNOWN_NON_GEOM and prim.IsA(UsdGeom.Gprim):
                self.report.skipped_types[type_name] += 1
        return out

    def _collect_instancer(self, prim, rel):
        t = self.time
        path = prim.GetPath().pathString
        pi = UsdGeom.PointInstancer(prim)
        protos = pi.GetPrototypesRel().GetForwardedTargets()
        proto_idx = np.array(pi.GetProtoIndicesAttr().Get(t) or [], dtype=np.int64)
        if not protos or len(proto_idx) == 0:
            return []
        xforms = pi.ComputeInstanceTransformsAtTime(
            t, t, UsdGeom.PointInstancer.ExcludeProtoXform, UsdGeom.PointInstancer.IgnoreMask)
        if xforms is None or len(xforms) != len(proto_idx):
            self.report.warn(f"{path}: PointInstancer arrays are inconsistent; skipped")
            return []
        inst = np.array(xforms, dtype=np.float64).reshape(-1, 4, 4)
        authored = np.arange(len(proto_idx))  # instance index as authored, before masking
        mask = pi.ComputeMaskAtTime(t)
        if mask is not None and len(mask) == len(proto_idx):
            keep = np.array(mask, dtype=bool)
            inst, proto_idx, authored = inst[keep], proto_idx[keep], authored[keep]
        if ((proto_idx < 0) | (proto_idx >= len(protos))).any():
            self.report.warn(f"{path}: protoIndices out of range ignored")

        world = np.array(self.xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        if rel is not None:
            world = world @ rel
        self.stats["instancers"] += 1

        out = []
        for p, proto_path in enumerate(protos):
            sel = np.nonzero(proto_idx == p)[0]
            if len(sel) == 0:
                continue
            proto = self.stage.GetPrimAtPath(proto_path)
            if not proto:
                self.report.warn(f"{path}: prototype {proto_path} not found")
                continue
            items = self._prototype_items(proto)
            base = inst[sel] @ world  # (k, 4, 4): instance -> PointInstancer -> world
            label = f"{path}/{proto_path.name}"
            for name, mesh_index, mats, n_tris, src in items:
                combined = (mats[:, None] @ base[None]).reshape(-1, 4, 4)
                # combined[i * k + j] is prototype matrix i placed by instance j
                source = {"path": path, "ids": np.tile(authored[sel], len(mats)),
                          "proto": src.get("proto", src["path"])}
                out.append((label, mesh_index, combined, n_tris, source))
        return out

    def _prototype_items(self, proto):
        """Items of a prototype subtree, relative to the prototype's parent.

        Per the PointInstancer spec the prototype root's own transform applies,
        but none of its ancestors' do.
        """
        key = (proto.GetPrimInPrototype().GetPath() if proto.IsInstanceProxy() else proto.GetPath()).pathString
        cached = self.proto_cache.get(key)
        if cached is not None:
            return cached
        if key in self._active_protos:
            self.report.warn(f"Prototype {key} instances itself; recursion stopped")
            return []
        self._active_protos.add(key)
        try:
            rel = np.linalg.inv(np.array(self.xf_cache.GetParentToWorldTransform(proto), dtype=np.float64))
            items = self._collect(proto, rel)
        finally:
            self._active_protos.discard(key)
        self.proto_cache[key] = items
        return items

    def _emit(self, item) -> list[int]:
        """Write glTF nodes for one collected item."""
        name, mesh_index, mats, n_tris, source = item
        n = len(mats)
        self.stats["mesh_prims"] += n
        self.stats["triangles"] += n * n_tris
        # the prim the viewer's scene graph selects when this is picked
        extras = {"usdPath": source["path"]}
        if source.get("proto"):
            extras["usdProto"] = source["proto"]
        ids = source.get("ids")
        if n == 1:
            if ids is not None:
                extras["usdInstance"] = int(ids[0])
            node = {"name": name, "mesh": mesh_index, "extras": extras}
            m = mats[0]
            if not np.allclose(m, np.eye(4)):
                node["matrix"] = [float(x) for x in m.reshape(-1)]
            return [self._add_with_lods(node)]

        self.stats["instances"] += n
        trs = _decompose_trs(mats)
        if trs is None:
            # shear or mirroring: fall back to one node per instance
            nodes = []
            for i, m in enumerate(mats):
                node_extras = dict(extras)
                if ids is not None:
                    node_extras["usdInstance"] = int(ids[i])
                nodes.append(self._add_with_lods({
                    "name": f"{name}[{i}]", "mesh": mesh_index, "extras": node_extras,
                    "matrix": [float(x) for x in m.reshape(-1)]}))
            return nodes
        t, r, s = trs
        ext = {"attributes": {
            "TRANSLATION": self.glb.add_attribute(t, target=None),
            "ROTATION": self.glb.add_attribute(r, target=None),
            "SCALE": self.glb.add_attribute(s, target=None),
        }}
        used = self.glb.gltf.setdefault("extensionsUsed", [])
        if "EXT_mesh_gpu_instancing" not in used:
            used.append("EXT_mesh_gpu_instancing")
        if ids is not None:
            # authored instance index per glTF instance (float: exact up to 2^24)
            extras["usdInstanceIds"] = self.glb.add_attribute(np.asarray(ids, dtype=np.float32), target=None)
        return [self._add_with_lods({"name": name, "mesh": mesh_index, "extras": extras,
                                     "extensions": {"EXT_mesh_gpu_instancing": ext}})]

    def _add_with_lods(self, node: dict) -> int:
        """Add a node; if its mesh has LODs, add one sibling node per level to the
        LOD scene, sharing the transform / instance data. The viewer pairs them up
        through extras.usdvLod = {group, level}."""
        lods = self.lods.get(node["mesh"])
        if not lods:
            return self.glb.add_node(node)
        group = len(self.lod_groups)
        self.lod_groups.append(node["name"])
        extras = node.get("extras", {})
        base = self.glb.add_node({**node, "extras": {**extras, "usdvLod": {"group": group, "level": 0}}})
        for level, mesh in enumerate(lods, 1):
            self.lod_nodes.append(self.glb.add_node({
                **node, "name": f"{node['name']} (LOD{level})", "mesh": mesh,
                "extras": {**extras, "usdvLod": {"group": group, "level": level}}}))
        return base

    def run(self) -> tuple[bytes, dict]:
        stage = self.stage
        up = UsdGeom.GetStageUpAxis(stage)
        mpu = UsdGeom.GetStageMetersPerUnit(stage)

        children: list[int] = []
        for item in self._collect(stage.GetPseudoRoot()):
            children.extend(self._emit(item))

        s = float(mpu)
        if up == UsdGeom.Tokens.z:
            root_matrix = [s, 0, 0, 0, 0, 0, -s, 0, 0, s, 0, 0, 0, 0, 0, 1]
        else:
            root_matrix = [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]
        root = {"name": "usd_stage_root", "children": children}
        if root_matrix != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]:
            root["matrix"] = root_matrix
        self.glb.add_node(root, root=True)
        if self.lod_nodes:
            # a second glTF scene: other glTF viewers show only the default scene
            lod_root = {**root, "name": "usd_stage_lods", "children": self.lod_nodes}
            self.glb.add_scene("usdv_lods", [self.glb.add_node(lod_root)])

        for t, n in self.report.skipped_types.items():
            self.report.warn(f"{n} {t} prim(s) skipped (not supported yet)")

        info = {
            "upAxis": str(up),
            "metersPerUnit": mpu,
            "defaultPrim": stage.GetDefaultPrim().GetPath().pathString if stage.GetDefaultPrim() else None,
            "purpose": self.purpose,
            "stats": {
                "pointInstancers": self.stats["instancers"],
                "instances": self.stats["instances"],
                "meshPrims": self.stats["mesh_prims"],
                "uniqueMeshes": len(self.glb.gltf["meshes"]) - sum(len(v) for v in self.lods.values()),
                "lodMeshes": sum(len(v) for v in self.lods.values()),
                "triangles": self.stats["triangles"],
                "uniqueTriangles": self.stats["unique_triangles"],
                "materials": len(self.glb.gltf["materials"]),
                "textures": len(self.glb.gltf["images"]),
                "lodGroups": len(self.lod_groups),
                "lodTriangles": self.stats["lod_triangles"],
                "cameras": self.stats["cameras"],
            },
            "warnings": self.report.as_list(),
        }
        return self.glb.to_bytes(), info

    # -- geometry ----------------------------------------------------------

    def _topology(self, prim, type_name):
        """Return (points, counts, indices, holes, smooth, left_handed, double_sided)."""
        t = self.time
        gprim = UsdGeom.Gprim(prim)
        double_sided = bool(gprim.GetDoubleSidedAttr().Get(t))
        left = gprim.GetOrientationAttr().Get(t) == UsdGeom.Tokens.leftHanded

        if type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get(t)
            counts = mesh.GetFaceVertexCountsAttr().Get(t)
            indices = mesh.GetFaceVertexIndicesAttr().Get(t)
            if not points or not counts or not indices:
                return None
            holes = mesh.GetHoleIndicesAttr().Get(t)
            scheme = mesh.GetSubdivisionSchemeAttr().Get(t)
            return (np.array(points, dtype=np.float64), np.array(counts), np.array(indices),
                    np.array(holes or [], dtype=np.int64), scheme != UsdGeom.Tokens.none,
                    left, double_sided)

        def get(attr_name, default):
            v = prim.GetAttribute(attr_name).Get(t)
            return default if v is None else v

        axis = get("axis", "Z")
        if type_name == "Cube":
            pts, counts, idx = _gen_cube(get("size", 2.0))
        elif type_name == "Sphere":
            pts, counts, idx = _gen_sphere(get("radius", 1.0))
        elif type_name == "Cylinder":
            r = get("radius", 1.0)
            pts, counts, idx = _gen_cylinder(get("radiusBottom", r), get("radiusTop", r), get("height", 2.0))
        elif type_name == "Cone":
            pts, counts, idx = _gen_cylinder(get("radius", 1.0), 0.0, get("height", 2.0))
        elif type_name == "Plane":
            pts, counts, idx = _gen_plane(get("width", 2.0), get("length", 2.0))
            double_sided = bool(get("doubleSided", False))
        else:
            return None
        if type_name != "Cube" and axis != "Z":
            pts = pts @ _axis_matrix(axis).T
        # procedural gprims: smooth normals over shared points, hard edges via split points
        return pts, counts, idx, np.zeros(0, dtype=np.int64), True, left, double_sided

    def _primvar(self, prim, name, inherited=False):
        api = UsdGeom.PrimvarsAPI(prim)
        pv = api.FindPrimvarWithInheritance(name) if inherited else api.GetPrimvar(name)
        if not pv or not pv.HasValue():
            return None, None
        vals = pv.ComputeFlattened(self.time)
        if vals is None or len(vals) == 0:
            return None, None
        return np.array(vals, dtype=np.float64), pv.GetInterpolation()

    def _gprim_mesh(self, prim, type_name):
        """glTF mesh for a gprim (shared between instances). Returns (mesh_index, n_tris)."""
        topo = self._topology(prim, type_name)
        if topo is None:
            return None
        points, counts, indices, holes, smooth, left, double_sided = topo
        path = prim.GetPath().pathString

        if counts.sum() != len(indices):
            self.report.warn(f"{path}: faceVertexCounts do not match faceVertexIndices; skipped")
            return None
        if len(indices) and (indices.max() >= len(points) or indices.min() < 0):
            self.report.warn(f"{path}: face indices out of range; skipped")
            return None

        # material groups: 0 = whole prim, i+1 = GeomSubset i
        bind_api = UsdShade.MaterialBindingAPI(prim)
        group_keys = [self._material_key(prim, prim, double_sided)]
        face_group = np.zeros(len(counts), dtype=np.int64)
        for subset in bind_api.GetMaterialBindSubsets():
            sidx = np.array(subset.GetIndicesAttr().Get(self.time) or [], dtype=np.int64)
            sidx = sidx[(sidx >= 0) & (sidx < len(counts))]
            group_keys.append(self._material_key(subset.GetPrim(), prim, double_sided))
            face_group[sidx] = len(group_keys) - 1

        proto_path = prim.GetPrimInPrototype().GetPath().pathString if prim.IsInstanceProxy() else path
        cache_key = (proto_path, tuple(group_keys))
        mesh_index = self.mesh_cache.get(cache_key)

        if mesh_index is None:
            mesh_index, n_tris = self._build_mesh(prim, path, points, counts, indices, holes, smooth, left,
                                                  face_group, group_keys)
            if mesh_index is None:
                return None
            self.mesh_cache[cache_key] = mesh_index
            self.stats["unique_triangles"] += n_tris
            self.mesh_cache[("tris", mesh_index)] = n_tris
        return mesh_index, self.mesh_cache[("tris", mesh_index)]

    def _build_mesh(self, prim, path, points, counts, indices, holes, smooth, left, face_group, group_keys):
        corners, face_of_tri = _triangulate(counts)
        if len(holes):
            keep = ~np.isin(face_of_tri, holes)
            corners, face_of_tri = corners[keep], face_of_tri[keep]
        if len(corners) == 0:
            return None, 0
        if left:
            corners = corners[:, [0, 2, 1]]

        corner_idx = corners.reshape(-1)
        face_idx = np.repeat(face_of_tri, 3)
        point_idx = indices[corner_idx]
        n_points, n_corners, n_faces = len(points), len(indices), len(counts)
        expand = lambda v, i: _expand(v, i, n_points, n_corners, n_faces, point_idx, corner_idx, face_idx)

        attrs = {"POSITION": points[point_idx]}

        # normals: primvars:normals wins over the normals attribute
        normals, n_interp = self._primvar(prim, "normals")
        if normals is None and prim.IsA(UsdGeom.PointBased):
            pb = UsdGeom.PointBased(prim)
            raw = pb.GetNormalsAttr().Get(self.time)
            if raw:
                normals, n_interp = np.array(raw, dtype=np.float64), pb.GetNormalsInterpolation()
        normals = expand(normals, n_interp)
        if normals is None:
            tri_points = point_idx.reshape(-1, 3)
            if smooth:
                normals = _smooth_normals(points, tri_points)[point_idx]
            else:
                p = points[tri_points]
                fn = _normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
                normals = np.repeat(fn, 3, axis=0)
        else:
            normals = _normalize(normals)
        attrs["NORMAL"] = _fix_zero_normals(normals, points[point_idx])

        # texture coordinates
        uv_name = self._uv_name_for(group_keys)
        uvs, uv_interp = self._primvar(prim, uv_name) if uv_name else (None, None)
        if uvs is None:
            for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
                if pv.GetTypeName() in (Sdf.ValueTypeNames.TexCoord2fArray, Sdf.ValueTypeNames.Float2Array):
                    vals = pv.ComputeFlattened(self.time)
                    if vals:
                        uvs, uv_interp = np.array(vals, dtype=np.float64), pv.GetInterpolation()
                        break
        uvs = expand(uvs, uv_interp)
        if uvs is not None and uvs.ndim == 2 and uvs.shape[1] >= 2:
            uv = np.array(uvs[:, :2], dtype=np.float64)
            uv[:, 1] = 1.0 - uv[:, 1]
            attrs["TEXCOORD_0"] = uv

        # vertex colours (only meaningful when no material is bound)
        if any(k[0] == "displayColor" and k[2] for k in group_keys):
            colors, c_interp = self._primvar(prim, "displayColor", inherited=True)
            colors = expand(colors, c_interp)
            if colors is not None:
                attrs["COLOR_0"] = np.clip(colors[:, :3], 0, 1)

        n_tris = len(corners)
        primitives = []
        lod_sources = []
        tri_group = face_group[face_of_tri]
        for g, key in enumerate(group_keys):
            sel = np.nonzero(tri_group == g)[0]
            if len(sel) == 0:
                continue
            corner_sel = (sel[:, None] * 3 + np.arange(3)).reshape(-1)
            sub = {k: np.asarray(v)[corner_sel] for k, v in attrs.items()}
            if "COLOR_0" in sub and not (key[0] == "displayColor" and key[2]):
                del sub["COLOR_0"]
            welded, index_buf = _weld(sub)
            prim_attrs = {k: self.glb.add_attribute(v) for k, v in welded.items()}
            primitives.append({
                "attributes": prim_attrs,
                "indices": self.glb.add_indices(index_buf),
                "material": self._material_for_key(key),
                "mode": 4,
            })
            lod_sources.append((welded, index_buf, primitives[-1]))
        mesh_index = self.glb.add_mesh(path, primitives)
        if LOD_ENABLED and n_tris >= LOD_MIN_TRIANGLES:
            self.lods[mesh_index] = self._build_lods(path, lod_sources)
        return mesh_index, n_tris

    # -- level of detail ---------------------------------------------------

    def _build_lods(self, path, sources) -> list[int]:
        """Simplified versions of a mesh, each ~1/4 of the previous triangle count.

        Normals are recomputed (smooth); UVs and vertex colours are carried over
        from a source vertex that collapsed into each kept vertex.
        """
        try:
            import fast_simplification as fs
        except ImportError:  # pragma: no cover - optional dependency
            self.report.warn("fast-simplification is not installed; no LODs generated")
            return []

        current = []
        for attrs, index_buf, prim in sources:
            pos = np.asarray(attrs["POSITION"], dtype=np.float64)
            # weld by position only so seams (split normals / UVs) don't open as cracks
            points, inv = np.unique(pos, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            extra = {}
            for k in ("TEXCOORD_0", "COLOR_0"):
                if k in attrs:
                    v = np.zeros((len(points), attrs[k].shape[1]), dtype=np.float32)
                    v[inv] = attrs[k]
                    extra[k] = v
            tris = inv[np.asarray(index_buf, dtype=np.int64).reshape(-1, 3)]
            current.append({"points": points, "tris": tris, "extra": extra, "last": prim, "keep": False})

        levels: list[int] = []
        prev_total = sum(len(c["tris"]) for c in current)
        for _ in range(LOD_LEVELS):
            nxt = []
            for c in current:
                if len(c["tris"]) < 256:
                    nxt.append({**c, "keep": True})  # too small to bother: reuse its last level
                    continue
                try:
                    pts, faces, coll = fs.simplify(c["points"], c["tris"].astype(np.int64),
                                                   target_reduction=0.75, return_collapses=True)
                except Exception as exc:  # pragma: no cover - defensive
                    self.report.warn(f"{path}: LOD generation failed ({exc})")
                    return levels
                extra = {}
                if c["extra"]:
                    _, _, mapping = fs.replay_simplification(
                        c["points"].astype(np.float32), c["tris"].astype(np.int64), coll)
                    if len(mapping) == len(c["points"]) and mapping.max(initial=-1) < len(pts):
                        for k, v in c["extra"].items():
                            out = np.zeros((len(pts), v.shape[1]), dtype=np.float32)
                            out[mapping] = v
                            extra[k] = out
                nxt.append({"points": pts, "tris": np.asarray(faces, dtype=np.int64), "extra": extra,
                            "last": c["last"], "keep": False})
            total = sum(len(c["tris"]) for c in nxt)
            if total == 0 or total > prev_total * 0.6:
                break  # the mesh won't simplify further in a useful way
            primitives = []
            for c in nxt:
                if c["keep"]:
                    primitives.append(c["last"])
                    continue
                normals = _smooth_normals(c["points"], c["tris"])
                attrs = {"POSITION": self.glb.add_attribute(c["points"]),
                         "NORMAL": self.glb.add_attribute(normals)}
                for k, v in c["extra"].items():
                    attrs[k] = self.glb.add_attribute(v)
                prim = {"attributes": attrs, "indices": self.glb.add_indices(c["tris"]),
                        "material": c["last"]["material"], "mode": 4}
                primitives.append(prim)
                c["last"] = prim
            levels.append(self.glb.add_mesh(f"{path} (LOD{len(levels) + 1})", primitives))
            self.stats["lod_triangles"] += total
            prev_total = total
            current = nxt
            if total < 2000:
                break
        return levels

    # -- materials ---------------------------------------------------------

    def _material_key(self, bind_prim, geom_prim, double_sided):
        """Hashable description of the material a prim (or subset) renders with."""
        mat, _rel = UsdShade.MaterialBindingAPI(bind_prim).ComputeBoundMaterial()
        if mat:
            mprim = mat.GetPrim()
            # materials inside an instance live in the shared prototype
            mpath = mprim.GetPrimInPrototype().GetPath() if mprim.IsInstanceProxy() else mprim.GetPath()
            return ("material", mpath.pathString, double_sided)
        colors, interp = self._primvar(geom_prim, "displayColor", inherited=True)
        opacity, _ = self._primvar(geom_prim, "displayOpacity", inherited=True)
        alpha = float(opacity[0]) if opacity is not None else 1.0
        if colors is None:
            return ("displayColor", (0.7, 0.7, 0.7, alpha), False, double_sided)
        if interp == "constant" or len(colors) == 1:
            c = tuple(round(float(x), 5) for x in colors[0][:3])
            return ("displayColor", c + (alpha,), False, double_sided)
        # varying displayColor -> white base colour multiplied by COLOR_0
        return ("displayColor", (1.0, 1.0, 1.0, alpha), True, double_sided)

    def _uv_name_for(self, group_keys):
        for key in group_keys:
            if key[0] == "material":
                surf = self._preview_surface(UsdShade.Material.Get(self.stage, key[1]))
                if surf is not None:
                    for name in ("diffuseColor", "normal", "roughness", "metallic", "opacity", "emissiveColor"):
                        tex = self._input_texture(surf, name)
                        if tex is not None:
                            return tex.get("varname") or "st"
        return "st"

    def _material_for_key(self, key):
        idx = self.material_index.get(key)
        if idx is not None:
            return idx
        if key[0] == "material":
            mat = self._convert_material(key[1])
        else:
            color = list(key[1])
            mat = {
                "name": "displayColor",
                "pbrMetallicRoughness": {"baseColorFactor": color, "metallicFactor": 0.0, "roughnessFactor": 0.8},
            }
            if color[3] < 1.0:
                mat["alphaMode"] = "BLEND"
        if key[-1]:
            mat["doubleSided"] = True
        # every instanced prototype carries its own copy of a shared material;
        # identical results collapse into one glTF material
        import json
        sig = json.dumps({k: v for k, v in mat.items() if k != "name"}, sort_keys=True)
        idx = self.material_index.get(("sig", sig))
        if idx is None:
            idx = self.glb.add_material(mat)
            self.material_index[("sig", sig)] = idx
        self.material_index[key] = idx
        return idx

    def _mtlx_surface(self, material):
        """A MaterialX standard_surface / open_pbr_surface on any render context."""
        if not material:
            return None
        for out in material.GetSurfaceOutputs():
            sources = out.GetConnectedSources()[0]
            for src in sources:
                s = UsdShade.Shader(src.source.GetPrim())
                sid = str(s.GetIdAttr().Get() or "") if s else ""
                if sid.startswith(("ND_standard_surface", "ND_open_pbr_surface")):
                    return s
        return None

    def _mx_trace(self, shader, name, channel=None, depth=0):
        """Follow a MaterialX input upstream.

        Returns ("value", v), ("texture", info) or (None, None). Understands
        image-like nodes (a `file` input, or MaterialX triplanarprojection's
        filex / filey / filez; Karma's triplanar nodes included) and
        passes through convert / extract / normalmap / transform nodes.
        """
        inp = shader.GetInput(name)
        if not inp or depth > 24:
            return None, None
        attrs = inp.GetValueProducingAttributes()
        if not attrs:
            return None, None
        attr = attrs[0]
        if not UsdShade.Output.IsOutput(attr):
            v = attr.Get(self.time)
            return ("value", v) if v is not None else (None, None)
        node = UsdShade.Shader(attr.GetPrim())
        if not node:
            return None, None
        nid = str(node.GetIdAttr().Get() or "")
        # image nodes read `file`; MaterialX's own triplanarprojection reads one
        # image per axis (filex / filey / filez), which artists usually set alike
        file_input = next((n for n in ("file", "filex", "filey", "filez") if node.GetInput(n)), None)
        if file_input:
            kind, f = self._mx_trace(node, file_input, depth=depth + 1)
            if kind != "value" or not isinstance(f, Sdf.AssetPath):
                return None, None
            if file_input == "filex":
                others = {self._mx_trace(node, n, depth=depth + 1)[1] for n in ("filey", "filez") if node.GetInput(n)}
                if any(isinstance(o, Sdf.AssetPath) and o.path != f.path for o in others):
                    self.report.warn(f"{_display(node.GetPath().pathString)}: a different image per axis "
                                     "is shown with the X image on every axis")
            info = {
                "path": f.resolvedPath or None,
                "authored": f.path,
                "colorSpace": self._asset_colorspace(node, file_input),
                "channel": channel or "rgb",
                "wrapS": REPEAT,
                "wrapT": REPEAT,
                "scale": (1, 1, 1, 1),
                "varname": None,
                "nodeId": nid,
                "triplanar": None,
            }
            if "triplanar" in nid.lower():
                info["triplanar"] = self._mx_triplanar(node, depth)
            return "texture", info
        if "_separate" in nid:  # separate2/3/4: outx|outr -> r, outy|outg -> g, ...
            out = attr.GetBaseName()
            ch = {"x": "r", "y": "g", "z": "b", "w": "a"}.get(out[-1:], out[-1:])
            return self._mx_trace(node, "in", ch if ch in "rgba" else None, depth + 1)
        if "_extract_" in nid:
            k, idx = self._mx_trace(node, "index", depth=depth + 1)
            idx = int(idx) if k == "value" else 0
            return self._mx_trace(node, "in", "rgba"[max(0, min(idx, 3))], depth + 1)
        if "_constant_" in nid:
            return self._mx_trace(node, "value", channel, depth + 1)
        for upstream in ("in", "in1", "input", "fg"):
            if node.GetInput(upstream):
                return self._mx_trace(node, upstream, channel, depth + 1)
        return None, None

    @staticmethod
    def _asset_colorspace(shader, name):
        """colorSpace metadata on a file input or on the attribute it is connected to."""
        inp = shader.GetInput(name)
        if not inp:
            return None
        for a in [inp.GetAttr()] + list(inp.GetValueProducingAttributes()):
            cs = a.GetColorSpace() if a.HasColorSpace() else ""
            if cs:
                return str(cs)
        return None

    def _mx_triplanar(self, node, depth):
        """Projection settings of a triplanar texture node: position scale + space."""
        scale, space = 1.0, "object"
        pos = node.GetInput("position")
        attrs = pos.GetValueProducingAttributes() if pos else []
        if attrs and UsdShade.Output.IsOutput(attrs[0]):
            n = UsdShade.Shader(attrs[0].GetPrim())
            # position node, optionally scaled by a multiply node
            for _ in range(4):
                nid = str(n.GetIdAttr().Get() or "")
                if "_position_" in nid:
                    k, sp = self._mx_trace(n, "space", depth=depth + 1)
                    space = "world" if k == "value" and str(sp) == "world" else "object"
                    break
                if "_multiply_" in nid:
                    k, v = self._mx_trace(n, "in2", depth=depth + 1)
                    if k == "value":
                        try:
                            scale *= float(v)
                        except TypeError:
                            scale *= float(v[0])
                    a = n.GetInput("in1").GetValueProducingAttributes() if n.GetInput("in1") else []
                    if not a or not UsdShade.Output.IsOutput(a[0]):
                        break
                    n = UsdShade.Shader(a[0].GetPrim())
                    continue
                break
        k, inv = self._mx_trace(node, "invert", depth=depth + 1)
        return {"scale": scale, "space": space, "invert": bool(inv) if k == "value" else False}

    def _convert_mtlx(self, mat_path, surf, out):
        pbr = out["pbrMetallicRoughness"]
        sid = str(surf.GetIdAttr().Get())
        openpbr = sid.startswith("ND_open_pbr_surface")
        names = {
            "base": "base_weight" if openpbr else "base",
            "base_color": "base_color",
            "metalness": "base_metalness" if openpbr else "metalness",
            "roughness": "specular_roughness",
            "normal": "geometry_normal" if openpbr else "normal",
            "emission": "emission_luminance" if openpbr else "emission",
            "emission_color": "emission_color",
            "opacity": "geometry_opacity" if openpbr else "opacity",
        }

        def get(key, default):
            kind, v = self._mx_trace(surf, names[key])
            if kind == "texture":
                return default, v
            return (default if v is None else v), None

        base, _ = get("base", 1.0)
        color, color_tex = get("base_color", (0.8, 0.8, 0.8))
        metal, metal_tex = get("metalness", 0.0)
        rough, rough_tex = get("roughness", 0.2)
        _, normal_tex = get("normal", None)
        opacity, _ = get("opacity", 1.0)
        base = float(base)
        try:
            alpha = float(min(opacity)) if hasattr(opacity, "__len__") else float(opacity)
        except (TypeError, ValueError):
            alpha = 1.0
        if color_tex:
            pbr["baseColorFactor"] = [base, base, base, alpha]
            tex = self._texture_ref(color_tex, mat_path, "baseColor")
            if tex is not None:
                pbr["baseColorTexture"] = tex
        else:
            pbr["baseColorFactor"] = [float(color[0]) * base, float(color[1]) * base, float(color[2]) * base, alpha]
        if alpha < 1.0:
            out["alphaMode"] = "BLEND"
        pbr["metallicFactor"] = float(metal)
        pbr["roughnessFactor"] = float(rough)
        if metal_tex or rough_tex:
            tex = self._packed_mr_texture(metal_tex, rough_tex, mat_path)
            if tex is not None:
                pbr["metallicRoughnessTexture"] = tex
                if metal_tex:
                    pbr["metallicFactor"] = 1.0
                if rough_tex:
                    pbr["roughnessFactor"] = 1.0
        if normal_tex:
            tex = self._texture_ref(normal_tex, mat_path, "normal")
            if tex is not None:
                out["normalTexture"] = tex
        emission, _ = get("emission", 0.0)
        ecolor, _ = get("emission_color", (1, 1, 1))
        if float(emission) > 0:
            out["emissiveFactor"] = [min(float(c) * float(emission), 1.0) for c in ecolor[:3]]

        # triplanar projection is not expressible in glTF: the viewer reads it from extras
        tri = next((t["triplanar"] for t in (color_tex, rough_tex, metal_tex, normal_tex)
                    if t and t.get("triplanar")), None)
        if tri:
            mpu = UsdGeom.GetStageMetersPerUnit(self.stage)
            scale = tri["scale"] / mpu if tri["space"] == "world" else tri["scale"]
            out["extras"] = {"usdvTriplanar": {
                "scale": scale, "space": tri["space"],
                "flipNormalY": bool(normal_tex and normal_tex["triplanar"] and normal_tex["triplanar"]["invert"]),
            }}
            ids = sorted({t["nodeId"] for t in (color_tex, rough_tex, metal_tex, normal_tex) if t})
            if any("hex" in i for i in ids):
                self.report.warn(f"Material {_display(mat_path)}: hex-tiled triplanar shown as plain triplanar")
        return out

    def _preview_surface(self, material):
        if not material:
            return None
        shader, _, _ = material.ComputeSurfaceSource()
        if not shader:
            # fall back to any render context that resolves to a UsdPreviewSurface
            for out in material.GetSurfaceOutputs():
                for src in out.GetConnectedSources()[0]:
                    s = UsdShade.Shader(src.source.GetPrim())
                    if s and s.GetIdAttr().Get() == "UsdPreviewSurface":
                        return s
            return None
        return shader if shader.GetIdAttr().Get() == "UsdPreviewSurface" else None

    def _resolve_input(self, shader, name):
        """Return ("value", v) / ("texture", info) / (None, None) for a shader input."""
        inp = shader.GetInput(name)
        if not inp:
            return None, None
        attrs = inp.GetValueProducingAttributes()
        if not attrs:
            return None, None
        attr = attrs[0]
        if UsdShade.Output.IsOutput(attr):
            src = UsdShade.Shader(attr.GetPrim())
            if src and src.GetIdAttr().Get() == "UsdUVTexture":
                channel = attr.GetBaseName()
                return "texture", self._texture_info(src, channel)
            return None, None
        return "value", attr.Get(self.time)

    def _input_texture(self, shader, name):
        kind, val = self._resolve_input(shader, name)
        return val if kind == "texture" else None

    def _texture_info(self, tex, channel):
        def val(name, default=None):
            kind, v = self._resolve_input(tex, name)
            return default if v is None or kind != "value" else v

        f = val("file")
        source_cs = str(val("sourceColorSpace", "") or "")
        info = {
            "path": f.resolvedPath if isinstance(f, Sdf.AssetPath) else None,
            "authored": f.path if isinstance(f, Sdf.AssetPath) else None,
            "colorSpace": (source_cs if source_cs not in ("", "auto") else None)
                          or self._asset_colorspace(tex, "file"),
            "channel": channel,
            "wrapS": _WRAP.get(str(val("wrapS", "repeat")), REPEAT),
            "wrapT": _WRAP.get(str(val("wrapT", "repeat")), REPEAT),
            "scale": tuple(val("scale", (1, 1, 1, 1))),
            "varname": None,
        }
        st = tex.GetInput("st")
        if st:
            for a in st.GetValueProducingAttributes():
                if UsdShade.Output.IsOutput(a):
                    reader = UsdShade.Shader(a.GetPrim())
                    vn = reader.GetInput("varname") if reader else None
                    if vn:
                        produced = vn.GetValueProducingAttributes()
                        v = produced[0].Get(self.time) if produced else vn.Get(self.time)
                        if v:
                            info["varname"] = str(v)
        return info

    def _convert_material(self, mat_path):
        material = UsdShade.Material.Get(self.stage, mat_path)
        out = {"name": mat_path}
        pbr = {"baseColorFactor": [0.7, 0.7, 0.7, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}
        out["pbrMetallicRoughness"] = pbr
        # MaterialX wins over UsdPreviewSurface: DCCs (e.g. Houdini) often add an
        # auto-generated preview shader that is only a flat-colour approximation
        mtlx = self._mtlx_surface(material)
        if mtlx is not None:
            return self._convert_mtlx(mat_path, mtlx, out)
        surf = self._preview_surface(material)
        if surf is None:
            self.report.warn(f"Material {_display(mat_path)} has no UsdPreviewSurface or MaterialX standard_surface "
                             "(MDL not supported); using grey")
            return out

        def value(name, default):
            kind, v = self._resolve_input(surf, name)
            return v if kind == "value" and v is not None else default

        def texture(name):
            return self._input_texture(surf, name)

        # base colour + opacity
        diffuse = value("diffuseColor", Gf.Vec3f(0.18, 0.18, 0.18))
        opacity = float(value("opacity", 1.0))
        diffuse_tex = texture("diffuseColor")
        opacity_tex = texture("opacity")
        pbr["baseColorFactor"] = [float(diffuse[0]), float(diffuse[1]), float(diffuse[2]), opacity]
        if diffuse_tex:
            s = diffuse_tex["scale"]
            pbr["baseColorFactor"] = [float(s[0]), float(s[1]), float(s[2]), opacity]
        if diffuse_tex or opacity_tex:
            if opacity_tex:
                pbr["baseColorFactor"][3] = 1.0
            tex = self._base_color_texture(diffuse_tex, opacity_tex, mat_path)
            if tex is not None:
                pbr["baseColorTexture"] = tex
        threshold = float(value("opacityThreshold", 0.0))
        if opacity < 1.0 or opacity_tex:
            if threshold > 0:
                out["alphaMode"] = "MASK"
                out["alphaCutoff"] = threshold
            else:
                out["alphaMode"] = "BLEND"

        if int(value("useSpecularWorkflow", 0)):
            self.report.warn(f"Material {_display(mat_path)}: specular workflow approximated as metallic/roughness")

        # metallic / roughness
        metallic = float(value("metallic", 0.0))
        roughness = float(value("roughness", 0.5))
        pbr["metallicFactor"], pbr["roughnessFactor"] = metallic, roughness
        m_tex, r_tex = texture("metallic"), texture("roughness")
        if m_tex or r_tex:
            tex = self._packed_mr_texture(m_tex, r_tex, mat_path)
            if tex is not None:
                pbr["metallicRoughnessTexture"] = tex
                if m_tex:
                    pbr["metallicFactor"] = 1.0
                if r_tex:
                    pbr["roughnessFactor"] = 1.0

        # normal / occlusion / emissive
        n_tex = texture("normal")
        if n_tex:
            tex = self._texture_ref(n_tex, mat_path, "normal")
            if tex is not None:
                out["normalTexture"] = tex
        o_tex = texture("occlusion")
        if o_tex:
            tex = self._channel_texture(o_tex, mat_path, "occlusion")
            if tex is not None:
                out["occlusionTexture"] = tex
        emissive = value("emissiveColor", Gf.Vec3f(0, 0, 0))
        e_tex = texture("emissiveColor")
        e = [float(emissive[0]), float(emissive[1]), float(emissive[2])]
        if e_tex:
            tex = self._texture_ref(e_tex, mat_path, "emissive")
            if tex is not None:
                out["emissiveTexture"] = tex
                e = [1.0, 1.0, 1.0]
        if max(e) > 0:
            strength = max(e)
            if strength > 1.0:
                out.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": strength}
                self.glb.gltf.setdefault("extensionsUsed", [])
                if "KHR_materials_emissive_strength" not in self.glb.gltf["extensionsUsed"]:
                    self.glb.gltf["extensionsUsed"].append("KHR_materials_emissive_strength")
                e = [x / strength for x in e]
            out["emissiveFactor"] = e
        return out

    # -- textures ----------------------------------------------------------

    def _read_asset(self, resolved: str | None) -> bytes | None:
        if not resolved:
            return None
        try:
            if resolved.endswith("]") and "[" in resolved:  # package-relative (usdz)
                outer, inner = resolved[:-1].split("[", 1)
                with zipfile.ZipFile(outer) as z:
                    return z.read(inner)
            with open(resolved, "rb") as fh:
                return fh.read()
        except (OSError, KeyError, zipfile.BadZipFile):
            return None

    def _load_pil(self, info, mat_path, purpose="data"):
        from PIL import Image

        data = self._read_asset(info["path"])
        if data is None:
            self.report.warn(f"Texture not found: {info['authored']} (material {_display(mat_path)})")
            return None
        if exr.is_exr(data):
            return self._load_exr(data, info, purpose)
        try:
            Image.MAX_IMAGE_PIXELS = None  # local production textures (8k/16k) are expected
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception:
            self.report.warn(f"Unsupported texture format: {info['authored']}")
            return None

    def _load_exr(self, data, info, purpose):
        if not exr.available():
            if not getattr(self, "_exr_warned", False):
                self._exr_warned = True
                self.report.warn("EXR textures skipped: install the OpenEXR package (pip install OpenEXR)")
            return None
        try:
            px, kind = exr.read(data, info["path"])
            return exr.to_image(px, kind, color=purpose in _COLOR_PURPOSES, normal=purpose == "normal",
                                colorspace=info.get("colorSpace"), max_size=MAX_TEXTURE_SIZE)
        except Exception as e:  # noqa: BLE001 - corrupt or exotic files only cost the texture
            self.report.warn(f"Could not read EXR texture {info['authored']}: {e}")
            return None

    @staticmethod
    def _fit(img):
        if MAX_TEXTURE_SIZE and max(img.size) > MAX_TEXTURE_SIZE:
            from PIL import Image
            k = MAX_TEXTURE_SIZE / max(img.size)
            img = img.resize((max(1, round(img.size[0] * k)), max(1, round(img.size[1] * k))),
                             Image.Resampling.LANCZOS)
        return img

    def _encode(self, img) -> tuple[bytes, str]:
        """Downscale if needed; JPEG for opaque images, PNG when there is alpha."""
        img = self._fit(img)
        buf = io.BytesIO()
        if "A" in img.getbands():
            img.save(buf, format="PNG", optimize=False)
            return buf.getvalue(), "image/png"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=92, subsampling=0)
        return buf.getvalue(), "image/jpeg"

    def _image_from_key(self, key, producer):
        idx = self.image_index.get(key)
        if idx is None:
            result = _TEXTURE_CACHE.get(key, producer)
            if result is None:
                self.image_index[key] = -1
                return None
            data, mime = result
            idx = self.glb.add_image(data, mime, name=os.path.basename(str(key[1]) if len(key) > 1 else "tex"))
            self.image_index[key] = idx
        return None if idx < 0 else idx

    def _texture_ref(self, info, mat_path, purpose):
        """Use the image file directly (rgb channels)."""

        def produce():
            data = self._read_asset(info["path"])
            if data is None:
                self.report.warn(f"Texture not found: {info['authored']} (material {_display(mat_path)})")
                return None
            mime = ("image/png" if data[:8] == b"\x89PNG\r\n\x1a\n"
                    else "image/jpeg" if data[:3] == b"\xff\xd8\xff" else None)
            if mime:
                from PIL import Image
                try:
                    size = Image.open(io.BytesIO(data)).size
                except Exception:
                    size = (0, 0)
                if not MAX_TEXTURE_SIZE or max(size) <= MAX_TEXTURE_SIZE:
                    return data, mime
            img = self._load_pil(info, mat_path, purpose)
            if img is None:
                return None
            return self._encode(img.convert("RGBA" if "A" in img.getbands() else "RGB"))

        # the same float file can need a different 8-bit encoding per role
        image = self._image_from_key(("file", info["path"], purpose in _COLOR_PURPOSES, purpose == "normal"),
                                     produce)
        if image is None:
            return None
        sampler = self.glb.add_sampler(info["wrapS"], info["wrapT"])
        return {"index": self.glb.add_texture(image, sampler)}

    @staticmethod
    def _channel(img, channel):
        bands = img.getbands()
        ch = {"r": "R", "g": "G", "b": "B", "a": "A", "rgb": "R"}.get(channel, "R")
        if img.mode in ("L", "I", "F", "I;16"):
            return img.convert("L")
        if img.mode == "LA":
            return img.getchannel("A" if ch == "A" else "L")
        if img.mode == "P":
            img = img.convert("RGBA")
            bands = img.getbands()
        if ch not in bands:
            ch = bands[0]
        return img.getchannel(ch)

    def _channel_texture(self, info, mat_path, purpose):
        """Single-channel input -> image with that channel in R."""
        if info["channel"] == "r":
            return self._texture_ref(info, mat_path, purpose)

        def produce():
            img = self._load_pil(info, mat_path, purpose)
            if img is None:
                return None
            from PIL import Image
            ch = self._channel(img, info["channel"])
            return self._encode(Image.merge("RGB", (ch, ch, ch)))

        image = self._image_from_key(("chan", info["path"], info["channel"]), produce)
        if image is None:
            return None
        return {"index": self.glb.add_texture(image, self.glb.add_sampler(info["wrapS"], info["wrapT"]))}

    def _base_color_texture(self, diffuse, opacity, mat_path):
        if diffuse and (not opacity or (opacity["path"] == diffuse["path"] and opacity["channel"] == "a")):
            return self._texture_ref(diffuse, mat_path, "baseColor")

        def produce():
            from PIL import Image
            base = self._load_pil(diffuse, mat_path, "baseColor") if diffuse else None
            op = self._load_pil(opacity, mat_path) if opacity else None
            if base is None and op is None:
                return None
            size = (base or op).size
            rgb = base.convert("RGB") if base is not None else Image.new("RGB", size, (255, 255, 255))
            alpha = self._channel(op, opacity["channel"]).resize(size) if op is not None else Image.new("L", size, 255)
            r, g, b = rgb.split()
            return self._encode(Image.merge("RGBA", (r, g, b, alpha)))

        key = ("base", diffuse and diffuse["path"], opacity and (opacity["path"], opacity["channel"]))
        image = self._image_from_key(key, produce)
        if image is None:
            return None
        ref = diffuse or opacity
        return {"index": self.glb.add_texture(image, self.glb.add_sampler(ref["wrapS"], ref["wrapT"]))}

    def _packed_mr_texture(self, metal, rough, mat_path):
        # already packed glTF-style (roughness in G, metal in B, e.g. an ORM/AORM map)
        if rough and rough["channel"] == "g" and (
                metal is None or (metal["path"] == rough["path"] and metal["channel"] == "b")):
            return self._texture_ref(rough, mat_path, "metallicRoughness")

        def produce():
            from PIL import Image
            m = self._load_pil(metal, mat_path) if metal else None
            r = self._load_pil(rough, mat_path) if rough else None
            if m is None and r is None:
                return None
            size = (r or m).size
            g = self._channel(r, rough["channel"]).resize(size) if r is not None else Image.new("L", size, 255)
            b = self._channel(m, metal["channel"]).resize(size) if m is not None else Image.new("L", size, 255)
            zero = Image.new("L", size, 255)
            return self._encode(Image.merge("RGB", (zero, g, b)))

        key = ("mr", metal and (metal["path"], metal["channel"]), rough and (rough["path"], rough["channel"]))
        image = self._image_from_key(key, produce)
        if image is None:
            return None
        ref = rough or metal
        return {"index": self.glb.add_texture(image, self.glb.add_sampler(ref["wrapS"], ref["wrapT"]))}


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def dependency_report(root_path: str) -> dict:
    """Layers, assets and unresolved paths reachable from the root layer."""
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(root_path))
    except Exception as exc:  # pragma: no cover - defensive
        return {"layers": [], "assets": [], "unresolved": [], "error": str(exc)}
    def norm(p: str) -> str:  # USD mixes separators on Windows; report forward slashes
        return p.replace(os.sep, "/") if os.sep != "/" else p

    return {
        "layers": sorted(norm(l.realPath or l.identifier) for l in layers),
        "assets": sorted(norm(str(a)) for a in assets),
        "unresolved": sorted(norm(str(u)) for u in unresolved),
    }


def variant_session_layer(variants: dict | None):
    """An anonymous session layer selecting variants: {prim path: {set: variant}}.

    Opening a stage with it makes the selections part of the first composition,
    so nothing is composed twice, and the files on disk are never touched.
    """
    session = Sdf.Layer.CreateAnonymous("usdv-variants.usda")
    for prim_path, selections in (variants or {}).items():
        if not Sdf.Path.IsValidPathString(prim_path) or not Sdf.Path(prim_path).IsPrimPath():
            raise ConversionError(f"Not a prim path: {prim_path}")
        spec = Sdf.CreatePrimInLayer(session, Sdf.Path(prim_path))
        for name, value in selections.items():
            spec.variantSelections[str(name)] = str(value)
    return session


def open_stage(root_path: str, variants: dict | None = None):
    if not os.path.isfile(root_path):
        raise ConversionError(f"Root layer not found: {root_path}")
    try:
        stage = Usd.Stage.Open(Sdf.Layer.FindOrOpen(root_path), variant_session_layer(variants),
                               Usd.Stage.LoadAll)
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"USD could not open the stage: {exc}") from exc
    if not stage:
        raise ConversionError("USD could not open the stage")
    return stage


def convert(root_path: str, purpose: str = "render", variants: dict | None = None) -> tuple[bytes, dict]:
    """Open root_path as a USD stage and return (glb_bytes, report).

    purpose: "render" shows render + default geometry (what a renderer sees),
    "proxy" shows proxy + default geometry (lightweight viewport stand-ins).
    variants: variant selections to override, {prim path: {set: variant}}.
    """
    stage = open_stage(root_path, variants)

    composition_errors = []
    try:
        composition_errors = [str(e) for e in stage.GetCompositionErrors()][:200]
    except Exception:
        pass

    glb_bytes, info = _Converter(stage, purpose).run()
    info["compositionErrors"] = composition_errors
    info["variantSets"] = _variant_summary(stage)
    info["defaultVariants"] = default_prim_variants(stage)
    info["variants"] = variants or {}
    return glb_bytes, info


def default_prim_variants(stage) -> dict | None:
    """The default prim's variant sets, with what is selected now and the options.

    Without a default prim, a stage with a single root prim uses that one.
    """
    prim = stage.GetDefaultPrim()
    if not prim:
        roots = [p for p in stage.GetPseudoRoot().GetChildren() if p.IsActive() and not p.IsAbstract()]
        prim = roots[0] if len(roots) == 1 else None
    if not prim:
        return None
    vsets = prim.GetVariantSets()
    sets = [{"name": name, "selection": vsets.GetVariantSelection(name) or None,
             "options": list(vsets.GetVariantSet(name).GetVariantNames())}
            for name in vsets.GetNames()]
    sets = [v for v in sets if v["options"]]
    return {"prim": prim.GetPath().pathString, "sets": sets} if sets else None


def _variant_summary(stage, limit=200):
    out = []
    for prim in stage.Traverse():
        vsets = prim.GetVariantSets()
        for name in vsets.GetNames():
            vs = vsets.GetVariantSet(name)
            out.append({
                "prim": prim.GetPath().pathString,
                "set": name,
                "selection": vs.GetVariantSelection(),
                "options": list(vs.GetVariantNames()),
            })
            if len(out) >= limit:
                return out
    return out
