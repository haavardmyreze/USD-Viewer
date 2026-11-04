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
  * upAxis + metersPerUnit (output is Y-up, metres)
"""

from __future__ import annotations

import io
import math
import os
import zipfile
from collections import Counter

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

from .glb import GlbBuilder

CONVERTER_VERSION = "1"

# glTF sampler wrap modes
REPEAT, CLAMP, MIRROR = 10497, 33071, 33648
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


class _Converter:
    def __init__(self, stage: Usd.Stage) -> None:
        self.stage = stage
        self.time = Usd.TimeCode.EarliestTime()
        self.report = _Report()
        self.glb = GlbBuilder()
        self.xf_cache = UsdGeom.XformCache(self.time)
        self.material_index: dict = {}
        self.image_index: dict = {}
        self.mesh_cache: dict = {}
        self.stats = Counter()

    # -- traversal ---------------------------------------------------------

    def run(self) -> tuple[bytes, dict]:
        stage = self.stage
        up = UsdGeom.GetStageUpAxis(stage)
        mpu = UsdGeom.GetStageMetersPerUnit(stage)

        children: list[int] = []
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        it = iter(Usd.PrimRange(stage.GetPseudoRoot(), predicate))
        for prim in it:
            if prim.IsPseudoRoot():
                continue
            type_name = prim.GetTypeName()
            if type_name in _SKIP_SUBTREE_TYPES:
                it.PruneChildren()
                continue
            img = UsdGeom.Imageable(prim)
            if img:
                if img.ComputeVisibility(self.time) == UsdGeom.Tokens.invisible:
                    it.PruneChildren()
                    continue
                purpose = img.ComputePurpose()
                if purpose in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                    it.PruneChildren()
                    continue
            if type_name == "PointInstancer":
                self.report.skipped_types[type_name] += 1
                it.PruneChildren()
                continue
            if type_name in _GPRIM_TYPES:
                node = self._convert_gprim(prim, type_name)
                if node is not None:
                    children.append(node)
            elif type_name == "Camera":
                self.stats["cameras"] += 1
            elif type_name not in _KNOWN_NON_GEOM and prim.IsA(UsdGeom.Gprim):
                self.report.skipped_types[type_name] += 1

        s = float(mpu)
        if up == UsdGeom.Tokens.z:
            root_matrix = [s, 0, 0, 0, 0, 0, -s, 0, 0, s, 0, 0, 0, 0, 0, 1]
        else:
            root_matrix = [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]
        root = {"name": "usd_stage_root", "children": children}
        if root_matrix != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]:
            root["matrix"] = root_matrix
        self.glb.add_node(root, root=True)

        for t, n in self.report.skipped_types.items():
            self.report.warn(f"{n} {t} prim(s) skipped (not supported yet)")

        info = {
            "upAxis": str(up),
            "metersPerUnit": mpu,
            "defaultPrim": stage.GetDefaultPrim().GetPath().pathString if stage.GetDefaultPrim() else None,
            "stats": {
                "meshPrims": self.stats["mesh_prims"],
                "uniqueMeshes": len(self.glb.gltf["meshes"]),
                "triangles": self.stats["triangles"],
                "uniqueTriangles": self.stats["unique_triangles"],
                "materials": len(self.glb.gltf["materials"]),
                "textures": len(self.glb.gltf["images"]),
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

    def _convert_gprim(self, prim, type_name):
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
        self.stats["triangles"] += self.mesh_cache[("tris", mesh_index)]
        self.stats["mesh_prims"] += 1

        node = {"name": path, "mesh": mesh_index}
        m = self.xf_cache.GetLocalToWorldTransform(prim)
        if m != Gf.Matrix4d(1.0):
            node["matrix"] = [float(m[r][c]) for r in range(4) for c in range(4)]
        return self.glb.add_node(node)

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
        return self.glb.add_mesh(path, primitives), n_tris

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
        idx = self.glb.add_material(mat)
        self.material_index[key] = idx
        return idx

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
        info = {
            "path": f.resolvedPath if isinstance(f, Sdf.AssetPath) else None,
            "authored": f.path if isinstance(f, Sdf.AssetPath) else None,
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
        surf = self._preview_surface(material)
        out = {"name": mat_path}
        pbr = {"baseColorFactor": [0.7, 0.7, 0.7, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}
        out["pbrMetallicRoughness"] = pbr
        if surf is None:
            self.report.warn(f"Material {mat_path} has no UsdPreviewSurface (MDL/MaterialX not supported yet); "
                             "using grey")
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
            self.report.warn(f"Material {mat_path}: specular workflow approximated as metallic/roughness")

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

    def _load_pil(self, info, mat_path):
        from PIL import Image

        data = self._read_asset(info["path"])
        if data is None:
            self.report.warn(f"Texture not found: {info['authored']} (material {mat_path})")
            return None
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception:
            self.report.warn(f"Unsupported texture format: {info['authored']}")
            return None

    def _encode_png(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()

    def _image_from_key(self, key, producer):
        idx = self.image_index.get(key)
        if idx is None:
            result = producer()
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
                self.report.warn(f"Texture not found: {info['authored']} (material {mat_path})")
                return None
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                return data, "image/png"
            if data[:3] == b"\xff\xd8\xff":
                return data, "image/jpeg"
            img = self._load_pil(info, mat_path)
            if img is None:
                return None
            return self._encode_png(img.convert("RGBA" if "A" in img.getbands() else "RGB")), "image/png"

        image = self._image_from_key(("file", info["path"]), produce)
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
            img = self._load_pil(info, mat_path)
            if img is None:
                return None
            from PIL import Image
            ch = self._channel(img, info["channel"])
            return self._encode_png(Image.merge("RGB", (ch, ch, ch))), "image/png"

        image = self._image_from_key(("chan", info["path"], info["channel"]), produce)
        if image is None:
            return None
        return {"index": self.glb.add_texture(image, self.glb.add_sampler(info["wrapS"], info["wrapT"]))}

    def _base_color_texture(self, diffuse, opacity, mat_path):
        if diffuse and (not opacity or (opacity["path"] == diffuse["path"] and opacity["channel"] == "a")):
            return self._texture_ref(diffuse, mat_path, "baseColor")

        def produce():
            from PIL import Image
            base = self._load_pil(diffuse, mat_path) if diffuse else None
            op = self._load_pil(opacity, mat_path) if opacity else None
            if base is None and op is None:
                return None
            size = (base or op).size
            rgb = base.convert("RGB") if base is not None else Image.new("RGB", size, (255, 255, 255))
            alpha = self._channel(op, opacity["channel"]).resize(size) if op is not None else Image.new("L", size, 255)
            r, g, b = rgb.split()
            return self._encode_png(Image.merge("RGBA", (r, g, b, alpha))), "image/png"

        key = ("base", diffuse and diffuse["path"], opacity and (opacity["path"], opacity["channel"]))
        image = self._image_from_key(key, produce)
        if image is None:
            return None
        ref = diffuse or opacity
        return {"index": self.glb.add_texture(image, self.glb.add_sampler(ref["wrapS"], ref["wrapT"]))}

    def _packed_mr_texture(self, metal, rough, mat_path):
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
            return self._encode_png(Image.merge("RGB", (zero, g, b))), "image/png"

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
    return {
        "layers": sorted(l.realPath or l.identifier for l in layers),
        "assets": sorted(str(a) for a in assets),
        "unresolved": sorted(str(u) for u in unresolved),
    }


def convert(root_path: str) -> tuple[bytes, dict]:
    """Open root_path as a USD stage and return (glb_bytes, report)."""
    if not os.path.isfile(root_path):
        raise ConversionError(f"Root layer not found: {root_path}")
    try:
        stage = Usd.Stage.Open(root_path, Usd.Stage.LoadAll)
    except Exception as exc:
        raise ConversionError(f"USD could not open the stage: {exc}") from exc
    if not stage:
        raise ConversionError("USD could not open the stage")

    composition_errors = []
    try:
        composition_errors = [str(e) for e in stage.GetCompositionErrors()][:200]
    except Exception:
        pass

    glb_bytes, info = _Converter(stage).run()
    info["compositionErrors"] = composition_errors
    info["variantSets"] = _variant_summary(stage)
    return glb_bytes, info


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
