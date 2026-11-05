"""Uniform Catmull-Clark refinement for USD subdivision meshes.

USD meshes default to subdivisionScheme = catmullClark: renderers (Karma, Storm at
higher complexity) show the smooth surface, not the control cage. This refines
the cage a few levels, following OpenSubdiv's rules where they matter for the look:

- smooth, boundary and crease rules for vertices and edges;
- interpolateBoundary edgeAndCorner (USD's default; boundary corners stay put),
  edgeOnly and none (treated like edgeOnly);
- creases (creaseIndices / creaseLengths / creaseSharpnesses), semi-sharp ones
  blended and decremented per level; corners (cornerIndices / cornerSharpnesses);
- "bilinear" refines linearly; "loop" (triangle meshes) is refined as Catmull-Clark.

Face-varying data (UVs) is interpolated linearly per face, i.e. USD's
faceVaryingLinearInterpolation "all" rather than the default "cornersPlus1".
Positions are not pushed to the limit surface: at level 2 the difference is small.

Refinement is linear in the vertex data, so every level is kept as a set of index
arrays and replayed for each primvar (vertex, faceVarying, uniform).
"""
from __future__ import annotations

import numpy as np


class _Level:
    """One refinement step of a polygon mesh into quads."""

    def __init__(self, n_points, counts, indices, edge_sharp: dict, corner_sharp: np.ndarray,
                 boundary: str, bilinear: bool):
        counts = np.asarray(counts, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        self.n_points = n_points
        self.n_faces = len(counts)
        self.n_corners = len(indices)
        self.bilinear = bilinear
        starts = np.zeros(self.n_faces, dtype=np.int64)
        if self.n_faces > 1:
            starts[1:] = np.cumsum(counts)[:-1]
        face_of = np.repeat(np.arange(self.n_faces), counts)
        local = np.arange(self.n_corners) - starts[face_of]
        nxt = starts[face_of] + (local + 1) % counts[face_of]
        prv = starts[face_of] + (local - 1) % counts[face_of]
        self.face_of, self.counts, self.indices = face_of, counts, indices
        self.nxt, self.prv = nxt, prv

        # edges: one per unordered vertex pair; corner c owns edge (c -> next corner)
        a, b = indices, indices[nxt]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        keys = lo * (n_points + 1) + hi
        ukeys, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        self.edge_of_corner = inv.reshape(-1)
        n_edges = len(ukeys)
        self.n_edges = n_edges
        self.e0, self.e1 = lo[first], hi[first]
        self.edge_keys = ukeys
        nf = np.bincount(self.edge_of_corner, minlength=n_edges)
        # the (up to) two faces of each edge
        order = np.argsort(self.edge_of_corner, kind="stable")
        sorted_e = self.edge_of_corner[order]
        first_pos = np.searchsorted(sorted_e, np.arange(n_edges))
        self.ef0 = face_of[order[first_pos]]
        second = np.minimum(first_pos + 1, len(order) - 1)
        self.ef1 = np.where(nf >= 2, face_of[order[second]], self.ef0)

        # sharpness per edge: boundary and non-manifold edges are infinitely sharp
        sharp = np.zeros(n_edges)
        if edge_sharp:
            k = np.fromiter(edge_sharp.keys(), dtype=np.int64, count=len(edge_sharp))
            s = np.fromiter(edge_sharp.values(), dtype=np.float64, count=len(edge_sharp))
            pos = np.searchsorted(ukeys, k)
            ok = (pos < n_edges) & (ukeys[np.minimum(pos, n_edges - 1)] == k)
            sharp[pos[ok]] = np.maximum(sharp[pos[ok]], s[ok])
        sharp[nf != 2] = np.inf
        self.sharp = sharp
        self.edge_w = np.clip(sharp, 0.0, 1.0)  # 0 smooth .. 1 sharp

        # vertex classification
        is_sharp = sharp > 0
        n = n_points
        self.valence = np.bincount(np.concatenate([self.e0, self.e1]), minlength=n)
        self.faces_at = np.bincount(indices, minlength=n)
        sharp_e = np.nonzero(is_sharp)[0]
        self.n_sharp = np.bincount(np.concatenate([self.e0[sharp_e], self.e1[sharp_e]]), minlength=n)
        s_sum = np.bincount(np.concatenate([self.e0[sharp_e], self.e1[sharp_e]]),
                            weights=np.concatenate([self.edge_w[sharp_e]] * 2), minlength=n)
        self.crease_w = np.where(self.n_sharp > 0, s_sum / np.maximum(self.n_sharp, 1), 0.0)
        boundary_v = np.bincount(np.concatenate([self.e0[nf == 1], self.e1[nf == 1]]), minlength=n) > 0
        corner = (self.n_sharp > 2) | (corner_sharp > 0)
        if boundary == "edgeAndCorner":
            corner |= boundary_v & (self.faces_at == 1)
        self.corner = corner
        self.crease = (self.n_sharp == 2) & ~corner
        self.sharp_e = sharp_e
        self.isolated = self.faces_at == 0

        # topology of the refined mesh: one quad per corner
        E = n_edges
        v_new = indices
        e_next = n + self.edge_of_corner
        f_new = n + E + face_of
        e_prev = n + self.edge_of_corner[prv]
        self.new_indices = np.stack([v_new, e_next, f_new, e_prev], axis=1).reshape(-1)
        self.new_counts = np.full(self.n_corners, 4, dtype=np.int64)
        self.parent_face = face_of
        self.new_n_points = n + E + self.n_faces

        # sharpness carried to the next level: both halves of a crease edge, minus one
        child = {}
        keep = np.nonzero((sharp > 1) & np.isfinite(sharp) & (nf == 2))[0]
        if len(keep):
            mid = n + keep
            m = self.new_n_points + 1
            for x, y in ((self.e0[keep], mid), (mid, self.e1[keep])):
                kk = np.minimum(x, y) * m + np.maximum(x, y)
                child.update(zip(kk.tolist(), (sharp[keep] - 1).tolist()))
        self.child_sharp = child
        # corners stay pinned at every level (semi-sharp corners are treated as sharp)
        self.child_corner = np.concatenate([corner_sharp, np.zeros(E + self.n_faces)])

    # -- data ---------------------------------------------------------------

    def face_points(self, V):
        F = np.zeros((self.n_faces,) + V.shape[1:])
        np.add.at(F, self.face_of, V[self.indices])
        return F / self.counts.reshape((-1,) + (1,) * (V.ndim - 1))

    def vertex(self, V):
        """Refine per-point data (positions and vertex-interpolated primvars)."""
        V = np.asarray(V, dtype=np.float64)
        shape1 = V.shape[1:]
        F = self.face_points(V)
        a, b = V[self.e0], V[self.e1]
        mid = (a + b) * 0.5
        if self.bilinear:
            return np.concatenate([V, mid, F])
        smooth_e = (a + b + F[self.ef0] + F[self.ef1]) * 0.25
        w = self.edge_w.reshape((-1,) + (1,) * len(shape1))
        E = smooth_e * (1 - w) + mid * w

        n = self.n_points
        bc = lambda idx, vals: _scatter(n, idx, vals)  # noqa: E731
        val = self.valence.reshape((-1,) + (1,) * len(shape1)).astype(np.float64)
        fa = self.faces_at.reshape(val.shape).astype(np.float64)
        Q = bc(self.indices, F[self.face_of]) / np.maximum(fa, 1)
        R = (bc(self.e0, mid) + bc(self.e1, mid)) / np.maximum(val, 1)
        smooth_v = (Q + 2 * R + (val - 3) * V) / np.maximum(val, 1)
        smooth_v = np.where(val >= 3, smooth_v, V)
        se = self.sharp_e
        nb = bc(self.e0[se], V[self.e1[se]]) + bc(self.e1[se], V[self.e0[se]])
        crease_v = 0.75 * V + 0.125 * nb
        cw = self.crease_w.reshape(val.shape)
        crease_v = smooth_v * (1 - cw) + crease_v * cw
        out = np.where(self.crease.reshape(val.shape), crease_v, smooth_v)
        out = np.where((self.corner | self.isolated).reshape(val.shape), V, out)
        return np.concatenate([out, E, F])

    def facevarying(self, C):
        """Refine per-corner data, linearly within each face."""
        C = np.asarray(C, dtype=np.float64)
        Fc = np.zeros((self.n_faces,) + C.shape[1:])
        np.add.at(Fc, self.face_of, C)
        Fc /= self.counts.reshape((-1,) + (1,) * (C.ndim - 1))
        quad = np.stack([C, (C + C[self.nxt]) * 0.5, Fc[self.face_of], (C + C[self.prv]) * 0.5], axis=1)
        return quad.reshape((-1,) + C.shape[1:])

    def uniform(self, U):
        return np.asarray(U)[self.parent_face]


def _scatter(n, idx, vals):
    out = np.zeros((n,) + vals.shape[1:])
    np.add.at(out, idx, vals)
    return out


class Refinement:
    """Several levels of refinement of one mesh, replayable on its primvars."""

    def __init__(self, n_points, counts, indices, levels, *, scheme="catmullClark", boundary="edgeAndCorner",
                 crease_indices=None, crease_lengths=None, crease_sharpness=None,
                 corner_indices=None, corner_sharpness=None):
        bilinear = scheme == "bilinear"
        edge_sharp = {}
        if crease_indices is not None and crease_lengths is not None and len(crease_indices):
            ci = np.asarray(crease_indices, dtype=np.int64)
            cl = np.asarray(crease_lengths, dtype=np.int64)
            cs = np.asarray(crease_sharpness if crease_sharpness is not None else [], dtype=np.float64)
            per_edge = len(cs) == int(np.sum(np.maximum(cl - 1, 0)))
            pos = 0
            k = 0
            m = n_points + 1
            for li, length in enumerate(cl):
                chain = ci[pos:pos + length]
                for j in range(len(chain) - 1):
                    if per_edge:
                        s = cs[k] if k < len(cs) else 0.0
                    else:
                        s = cs[li] if li < len(cs) else 0.0
                    k += 1
                    x, y = int(chain[j]), int(chain[j + 1])
                    key = min(x, y) * m + max(x, y)
                    edge_sharp[key] = max(edge_sharp.get(key, 0.0), 10.0 if s >= 10 else float(s))
                pos += length
        corner = np.zeros(n_points)
        if corner_indices is not None and len(corner_indices):
            cs = np.asarray(corner_sharpness if corner_sharpness is not None else [], dtype=np.float64)
            for i, v in enumerate(np.asarray(corner_indices, dtype=np.int64)):
                if 0 <= v < n_points:
                    corner[v] = max(corner[v], cs[i] if i < len(cs) else 10.0)
        if boundary not in ("edgeAndCorner", "edgeOnly"):
            boundary = "edgeOnly"
        self.levels: list[_Level] = []
        n, c, ix = n_points, counts, indices
        for _ in range(levels):
            lv = _Level(n, c, ix, edge_sharp, corner, boundary, bilinear)
            self.levels.append(lv)
            n, c, ix = lv.new_n_points, lv.new_counts, lv.new_indices
            edge_sharp = lv.child_sharp  # keyed with the next level's point count
            corner = lv.child_corner
        self.n_points, self.counts, self.indices = n, c, ix
        parent = np.arange(len(counts))
        for lv in self.levels:
            parent = parent[lv.parent_face]
        self.parent_face = parent

    def vertex(self, V):
        V = np.asarray(V, dtype=np.float64)
        for lv in self.levels:
            V = lv.vertex(V)
        return V

    def facevarying(self, C):
        C = np.asarray(C, dtype=np.float64)
        for lv in self.levels:
            C = lv.facevarying(C)
        return C

    def uniform(self, U):
        return np.asarray(U)[self.parent_face]

