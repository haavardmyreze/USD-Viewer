"""UsdGeomBasisCurves -> polylines -> tubes (or line segments).

Linear and cubic curves (bezier, bspline, catmullRom; nonperiodic, periodic and
pinned wrap) are evaluated into polylines, then swept into round tubes using the
authored widths, as renderers show them. Curves without widths, or too many to
sweep, become GL lines.
"""
from __future__ import annotations

import numpy as np

SAMPLES_PER_SEGMENT = 6

_BASIS = {
    "bezier": lambda t: np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3], -1),
    "bspline": lambda t: np.stack([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4,
                                   -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3], -1) / 6.0,
    "catmullRom": lambda t: np.stack([-t ** 3 + 2 * t ** 2 - t, 3 * t ** 3 - 5 * t ** 2 + 2,
                                      -3 * t ** 3 + 4 * t ** 2 + t, t ** 3 - t ** 2], -1) * 0.5,
}


def _segments(counts, ctype, basis, wrap):
    """Control point indices (S, 4) of every cubic segment, and the curve of each."""
    seg_idx, seg_curve = [], []
    start = 0
    step = 3 if basis == "bezier" else 1
    for c, n in enumerate(counts):
        n = int(n)
        ids = np.arange(start, start + n)
        start += n
        if wrap == "periodic":
            if basis == "bezier":
                m = n // 3
                idx = (np.arange(m)[:, None] * 3 + np.arange(4)[None, :]) % n
            else:
                idx = (np.arange(n)[:, None] + np.arange(4)[None, :]) % n
        else:
            if wrap == "pinned" and basis != "bezier":
                # phantom end points (OpenSubdiv / Hydra convention) so the curve reaches its ends
                ids = np.concatenate([[-1 - ids[0]], ids, [-1 - ids[-1]]])  # negative: phantom
            m = (len(ids) - 1) // 3 if basis == "bezier" else len(ids) - 3
            if m <= 0:
                continue
            idx = ids[np.arange(m)[:, None] * step + np.arange(4)[None, :]]
            seg_idx.append(idx)
            seg_curve.append(np.full(m, c))
            continue
        seg_idx.append(ids[idx])
        seg_curve.append(np.full(len(idx), c))
    if not seg_idx:
        return np.zeros((0, 4), np.int64), np.zeros(0, np.int64)
    return np.concatenate(seg_idx), np.concatenate(seg_curve)


def polylines(points, counts, ctype="linear", basis="bezier", wrap="nonperiodic", vertex_data=(),
              samples=SAMPLES_PER_SEGMENT):
    """Evaluate curves to polylines.

    vertex_data: per control point arrays evaluated like the positions.
    Returns (P, curve_of_point, [data...], src) where src gives, per polyline point,
    (curve, parameter along the curve 0..1) for uniform / varying data.
    """
    points = np.asarray(points, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    data = [np.asarray(d, dtype=np.float64).reshape(len(points), -1) for d in vertex_data]
    if ctype != "cubic" or basis not in _BASIS:
        out_p, out_c, out_t = [], [], []
        out_d = [[] for _ in data]
        start = 0
        for c, n in enumerate(counts):
            ids = np.arange(start, start + int(n))
            start += int(n)
            if wrap == "periodic" and len(ids) > 2:
                ids = np.append(ids, ids[0])
            if len(ids) < 2:
                continue
            out_p.append(points[ids])
            out_c.append(np.full(len(ids), c))
            out_t.append(np.linspace(0, 1, len(ids)))
            for k, d in enumerate(data):
                out_d[k].append(d[ids])
        if not out_p:
            return np.zeros((0, 3)), np.zeros(0, np.int64), [np.zeros((0, d.shape[1])) for d in data], np.zeros(0)
        return (np.concatenate(out_p), np.concatenate(out_c), [np.concatenate(x) for x in out_d],
                np.concatenate(out_t))

    seg, seg_curve = _segments(counts, ctype, basis, wrap)
    if not len(seg):
        return np.zeros((0, 3)), np.zeros(0, np.int64), [np.zeros((0, d.shape[1])) for d in data], np.zeros(0)

    def ctrl(arr):
        """Control values per segment (S, 4, k), resolving pinned phantom points."""
        real = np.where(seg < 0, -1 - seg, seg)
        v = arr[real]
        phantom = seg < 0
        if phantom.any():
            # phantom = 2 * end - neighbour (reflection), neighbour is the next / previous point
            s, j = np.nonzero(phantom)
            # the first phantom mirrors the second real point, the last one the second to last
            nb = np.where(j == 0, seg[s, 2], seg[s, 1])
            nb = np.where(nb < 0, -1 - nb, nb)
            v[s, j] = 2 * arr[real[s, j]] - arr[nb]
        return v

    t = np.arange(samples) / samples
    B = _BASIS[basis](t)  # (k, 4)
    # last segment of each curve also gets its end point
    last = np.r_[seg_curve[1:] != seg_curve[:-1], True]
    Bend = _BASIS[basis](np.array([1.0]))[0]

    def evaluate(arr):
        cv = ctrl(arr)  # (S, 4, k)
        body = np.einsum("tj,sjk->stk", B, cv)  # (S, samples, k)
        end = np.einsum("j,sjk->sk", Bend, cv[last])
        parts = []
        ends_at = np.nonzero(last)[0]
        e = 0
        prev = 0
        for i in ends_at:
            parts.append(body[prev:i + 1].reshape(-1, arr.shape[1]))
            parts.append(end[e:e + 1])
            e += 1
            prev = i + 1
        return np.concatenate(parts)

    P = evaluate(points)
    curve = []
    param = []
    prev = 0
    for i in np.nonzero(last)[0]:
        c = seg_curve[i]
        n_seg = i + 1 - prev
        curve.append(np.full(n_seg * samples + 1, c))
        param.append(np.linspace(0, 1, n_seg * samples + 1))
        prev = i + 1
    return P, np.concatenate(curve), [evaluate(d) for d in data], np.concatenate(param)


def tubes(P, curve, radius, sides):
    """Sweep polylines into tubes. Returns (positions, normals, triangles, source point)."""
    n = len(P)
    if n < 2:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    same_next = np.r_[curve[1:] == curve[:-1], False]
    same_prev = np.r_[False, curve[1:] == curve[:-1]]
    nxt = np.where(same_next, np.arange(n) + 1, np.arange(n))
    prv = np.where(same_prev, np.arange(n) - 1, np.arange(n))
    T = P[nxt] - P[prv]
    L = np.linalg.norm(T, axis=1, keepdims=True)
    T = np.where(L > 1e-20, T / np.maximum(L, 1e-20), [[0.0, 0.0, 1.0]])
    # one reference direction per curve: the axis least aligned with its overall direction
    n_curves = int(curve.max()) + 1
    span = np.zeros((n_curves, 3))
    np.add.at(span, curve, np.abs(T))
    ref = np.eye(3)[np.argmin(span, axis=1)][curve]
    N = np.cross(T, ref)
    bad = np.linalg.norm(N, axis=1) < 1e-6
    if bad.any():
        N[bad] = np.cross(T[bad], np.eye(3)[np.argmin(np.abs(T[bad]), axis=1)])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    Bv = np.cross(T, N)
    ang = 2 * np.pi * np.arange(sides) / sides
    dirs = np.cos(ang)[None, :, None] * N[:, None, :] + np.sin(ang)[None, :, None] * Bv[:, None, :]  # (n, s, 3)
    pos = P[:, None, :] + dirs * np.asarray(radius, dtype=np.float64).reshape(-1, 1, 1)
    # quads between ring i and ring i + 1 of the same curve
    i = np.nonzero(same_next)[0]
    s = np.arange(sides)
    a = i[:, None] * sides + s[None, :]
    b = i[:, None] * sides + (s[None, :] + 1) % sides
    c = a + sides
    d = b + sides
    # counter-clockwise seen from outside: (b - a) x (c - a) points along +N
    tris = np.concatenate([np.stack([a, b, c], -1), np.stack([b, d, c], -1)], axis=1).reshape(-1, 3)
    src = np.repeat(np.arange(n), sides)
    return pos.reshape(-1, 3), dirs.reshape(-1, 3), tris, src


def line_segments(curve):
    """Index pairs joining consecutive polyline points of the same curve."""
    i = np.nonzero(curve[1:] == curve[:-1])[0]
    return np.stack([i, i + 1], -1)
