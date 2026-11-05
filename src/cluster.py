"""Pack several copies of one object (e.g. a watch charging stand) as tightly as
possible on a single shared base.

The spacing is not set by the bounding box of the object but by what has to stay
free around it: the object itself, the watch sitting on it, and the path the watch
takes when it is put on (slid in along the charger axis).  Copies may interleave
as long as none of those volumes touch.

Pipeline (all millimetres, objects centred on XY with their bottom at z = 0):

1. :func:`detect_watch_cradle` finds the charger puck recess in the mesh (a tilted
   planar face with a circular pocket) and turns it into a :class:`WatchSpec`.
2. :class:`ClusterPlanner` voxelises the object and the watch (2 mm), builds a
   collision map for every XY offset between two copies with FFT cross-correlation
   and greedily places copies where the bounding box grows least.
3. :func:`cluster_outline` merges the footprints into the base outline that hugs
   the cluster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh


# --------------------------------------------------------------------------- #
# Watch model
# --------------------------------------------------------------------------- #
@dataclass
class WatchSpec:
    """A watch sitting on its charger: a disc whose back touches the puck face."""
    axis: np.ndarray                 # unit normal pointing out of the puck face, towards the watch
    face_center: np.ndarray          # centre of the puck face
    diameter: float = 47.0           # Galaxy Watch Ultra
    thickness: float = 14.4

    @property
    def center(self) -> np.ndarray:
        return self.face_center + self.axis * self.thickness / 2

    def transformed(self, T: np.ndarray) -> "WatchSpec":
        R = T[:3, :3]
        return WatchSpec(R @ self.axis, (T @ np.append(self.face_center, 1.0))[:3], self.diameter, self.thickness)

    def top_z(self) -> float:
        """Highest point of the installed watch."""
        nz = float(self.axis[2])
        return float(self.center[2] + (self.diameter / 2) * math.sqrt(max(0.0, 1 - nz * nz)) + (self.thickness / 2) * abs(nz))


def detect_watch_cradle(mesh: trimesh.Trimesh, diameter: float = 47.0, thickness: float = 14.4) -> Optional[WatchSpec]:
    """Find the charger-puck recess: the biggest tilted planar face in the upper part of the
    object, plus the circular pocket in front of it.  ``None`` if there is none."""
    try:
        zmax = float(mesh.bounds[1][2])
        fac, ar, nr = mesh.facets, mesh.facets_area, mesh.facets_normal
        if len(fac) == 0:
            return None
        cen = np.array([mesh.triangles_center[f].mean(0) for f in fac])
        ok = (ar > 80) & (nr[:, 2] > 0.25) & (nr[:, 2] < 0.95) & (cen[:, 2] > 0.35 * zmax)
        if not ok.any():
            return None
        groups: Dict[tuple, List[int]] = {}
        for i in np.where(ok)[0]:
            groups.setdefault(tuple(np.round(nr[i], 1)), []).append(i)
        best = max(groups.values(), key=lambda g: ar[g].sum())
        n = nr[best].mean(0)
        n = n / np.linalg.norm(n)
        V = mesh.vertices
        up = V[:, 2] > 0.35 * zmax
        rim = float((V[up] @ n).max())
        t = np.cross([0.0, 0.0, 1.0], n)
        if np.linalg.norm(t) < 1e-6:
            t = np.array([1.0, 0.0, 0.0])
        t = t / np.linalg.norm(t)
        u = np.cross(n, t)
        fc, fn = mesh.triangles_center, mesh.face_normals
        fd = fc @ n
        sel = (np.abs(fn @ n) < 0.3) & (fd > rim - 12) & (fd < rim + 0.3)
        if sel.sum() < 20:
            return None
        P = np.c_[fc[sel] @ t, fc[sel] @ u]
        c0 = (cen[best] * ar[best, None]).sum(0) / ar[best].sum()
        guess, r = np.array([c0 @ t, c0 @ u]), 15.0
        Q = P
        for it in range(6):
            d = np.linalg.norm(P - guess, axis=1)
            keep = d < 22 if it == 0 else np.abs(d - r) < 2.5
            Q = P[keep]
            if len(Q) < 12:
                return None
            A = np.c_[2 * Q, np.ones(len(Q))]
            cx, cy, c = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
            guess, r = np.array([cx, cy]), math.sqrt(max(c + cx * cx + cy * cy, 0.0))
        if not 8.0 <= r <= 30.0:
            return None
        face_center = guess[0] * t + guess[1] * u + n * rim
        return WatchSpec(n, face_center, diameter, thickness)
    except Exception:                                    # detection is best effort
        return None


# --------------------------------------------------------------------------- #
# Voxels and collision map
# --------------------------------------------------------------------------- #
def _grid(lo: np.ndarray, hi: np.ndarray, pitch: float) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    origin = np.floor(lo / pitch) * pitch - 2 * pitch
    shape = tuple(int(math.ceil((h - o) / pitch)) + 3 for h, o in zip(hi, origin))
    return origin, shape  # type: ignore[return-value]


def _stand_voxels(mesh: trimesh.Trimesh, origin: np.ndarray, shape, pitch: float) -> np.ndarray:
    vox = mesh.voxelized(pitch).fill()
    pts = np.asarray(vox.points)
    idx = np.floor((pts - origin) / pitch + 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    out = np.zeros(shape, dtype=bool)
    idx = idx[ok]
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def _watch_voxels(w: WatchSpec, origin: np.ndarray, shape, pitch: float, margin: float, sweep: float) -> Tuple[np.ndarray, np.ndarray]:
    """(installed, swept) boolean grids.  ``swept`` additionally covers sliding the watch in
    ``sweep`` mm along the charger axis."""
    gx, gy, gz = np.meshgrid(*[origin[i] + pitch * np.arange(shape[i]) for i in range(3)], indexing="ij")
    p = np.stack([gx, gy, gz], axis=-1) - w.center
    a = p @ w.axis
    r = np.linalg.norm(p - a[..., None] * w.axis, axis=-1)
    inside_r = r <= w.diameter / 2 + margin
    inst = inside_r & (np.abs(a) <= w.thickness / 2 + margin)
    swept = inside_r & (a >= -w.thickness / 2 - margin) & (a <= w.thickness / 2 + sweep)
    return inst, swept


def _dilate(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return a
    from scipy import ndimage
    return ndimage.binary_dilation(a, structure=np.ones((3, 3, 3), dtype=bool), iterations=n)


def _xcorr(A: np.ndarray, B: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """C[d] = sum over x, y, z of A[p] * B[p - d] (z aligned: copies stand on one base)."""
    FA = np.fft.rfft2(A.astype(np.float32), s=size, axes=(0, 1))
    FB = np.fft.rfft2(B.astype(np.float32), s=size, axes=(0, 1))
    return np.fft.irfft2((FA * np.conj(FB)).sum(axis=2), s=size)


@dataclass
class Unit:
    """One orientation of the object with everything needed to place copies of it."""
    rz: int                          # extra rotation about Z (degrees) applied to the user's orientation
    mesh: trimesh.Trimesh            # centred on XY, bottom at z = 0
    watch: Optional[WatchSpec]
    conflict: np.ndarray             # (2nx, 2ny) bool, index = offset in voxels (wraps for negatives)
    footprint: Tuple[int, int, int, int]    # voxel bounds of stand+installed watch (x0, x1, y0, y1)
    origin: np.ndarray
    pitch: float
    height: float                    # tallest point of stand or watch


def _rotate_centered(mesh: trimesh.Trimesh, watch: Optional[WatchSpec], rz: int):
    m = mesh.copy()
    T = trimesh.transformations.rotation_matrix(math.radians(rz), [0, 0, 1])
    m.apply_transform(T)
    lo, hi = m.bounds
    T2 = np.eye(4)
    T2[:3, 3] = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
    m.apply_transform(T2)
    return m, (watch.transformed(T2 @ T) if watch is not None else None)


def build_unit(mesh: trimesh.Trimesh, watch: Optional[WatchSpec], rz: int, pitch: float = 2.0,
               gap: float = 3.0, sweep: float = 40.0, margin: float = 1.5) -> Unit:
    """``gap`` is the free space kept between neighbouring copies; ``margin`` extra room round the watch."""
    m, w = _rotate_centered(mesh, watch, rz)
    lo, hi = m.bounds[0].copy(), m.bounds[1].copy()
    if w is not None:                                    # grid must contain the swept watch
        c, R, t = w.center, w.diameter / 2 + margin + 2, w.thickness / 2 + sweep + margin
        ext = R + t
        lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
        lo[2] = min(lo[2], 0.0)
    origin, shape = _grid(lo, hi, pitch)
    S0 = _stand_voxels(m, origin, shape, pitch)
    grow = int(round(gap / 2 / pitch))
    S = _dilate(S0, grow)
    if w is not None:
        Wi0, _ = _watch_voxels(w, origin, shape, pitch, 0.0, sweep)
        Wi, Ws = _watch_voxels(w, origin, shape, pitch, margin, sweep)
        Wi, Ws = _dilate(Wi, grow), _dilate(Ws, grow)
    else:
        Wi0 = Wi = Ws = np.zeros(shape, dtype=bool)
    occ = S | Wi
    size = (2 * shape[0], 2 * shape[1])
    conf = (_xcorr(occ, occ, size) + _xcorr(Ws, occ, size) + _xcorr(occ, Ws, size)) > 0.5
    xs, ys = np.where((S0 | Wi0).any(axis=2))              # real extent, without the safety margins
    height = float(m.bounds[1][2])
    if w is not None:
        height = max(height, w.top_z())
    return Unit(rz, m, w, conf, (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())), origin, pitch, height)


def _beam_pack(unit: Unit, count: int, fits=None, width: int = 8, expand: int = 10) -> Optional[List[Tuple[int, int]]]:
    """Positions (voxel offsets) of ``count`` copies minimising the bounding-box area.

    Beam search: every partial arrangement is extended with its ``expand`` best new positions and
    the ``width`` best results survive, so one early unlucky placement cannot ruin the layout.
    ``fits(w_array, h_array) -> bool_array`` (sizes in mm) rejects arrangements that are too big;
    returns None if every arrangement is rejected.
    """
    conf = unit.conflict
    NX, NY = conf.shape
    nx, ny = NX // 2, NY // 2
    fx0, fx1, fy0, fy1 = unit.footprint
    pitch = unit.pitch
    dxs, dys = np.meshgrid(np.arange(-(nx - 1), nx), np.arange(-(ny - 1), ny), indexing="ij")
    free = ~conf[dxs % NX, dys % NY]
    offs = np.c_[dxs[free], dys[free]]                       # every collision-free offset between two copies

    def size_of(allp):
        x0, x1 = allp[..., 0].min(0) + fx0, allp[..., 0].max(0) + fx1
        y0, y1 = allp[..., 1].min(0) + fy0, allp[..., 1].max(0) + fy1
        return (x1 - x0 + 1) * pitch, (y1 - y0 + 1) * pitch

    beam = [(0.0, np.zeros((1, 2), dtype=int))]
    for _ in range(count - 1):
        nxt: Dict[tuple, Tuple[float, np.ndarray]] = {}
        for _, pos in beam:
            cand = np.unique((pos[:, None, :] + offs[None, :, :]).reshape(-1, 2), axis=0)
            ok = np.ones(len(cand), dtype=bool)
            for p in pos:
                d = cand - p
                inside = (np.abs(d[:, 0]) < nx) & (np.abs(d[:, 1]) < ny)
                hit = np.zeros(len(cand), dtype=bool)
                hit[inside] = conf[d[inside, 0] % NX, d[inside, 1] % NY]
                ok &= ~hit
            cand = cand[ok]
            if len(cand) == 0:
                continue
            allp = np.concatenate([np.repeat(pos[:, None, :], len(cand), axis=1), cand[None, :, :]], axis=0)
            w, h = size_of(allp)
            valid = np.ones(len(cand), dtype=bool) if fits is None else np.asarray(fits(w, h), dtype=bool)
            score = np.where(valid, w * h + 0.01 * (w + h), np.inf)
            for i in np.argsort(score)[:expand]:
                if not np.isfinite(score[i]):
                    break
                new = np.vstack([pos, cand[i]])
                norm = new - new.min(axis=0)
                key = tuple(sorted(map(tuple, norm.tolist())))
                if key not in nxt or score[i] < nxt[key][0]:
                    nxt[key] = (float(score[i]), new)
        if not nxt:
            return None
        beam = sorted(nxt.values(), key=lambda t: t[0])[:width]
    return [(int(a), int(b)) for a, b in beam[0][1]]


def _lattice_pack(unit: Unit, count: int, fits=None, bins: int = 72) -> Optional[List[Tuple[int, int]]]:
    """Best regular arrangement: copies at i*A + j*B, ``c`` per row.  A and B are the tightest
    collision-free offsets in each compass direction, so rows and columns sit exactly at the
    contact distance.  Neat to build and often smaller than anything the beam search finds."""
    conf = unit.conflict
    NX, NY = conf.shape
    nx, ny = NX // 2, NY // 2
    fx0, fx1, fy0, fy1 = unit.footprint
    pitch = unit.pitch
    dxs, dys = np.meshgrid(np.arange(-(nx - 1), nx), np.arange(-(ny - 1), ny), indexing="ij")
    free = ~conf[dxs % NX, dys % NY]
    free[nx - 1, ny - 1] = False
    offs = np.c_[dxs[free], dys[free]]
    if len(offs) == 0:
        return None
    ang = np.arctan2(offs[:, 1], offs[:, 0])
    norm = np.hypot(offs[:, 0], offs[:, 1])
    b = ((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    frontier = []
    for k in range(bins):
        m = np.where(b == k)[0]
        if len(m):
            frontier.append(offs[m[np.argmin(norm[m])]])
    F = np.array(frontier)
    nf = len(F)

    def is_free(d):                                            # d: (..., 2) int offsets
        d = np.asarray(d)
        inside = (np.abs(d[..., 0]) < nx) & (np.abs(d[..., 1]) < ny)
        out = np.ones(d.shape[:-1], dtype=bool)
        out[inside] = ~conf[d[..., 0][inside] % NX, d[..., 1][inside] % NY]
        return out

    # A - B must be collision free for any arrangement with two rows and two columns (A + B too from four copies)
    plus, minus = is_free(F[:, None, :] + F[None, :, :]), is_free(F[:, None, :] - F[None, :, :])
    cross_ok = (F[:, None, 0] * F[None, :, 1] - F[:, None, 1] * F[None, :, 0]) != 0
    best, best_score = None, np.inf
    seen = set()

    def consider(P):
        nonlocal best, best_score
        key = tuple(sorted(map(tuple, (P - P.min(0)).tolist())))
        if key in seen:
            return
        seen.add(key)
        D = P[:, None, :] - P[None, :, :]
        inside = (np.abs(D[..., 0]) < nx) & (np.abs(D[..., 1]) < ny)
        np.fill_diagonal(inside, False)
        if conf[D[..., 0][inside] % NX, D[..., 1][inside] % NY].any():
            return
        w = (P[:, 0].max() - P[:, 0].min() + fx1 - fx0 + 1) * pitch
        h = (P[:, 1].max() - P[:, 1].min() + fy1 - fy0 + 1) * pitch
        if fits is not None and not bool(np.asarray(fits(np.array([w]), np.array([h])))[0]):
            return
        sc = w * h + 0.01 * (w + h)
        if sc < best_score:
            best, best_score = P.copy(), sc

    idx = np.arange(count)
    for i in range(nf):
        consider(idx[:, None] * F[i][None])                    # one straight row
        if count < 3:
            continue
        for j in np.where(cross_ok[i] & minus[i] & (plus[i] | (count < 4)))[0]:
            for c in range(2, count):
                if count > c:
                    consider((idx % c)[:, None] * F[i][None] + (idx // c)[:, None] * F[j][None])
    return None if best is None else [(int(x), int(y)) for x, y in best]


@dataclass
class ClusterPlan:
    count: int
    rz: int
    offsets: List[Tuple[float, float]]     # copy offsets in mm relative to copy 0 (in the rotated, centred frame)
    unit: Unit
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) mm of stands + installed watches, same frame
    height: float                          # tallest point above the base top

    @property
    def size(self) -> Tuple[float, float]:
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]


class ClusterPlanner:
    """Plans the tightest arrangement of copies of one object."""

    def __init__(self, mesh: trimesh.Trimesh, watch: Optional[WatchSpec] = None, pitch: float = 2.0,
                 gap: float = 3.0, sweep: float = 40.0, rotations: Sequence[int] = (0, 90, 180, 270)):
        self.pitch = pitch
        self.units = [build_unit(mesh, watch, rz, pitch, gap, sweep) for rz in rotations]

    def plan(self, count: int, max_wh: Optional[Tuple[float, float]] = None, fits=None) -> Optional[ClusterPlan]:
        """Smallest-area arrangement over all tried orientations.  ``max_wh`` limits the bounding box
        (width, height in the frame of the case); ``fits(w, h)`` is an extra vectorised test (printer bed).
        ``None`` if no arrangement satisfies both."""
        if max_wh is not None:
            lim = fits
            fits = (lambda w, h: (w <= max_wh[0] + 1e-6) & (h <= max_wh[1] + 1e-6) & (True if lim is None else lim(w, h)))
        best: Optional[ClusterPlan] = None
        for u in self.units:
            for pos in (_lattice_pack(u, count, fits),):
                if pos is None:
                    continue
                p = np.array(pos)
                fx0, fx1, fy0, fy1 = u.footprint
                pitch = u.pitch
                x0, x1 = u.origin[0] + (p[:, 0].min() + fx0) * pitch, u.origin[0] + (p[:, 0].max() + fx1 + 1) * pitch
                y0, y1 = u.origin[1] + (p[:, 1].min() + fy0) * pitch, u.origin[1] + (p[:, 1].max() + fy1 + 1) * pitch
                plan = ClusterPlan(count, u.rz, [(float(a * pitch), float(b * pitch)) for a, b in pos], u,
                                   (float(x0), float(y0), float(x1), float(y1)), u.height)
                if best is None or (plan.size[0] * plan.size[1], sum(plan.size)) < (best.size[0] * best.size[1], sum(best.size)):
                    best = plan
        return best


# --------------------------------------------------------------------------- #
# Footprints and base outline
# --------------------------------------------------------------------------- #
def mesh_footprint(mesh: trimesh.Trimesh):
    """Top-down outline of a mesh (union of projected triangles, holes filled)."""
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    tris = mesh.triangles[:, :, :2]
    keep = np.abs(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])) > 1e-9
    u = shapely.union_all(shapely.polygons(tris[keep]))
    if isinstance(u, MultiPolygon):
        u = u.buffer(0.3).buffer(-0.3)
    if isinstance(u, MultiPolygon):
        u = u.convex_hull
    return Polygon(u.exterior)


def cluster_outline(plan: ClusterPlan, margin: float = 3.0, close: float = 10.0):
    """Base outline hugging all copies: union of the footprints, grown by ``margin`` and
    closed up (``close`` mm) so narrow slits between neighbours are filled."""
    from shapely import affinity
    from shapely.ops import unary_union

    fp = mesh_footprint(plan.unit.mesh)
    polys = [affinity.translate(fp, dx, dy) for dx, dy in plan.offsets]
    poly = unary_union(polys).buffer(margin, join_style="round", quad_segs=8)
    if close > 0:
        poly = poly.buffer(close, join_style="round", quad_segs=8).buffer(-close, join_style="round", quad_segs=8)
    if poly.geom_type == "MultiPolygon":
        poly = poly.convex_hull
    from shapely.geometry import Polygon
    return Polygon(poly.exterior).simplify(0.1)


# --------------------------------------------------------------------------- #
# Loading, watch footprint, caching
# --------------------------------------------------------------------------- #
def load_unit_mesh(stl_path: str, rotation_deg=(0, 0, 0)) -> trimesh.Trimesh:
    """Oriented, cleaned mesh centred on XY with its bottom at z = 0 (the frame the planner works in)."""
    from . import geometry as geo

    mesh = geo._load_oriented_mesh(stl_path, rotation_deg)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def watch_footprint(w: WatchSpec):
    """Top-down outline of the installed watch."""
    from shapely.geometry import MultiPoint

    n = w.axis
    t = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(t) < 1e-6:
        t = np.array([1.0, 0.0, 0.0])
    t = t / np.linalg.norm(t)
    u = np.cross(n, t)
    ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    pts = []
    for side in (-0.5, 0.5):
        c = w.center + n * side * w.thickness
        for a in ang:
            q = c + (w.diameter / 2) * (math.cos(a) * t + math.sin(a) * u)
            pts.append((q[0], q[1]))
    return MultiPoint(pts).convex_hull


_PLANNERS: Dict[tuple, Tuple[trimesh.Trimesh, Optional[WatchSpec], "ClusterPlanner"]] = {}


def get_planner(stl_path: str, rotation_deg=(0, 0, 0), diameter: float = 47.0, thickness: float = 14.4,
                gap: float = 3.0, use_watch: bool = True):
    """Cached ``(mesh, watch, planner)``; building the voxel maps takes a couple of seconds."""
    import os

    key = (stl_path, os.path.getmtime(stl_path), tuple(int(r) for r in rotation_deg), round(diameter, 2),
           round(thickness, 2), round(gap, 2), bool(use_watch))
    if key not in _PLANNERS:
        mesh = load_unit_mesh(stl_path, rotation_deg)
        watch = detect_watch_cradle(mesh, diameter, thickness) if use_watch else None
        if len(_PLANNERS) > 6:
            _PLANNERS.pop(next(iter(_PLANNERS)))
        _PLANNERS[key] = (mesh, watch, ClusterPlanner(mesh, watch, gap=gap))
    return _PLANNERS[key]
