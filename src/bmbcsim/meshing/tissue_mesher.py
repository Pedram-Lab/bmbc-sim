"""Tetrahedral mesh of a packed-cell tissue in a box, from per-cell signed distance functions.

Pipeline (see `surface_mesh` and `volume_mesh`):

1. Membranes: one triangle mesh per cell (marching cubes, isotropic remeshing, exact projection), cut at the box
   walls, cleaned up along the cut, and corrected for the volume lost to faceting (secant offset).
2. Walls: one constrained Delaunay triangulation per box face, with the membrane outlines and the box edges as
   fixed segments. Each wall region is named `cell_boundary` (inside a cell) or after the wall (ECS).
3. Volume: TetGen fills the closed multi-domain surface without changing it, the tets are labelled per domain, and
   netgen optimizes the interior.

A cell is any object with the `Cell` interface, e.g. `bmbcsim.geometry.cells.RoundedPolytope`. The names and the
file I/O of tissue meshes are in `bmbcsim.meshing.tissue_io`. Meshing needs the `geometry` extra (pymeshlab,
triangle, tetgen).
"""
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import ngsolve as ngs
import pyvista as pv
from netgen.meshing import MeshingParameters
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bmbcsim.logging import logger
from bmbcsim.meshing.tissue_io import (CELL0, ECS, OUTSIDE, WALLS, _face_key, box_walls, membrane_name,
                                       tissue_mesh_from_labels)


class MeshingError(RuntimeError):
    """The mesher could not mesh this geometry (a retry with other mesh parameters may succeed)."""


class Cell(Protocol):
    """A closed cell surface given by an exact signed distance function."""

    def phi(self, X: np.ndarray, band: float = np.inf) -> np.ndarray:
        """Signed distance (N,) of the points X (N, 3), negative inside; exact below `band`, elsewhere a lower
        bound of at least `band`."""

    def project(self, X: np.ndarray) -> np.ndarray:
        """Closest surface points (N, 3)."""

    def normal(self, X: np.ndarray) -> np.ndarray:
        """Outward unit normals (N, 3) of the surface at the closest points."""

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Corners (lo, hi) of a box that contains the cell."""


# --- membranes --------------------------------------------------------------------------------------------------

def _edges(F):
    E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    return np.unique(E, axis=0, return_counts=True)


def _compact(P, F, *per_point):
    """Drop points that no triangle uses."""
    used = np.unique(F)
    remap = np.full(len(P), -1); remap[used] = np.arange(len(used))
    return (P[used], remap[F], *(a[used] for a in per_point))


def _merge_outside(P, F, mask, B, tol=1e-9):
    """A border point that the projection put beyond another wall lies past a box edge: merge it into the
    neighbouring corner point (a point on two walls)."""
    for _ in range(100):
        out = np.nonzero(np.any((P < -tol) | (P > B + tol), axis=1))[0]
        if len(out) == 0:
            return P, F, mask
        j = out[0]
        nb = np.unique(F[np.any(F == j, axis=1)])
        corners = [i for i in nb if i != j and bin(mask[i]).count("1") > 1]
        if not corners and bin(mask[j]).count("1") > 1:    # a box-edge point past the corner: it becomes the corner
            for k, (a, val, side) in enumerate(box_walls(B)):
                if side * (P[j, a] - val) > tol:
                    P[j, a] = val; mask[j] |= 1 << k
            continue
        if not corners:
            raise RuntimeError(f"membrane point outside the box without a corner neighbour: {P[j]}")
        i = min(corners, key=lambda c: np.linalg.norm(P[c] - P[j]))
        F = F[~(np.any(F == i, axis=1) & np.any(F == j, axis=1))]
        F[F == j] = i
        P, F, mask = _compact(P, F, mask)
    raise RuntimeError("could not merge the membrane points outside the box")


def _drop_in_wall(P, F, mask):
    """Remove membrane triangles whose three points lie on one wall (they would lie inside the wall)."""
    keep = (mask[F[:, 0]] & mask[F[:, 1]] & mask[F[:, 2]]) == 0
    if keep.all():
        return P, F, mask
    return _compact(P, F[keep], mask)


def _orientation(P, F, cell):
    X = P[F]
    return np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), cell.normal(X.mean(1)))


def _fix_flips(P, F, cell, max_iter=50):
    """Collapse the shortest edge of triangles folded over by the projection (normal against the surface normal)."""
    for _ in range(max_iter):
        s = _orientation(P, F, cell)
        if np.sum(s < 0) > len(F) / 2:
            F = F[:, [1, 0, 2]]; s = -s
        bad = np.nonzero(s <= 0)[0]
        if len(bad) == 0:
            break
        f = F[bad[0]]
        L = [np.linalg.norm(P[f[k]] - P[f[(k + 1) % 3]]) for k in range(3)]
        k = int(np.argmin(L)); i, j = f[k], f[(k + 1) % 3]
        P = P.copy(); P[i] = cell.project(((P[i] + P[j]) / 2)[None])[0]
        F = F[~(np.any(F == i, axis=1) & np.any(F == j, axis=1))]
        F[F == j] = i
        P, F = _compact(P, F)
    return P, F


def _remesh(P, F, h, iters):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(P, F))
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(h), iterations=iters, featuredeg=60)
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix(), float), np.asarray(m.face_matrix(), np.int64)


def _collapse_border(P, F, mask, lmin, cell):
    """Collapse border edges shorter than lmin (the wall cut leaves very short ones)."""
    P, F, mask = P.copy(), F.copy(), mask.copy()
    while True:
        E, cnt = _edges(F)
        bnd = E[cnt == 1]
        L = np.linalg.norm(P[bnd[:, 0]] - P[bnd[:, 1]], axis=1)
        done = True
        for e in np.argsort(L):
            if L[e] >= lmin:
                break
            i, j = bnd[e]
            ci, cj = bin(mask[i]).count("1") > 1, bin(mask[j]).count("1") > 1
            if ci and cj:
                continue
            keep, drop = (j, i) if cj else (i, j)
            ti = np.any(F == keep, axis=1)
            tj = np.any(F == drop, axis=1)
            nb_k, nb_d = set(F[ti].ravel()) - {keep}, set(F[tj].ravel()) - {drop}
            if len((nb_k & nb_d) - {keep, drop}) != 1:     # would make the membrane non-manifold
                continue
            F2 = F[~(ti & tj)].copy()
            F2[F2 == drop] = keep
            moved = np.any(F2 == keep, axis=1)                # reject collapses that fold a triangle over
            s0 = _orientation(P, F[~(ti & tj)][moved], cell)
            s1 = _orientation(P, F2[moved], cell)
            if np.any((s0 > 0) & (s1 <= 0)):
                continue
            if not (mask[keep] & mask[drop]):               # edge across a box edge: the kept point becomes a corner
                mask[keep] |= mask[drop]
            F = F2
            done = False
            break
        if done:
            break
    return _compact(P, F, mask)


def _cut(P, F, mask, a, val, side, bit, cell, s_snap):
    """Keep the part of the membrane with side * (x_a - val) <= 0.

    Points within s_snap of the wall first move onto it along the surface, so the cut makes no slivers. New points
    get the wall bit in mask.
    """
    P, mask = P.copy(), mask.copy()
    d = side * (P[:, a] - val)
    near = np.nonzero((np.abs(d) < s_snap) & (mask == 0))[0]
    if len(near):
        X = P[near].copy()
        for _ in range(30):
            X[:, a] = val
            X = cell.project(X)
        X[:, a] = val
        ok = np.linalg.norm(X - P[near], axis=1) < 1.5 * s_snap    # surface nearly parallel to the wall: no snap
        P0 = P.copy()
        P[near[ok]] = X[ok]
        n0 = np.cross(P0[F][:, 1] - P0[F][:, 0], P0[F][:, 2] - P0[F][:, 0])
        while True:                                                  # undo snaps that fold a triangle over
            Y = P[F]
            fl = np.einsum("ij,ij->i", np.cross(Y[:, 1] - Y[:, 0], Y[:, 2] - Y[:, 0]), n0) <= 0
            undo = np.intersect1d(np.unique(F[fl]), near[ok])
            if len(undo) == 0:
                break
            P[undo] = P0[undo]
            ok[np.isin(near, undo)] = False
        mask[near[ok]] |= bit; d[near[ok]] = 0.0
    d[(mask & bit) > 0] = 0.0
    newP, newM, cut_id = [], [], {}

    def cut_point(i, j):
        key = (min(i, j), max(i, j))
        if key not in cut_id:
            t = d[i] / (d[i] - d[j])
            cut_id[key] = len(P) + len(newP)
            newP.append(P[i] + t * (P[j] - P[i])); newM.append(mask[i] & mask[j] | bit)
        return cut_id[key]

    out = []
    for f in F:
        df = d[f]
        if np.all(df <= 0):
            if not np.all(df == 0):
                out.append(f)
            continue
        if np.all(df >= 0):
            continue
        poly = []                                   # clip one triangle by the plane
        for k in range(3):
            i, j = f[k], f[(k + 1) % 3]
            if d[i] <= 0:
                poly.append(i)
            if (d[i] < 0 < d[j]) or (d[j] < 0 < d[i]):
                poly.append(cut_point(i, j))
        if len(poly) == 3:
            out.append(poly)
        elif len(poly) == 4:                        # split the quad along its shorter diagonal
            Q = np.array([P[x] if x < len(P) else newP[x - len(P)] for x in poly])
            if np.linalg.norm(Q[0] - Q[2]) <= np.linalg.norm(Q[1] - Q[3]):
                out += [[poly[0], poly[1], poly[2]], [poly[0], poly[2], poly[3]]]
            else:
                out += [[poly[0], poly[1], poly[3]], [poly[1], poly[2], poly[3]]]
    if newP:
        P = np.r_[P, np.array(newP)]; mask = np.r_[mask, np.array(newM, np.int64)]
    F = np.array(out, np.int64).reshape(-1, 3)
    # snapped points that only touch the wall (inside the membrane, not on its border): move them off the wall
    E, cnt = _edges(F)
    on_bnd = np.zeros(len(P), bool); on_bnd[E[cnt == 1].ravel()] = True
    touch = ((mask & bit) > 0) & ~on_bnd
    if touch.any():
        P[touch, a] = val - side * 0.5 * s_snap
        mask[touch] &= ~bit
    return _compact(P, F, mask)


def _project_border(P, mask, planes, cell):
    """Move border points onto the surface within their wall plane(s) (alternating projections)."""
    P = P.copy()
    sel = mask > 0
    X = P[sel]
    for _ in range(40):
        for k, (a, val) in enumerate(planes):
            X[(mask[sel] >> k) & 1 == 1, a] = val
        X = cell.project(X)
    for k, (a, val) in enumerate(planes):
        X[(mask[sel] >> k) & 1 == 1, a] = val
    P[sel] = X
    return P


def _secant_offset(P, F, mask, cell, B, h, max_offset):
    """Move points outward so that the faceted membrane encloses the exact cell volume.

    The gap of a triangle to the surface is the mean of -phi at its edge midpoints (exact on average for a
    quadratic surface); each point moves by the area-weighted mean gap of its triangles, at most
    `max_offset`. Points on walls move within their wall plane(s); no point moves towards a wall.
    """
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    gap_t = -cell.phi((P[E[:, 0]] + P[E[:, 1]]) / 2).reshape(3, -1).mean(0)
    X = P[F]
    area = np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1) / 2
    wsum = np.bincount(F.ravel(), np.repeat(area, 3), len(P))
    gap_v = np.bincount(F.ravel(), np.repeat(area * gap_t, 3), len(P)) / np.maximum(wsum, 1e-30)
    gap_v = np.clip(gap_v, -max_offset, max_offset)
    d = gap_v[:, None] * cell.normal(P)
    for k, (a, val, side) in enumerate(box_walls(B)):
        d[(mask >> k) & 1 == 1, a] = 0.0
        d[(side * d[:, a] > 0) & (np.abs(P[:, a] - val) < h), a] = 0.0
    s0 = np.sign(np.sum(_orientation(P, F, cell)))
    for _ in range(10):     # keep the points of triangles the offset would flip (sharp creases of non-convex cells)
        flip = s0 * _orientation(P + d, F, cell) <= 0
        if not flip.any():
            break
        d[F[flip].ravel()] = 0.0
    return P + d


def _insert_corners(P, F, mask):
    """Split the triangle at each border edge whose ends share no wall (it cuts across a box edge)."""
    E, cnt = _edges(F)
    if np.any(cnt > 2):
        raise RuntimeError("non-manifold membrane")
    F = [list(f) for f in F]
    tri_of = {}
    for t, f in enumerate(F):
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            tri_of[tuple(sorted(e))] = (t, e)
    P, mask = list(P), list(mask)
    for i, j in E[cnt == 1]:
        if mask[i] & mask[j]:
            continue
        t, (u, v) = tri_of[(min(i, j), max(i, j))]
        q = [x for x in F[t] if x not in (u, v)][0]
        m = len(P)
        P.append((P[u] + P[v]) / 2); mask.append(mask[u] | mask[v])
        F[t] = [u, m, q]
        F.append([m, v, q])
    return np.array(P), np.array(F, np.int64), np.array(mask)


def _check_outlines(P, F, mask):
    """Raise if a membrane outline on a wall crosses itself."""
    E, cnt = _edges(F)
    inner = E[cnt == 2]
    if np.any(mask[inner[:, 0]] & mask[inner[:, 1]]):
        raise RuntimeError("inner membrane edge on a wall (membrane touches the wall across a neck)")
    bnd = E[cnt == 1]
    cr = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    for k in range(6):
        S = bnd[((mask[bnd[:, 0]] & mask[bnd[:, 1]]) >> k) & 1 == 1]
        if len(S) < 4:
            continue
        u, v = [x for x in range(3) if x != k // 2]
        A, Bq = P[S[:, 0]][:, [u, v]], P[S[:, 1]][:, [u, v]]
        d = Bq - A
        den = cr(d[:, None], d[None])
        w_ = A[None] - A[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cr(w_, d[None]) / den
            s = cr(w_, d[:, None]) / den
        share = (S[:, None, :, None] == S[None, :, None, :]).any((2, 3))
        hit = (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9) & ~share & (np.abs(den) > 1e-15)
        if hit.any():
            raise RuntimeError(f"membrane outline on wall {WALLS[k]} crosses itself")


def _split_wall_edges(P, F, mask, cell, B):
    """Split inner edges whose ends lie on the same wall (the membrane spans a neck along the wall): the midpoint
    moves onto the surface, off the wall, so that the wall patch cannot reuse the edge."""
    for _ in range(100):
        E, cnt = _edges(F)
        inner = E[cnt == 2]
        hit = inner[(mask[inner[:, 0]] & mask[inner[:, 1]]) > 0]
        if len(hit) == 0:
            break
        i, j = hit[0]
        m = cell.project(((P[i] + P[j]) / 2)[None])[0]
        if np.any(m < 1e-9) or np.any(m > B - 1e-9):
            raise RuntimeError("inner membrane edge on a wall: its midpoint projects onto or outside the wall")
        k = len(P)
        P = np.vstack([P, m]); mask = np.r_[mask, 0]
        out = []
        for f in F:
            f = list(f)
            if i in f and j in f:
                x = f.index(i)
                if f[(x + 1) % 3] == j:          # i -> j -> q
                    q = f[(x + 2) % 3]; out += [[i, k, q], [k, j, q]]
                else:                            # j -> i -> q
                    q = f[(x + 1) % 3]; out += [[j, k, q], [k, i, q]]
            else:
                out.append(f)
        F = np.array(out, np.int64)
    return P, F, mask


def membrane(cell, box, h, max_offset=np.inf, iters=10, snap=0.3, collapse=0.3):
    """Triangle mesh of one cell's surface inside the box.

    Args:
        cell: The cell (see the module docstring).
        box: Box size (3,).
        h: Target edge length.
        max_offset: Bound for the volume correction (see `surface_mesh`).
        iters: Remeshing iterations.
        snap: Points within snap * h of a wall move onto it before the cut.
        collapse: Border edges shorter than collapse * h are collapsed.

    Returns:
        Points (N, 3), triangles (M, 3) oriented out of the cell, and per point a bitmask of the walls it lies on
        (bit k = WALLS[k]; 0 = on no wall).
    """
    B = np.asarray(box, float)
    hg = min(h / 3, 0.08)
    lo, hi = cell.bounds()
    lo, hi = lo - 2 * hg, hi + 2 * hg
    img = pv.ImageData(dimensions=np.ceil((hi - lo) / hg).astype(int) + 1, spacing=(hg,) * 3, origin=lo)
    img.point_data["phi"] = cell.phi(np.asarray(img.points), band=3 * hg)
    s = img.contour([0.0], scalars="phi", method="flying_edges").triangulate().clean()
    P, F = _remesh(cell.project(np.asarray(s.points, float)), s.regular_faces, h, iters)  # closed: no border issues
    P, F = _fix_flips(cell.project(P), F, cell)
    walls = box_walls(B)
    planes = [(a, val) for a, val, _ in walls]
    mask = np.zeros(len(P), np.int64)
    for k, (a, val, side) in enumerate(walls):
        if np.any(side * (P[:, a] - val) > 0):
            P, F, mask = _cut(P, F, mask, a, val, side, 1 << k, cell, snap * h)
    P, F, mask = _drop_in_wall(P, F, mask)
    P = _project_border(P, mask, planes, cell)
    P, F, mask = _drop_in_wall(*_collapse_border(P, F, mask, collapse * h, cell))
    P, F, mask = _insert_corners(P, F, mask)
    P = _project_border(P, mask, planes, cell)
    P, F, mask = _drop_in_wall(*_merge_outside(*_drop_in_wall(P, F, mask), B))
    P = _project_border(P, mask, planes, cell)
    P, F, mask = _drop_in_wall(*_collapse_border(P, F, mask, collapse * h, cell))  # projection can reorder points
    P = _project_border(P, mask, planes, cell)
    P, F, mask = _split_wall_edges(P, F, mask, cell, B)
    P = _secant_offset(P, F, mask, cell, B, h, max_offset)
    s = _orientation(P, F, cell)
    if np.sum(s < 0) > len(F) / 2:
        F = F[:, [1, 0, 2]]; s = -s
    if np.any(s <= 0):
        raise RuntimeError(f"membrane has {np.sum(s <= 0)} flipped triangles")
    _check_outlines(P, F, mask)
    return P, F, mask


def membrane_robust(cell, box, h, max_offset=np.inf,
                    variants=((1.0, 0.3), (0.95, 0.3), (1.05, 0.25), (0.9, 0.35), (1.1, 0.2))):
    """`membrane` with retries at a slightly different h and snap distance. Returns (P, F, mask, attempt)."""
    err = None
    for attempt, (fh, snap) in enumerate(variants):
        try:
            return (*membrane(cell, box, h * fh, max_offset, snap=snap), attempt)
        except Exception as e:
            err = e
    raise RuntimeError(f"membrane failed in all {len(variants)} variants: {err}")


# --- walls and closed surface -------------------------------------------------------------------------------------

@dataclass
class SurfaceMesh:
    """Closed multi-domain surface mesh; each triangle's normal points out of domain `domin` into `domout`.

    Domain numbers: see `bmbcsim.meshing.tissue_io.CELL0`.
    """
    points: np.ndarray   # (N, 3)
    tris: np.ndarray     # (M, 3)
    domin: np.ndarray    # (M,)
    domout: np.ndarray   # (M,)

    @property
    def n_cells(self):
        return int(self.domin.max()) - CELL0 + 1

    def ecs_fraction(self, box):
        """ECS volume fraction enclosed by the surface (divergence theorem). `volume_mesh` keeps the surface, so
        the volume mesh has the same ECS fraction unless it merges a sealed ECS pocket into a cell."""
        X = self.points[self.tris]
        vol = np.einsum("ij,ij->i", X[:, 0], np.cross(X[:, 1], X[:, 2])) / 6
        return (vol[self.domin == ECS].sum() - vol[self.domout == ECS].sum()) / np.prod(box)

    def open_domains(self):
        """Domains whose boundary is not closed and consistently oriented: {domain: (open edges, duplicate edges)}."""
        N = len(self.points)
        bad = {}
        for d in np.unique(np.r_[self.domin, self.domout]):
            if d == OUTSIDE:
                continue
            T = np.r_[self.tris[self.domin == d], self.tris[self.domout == d][:, [1, 0, 2]]]
            E = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
            fw, bw = np.unique(E[:, 0] * N + E[:, 1]), np.unique(E[:, 1] * N + E[:, 0])
            n_open = len(np.setxor1d(fw, bw, assume_unique=True))
            if n_open or len(fw) != len(E):
                bad[int(d)] = (n_open, len(E) - len(fw))
        return bad


def _edge_keys(E, n):
    """Integer key of each undirected edge (rows of E) with point ids below n."""
    E = np.sort(E, axis=1)
    return E[:, 0] * n + E[:, 1]


class _SurfaceBuilder:
    """Assembles the closed surface: the membranes, points along the box edges, one CDT per wall, and finally the
    new wall points on membrane outlines inserted into the membranes."""

    def __init__(self, cells, box, h):
        self.cells, self.B, self.h = cells, np.asarray(box, float), h
        self.G = []                       # points
        self.mem_tris = []                # per cell: membrane triangles
        self.walls = []                   # wall regions: (triangles, domin, domout)
        self.border_pts = []              # (point id, wall mask, cell)
        self.outline_dir = {}             # membrane outline edge (sorted) -> (start, end) in membrane orientation
        self.owner = {}                   # membrane border point -> cell
        self.seg_by_wall = [[] for _ in range(6)]
        self.chain = {}                   # box edge -> point ids along it
        self.splits = {}                  # membrane outline edge -> new wall points on it (ordered from its first end)
        self.used_new = set()
        self.n_yy = 0

    def add_membranes(self, mems):
        for kc, (P, F, mask, _) in enumerate(mems):
            off = len(self.G)
            self.G += list(P)
            self.mem_tris.append(F + off)
            E, cnt = _edges(F)
            bnd = E[cnt == 1]
            D = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
            for x, y in D[np.isin(_edge_keys(D, len(P)), _edge_keys(bnd, len(P)))]:
                self.outline_dir[(min(x, y) + off, max(x, y) + off)] = (x + off, y + off)
            for i, j in bnd:
                common = mask[i] & mask[j]
                for k in range(6):
                    if (common >> k) & 1:
                        self.seg_by_wall[k].append((i + off, j + off))
            on_wall = np.nonzero(mask)[0]
            self.border_pts += [(i + off, mask[i], kc) for i in on_wall]
            self.owner.update({i + off: kc for i in on_wall})

    def add_box_edges(self):
        """Corners and membrane points on each box edge, gaps filled at spacing ~h."""
        G, B, h = self.G, self.B, self.h
        corner_id = {tuple(G[g]): g for g, m, _ in self.border_pts if bin(m).count("1") == 3}  # membrane point
        for c in range(3):
            ea, eb = [x for x in range(3) if x != c]
            for va in (0.0, B[ea]):
                for vb in (0.0, B[eb]):
                    bits = (1 << (2 * ea + (va > 0))) | (1 << (2 * eb + (vb > 0)))
                    pts = []
                    for t in (0.0, B[c]):
                        X = np.zeros(3); X[c] = t; X[ea] = va; X[eb] = vb
                        if tuple(X) not in corner_id:
                            corner_id[tuple(X)] = len(G); G.append(X)
                        pts.append((t, corner_id[tuple(X)]))
                    pts = sorted(set(pts + [(G[g][c], g) for g, m, _ in self.border_pts if (m & bits) == bits]))
                    ids = [pts[0][1]]
                    for (ta, ga), (tb, gb) in zip(pts[:-1], pts[1:]):
                        nfill = max(int(round((tb - ta) / h)), 1)
                        for s in range(1, nfill):
                            X = np.zeros(3); X[c] = ta + (tb - ta) * s / nfill; X[ea] = va; X[eb] = vb
                            ids.append(len(G)); G.append(X)
                        ids.append(gb)
                    self.chain[(c, ea, eb, va, vb)] = ids

    def add_wall(self, k, a, val, side, G0):
        """CDT of wall k with the membrane outlines and box edges as segments, its regions labelled.

        New points on membrane outlines are recorded in `splits`. G0: the points before any wall was added.
        """
        import triangle as tr
        B, h = self.B, self.h
        u, v = [x for x in range(3) if x != a]
        segs = list(self.seg_by_wall[k])
        n_out = len(segs)
        for (c, ea, eb, va, vb), ids in self.chain.items():
            if (ea == a and va == val) or (eb == a and vb == val):
                segs += list(zip(ids[:-1], ids[1:]))
        gids = np.unique(np.array(segs).ravel())
        loc = np.full(len(G0), -1); loc[gids] = np.arange(len(gids))
        V2 = G0[gids][:, [u, v]]
        S = loc[np.array(segs)]
        marks = np.r_[np.arange(n_out) + 2, np.ones(len(segs) - n_out, int)]
        # new points may split membrane outlines (Y), not box edges, which the neighbouring wall shares
        # a cascade can run to tens of millions of points before it ends: cap the Steiner points at ~10x the
        # expected number
        n_max = int(10 * (B[u] * B[v] / (0.3 * h * h) + len(V2)))
        out = tr.triangulate(dict(vertices=V2, segments=S, segment_markers=marks), f"pq30Ya{0.6 * h * h:.6f}S{n_max}")
        sub = out["segments"][out["segment_markers"].ravel() >= 2]
        if len(out["vertices"]) - len(V2) >= n_max or (len(sub) and np.linalg.norm(
                out["vertices"][sub[:, 0]] - out["vertices"][sub[:, 1]], axis=1).min() < h / 4):
            # refinement cascade along an outline (shallow crossing or thin ECS gap): no outline splits on this
            # wall; with YY, an angle bound (q30, even q20) makes Triangle reject the points near outlines and leave
            # large fans (up to 3.6 um^2), so refine by area only
            out = tr.triangulate(dict(vertices=V2, segments=S, segment_markers=marks), f"pYYa{0.6 * h * h:.6f}")
            self.n_yy += 1
        W, T = out["vertices"], out["triangles"]
        if len(W) < len(gids) or not np.allclose(W[:len(gids)], V2):
            raise MeshingError(f"wall {WALLS[k]}: triangle moved input points")
        X3 = np.zeros((len(W), 3)); X3[:, a] = val; X3[:, u] = W[:, 0]; X3[:, v] = W[:, 1]
        g_all = np.r_[gids, len(self.G) + np.arange(len(W) - len(gids))]
        self.G += list(X3[len(gids):])
        S_out, m_out = out["segments"], out["segment_markers"].ravel()
        for mk in np.unique(m_out[m_out >= 2]):
            sub = S_out[m_out == mk]
            if len(sub) == 1:
                continue
            gi, gj = segs[mk - 2]
            inner = np.setdiff1d(np.unique(sub), loc[[gi, gj]])
            order = np.argsort(np.linalg.norm(W[inner] - V2[loc[gi]], axis=1))
            self.splits[(gi, gj)] = list(g_all[inner[order]])
            if not self.used_new.isdisjoint(self.splits[(gi, gj)]):
                raise MeshingError(f"wall {WALLS[k]}: membrane outlines cross near {G0[gi]}")
            self.used_new.update(self.splits[(gi, gj)])
        # regions: flood fill across edges that are no segments
        Es = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)
        tid = np.tile(np.arange(len(T)), 3)
        o = np.lexsort((Es[:, 1], Es[:, 0])); Es, tid = Es[o], tid[o]
        same = np.all(Es[1:] == Es[:-1], axis=1)
        free = ~np.isin(_edge_keys(Es[:-1][same], len(W)), _edge_keys(S_out, len(W)))
        pa, pb = tid[:-1][same][free], tid[1:][same][free]
        nc, comp = connected_components(coo_matrix((np.ones(len(pa)), (pa, pb)), shape=(len(T),) * 2),
                                        directed=False)
        nrm = np.cross(X3[T[:, 1]] - X3[T[:, 0]], X3[T[:, 2]] - X3[T[:, 0]])[:, a]
        T = np.where((side * nrm < 0)[:, None], T[:, [1, 0, 2]], T)     # normal out of the box
        # side of the membrane outlines: a cell's closed surface runs along an outline edge once in the membrane
        # and once, reversed, in its wall patch (normal out of the box); the ECS wall triangle runs along it forwards
        outline = m_out >= 2
        mark_of = dict(zip(_edge_keys(S_out[outline], len(W)), m_out[outline]))
        De = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1).reshape(-1, 2)
        ke = _edge_keys(De, len(W))
        hit = np.nonzero(np.isin(ke, list(mark_of)))[0]
        side_vote = [set() for _ in range(nc)]      # per region: (cell, inside)
        for e in hit:
            x, y = De[e]
            gi, gj = segs[mark_of[ke[e]] - 2]
            p0, p1 = self.outline_dir[(min(gi, gj), max(gi, gj))]
            side_vote[comp[e // 3]].add((self.owner[p0], float(np.dot(X3[y] - X3[x], G0[p1] - G0[p0])) < 0))
        C = X3[T].mean(1)
        cand = sorted({kc for _, m, kc in self.border_pts if (m >> k) & 1})
        for ci in range(nc):
            idx = np.nonzero(comp == ci)[0]
            if side_vote[ci]:
                votes = sorted({kc for kc, inside in side_vote[ci] if inside})
                if votes and any(not inside for _, inside in side_vote[ci]):
                    raise MeshingError(f"wall {WALLS[k]}: region inside and outside of membrane outlines")
            else:                                   # no outline on the boundary: inside test
                votes = [kc for kc in cand if np.mean(self.cells[kc].phi(C[idx]) < 0) > 0.5]
            if len(votes) > 1:
                raise MeshingError(f"wall {WALLS[k]}: region claimed by cells {votes}")
            self.walls.append((g_all[T[idx]], CELL0 + votes[0], OUTSIDE) if votes else (g_all[T[idx]], ECS, OUTSIDE))

    def insert_splits(self):
        """Insert the new wall points on membrane outlines into the membrane triangles (fan from the opposite point).

        Returns:
            The membrane triangles and the cell of each.
        """
        M = [list(f) for f in np.concatenate(self.mem_tris)]
        cell_of = list(np.repeat(np.arange(len(self.mem_tris)), [len(F) for F in self.mem_tris]))
        edge_tri = {}

        def register(t):
            f = M[t]
            for x in range(3):
                edge_tri[(min(f[x], f[(x + 1) % 3]), max(f[x], f[(x + 1) % 3]))] = t

        for t in range(len(M)):
            register(t)
        for (gi, gj), pts in self.splits.items():   # one at a time: a triangle can have two split outline edges
            t = edge_tri[(min(gi, gj), max(gi, gj))]
            f = M[t]
            x = [y for y in range(3) if {f[y], f[(y + 1) % 3]} == {gi, gj}][0]
            uu, vv, q = f[x], f[(x + 1) % 3], f[(x + 2) % 3]
            ch = [uu] + (pts if uu == gi else pts[::-1]) + [vv]
            M[t] = [ch[0], ch[1], q]
            register(t)
            for s in range(1, len(ch) - 1):
                M.append([ch[s], ch[s + 1], q]); cell_of.append(cell_of[t])
                register(len(M) - 1)
        return np.array(M, np.int64), np.array(cell_of)

    def build(self, mems):
        self.add_membranes(mems)
        self.add_box_edges()
        G0 = np.asarray(self.G)
        for k, (a, val, side) in enumerate(box_walls(self.B)):
            self.add_wall(k, a, val, side, G0)
        M, cell_of = self.insert_splits()
        sizes = [len(T) for T, _, _ in self.walls]
        return SurfaceMesh(np.asarray(self.G), np.concatenate([M] + [T for T, _, _ in self.walls]),
                           np.r_[CELL0 + cell_of, np.repeat([d for _, d, _ in self.walls], sizes)],
                           np.r_[np.full(len(M), ECS), np.repeat([d for _, _, d in self.walls], sizes)])


def surface_mesh(cells, box, h, max_offset=np.inf, workers=8):
    """Closed multi-domain surface mesh of the cells in the box.

    Each membrane is corrected outward for the volume lost to faceting, by at most `max_offset`, which keeps
    neighbouring membranes apart (see `bmbcsim.geometry.cells.CellSet`).

    Returns:
        The `SurfaceMesh`.
    """
    t0 = time.time()
    B = np.asarray(box, float)
    if workers > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            mems = pool.starmap(membrane_robust, [(c, B, h, max_offset) for c in cells])
    else:
        mems = [membrane_robust(c, B, h, max_offset) for c in cells]
    t_mem = time.time() - t0
    builder = _SurfaceBuilder(cells, B, h)
    surf = builder.build(mems)
    bad = surf.open_domains()
    if bad:
        raise MeshingError(f"open or inconsistent domain boundaries (domain: open, duplicate edges): {bad}")
    logger.info(f"surface: {len(cells)} cells, {len(surf.tris)} triangles, {len(surf.points)} points, "
                f"{builder.n_yy} walls without outline splits, {sum(m[3] for m in mems)} membrane retries "
                f"(membranes {t_mem:.0f} s, total {time.time() - t0:.0f} s)")
    return surf


# --- volume -------------------------------------------------------------------------------------------------------

def volume_mesh(surface, maxvol=0.15, minratio=1.5, mindihedral=10.0, optsteps=5):
    """Tet mesh of a `SurfaceMesh`: TetGen, then netgen optimization of the interior.

    Tets are grouped into connected components across faces that are not surface triangles; each component is
    labelled by the oriented surface triangles around it.
    """
    import tetgen
    t0 = time.time()
    G, tris, din, dout, n_cells = surface.points, surface.tris, surface.domin, surface.domout, surface.n_cells
    tg = tetgen.TetGen(np.ascontiguousarray(G), np.ascontiguousarray(tris, np.int32))
    nodes, elem, _, _ = tg.tetrahedralize(plc=True, quality=True, nobisect=True,
                                          minratio=minratio, mindihedral=mindihedral, maxvolume=maxvol)
    if len(nodes) < len(G) or not np.allclose(nodes[:len(G)], G):
        raise RuntimeError("TetGen moved surface points")
    Tet = elem.astype(np.int64)
    N = len(nodes)
    faces = np.r_[Tet[:, [1, 2, 3]], Tet[:, [0, 2, 3]], Tet[:, [0, 1, 3]], Tet[:, [0, 1, 2]]]
    opp = np.r_[Tet[:, 0], Tet[:, 1], Tet[:, 2], Tet[:, 3]]
    ft = np.tile(np.arange(len(Tet)), 4)
    fk = _face_key(faces, N)
    o = np.argsort(fk, kind="stable"); fk, ft, opp = fk[o], ft[o], opp[o]
    sk = _face_key(tris, N)
    inner = np.nonzero((fk[1:] == fk[:-1]) & ~np.isin(fk[1:], sk))[0]
    nc, comp = connected_components(coo_matrix((np.ones(len(inner)), (ft[inner], ft[inner + 1])),
                                               shape=(len(Tet),) * 2), directed=False)
    # each surface triangle votes: the tet behind it (against the normal) is in domin, the one in front in domout
    Y = G[tris]
    nrm = np.cross(Y[:, 1] - Y[:, 0], Y[:, 2] - Y[:, 0])
    j0 = np.searchsorted(fk, sk)
    votes = np.zeros((nc, CELL0 + n_cells))
    for off in (0, 1):
        jj = np.minimum(j0 + off, len(fk) - 1)
        ok = fk[jj] == sk
        side = np.einsum("ij,ij->i", nrm, nodes[opp[jj]] - Y[:, 0])
        np.add.at(votes, (comp[ft[jj[ok]]], np.where(side < 0, din, dout)[ok]), 1)
    reg = votes.argmax(1)
    purity = votes.max(1) / np.maximum(votes.sum(1), 1)
    if np.any(reg == OUTSIDE) or purity.min() < 0.99:
        raise RuntimeError(f"tet regions: {np.sum(reg == OUTSIDE)} unlabelled components, min purity {purity.min():.3f}")
    ecs = np.nonzero(reg == ECS)[0]
    if len(ecs) > 1:
        # sealed ECS pockets (e.g. at a box corner that a cell crosses on all three walls) join the cell around them
        X = nodes[Tet]
        vol = np.bincount(comp, np.abs(np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]),
                                                 X[:, 3] - X[:, 0])) / 6, nc)
        pair = (fk[1:] == fk[:-1]) & np.isin(fk[1:], sk)            # the two tets at each interior surface triangle
        ca, cb = comp[ft[:-1][pair]], comp[ft[1:][pair]]
        for c in ecs[np.argsort(vol[ecs])[:-1]]:
            nb = np.r_[cb[ca == c], ca[cb == c]]
            reg[c] = np.bincount(reg[nb], minlength=CELL0 + n_cells).argmax()
            logger.warning(f"sealed ECS pocket of {vol[c]:.2g} um^3 added to cell_{reg[c] - CELL0}")
    treg = reg[comp]
    if set(np.unique(treg)) != set(range(ECS, CELL0 + n_cells)):
        raise RuntimeError("a domain received no tets")
    t1 = time.time()
    mesh = tissue_mesh_from_labels(nodes, Tet, treg - CELL0)   # keep alive: its ngmesh dies with it
    mesh.ngmesh.OptimizeVolumeMesh(MeshingParameters(optsteps3d=optsteps))
    mesh = ngs.Mesh(mesh.ngmesh.Copy())
    logger.info(f"volume: {mesh.ne} tets, {mesh.nv} points (TetGen {t1 - t0:.0f} s, "
                f"netgen optimize {time.time() - t1:.0f} s)")
    return mesh
