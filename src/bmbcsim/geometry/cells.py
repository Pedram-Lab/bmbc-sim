"""Building blocks of the packed-cell generators (`rounded_voronoi`, `tube_cells`).

A cell (or a piece of one) is a convex polytope rounded by a radius r: an exact signed distance function with an
exact closest-point projection, which `bmbcsim.meshing.tissue_mesher` meshes. Halfspaces are rows (n, c) of
{x: n . x + c <= 0} with unit normals n.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree


@dataclass
class CellSet:
    """The cells from a generator at gap w; `labels` and `faces` do not depend on w.

    Attributes:
        cells: Objects with the cell interface of `bmbcsim.meshing.tissue_mesher.Cell`.
        labels: A label of each cell that identifies it across w (seed index, or chain component for tubes).
        faces: Per cell, the triangles (M, 3, 3) of its raw Voronoi faces inside the box and off the walls.
        stats: Diagnostics of the generator.
    """
    cells: list
    labels: np.ndarray
    faces: list
    stats: dict = field(default_factory=dict)


def voronoi_seeds(box, d, seed, n_lloyd=20):
    """Lloyd-relaxed seed points in the box padded by one cell diameter.

    Args:
        box: Box size (3,); the box spans [0, box].
        d: Target cell diameter.
        seed: Random seed.
        n_lloyd: Number of Lloyd iterations (Monte Carlo centroids).
    """
    rng = np.random.default_rng(seed)
    lo, hi = -d * np.ones(3), np.asarray(box) + d
    n = int(round(np.prod(hi - lo) / (np.pi / 6 * d**3)))
    pts = rng.uniform(lo, hi, (n, 3))
    s = rng.uniform(lo, hi, (100 * n, 3))
    for _ in range(n_lloyd):
        lab = cKDTree(pts).query(s)[1]
        pts = np.array([s[lab == i].mean(0) if np.any(lab == i) else pts[i] for i in range(n)])
    return pts


def box_halfspaces(lo, hi):
    """Halfspaces (6, 4) of the box [lo, hi]."""
    return np.array([[-1, 0, 0, lo[0]], [1, 0, 0, -hi[0]], [0, -1, 0, lo[1]], [0, 1, 0, -hi[1]],
                     [0, 0, -1, lo[2]], [0, 0, 1, -hi[2]]], float)


def chebyshev_center(hs):
    """Chebyshev centre and radius of the polytope of the halfspaces hs, or None if it is (nearly) empty."""
    A, b = hs[:, :3], -hs[:, 3]
    res = linprog([0, 0, 0, -1], A_ub=np.c_[A, np.linalg.norm(A, axis=1)], b_ub=b,
                  bounds=[(None, None)] * 3 + [(0, None)])
    return (res.x[:3], res.x[3]) if res.success and res.x[3] > 1e-3 else None


def facet_triangles(V, hs):
    """Triangulated boundary of the convex polytope with vertices V, cut out by the halfspaces hs.

    Args:
        V: Vertices (N, 3) of the polytope of hs.
        hs: Halfspaces (K, 4).

    Returns:
        (tris, k): the triangles (M, 3, 3) and the halfspace (row of hs) each one lies on.
    """
    tris = V[ConvexHull(V).simplices]
    c = tris.mean(1)
    return tris, np.argmin(np.abs(c @ hs[:, :3].T + hs[:, 3]), axis=1)


def closest_on_triangles(X, A, B, C, chunk=4096):
    """Closest point to each X (N, 3) on the union of the triangles (A, B, C), each (M, 3).

    Uses the Voronoi-region test from Ericson, Real-Time Collision Detection, 5.1.5.
    """
    out = np.empty_like(X)
    ab, ac, bc = B - A, C - A, C - B
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk, None, :]
        ap, bp, cp = x - A, x - B, x - C
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ab = d1 / (d1 - d3)
            t_ac = d2 / (d2 - d6)
            t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            den = 1.0 / (va + vb + vc)
            v, w = vb * den, vc * den
        conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0),
                 (d6 >= 0) & (d5 <= d6), (vb <= 0) & (d2 >= 0) & (d6 <= 0),
                 (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
        Q = np.broadcast_to(A + ab * v[..., None] + ac * w[..., None], ap.shape).copy()   # face interior
        cands = [A, B, A + ab * t_ab[..., None], C, A + ac * t_ac[..., None], B + bc * t_bc[..., None]]
        for cnd, q in reversed(list(zip(conds, cands))):                                 # first true wins
            Q = np.where(cnd[..., None], np.broadcast_to(q, ap.shape), Q)
        j = np.argmin(((x - Q) ** 2).sum(-1), axis=1)
        out[s:s + chunk] = Q[np.arange(len(j)), j]
    return out


class RoundedPolytope:
    """The set {x: dist(x, P) <= r} for the bounded convex polytope P of the halfspaces hs.

    Args:
        hs: Halfspaces (K, 4): the first `n_faces` rows are the cell's faces, the rest only bound P far away.
        p: A point inside P (e.g. its Chebyshev centre).
        r: Rounding radius.
        n_faces: Number of face rows; default all.
    """

    def __init__(self, hs, p, r, n_faces=None):
        self.hs, self.p, self.r = hs, p, r
        self.n_faces = len(hs) if n_faces is None else n_faces
        self.V = HalfspaceIntersection(hs, p).intersections
        hull = ConvexHull(self.V)
        self.A, self.B, self.C = (self.V[hull.simplices[:, i]] for i in range(3))

    @property
    def face_normals(self):
        """Outward normals (n_faces, 3) of the faces."""
        return self.hs[:self.n_faces, :3]

    def translated(self, t):
        """The cell translated by t."""
        hs = self.hs.copy()
        hs[:, 3] -= hs[:, :3] @ t
        return RoundedPolytope(hs, self.p + t, self.r, self.n_faces)

    def _plane(self, X):
        return (X @ self.hs[:, :3].T + self.hs[:, 3]).max(1)

    def phi(self, X, band=np.inf):
        """Signed distance (negative inside); exact where the plane bound is below `band`, else that lower bound."""
        g = self._plane(X)
        d = g.copy()
        out = (g > 0) & (g < band)
        if out.any():
            d[out] = np.linalg.norm(X[out] - closest_on_triangles(X[out], self.A, self.B, self.C), axis=1)
        return d - self.r

    def project(self, X):
        """Closest point on the surface; X must lie outside the core, i.e. phi > -r."""
        if len(X) == 0:
            return X
        Q = closest_on_triangles(X, self.A, self.B, self.C)
        u = X - Q
        u[self._plane(X) < 0] *= -1                 # inside the core: go outward
        n = np.linalg.norm(u, axis=1)
        if n.min() <= 1e-12:
            raise ValueError("point on the core polytope boundary")
        return Q + self.r * u / n[:, None]

    def normal(self, X):
        """Outward unit normal of the surface at the closest point (X outside the core)."""
        u = X - closest_on_triangles(X, self.A, self.B, self.C)
        return u / np.linalg.norm(u, axis=1)[:, None]

    def extent(self, nrm, off):
        """Support function: max over the cell of nrm . x + off."""
        return (self.V @ nrm).max() + off + self.r

    def bounds(self):
        return self.V.min(0) - self.r, self.V.max(0) + self.r


def rounded_polytope(faces, bound, r):
    """The polytope of the halfspaces `faces` and `bound` rounded by r, or None if it is (nearly) empty."""
    hs = np.vstack([faces, bound])
    cc = chebyshev_center(hs)
    return None if cc is None else RoundedPolytope(hs, cc[0], r, n_faces=len(faces))


def edge_margin(cell, B, n=400):
    """min |min phi| over the parts of the 12 box edges near the cell."""
    lo, hi = cell.bounds()
    m = np.inf
    for c in range(3):
        a, b = [x for x in range(3) if x != c]
        for va in (0.0, B[a]):
            for vb in (0.0, B[b]):
                if not (lo[a] - 0.1 < va < hi[a] + 0.1 and lo[b] - 0.1 < vb < hi[b] + 0.1):
                    continue
                t = np.linspace(max(lo[c], 0.0), min(hi[c], B[c]), n)
                if t[0] >= t[-1]:
                    continue
                X = np.zeros((n, 3)); X[:, c] = t; X[:, a] = va; X[:, b] = vb
                m = min(m, abs(cell.phi(X).min()))
    return m


def rounding_radii(r, r_range, n_r):
    """Candidate rounding radii r * r_range (n_r values), r itself first, then by distance to r."""
    cands = r * np.r_[1.0, np.linspace(*r_range, n_r)]
    return cands[np.argsort(np.abs(cands - r), kind="stable")]


def fit_rounding_and_shift(build, score, radii, w, rng, max_shift, n_shift):
    """The cell whose score is zero: the first rounding radius that gives one, else the best shifted cell.

    Shallow crossings and near misses of the box walls and edges make slivers in the mesh. If no radius avoids
    them, the best cell is translated by at most `max_shift` per axis (up to `n_shift` random tries), without
    moving any face outward by more than w/4, so a gap of w to a neighbour stays at least w/2.

    Args:
        build: rk -> the cell rounded by rk, or None if its core is empty.
        score: cell -> tuple of non-negative defects; all zero is good, smaller (lexicographically) is better.
        radii: Candidate rounding radii, in the order to try.
        w: Gap width to the neighbours.
        rng: Random generator for the translations.

    Returns:
        (cell, score, shifted), or None if `build` gives no cell for any radius.
    """
    best = None
    for rk in radii:
        cell = build(rk)
        if cell is None:
            continue
        sc = score(cell)
        if not any(sc):
            return cell, sc, False
        if best is None or sc < best[1]:
            best = (cell, sc)
    if best is None:
        return None
    c0 = best[0]
    N = c0.face_normals
    for _ in range(n_shift):
        t = rng.uniform(-max_shift, max_shift, 3)
        if np.max(N @ t) > w / 4:
            continue
        cell = c0.translated(t)
        sc = score(cell)
        if sc < best[1]:
            best = (cell, sc)
        if not any(sc):
            break
    return best[0], best[1], True
