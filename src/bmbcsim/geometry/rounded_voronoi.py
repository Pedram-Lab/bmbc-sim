"""Rounded Voronoi cells: a packed-cell tissue whose gap width w sets the ECS fraction.

Each cell is a Voronoi cell of Lloyd-relaxed seeds, eroded by w/2 + r_k and rounded by r_k (the Minkowski sum of
the eroded polytope with a ball). The result is an exact signed distance function with an exact closest-point
projection, which `bmbcsim.meshing.tissue_mesher` meshes. Two neighbours are exactly w apart, and the cell set
does not depend on w, so all w values give the same cells with the same numbering.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi, cKDTree


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
    """The set {x: dist(x, P) <= r} for the convex polytope P = {x: hs[:, :3] x + hs[:, 3] <= 0} around p.

    Attributes:
        max_offset: Upper bound for the mesher's volume correction (secant offset) of this cell.
    """

    def __init__(self, hs, p, r, max_offset=np.inf):
        self.hs, self.p, self.r, self.max_offset = hs, p, r, max_offset
        self.V = HalfspaceIntersection(hs, p).intersections
        hull = ConvexHull(self.V)
        self.A, self.B, self.C = (self.V[hull.simplices[:, i]] for i in range(3))

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


def _edge_margin(cell, B, n=400):
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


def make_cells(w, r, box, d=4.0, seed=0, v_drop=1.0, delta=0.05, r_range=(0.7, 1.3), n_r=25, max_shift=0.05,
               n_shift=200):
    """Rounded Voronoi cells with gap w inside the box [0, box].

    A Voronoi cell is kept if its raw (uneroded) volume inside the box is at least `v_drop`; this does not depend
    on w. Per cell, r_k is the value in r * r_range closest to r for which every wall and box edge is crossed or
    missed by at least `delta` (shallow crossings make slivers in the mesh). If no r_k works, the cell is shifted
    by at most `max_shift`, without approaching a neighbour by more than w/4. The flat faces, and hence the gaps,
    do not depend on r_k.

    Args:
        w: Gap width between neighbouring cells.
        r: Nominal rounding radius.
        box: Box size (3,).
        d: Target cell diameter for the seeds.
        seed: Random seed for the seeds.

    Returns:
        (cells, ids, info): the `RoundedPolytope` cells, the seed index of each cell, and a dict with the number
        of shifted cells and of cells that still miss the `delta` margin (`n_bad`).
    """
    B = np.asarray(box, float)
    pts = voronoi_seeds(B, d, seed)
    vor = Voronoi(pts)
    box_hs = np.array([[-1, 0, 0, 0], [1, 0, 0, -B[0]], [0, -1, 0, 0], [0, 1, 0, -B[1]], [0, 0, -1, 0],
                       [0, 0, 1, -B[2]]], float)
    cands = r * np.r_[1.0, np.linspace(*r_range, n_r)]
    cands = cands[np.argsort(np.abs(cands - r), kind="stable")]

    def halfspaces(i, erosion):
        p = pts[i]
        rp = vor.ridge_points[(vor.ridge_points == i).any(1)]
        q = pts[np.where(rp[:, 0] == i, rp[:, 1], rp[:, 0])]
        nrm = (q - p) / np.linalg.norm(q - p, axis=1)[:, None]
        return np.c_[nrm, -(np.einsum("ij,ij->i", nrm, (p + q) / 2) - erosion)]

    def margin(c):
        e = np.array([c.extent(h[:3], h[3]) for h in box_hs])    # > 0: crosses the wall
        return min(np.abs(e).min(), _edge_margin(c, B))

    cells, ids, n_bad, n_shifted = [], [], 0, 0
    for i, p in enumerate(pts):
        hs_in = np.vstack([halfspaces(i, 0.0), box_hs])
        A_, b_ = hs_in[:, :3], -hs_in[:, 3]                        # Chebyshev centre: an interior point
        res = linprog([0, 0, 0, -1], A_ub=np.c_[A_, np.linalg.norm(A_, axis=1)], b_ub=b_,
                      bounds=[(None, None)] * 3 + [(0, None)])
        if not res.success or res.x[3] < 1e-3:
            continue
        if ConvexHull(HalfspaceIntersection(hs_in, res.x[:3]).intersections).volume < v_drop:
            continue
        best = None
        for rk in cands:
            try:
                c = RoundedPolytope(halfspaces(i, w / 2 + rk), p, rk)
            except Exception:
                continue        # eroded core empty for this r_k
            m = margin(c)
            if best is None or m > best[0]:
                best = (m, c)
            if m >= delta:
                break
        if best is None:
            continue
        if best[0] < delta:     # r_k does not help (flat face or box edge): translate the cell
            rng = np.random.default_rng([seed, i])
            c0 = best[1]
            for _ in range(n_shift):
                t = rng.uniform(-max_shift, max_shift, 3)
                if np.max(c0.hs[:, :3] @ t) > w / 4:      # approaches a neighbour: keeps the gap >= w/2
                    continue
                hs = c0.hs.copy(); hs[:, 3] -= hs[:, :3] @ t
                c = RoundedPolytope(hs, p + t, c0.r)
                m = margin(c)
                if m > best[0]:
                    best = (m, c)
                if m >= delta:
                    break
            n_shifted += 1
        if best[0] < delta:
            n_bad += 1
        best[1].max_offset = 0.2 * w   # two neighbours (gap >= w/2 after shifts) never cross after the offset
        cells.append(best[1])
        ids.append(i)
    return cells, ids, dict(n_bad=n_bad, n_shifted=n_shifted)
