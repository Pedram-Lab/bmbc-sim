"""Tube-shaped cells: chains of Voronoi pieces, a dendrite-like alternative to `rounded_voronoi.make_cells`.

Lloyd-relaxed seeds at spacing d are linked greedily into chains (turning angle < `max_turn`, random length);
leftover short chains are attached to a neighbouring chain, which gives branches and spine-like stubs. Every chain
(split where it leaves and re-enters the box) is one cell: the union of its Voronoi pieces, each eroded by w/2 + r
only against other cells and rounded by r. Rounding a union equals the union of the rounded pieces, so the cell's
SDF is the min over its pieces (exact outside), and two cells stay at least w apart (see `make_tube_cells`). As for
the round cells, the cell set and numbering depend only on the seed, not on w.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi, cKDTree

from bmbcsim.geometry.cells import (box_halfspaces, chebyshev_center, edge_margin, facet_triangles,
                                    fit_rounding_and_shift, rounded_polytope, rounding_radii, voronoi_seeds)


def link_chains(pts, nbrs, rng, max_turn=45.0, len_range=(4, 16), min_len=3, noise=0.3):
    """Greedy chains through the seed adjacency graph.

    Returns:
        (lab, links): the chain label of every seed and the linked seed pairs (chain steps and attachments).
    """
    n = len(pts)
    links = []
    lab = np.full(n, -1)
    cos_max = np.cos(np.radians(max_turn))
    chains = []

    def grow(chain, t, L):
        while len(chain) < L:
            e = chain[-1]
            cand = [q for q in nbrs[e] if lab[q] < 0]
            if not cand:
                return
            steps = pts[cand] - pts[e]
            steps /= np.linalg.norm(steps, axis=1)[:, None]
            if t is None:
                k = rng.integers(len(cand))
            else:
                c = steps @ t
                ok = c > cos_max
                if not ok.any():
                    return
                k = int(np.argmax(np.where(ok, c + noise * rng.normal(size=len(c)), -np.inf)))
            lab[cand[k]] = len(chains)
            links.append((e, cand[k]))
            chain.append(cand[k])
            t = steps[k] if t is None else (t + steps[k]) / np.linalg.norm(t + steps[k])

    for s in rng.permutation(n):
        if lab[s] >= 0:
            continue
        lab[s] = len(chains)
        L = rng.integers(*len_range)
        fwd = [s]
        grow(fwd, None, L)
        back = [s]
        if len(fwd) > 1:
            t0 = pts[fwd[0]] - pts[fwd[1]]
            grow(back, t0 / np.linalg.norm(t0), L - len(fwd) + 1)
        chains.append(back[::-1] + fwd[1:])

    # attach short chains to the neighbouring chain they touch most (shortest first)
    size = np.bincount(lab)
    while True:
        short = [c for c in np.argsort(size) if 0 < size[c] < min_len]
        if not short:
            break
        c = short[0]
        mem = np.nonzero(lab == c)[0]
        touch = np.concatenate([lab[nbrs[i]] for i in mem])
        touch = touch[touch != c]
        long_ = touch[size[touch] >= min_len]
        tgt = np.bincount(long_ if len(long_) else touch).argmax()
        pairs = [(i, q) for i in mem for q in nbrs[i] if lab[q] == tgt]
        links.append(min(pairs, key=lambda e: np.linalg.norm(pts[e[0]] - pts[e[1]])))   # closest contact
        lab[mem] = tgt
        size[tgt] += size[c]; size[c] = 0
    _, lab = np.unique(lab, return_inverse=True)
    return lab, links


class UnionCell:
    """Union of `RoundedPolytope` pieces with a common r; the tissue-mesher cell interface."""

    def __init__(self, pieces, max_offset=np.inf):
        self.pieces, self.max_offset = pieces, max_offset
        self.r = max(p.r for p in pieces)
        self.lo = np.array([p.bounds()[0] for p in pieces])
        self.hi = np.array([p.bounds()[1] for p in pieces])

    def bounds(self):
        return self.lo.min(0), self.hi.max(0)

    @property
    def face_normals(self):
        """Outward normals of all pieces' faces."""
        return np.vstack([p.face_normals for p in self.pieces])

    def translated(self, t):
        """The cell translated by t."""
        return UnionCell([p.translated(t) for p in self.pieces], self.max_offset)

    def _near(self, X, k, pad):
        return np.all((X > self.lo[k] - pad) & (X < self.hi[k] + pad), axis=1)

    def _phi_all(self, X, pad, band=np.inf):
        """(N, K) piece SDFs (exact below `band`, see `RoundedPolytope.phi`), +inf beyond pad of a piece's box."""
        D = np.full((len(X), len(self.pieces)), np.inf)
        for k, p in enumerate(self.pieces):
            s = self._near(X, k, pad)
            if s.any():
                D[s, k] = p.phi(X[s], band=band)
        return D

    def phi(self, X, band=np.inf):
        pad = min(band, 1e3)
        d = np.full(len(X), pad)
        for k, p in enumerate(self.pieces):
            s = self._near(X, k, pad)
            if s.any():
                d[s] = np.minimum(d[s], p.phi(X[s], band=band))
        return d

    def _closest(self, X):
        """Closest surface point and the piece it lies on, for X near the surface."""
        D = self._phi_all(X, 1.0, band=1.0 + self.r)
        Q = np.full_like(X, np.nan)
        best = np.full(len(X), np.inf)
        kq = np.full(len(X), -1)
        for k, p in enumerate(self.pieces):
            s = np.isfinite(D[:, k]) & (D[:, k] > -p.r + 1e-9)       # outside piece k's core
            if not s.any():
                continue
            idx = np.nonzero(s)[0]
            q = p.project(X[idx])
            other = self._phi_all(q, 1e-6, band=self.r)      # only the sign matters
            other[:, k] = np.inf
            ok = other.min(1) >= -1e-9                                # not inside another piece
            dist = np.linalg.norm(q - X[idx], axis=1)
            take = ok & (dist < best[idx])
            Q[idx[take]], best[idx[take]], kq[idx[take]] = q[take], dist[take], k
        for i in np.nonzero(kq < 0)[0]:     # concave crease: alternate projections onto the pieces
            x = X[i:i + 1]
            k = int(np.argmin(np.where(D[i] > -np.array([p.r for p in self.pieces]), np.abs(D[i]), np.inf)))
            q = self.pieces[k].project(x)
            for _ in range(50):
                dq = self._phi_all(q, 1e-6, band=self.r)[0]
                j = int(np.argmin(dq))
                if dq[j] >= -1e-9:
                    break
                q, k = self.pieces[j].project(q), j
            Q[i], kq[i] = q[0], k
        return Q, kq

    def project(self, X):
        return X if len(X) == 0 else self._closest(X)[0]

    def normal(self, X, blend=0.1):
        """Outward unit normal; near a crease (pieces within `blend` of the nearest) the piece normals are blended."""
        Q, kq = self._closest(X)
        D = self._phi_all(Q, blend, band=blend + self.r)
        r = np.array([p.r for p in self.pieces])
        Wt = np.clip(1 - (D - D.min(1, keepdims=True)) / blend, 0, None)
        Wt[D <= -r + 1e-9] = 0                          # Q inside that piece's core: no normal there
        Wt[np.arange(len(X)), kq] = np.maximum(Wt[np.arange(len(X)), kq], 1e-3)
        n = np.zeros_like(X)
        for k, p in enumerate(self.pieces):
            s = Wt[:, k] > 0
            if s.any():
                n[s] += Wt[s, k, None] * p.normal(Q[s])
        return n / np.linalg.norm(n, axis=1)[:, None]


def wall_defects(cell, B, delta, hg=0.025):
    """Topology changes on the box walls: sum over walls of how much the number of connected components of
    {phi < t} differs between t = -delta, 0, delta (0 = every wall crossing is deep or missed by delta).

    Catches shallow crossings and near misses of single pieces and thin necks, which the per-wall extent of the
    whole cell does not see.
    """
    from scipy.ndimage import label
    lo, hi = cell.bounds()
    bad = 0
    for k in range(6):
        a, val = k // 2, (0.0 if k % 2 == 0 else B[k // 2])
        if not lo[a] - delta < val < hi[a] + delta:
            continue
        u, v = [x for x in range(3) if x != a]
        gu = np.arange(max(lo[u], 0.0), min(hi[u], B[u]) + hg, hg)
        gv = np.arange(max(lo[v], 0.0), min(hi[v], B[v]) + hg, hg)
        if len(gu) < 2 or len(gv) < 2:
            continue
        X = np.zeros((len(gu) * len(gv), 3)); X[:, a] = val
        X[:, u], X[:, v] = (g.ravel() for g in np.meshgrid(gu, gv, indexing="ij"))
        f = cell.phi(X, band=cell.r + 2 * delta).reshape(len(gu), len(gv))
        n = [label(f < t)[1] for t in (-delta, 0.0, delta)]
        bad += abs(n[0] - n[1]) + abs(n[2] - n[1])
    return bad


def make_tube_cells(w, r, box, d=2.0, seed=0, v_drop=1.0, delta=0.05, r_range=(0.7, 1.3), n_r=13, n_mid=1,
                    max_shift=0.05, n_shift=100, **chain_kw):
    """Tube cells with gap w inside the box [0, box].

    Gap: piece a (rounding r_a) is eroded by w/2 + r_a against every piece c of another cell whose Voronoi cell
    may come within w + 2 r_max of its own (seed distance below rho_a + rho_c + w + 2 r_max, rho = Voronoi
    circumradius), not only against its Voronoi neighbours. A ball of radius w/2 + r_a around any core point of a
    then stays on a's side of every such bisector, so the two surfaces are at least w apart, whatever shape the
    union has.

    Wall crossings (see `wall_defects`) are made robust by one r_k for the whole cell from r * r_range, then by a
    translation (see `fit_rounding_and_shift`).

    `n_mid` extra seeds on each link (same cell) make the tubes smoother and the creases between pieces shallower.

    Args:
        w: Gap width between neighbouring cells.
        r: Nominal rounding radius.
        box: Box size (3,); the box spans [0, box].
        d: Seed spacing (tube diameter).
        seed: Random seed for the seeds and the chains.

    Returns:
        (cells, ids, info): the `UnionCell` cells, a w-independent label of each cell, and a dict with chain/cell
        statistics and `faces`: per cell, the triangles (M, 3, 3) of the raw Voronoi faces inside the box between
        its pieces and those of other cells, a reference surface that does not depend on w (synapse placement).
    """
    B = np.asarray(box, float)
    rng = np.random.default_rng(seed)
    pts = voronoi_seeds(B, d, seed)

    def adjacency(pts):
        vor = Voronoi(pts)
        nbrs = [[] for _ in range(len(pts))]
        for a, b in vor.ridge_points:
            nbrs[a].append(b); nbrs[b].append(a)
        return vor, nbrs

    vor, nbrs = adjacency(pts)
    chain, links = link_chains(pts, nbrs, rng, **chain_kw)
    if n_mid and links:
        L = np.array(links)
        t = np.arange(1, n_mid + 1) / (n_mid + 1)
        mid = (pts[L[:, 0], None] * (1 - t[:, None]) + pts[L[:, 1], None] * t[:, None]).reshape(-1, 3)
        pts = np.vstack([pts, mid])
        chain = np.r_[chain, np.repeat(chain[L[:, 0]], n_mid)]
        vor, nbrs = adjacency(pts)
    n = len(pts)
    big = box_halfspaces(-2 * d * np.ones(3), B + 2 * d)
    box_hs = box_halfspaces(np.zeros(3), B)

    def vor_hs(i, js):
        q = pts[js]
        nrm = (q - pts[i]) / np.linalg.norm(q - pts[i], axis=1)[:, None]
        return nrm, np.einsum("ij,ij->i", nrm, (pts[i] + q) / 2)

    # pieces in the box (raw Voronoi cell clipped to the box), their circumradius, and shared faces in the box
    in_box, rho, clip_V, clip_hs = np.zeros(n, bool), np.zeros(n), {}, {}
    for i in range(n):
        nrm, off = vor_hs(i, nbrs[i])
        hs = np.vstack([np.c_[nrm, -off], box_hs])
        cc = chebyshev_center(hs)
        if cc is None:
            continue
        V = HalfspaceIntersection(hs, cc[0]).intersections
        in_box[i] = True
        clip_V[i], clip_hs[i] = V, hs
        Vb = HalfspaceIntersection(np.vstack([np.c_[nrm, -off], big]), pts[i]).intersections
        rho[i] = np.linalg.norm(Vb - pts[i], axis=1).max()
    kept = np.nonzero(in_box)[0]
    kid = {i: k for k, i in enumerate(kept)}
    ea, eb = [], []
    for a, b in vor.ridge_points:
        if in_box[a] and in_box[b] and chain[a] == chain[b]:
            nrm = (pts[b] - pts[a]) / np.linalg.norm(pts[b] - pts[a])
            on = clip_V[a][np.abs(clip_V[a] @ nrm - nrm @ (pts[a] + pts[b]) / 2) < 1e-9]
            if len(on) >= 3 and np.linalg.matrix_rank(on[1:] - on[0], tol=1e-6) == 2:
                ea.append(kid[a]); eb.append(kid[b])
    nc, comp = connected_components(coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(len(kept),) * 2),
                                    directed=False)
    cell_of = np.full(n, -1)
    cell_of[kept] = comp

    tree = cKDTree(pts[kept])
    radii = rounding_radii(r, r_range, n_r)
    r_max = radii.max()

    def piece(i, rk):
        """Piece i rounded by rk (None if its core is empty)."""
        js = list(nbrs[i])
        near = kept[tree.query_ball_point(pts[i], rho[i] + rho[kept].max() + w + 2 * r_max)]
        js += [c for c in near if cell_of[c] != cell_of[i] and c != i and c not in nbrs[i]
               and np.linalg.norm(pts[c] - pts[i]) < rho[i] + rho[c] + w + 2 * r_max]
        nrm, off = vor_hs(i, js)
        foreign = np.array([cell_of[j] != cell_of[i] and (in_box[j] or chain[j] != chain[i]) for j in js])
        return rounded_polytope(np.c_[nrm, -(off - foreign * (w / 2 + rk))], big, rk)

    def assemble(members, rk):
        """The cell of pieces rounded by rk (pieces with an empty core left out), or None if all are empty."""
        ok = [p for p in (piece(i, rk) for i in members) if p is not None]
        return UnionCell(ok, max_offset=0.2 * w) if ok else None

    def score(cell):
        return wall_defects(cell, B, delta), max(delta - edge_margin(cell, B), 0.0)

    cells, ids, faces, n_bad, n_lost, n_shifted, vols = [], [], [], 0, 0, 0, []
    for c in range(nc):
        members = kept[comp == c]
        vol = sum(ConvexHull(clip_V[i]).volume for i in members)
        if vol < v_drop:
            continue
        fit = fit_rounding_and_shift(lambda rk: assemble(members, rk), score, radii, w,
                                     np.random.default_rng([seed, c]), max_shift, n_shift)
        if fit is None:                           # dropping it would renumber the cells at this w only
            raise ValueError(f"cell {c}: every piece's eroded core is empty at w = {w}; w is too large")
        cell, sc, shifted = fit
        n_shifted += shifted
        n_bad += any(sc)
        n_lost += len(members) - len(cell.pieces)
        cells.append(cell); ids.append(c); vols.append(vol)
        tri = []
        for i in members:
            t, k = facet_triangles(clip_V[i], clip_hs[i])
            nb = np.r_[cell_of[nbrs[i]], c]               # the box walls (k >= len(nbrs[i])) count as own
            tri.append(t[nb[np.minimum(k, len(nbrs[i]))] != c])
        faces.append(np.concatenate(tri))
    if not cells:
        raise ValueError(f"no cell has a volume of at least {v_drop} in the box")
    n_pieces = np.array([len(c.pieces) for c in cells])
    info = dict(n_seeds=n, n_chains=chain.max() + 1, n_cells=len(cells), n_bad=n_bad, n_shifted=n_shifted,
                pieces_lost=n_lost, pieces=np.percentile(n_pieces, [0, 25, 50, 75, 100]).tolist(),
                volume_median=float(np.median(vols)), faces=faces)
    return cells, ids, info
