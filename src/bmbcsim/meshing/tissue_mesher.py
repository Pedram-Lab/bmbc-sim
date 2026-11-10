"""Tetrahedral mesh of a packed-cell tissue in a box, from per-cell signed distance functions.

Pipeline (see `surface_mesh` and `volume_mesh`):

1. Membranes: one triangle mesh per cell (marching cubes, isotropic remeshing, exact projection), cut at the box
   walls, cleaned up along the cut, and corrected for the volume lost to faceting (secant offset).
2. Walls: one constrained Delaunay triangulation per box face, with the membrane outlines and the box edges as
   fixed segments. Each wall region is named `cell_boundary` (inside a cell) or after the wall (ECS).
3. Volume: TetGen fills the closed multi-domain surface without changing it, the tets are labelled per domain, and
   netgen optimizes the interior.

A cell is any object with `phi(X, band)`, `project(X)`, `normal(X)`, `bounds()`, `r` and `max_offset`, e.g.
`bmbcsim.geometry.rounded_voronoi.RoundedPolytope`. Region names: `ecs`, `cell_k`; boundary names: `membrane_k`,
`cell_boundary`, `left`, `right`, `top`, `bottom`, `front`, `back` (x-, x+, y-, y+, z-, z+).

`save_tissue_mesh` / `load_tissue_mesh` store the mesh as a .vtu file with one `cell_id` per tet (-1 = ECS) and
rebuild the boundary names on load. Meshing needs the `geometry` extra (pymeshlab, triangle, tetgen); loading
does not.
"""
import time

import numpy as np
import ngsolve as ngs
import pyvista as pv
from netgen.meshing import FaceDescriptor, Mesh, MeshingParameters
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bmbcsim.logging import logger

WALLS = ["left", "right", "top", "bottom", "front", "back"]  # x-, x+, y-, y+, z-, z+


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


def _secant_offset(P, F, mask, cell, B, h):
    """Move points outward so that the faceted membrane encloses the exact cell volume.

    The gap of a triangle to the surface is the mean of -phi at its edge midpoints (exact on average for a
    quadratic surface); each point moves by the area-weighted mean gap of its triangles, at most
    `cell.max_offset`. Points on walls move within their wall plane(s); no point moves towards a wall.
    """
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    gap_t = -cell.phi((P[E[:, 0]] + P[E[:, 1]]) / 2).reshape(3, -1).mean(0)
    X = P[F]
    area = np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1) / 2
    wsum = np.bincount(F.ravel(), np.repeat(area, 3), len(P))
    gap_v = np.bincount(F.ravel(), np.repeat(area * gap_t, 3), len(P)) / np.maximum(wsum, 1e-30)
    gap_v = np.clip(gap_v, -cell.max_offset, cell.max_offset)
    d = gap_v[:, None] * cell.normal(P)
    for k in range(6):
        a, val, side = k // 2, (0.0 if k % 2 == 0 else B[k // 2]), (-1.0 if k % 2 == 0 else 1.0)
        d[(mask >> k) & 1 == 1, a] = 0.0
        d[(side * d[:, a] > 0) & (np.abs(P[:, a] - val) < h), a] = 0.0
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


def membrane(cell, box, h, iters=10, snap=0.3, collapse=0.3):
    """Triangle mesh of one cell's surface inside the box.

    Args:
        cell: The cell (see the module docstring).
        box: Box size (3,).
        h: Target edge length.
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
    img.point_data["phi"] = cell.phi(np.asarray(img.points), band=cell.r + 3 * hg)
    s = img.contour([0.0], scalars="phi", method="flying_edges").triangulate().clean()
    P, F = _remesh(cell.project(np.asarray(s.points, float)), s.regular_faces, h, iters)  # closed: no border issues
    P, F = _fix_flips(cell.project(P), F, cell)
    planes = [(k // 2, 0.0 if k % 2 == 0 else B[k // 2]) for k in range(6)]
    mask = np.zeros(len(P), np.int64)
    for k, (a, val) in enumerate(planes):
        side = -1.0 if k % 2 == 0 else 1.0
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
    P = _secant_offset(P, F, mask, cell, B, h)
    s = _orientation(P, F, cell)
    if np.sum(s < 0) > len(F) / 2:
        F = F[:, [1, 0, 2]]; s = -s
    if np.any(s <= 0):
        raise RuntimeError(f"membrane has {np.sum(s <= 0)} flipped triangles")
    _check_outlines(P, F, mask)
    return P, F, mask


def membrane_robust(cell, box, h, variants=((1.0, 0.3), (0.95, 0.3), (1.05, 0.25), (0.9, 0.35), (1.1, 0.2))):
    """`membrane` with retries at a slightly different h and snap distance. Returns (P, F, mask, attempt)."""
    err = None
    for attempt, (fh, snap) in enumerate(variants):
        try:
            return (*membrane(cell, box, h * fh, snap=snap), attempt)
        except Exception as e:
            err = e
    raise RuntimeError(f"membrane failed in all {len(variants)} variants: {err}")


# --- walls and closed surface -------------------------------------------------------------------------------------

def surface_mesh(cells, box, h, workers=8):
    """Closed multi-domain surface mesh of the cells in the box.

    Domain numbers: 0 = outside the box, 1 = ECS, k + 2 = cell k.

    Returns:
        (points, triangles, tags): tags[t] = (domin, domout, name); each triangle's normal points out of domin.
    """
    import triangle as tr
    t0 = time.time()
    B = np.asarray(box, float)
    if workers > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            mems = pool.starmap(membrane_robust, [(c, B, h) for c in cells])
    else:
        mems = [membrane_robust(c, B, h) for c in cells]
    t_mem = time.time() - t0

    G, tris, tags = [], [], []
    border_pts = []                   # (point id, wall mask, cell)
    seg_by_wall = [[] for _ in range(6)]
    for kc, (P, F, mask, _) in enumerate(mems):
        off = len(G)
        G += list(P)
        tris.append(F + off); tags += [(kc + 2, 1, f"membrane_{kc}")] * len(F)
        E, cnt = _edges(F)
        for i, j in E[cnt == 1]:
            common = mask[i] & mask[j]
            for k in range(6):
                if (common >> k) & 1:
                    seg_by_wall[k].append((i + off, j + off))
        border_pts += [(i + off, mask[i], kc) for i in np.nonzero(mask)[0]]

    # box edges: corners and membrane points on the edge, gaps filled at spacing ~h
    corner_id, chain = {}, {}
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
                pts += [(G[g][c], g) for g, m, _ in border_pts if (m & bits) == bits]
                pts.sort()
                ids = [pts[0][1]]
                for (ta, ga), (tb, gb) in zip(pts[:-1], pts[1:]):
                    nfill = max(int(round((tb - ta) / h)), 1)
                    for s in range(1, nfill):
                        X = np.zeros(3); X[c] = ta + (tb - ta) * s / nfill; X[ea] = va; X[eb] = vb
                        ids.append(len(G)); G.append(X)
                    ids.append(gb)
                chain[(c, ea, eb, va, vb)] = ids

    # walls: one CDT per wall; new points on membrane outlines are inserted into the membranes below
    splits = {}                       # membrane outline edge -> new wall points on it (ordered from its first end)
    used_new = set()
    n_yy = 0
    for k in range(6):
        a, val = k // 2, (0.0 if k % 2 == 0 else B[k // 2])
        u, v = [x for x in range(3) if x != a]
        segs = list(seg_by_wall[k])
        n_out = len(segs)
        for (c, ea, eb, va, vb), ids in chain.items():
            if (ea == a and va == val) or (eb == a and vb == val):
                segs += list(zip(ids[:-1], ids[1:]))
        Gk = np.asarray(G)
        gids = np.unique(np.array(segs).ravel())
        loc = np.full(len(G), -1); loc[gids] = np.arange(len(gids))
        V2 = Gk[gids][:, [u, v]]
        S = loc[np.array(segs)]
        marks = np.r_[np.arange(n_out) + 2, np.ones(len(segs) - n_out, int)]
        # new points may split membrane outlines (Y), not box edges, which the neighbouring wall shares
        out = tr.triangulate(dict(vertices=V2, segments=S, segment_markers=marks), f"pq30Ya{0.6 * h * h:.6f}")
        sub = out["segments"][out["segment_markers"].ravel() >= 2]
        if len(sub) and np.linalg.norm(out["vertices"][sub[:, 0]] - out["vertices"][sub[:, 1]], axis=1).min() < h / 4:
            # refinement cascade along an outline (shallow crossing or thin ECS gap): no outline splits on this
            # wall; with YY, an angle bound (q30, even q20) makes Triangle reject the points near outlines and leave
            # large fans (up to 3.6 um^2), so refine by area only
            out = tr.triangulate(dict(vertices=V2, segments=S, segment_markers=marks), f"pYYa{0.6 * h * h:.6f}")
            n_yy += 1
        W, T = out["vertices"], out["triangles"]
        if len(W) < len(gids) or not np.allclose(W[:len(gids)], V2):
            raise RuntimeError(f"wall {WALLS[k]}: triangle moved input points")
        new = np.zeros((len(W) - len(gids), 3)); new[:, a] = val; new[:, u] = W[len(gids):, 0]
        new[:, v] = W[len(gids):, 1]
        g_all = np.r_[gids, len(G) + np.arange(len(new))]
        G += list(new)
        S_out, m_out = out["segments"], out["segment_markers"].ravel()
        for mk in np.unique(m_out[m_out >= 2]):
            sub = S_out[m_out == mk]
            if len(sub) == 1:
                continue
            gi, gj = segs[mk - 2]
            inner = np.setdiff1d(np.unique(sub), loc[[gi, gj]])
            order = np.argsort(np.linalg.norm(W[inner] - V2[loc[gi]], axis=1))
            splits[(gi, gj)] = list(g_all[inner[order]])
            if not used_new.isdisjoint(splits[(gi, gj)]):
                raise RuntimeError(f"wall {WALLS[k]}: membrane outlines cross near {Gk[gi]}")
            used_new.update(splits[(gi, gj)])
        X3 = np.zeros((len(W), 3)); X3[:, a] = val; X3[:, u] = W[:, 0]; X3[:, v] = W[:, 1]
        # regions: flood fill across edges that are no segments
        Es = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)
        tid = np.tile(np.arange(len(T)), 3)
        o = np.lexsort((Es[:, 1], Es[:, 0])); Es, tid = Es[o], tid[o]
        same = np.all(Es[1:] == Es[:-1], axis=1)
        seg_set = set(map(tuple, np.sort(S_out, axis=1)))
        free = np.array([tuple(e) not in seg_set for e in Es[:-1][same]], bool)
        pa, pb = tid[:-1][same][free], tid[1:][same][free]
        nc, comp = connected_components(coo_matrix((np.ones(len(pa)), (pa, pb)), shape=(len(T),) * 2),
                                        directed=False)
        C = X3[T].mean(1)
        cand = sorted({kc for _, m, kc in border_pts if (m >> k) & 1})
        side = -1.0 if k % 2 == 0 else 1.0
        nrm = np.cross(X3[T[:, 1]] - X3[T[:, 0]], X3[T[:, 2]] - X3[T[:, 0]])[:, a]
        T = np.where((side * nrm < 0)[:, None], T[:, [1, 0, 2]], T)     # normal out of the box
        for ci in range(nc):
            idx = np.nonzero(comp == ci)[0]
            votes = [kc for kc in cand if np.mean(cells[kc].phi(C[idx]) < 0) > 0.5]
            if len(votes) > 1:
                raise RuntimeError(f"wall {WALLS[k]}: region claimed by cells {votes}")
            tris.append(g_all[T[idx]])
            tags += [(votes[0] + 2, 0, "cell_boundary") if votes else (1, 0, WALLS[k])] * len(idx)

    # insert the new wall points on membrane outlines into the membrane triangles (fan from the opposite point)
    M = [list(f) for f in np.concatenate(tris[:len(cells)])]
    nm = len(M)
    mtags = tags[:nm]
    edge_tri = {}

    def register(t):
        f = M[t]
        for x in range(3):
            edge_tri[(min(f[x], f[(x + 1) % 3]), max(f[x], f[(x + 1) % 3]))] = t

    for t in range(nm):
        register(t)
    for (gi, gj), pts in splits.items():   # one at a time: a triangle can have two split outline edges
        t = edge_tri[(min(gi, gj), max(gi, gj))]
        f = M[t]
        x = [y for y in range(3) if {f[y], f[(y + 1) % 3]} == {gi, gj}][0]
        uu, vv, q = f[x], f[(x + 1) % 3], f[(x + 2) % 3]
        ch = [uu] + (pts if uu == gi else pts[::-1]) + [vv]
        M[t] = [ch[0], ch[1], q]
        register(t)
        for s in range(1, len(ch) - 1):
            M.append([ch[s], ch[s + 1], q]); mtags.append(mtags[t])
            register(len(M) - 1)
    G = np.asarray(G)
    tris = np.concatenate([np.array(M, np.int64)] + tris[len(cells):])
    tags = mtags + tags[nm:]
    bad = check_closed(G, tris, tags)
    if bad:
        raise RuntimeError(f"open or inconsistent domain boundaries (domain: open, duplicate edges): {bad}")
    logger.info(f"surface: {len(cells)} cells, {len(tris)} triangles, {len(G)} points, {n_yy} walls without "
                f"outline splits, {sum(m[3] for m in mems)} membrane retries (membranes {t_mem:.0f} s, "
                f"total {time.time() - t0:.0f} s)")
    return G, tris, tags


def check_closed(G, tris, tags):
    """Domains whose boundary is not closed and consistently oriented: {domain: (open edges, duplicate edges)}."""
    dom = np.array([(a, b) for a, b, _ in tags])
    bad = {}
    for d in np.unique(dom):
        if d == 0:
            continue
        T = np.r_[tris[dom[:, 0] == d], tris[dom[:, 1] == d][:, [1, 0, 2]]]
        E = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
        fw = set(map(tuple, E)); bw = set(map(tuple, E[:, ::-1]))
        if len(fw ^ bw) or len(fw) != len(E):
            bad[int(d)] = (len(fw ^ bw), len(E) - len(fw))
    return bad


def surface_ecs_fraction(G, tris, tags, box):
    """ECS volume fraction enclosed by the surface mesh (divergence theorem); TetGen keeps the surface, so this is
    also the ECS fraction of the volume mesh."""
    X = G[tris]
    vol = np.einsum("ij,ij->i", X[:, 0], np.cross(X[:, 1], X[:, 2])) / 6
    din = np.array([t[0] for t in tags]); dout = np.array([t[1] for t in tags])
    return (vol[din == 1].sum() - vol[dout == 1].sum()) / np.prod(box)


# --- volume -------------------------------------------------------------------------------------------------------

def _face_key(F, N):
    F = np.sort(F, axis=1).astype(np.int64)
    return (F[:, 0] * N + F[:, 1]) * N + F[:, 2]


def volume_mesh(G, tris, tags, n_cells, maxvol=0.15, minratio=1.5, mindihedral=10.0, optsteps=5):
    """Tet mesh of a closed surface mesh from `surface_mesh`: TetGen, then netgen optimization of the interior.

    Tets are grouped into connected components across faces that are not surface triangles; each component is
    labelled by the oriented surface triangles around it.
    """
    import tetgen
    t0 = time.time()
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
    votes = np.zeros((nc, n_cells + 2))
    din = np.array([t[0] for t in tags]); dout = np.array([t[1] for t in tags])
    for off in (0, 1):
        jj = np.minimum(j0 + off, len(fk) - 1)
        ok = fk[jj] == sk
        side = np.einsum("ij,ij->i", nrm, nodes[opp[jj]] - Y[:, 0])
        np.add.at(votes, (comp[ft[jj[ok]]], np.where(side < 0, din, dout)[ok]), 1)
    reg = votes.argmax(1)
    purity = votes.max(1) / np.maximum(votes.sum(1), 1)
    if np.any(reg == 0) or purity.min() < 0.99:
        raise RuntimeError(f"tet regions: {np.sum(reg == 0)} unlabelled components, min purity {purity.min():.3f}")
    ecs = np.nonzero(reg == 1)[0]
    if len(ecs) > 1:
        # sealed ECS pockets (e.g. at a box corner that a cell crosses on all three walls) join the cell around them
        X = nodes[Tet]
        vol = np.bincount(comp, np.abs(np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]),
                                                 X[:, 3] - X[:, 0])) / 6, nc)
        pair = (fk[1:] == fk[:-1]) & np.isin(fk[1:], sk)            # the two tets at each interior surface triangle
        ca, cb = comp[ft[:-1][pair]], comp[ft[1:][pair]]
        for c in ecs[np.argsort(vol[ecs])[:-1]]:
            nb = np.r_[cb[ca == c], ca[cb == c]]
            reg[c] = np.bincount(reg[nb], minlength=n_cells + 2).argmax()
            logger.warning(f"sealed ECS pocket of {vol[c]:.2g} um^3 added to cell_{reg[c] - 2}")
    treg = reg[comp]
    if set(np.unique(treg)) != set(range(1, n_cells + 2)):
        raise RuntimeError("a domain received no tets")
    t1 = time.time()
    mesh = tissue_mesh_from_labels(nodes, Tet, treg - 2)   # keep alive: its ngmesh dies with it
    mesh.ngmesh.OptimizeVolumeMesh(MeshingParameters(optsteps3d=optsteps))
    mesh = ngs.Mesh(mesh.ngmesh.Copy())
    logger.info(f"volume: {mesh.ne} tets, {mesh.nv} points (TetGen {t1 - t0:.0f} s, "
                f"netgen optimize {time.time() - t1:.0f} s)")
    return mesh


def tissue_mesh_from_labels(nodes, tets, cell_id):
    """NGSolve mesh with the tissue names from labelled tets (cell_id -1 = ECS, k = cell k).

    Interfaces between the ECS and cell k become `membrane_k`; faces on the box (the bounding box of the nodes)
    become `cell_boundary` (cell side) or the wall name (ECS side).
    """
    nodes = np.asarray(nodes, float)
    tets = np.asarray(tets, np.int64).copy()
    treg = np.asarray(cell_id, np.int64) + 2         # netgen domain: 1 = ECS, k + 2 = cell k
    lo, hi = nodes.min(0), nodes.max(0)
    N = len(nodes)
    X = nodes[tets]
    sv = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
    tets[sv > 0] = tets[sv > 0][:, [1, 0, 2, 3]]     # netgen orientation
    faces = np.r_[tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]
    opp = np.r_[tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]]
    freg = np.tile(treg, 4)
    fk = _face_key(faces, N)
    o = np.lexsort((freg, fk))
    faces, fk, opp, freg = faces[o], fk[o], opp[o], freg[o]
    pair = np.r_[fk[1:] == fk[:-1], False]
    first = np.r_[True, ~pair[:-1]]                  # first face of each key
    single = first & ~pair
    iface = np.nonzero(pair & (freg != np.r_[freg[1:], -1]))[0]   # faces i, i + 1 in different domains
    if np.any((freg[iface] > 1) & (freg[iface + 1] > 1)):
        raise ValueError("cells touch: tissue meshes need ECS between all cells")
    # interfaces: domin = cell (freg[i + 1], the higher domain), domout = ECS; boundary faces: domout = 0
    F = np.r_[faces[iface], faces[single]]
    domin = np.r_[freg[iface + 1], freg[single]]
    domout = np.r_[freg[iface], np.zeros(single.sum(), np.int64)]
    opp_in = np.r_[opp[iface + 1], opp[single]]
    names = [f"membrane_{d - 2}" for d in domin[:len(iface)]]
    C = nodes[faces[single]].mean(1)
    tol = 1e-6 * np.max(hi - lo)
    on = np.c_[np.abs(C - lo) < tol, np.abs(C - hi) < tol][:, [0, 3, 1, 4, 2, 5]]   # WALLS order
    if not np.all(on.any(1)):
        raise ValueError("boundary face off the box: the mesh is not a box")
    names += ["cell_boundary" if d > 1 else WALLS[w] for d, w in zip(domin[len(iface):], on.argmax(1))]
    Y = nodes[F]
    s = np.einsum("ij,ij->i", np.cross(Y[:, 1] - Y[:, 0], Y[:, 2] - Y[:, 0]), nodes[opp_in] - Y[:, 0])
    F[s > 0] = F[s > 0][:, [1, 0, 2]]                # normal out of domin

    m = Mesh(dim=3)
    m.AddPoints(np.ascontiguousarray(nodes))
    for d in range(1, treg.max() + 1):
        m.SetMaterial(d, "ecs" if d == 1 else f"cell_{d - 2}")
        m.AddElements(dim=3, index=d, data=np.ascontiguousarray(tets[treg == d], np.int32), base=0)
    groups = {}
    for i, key in enumerate(zip(domin, domout, names)):
        groups.setdefault(key, []).append(i)
    for gi, ((di, do, name), idx) in enumerate(sorted(groups.items())):
        fd = m.Add(FaceDescriptor(surfnr=gi + 1, domin=int(di), domout=int(do), bc=gi + 1))
        m.SetBCName(fd - 1, name)
        m.AddElements(dim=2, index=fd, data=np.ascontiguousarray(F[idx], np.int32), base=0)
    return ngs.Mesh(m)


# --- file I/O -----------------------------------------------------------------------------------------------------

def save_tissue_mesh(path, mesh, **meta):
    """Write a tissue mesh as .vtu: tets with a `cell_id` (-1 = ECS, k = cell_k); `meta` goes to the field data."""
    ngm = mesh.ngmesh
    el = ngm.Elements3D().NumPy()
    tets = el["nodes"][:, :4].astype(np.int64) - 1
    names = ["ecs"] + [f"cell_{k}" for k in range(len(mesh.GetMaterials()) - 1)]
    if list(mesh.GetMaterials()) != names:
        raise ValueError(f"not a tissue mesh: materials {mesh.GetMaterials()}")
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, ngm.Coordinates())
    grid.cell_data["cell_id"] = el["index"].astype(np.int32) - 2
    for key, value in meta.items():
        grid.field_data[key] = np.atleast_1d(value)
    grid.save(path)


def load_tissue_mesh(path):
    """Read a .vtu file from `save_tissue_mesh`. Returns (mesh, meta), size-1 field data as scalars."""
    grid = pv.read(path)
    tets = grid.cells_dict[pv.CellType.TETRA]
    mesh = tissue_mesh_from_labels(grid.points, tets, grid.cell_data["cell_id"])
    meta = {k: (v.item() if v.size == 1 else v) for k, v in ((k, grid.field_data[k]) for k in grid.field_data)}
    return mesh, meta
