"""Rounded Voronoi cells: a packed-cell tissue whose gap width w sets the ECS fraction.

Each cell is a Voronoi cell of Lloyd-relaxed seeds, eroded by w/2 + r_k and rounded by r_k (the Minkowski sum of
the eroded polytope with a ball), see `bmbcsim.geometry.cells.RoundedPolytope`. Two neighbours are w apart (at
least w/2 where a cell was shifted), and the cell set does not depend on w, so all w values give the same cells
with the same numbering.
"""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi

from bmbcsim.geometry.cells import (CellSet, box_halfspaces, chebyshev_center, edge_margin, facet_triangles,
                                    fit_rounding_and_shift, rounded_polytope, rounding_radii, voronoi_seeds)


def make_rounded_voronoi_cells(w, r, box, d=4.0, seed=0, v_drop=1.0, delta=0.05, r_range=(0.7, 1.3), n_r=25,
                               max_shift=0.05, n_shift=200):
    """Rounded Voronoi cells with gap w inside the box [0, box].

    A Voronoi cell is kept if its raw (uneroded) volume inside the box is at least `v_drop`; this does not depend
    on w. Per cell, r_k is the value in r * r_range closest to r for which every wall and box edge is crossed or
    missed by at least `delta`; if there is none, the cell is shifted (see `fit_rounding_and_shift`). The flat
    faces, and hence the gaps, do not depend on r_k.

    Args:
        w: Gap width between neighbouring cells.
        r: Nominal rounding radius.
        box: Box size (3,).
        d: Target cell diameter for the seeds.
        seed: Random seed for the seeds.

    Returns:
        The `RoundedPolytope` cells, labelled by seed index; stats: the number of shifted cells and of cells that
        still miss the `delta` margin (`n_bad`).
    """
    B = np.asarray(box, float)
    pts = voronoi_seeds(B, d, seed)
    vor = Voronoi(pts)
    box_hs = box_halfspaces(np.zeros(3), B)
    big = box_halfspaces(-2 * d * np.ones(3), B + 2 * d)   # bounds the open Voronoi cells of the outermost seeds
    radii = rounding_radii(r, r_range, n_r)

    def halfspaces(i, erosion):
        p = pts[i]
        rp = vor.ridge_points[(vor.ridge_points == i).any(1)]
        q = pts[np.where(rp[:, 0] == i, rp[:, 1], rp[:, 0])]
        nrm = (q - p) / np.linalg.norm(q - p, axis=1)[:, None]
        return np.c_[nrm, -(np.einsum("ij,ij->i", nrm, (p + q) / 2) - erosion)]

    def score(c):
        e = np.array([c.extent(h[:3], h[3]) for h in box_hs])    # > 0: crosses the wall
        return (max(delta - min(np.abs(e).min(), edge_margin(c, B)), 0.0),)

    cells, ids, faces, n_bad, n_shifted = [], [], [], 0, 0
    for i in range(len(pts)):
        hs_in = np.vstack([halfspaces(i, 0.0), box_hs])
        cc = chebyshev_center(hs_in)
        if cc is None:
            continue
        V = HalfspaceIntersection(hs_in, cc[0]).intersections
        if ConvexHull(V).volume < v_drop:
            continue
        fit = fit_rounding_and_shift(lambda rk: rounded_polytope(halfspaces(i, w / 2 + rk), big, rk), score,
                                     radii, w, np.random.default_rng([seed, i]), max_shift, n_shift)
        if fit is None:         # dropping it would renumber the cells at this w only
            raise ValueError(f"cell of seed {i}: eroded core empty for every r_k at w = {w}; w is too large")
        cell, sc, shifted = fit
        n_shifted += shifted
        n_bad += any(sc)
        cells.append(cell)
        ids.append(i)
        tris, k = facet_triangles(V, hs_in)
        faces.append(tris[k < len(hs_in) - 6])    # not on the box walls
    if not cells:
        raise ValueError(f"no cell has a volume of at least {v_drop} in the box")
    # two neighbours (gap >= w/2 after shifts) never cross after the offset
    return CellSet(cells, np.array(ids), faces, 0.2 * w, dict(n_bad=n_bad, n_shifted=n_shifted))
