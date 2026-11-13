"""Generate tissue meshes (round or tube-shaped cells) with a target ECS volume fraction.

For each (seed, ecs_ratio), bisects the gap w between the cells until the ECS fraction of the surface mesh is
within ``--tol`` of the target, then builds the tet mesh once and writes ``tissue_s{seed}_ecs{NN}.vtu``
(see ``geometry_file``) to ``data/cells`` or ``data/tubes``. The cell set and numbering depend only on the
seed, not on w, so all ratios of one seed have the same cells. Simulations identify the geometry by (mesh_dir, seed, ecs_ratio).

The mesh also stores each cell's raw Voronoi faces (``ref_faces``, rows of 9 coordinates, and their cell
``ref_cell``), which do not depend on w; the simulation places synapses on them.

    uv run --extra geometry scripts/ongoing_work/tissue_kinetics/make_geometry.py --ecs-ratio 0.1 0.17 0.25 --seed 0 1
    uv run --extra geometry scripts/ongoing_work/tissue_kinetics/make_geometry.py --cells tube --ecs-ratio 0.1
"""
import argparse
import logging
import os
import time

import numpy as np


DEFAULTS = {"round": dict(d=4.0, r=0.5, h=0.3, w_bracket=[0.03, 0.5]),
            "tube": dict(d=2.0, r=0.25, h=0.25, w_bracket=[0.03, 0.3])}


def ecs_percent(ecs_ratio):
    """The ECS ratio in percent; a ValueError unless it is a whole percentage."""
    percent = round(100 * ecs_ratio)
    if abs(100 * ecs_ratio - percent) > 1e-6:
        raise ValueError(f"ecs_ratio must be a whole percentage, got {ecs_ratio}")
    return percent


def geometry_file(mesh_dir, seed, ecs_ratio):
    """Path of the mesh for this geometry seed and target ECS ratio (a whole percentage)."""
    return os.path.join(mesh_dir, f"tissue_s{seed}_ecs{ecs_percent(ecs_ratio):02d}.vtu")


def radius_ratio(X):
    """3 r_in / r_circ of the tets X (M, 4, 3): 1 for the regular tet, 0 for a flat one."""
    a, b, c = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]
    V = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6
    S = sum(np.linalg.norm(np.cross(X[:, j] - X[:, i], X[:, k] - X[:, i]), axis=1) / 2
            for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    num = np.linalg.norm((a * a).sum(1)[:, None] * np.cross(b, c) + (b * b).sum(1)[:, None] * np.cross(c, a)
                         + (c * c).sum(1)[:, None] * np.cross(a, b), axis=1)
    return 3 * (3 * V / S) / (num / np.maximum(12 * V, 1e-300))


def make_geometry(seed, ecs_ratio, args):
    import ngsolve as ngs
    from bmbcsim.geometry.rounded_voronoi import make_rounded_voronoi_cells
    from bmbcsim.geometry.tube_cells import make_tube_cells
    from bmbcsim.logging import logger
    from bmbcsim.meshing import tissue_mesher as tm

    box = np.array(args.box, float)
    make = make_tube_cells if args.cells == "tube" else make_rounded_voronoi_cells
    lo, hi = args.w_bracket
    for it in range(args.max_iter):
        w = (lo + hi) / 2
        cs = make(w, args.r, box, d=args.d, seed=seed)
        G, T, tags = tm.surface_mesh(cs.cells, box, args.h, workers=args.workers)
        alpha = tm.surface_ecs_fraction(G, T, tags, box)
        logger.info(f"seed {seed}, target {ecs_ratio}: w = {w:.5f} -> ECS fraction {alpha:.4f}")
        if abs(alpha - ecs_ratio) <= args.tol:
            break
        lo, hi = (w, hi) if alpha < ecs_ratio else (lo, w)
    else:
        raise RuntimeError(f"seed {seed}: no w in {args.w_bracket} gives ECS fraction {ecs_ratio} +- {args.tol} "
                           f"after {args.max_iter} bisection steps (last: w = {w:.5f}, {alpha:.4f})")

    mesh = tm.volume_mesh(G, T, tags, len(cs.cells))
    alpha_mesh = ngs.Integrate(1, mesh, definedon=mesh.Materials("ecs")) / box.prod()
    if abs(alpha_mesh - ecs_ratio) > args.tol:      # only a merged sealed ECS pocket changes it
        logger.warning(f"seed {seed}: mesh ECS fraction {alpha_mesh:.4f} misses the target {ecs_ratio}")
    el = mesh.ngmesh.Elements3D().NumPy()["nodes"][:, :4].astype(np.int64) - 1
    q = radius_ratio(np.array(mesh.ngmesh.Coordinates())[el])
    logger.info(f"seed {seed}, target {ecs_ratio}: {len(cs.cells)} cells ({cs.stats['n_shifted']} shifted, "
                f"{cs.stats['n_bad']} below the wall margin), {mesh.ne} tets, radius ratio min {q.min():.1e}, "
                f"0.1 % quantile {np.quantile(q, 1e-3):.3f}, median {np.median(q):.3f}")

    path = geometry_file(args.out, seed, ecs_ratio)
    faces = cs.faces
    tm.save_tissue_mesh(path, mesh, ecs_ratio=ecs_ratio, ecs_fraction=alpha_mesh, w=w, seed=seed, box=box,
                        cell_diam=args.d, rounding=args.r, mesh_size=args.h, cell_labels=cs.labels,
                        ref_faces=np.concatenate(faces).reshape(-1, 9),
                        ref_cell=np.repeat(np.arange(len(faces)), [len(f) for f in faces]))
    logger.info(f"-> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecs-ratio", type=float, nargs="+", required=True, help="target ECS volume fraction(s)")
    p.add_argument("--seed", type=int, nargs="+", default=[0], help="geometry seed(s)")
    p.add_argument("--out", help="output directory; default data/cells (round), data/tubes (tube)")
    p.add_argument("--box", type=float, nargs=3, default=[14.0, 14.0, 7.0], help="box size (um)")
    p.add_argument("--cells", choices=DEFAULTS, default="round", help="cell shape")
    p.add_argument("--d", type=float, help="cell diameter (um); default 4 (round), 2 (tube)")
    p.add_argument("--r", type=float, help="nominal rounding radius of the cells (um); default 0.5 / 0.25")
    p.add_argument("--h", type=float, help="surface mesh size (um); default 0.3 / 0.25")
    p.add_argument("--w-bracket", type=float, nargs=2, help="bisection bracket for w (um); default [0.03, 0.5] / [0.03, 0.3]")
    p.add_argument("--tol", type=float, default=1e-3, help="tolerance on the ECS fraction")
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--workers", type=int, default=8, help="processes for the per-cell membranes")
    args = p.parse_args()
    for k, v in DEFAULTS[args.cells].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.out = args.out or f"data/{'tubes' if args.cells == 'tube' else 'cells'}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)
    for ecs_ratio in args.ecs_ratio:
        ecs_percent(ecs_ratio)    # fail on a bad ratio before any meshing
    for seed in args.seed:
        for ecs_ratio in args.ecs_ratio:
            t0 = time.time()
            make_geometry(seed, ecs_ratio, args)
            print(f"seed {seed}, ecs {ecs_ratio}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":   # the membrane workers are spawned and re-import this module
    main()
