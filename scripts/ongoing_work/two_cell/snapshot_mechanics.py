"""Per-snapshot mechanics summary of a bmbcsim run: min J, ECS volume ratio, ECM_Ca range.

Usage: uv run scripts/ongoing_work/two_cell/snapshot_mechanics.py <result_dir>
"""
import sys

import numpy as np

from bmbcsim.simulation.result_io.result_loader import ResultLoader


def tet_volumes(points, cells):
    p = points[cells]
    return np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                            np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6


def main(path):
    loader = ResultLoader(path)
    ecs = np.array([not n.startswith("cell") for n in loader.regions])[loader.cell_to_region]
    print(f"regions: {loader.regions}; ECS elements: {ecs.sum()} of {ecs.size}")
    print("step  t[ms]   J_min   J_min_ecs  ecs_vol  ecmca_min  ecmca_max")
    vol0 = None
    for step in range(len(loader)):
        g = loader.load_snapshot(step)
        cells = g.cell_connectivity.reshape(-1, 4)
        if vol0 is None:
            vol0 = tet_volumes(g.points, cells)
        v = tet_volumes(g.points + g.point_data["deformation"], cells)
        J = v / vol0
        t = loader.snapshots[step][0].to("ms").value
        c = g.point_data["ECM_Ca"][np.unique(cells[ecs])] if "ECM_Ca" in g.point_data else [np.nan]
        print(f"{step:4d} {t:7.2f}  {J.min():.4f}  {J[ecs].min():.4f}   "
              f"{v[ecs].sum() / vol0[ecs].sum():.4f}  {np.min(c):.4f}  {np.max(c):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
