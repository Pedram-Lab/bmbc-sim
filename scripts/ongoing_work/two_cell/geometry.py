"""Two-cell test geometry: tissue cells 757 (clipped at half) and 758 in a 1 um slab.

The pair was chosen as the smallest interior side-by-side contact pair in
data/tissue_geometry.vtk (search in results/two_cell_candidates). The box lives in
a local frame whose x axis is the pair axis (758 -> 757); the constants below were
measured once at ecs_ratio 0.1 (slab through the cell centroids, 0.3 um ECS margin,
cell 757 cut at its in-plane mid-plane; gap 0.18 um) and are kept fixed so the geometry does not depend
on the ECS ratio of a run.
"""
import numpy as np
import ngsolve as ngs

from bmbcsim.geometry import TissueGeometry

CELLS = (757, 758)                     # 757 = clipped (red), 758 = complete (blue)
ROT_DEG, Z0 = -63.5, 16.24             # world -> local rotation about z, slab centre (cell centroids)
BOX_MIN = np.array([2.39, 13.98, -0.5])
BOX_MAX = np.array([6.59, 17.98, 0.5])
TARGET_CELL_DIAM = 4.0


def two_cell_geometry(ecs_ratio=0.1) -> TissueGeometry:
    """Preprocess the tissue exactly as tissue_kinetics does, then keep the pair in the local frame."""
    g = TissueGeometry.from_file("data/tissue_geometry.vtk")
    median_diam = float(np.median([np.subtract(c.bounds[1::2], c.bounds[::2]).max() for c in g.cells]))
    g = g.scale(TARGET_CELL_DIAM / median_diam).decimate(0.5).smooth(10).decimate(0.5)
    minc, _ = g.bounding_box()
    for c in g.cells:
        c.points -= minc
    g = g.shrink_cells(1 - ecs_ratio, jitter=0.0)
    a = np.radians(ROT_DEG)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    cells = []
    for i in CELLS:
        c = g.cells[i].copy()
        c.points[:, :2] = c.points[:, :2] @ rot.T
        c.points[:, 2] -= Z0
        cells.append(c)
    return TissueGeometry(cells)


def two_cell_mesh(ecs_ratio=0.1, n_refine=0, mesh_size=5.0) -> ngs.Mesh:
    """Mesh of the two-cell box; materials cell_0 (757, clipped), cell_1 (758), ecs; bnd membrane_0/1."""
    g = two_cell_geometry(ecs_ratio)
    mesh = g.to_ngs_mesh(mesh_size=mesh_size, min_coords=BOX_MIN, max_coords=BOX_MAX, projection_tol=0.02,
                         cell_names=["cell_0", "cell_1"], cell_bnd_names=["membrane_0", "membrane_1"])
    ecs = [m for m in mesh.GetMaterials() if m.startswith("ecs")]
    if ecs != ["ecs"]:
        raise RuntimeError(f"ECS is not one region: {ecs}")
    for _ in range(n_refine):
        mesh.Refine()
    return mesh


if __name__ == "__main__":
    import sys
    for n in range(int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 3):
        m = two_cell_mesh(n_refine=n)
        vol = {mat: ngs.Integrate(ngs.CoefficientFunction(1), m, definedon=m.Materials(mat)) for mat in m.GetMaterials()}
        print(f"refine {n}: ne={m.ne} nv={m.nv} " + " ".join(f"{k}={v:.3f}" for k, v in vol.items()), flush=True)
