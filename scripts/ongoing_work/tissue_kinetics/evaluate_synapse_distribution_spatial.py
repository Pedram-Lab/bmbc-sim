"""Compute per-synapse spatial metrics via the heat-method geodesic distance.

For each seed result in a sweep directory, the script:
  1. Loads the tet mesh and restricts it to the ECS subdomain.
  2. Builds an NGSolve P1 H1 space on that subdomain.
  3. For every synapse: solves the heat step and Poisson step ("The heat method
     for distance computation", Crane et al. 2017) to obtain a per-vertex geodesic
     distance field phi.
  4. Reads off, per synapse:
       * d_boundary  - min(phi) over ECS vertices on the outer simulation box
       * d_neighbor  - min over other synapses of phi at their anchor vertex
       * v_local[r]  - volume of {phi < r} via exact marching tetrahedra on the
                       piecewise-linear interpolant, one column per radius r

Output: a single pooled CSV at <sweep_dir>/spatial_metrics.csv.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import ngsolve as ngs
from dask.distributed import Client, as_completed
from scipy.integrate import quad
from scipy.spatial import cKDTree

from bmbcsim.meshing.netgen_vtk import pyvista_volume_to_netgen
from bmbcsim.utils import create_cluster
from evaluate_ecs_ratio import compute_local_ca, find_synapse_centers

RESULTS_ROOT = "results"
DEFAULT_RADII = (0.25, 0.5, 1.0, 2.0)  # um
# Note: Heat time step t = DEFAULT_HEAT_M * h_bar^2. m~30 propagates the heat
# globally while still resolving cell obstacles; tuned empirically on the
# synapse_distribution_ecs_25 sweep (median heat-method residual ~0.14 um, max
# ~0.33 um).


# ---------------------------------------------------------------------------
# Sweep / seed discovery
# ---------------------------------------------------------------------------

_SWEEP_PATTERN = re.compile(
    r"synapse_distribution(?:_ecs_\d+)?_\d{4}-\d{2}-\d{2}-\d{6}$"
)
_SEED_PATTERN = re.compile(r"tissue_kinetics_seed(\d+)_\d{4}-\d{2}-\d{2}-\d{6}$")


def find_latest_sweep_dir(results_root):
    dirs = [
        d for d in os.listdir(results_root)
        if _SWEEP_PATTERN.match(d)
        and os.path.isdir(os.path.join(results_root, d))
    ]
    if not dirs:
        raise RuntimeError("No synapse_distribution* directories found")
    return os.path.join(results_root, sorted(dirs)[-1])


def find_seed_dirs(sweep_dir):
    out = []
    for d in sorted(os.listdir(sweep_dir)):
        m = _SEED_PATTERN.match(d)
        if m and os.path.isdir(os.path.join(sweep_dir, d)):
            out.append((int(m.group(1)), os.path.join(sweep_dir, d)))
    return out


# ---------------------------------------------------------------------------
# ECS submesh -> NGSolve
# ---------------------------------------------------------------------------

def build_ecs_ngs_mesh(points, connectivity, ecs_indicator):
    """Restrict the tet mesh to the ECS subdomain and build an ngs.Mesh.

    Returns (mesh, ecs_vertex_idx, ecs_tets_local, ecs_points) where
    ecs_vertex_idx maps local ECS index -> global mesh-vertex index, and
    ecs_tets_local / ecs_points use the local (contiguous) ECS indexing.
    """
    ecs_vertex_mask = ecs_indicator > 0.5
    ecs_vertex_idx = np.where(ecs_vertex_mask)[0]
    remap = -np.ones(len(ecs_indicator), dtype=np.int64)
    remap[ecs_vertex_idx] = np.arange(len(ecs_vertex_idx))

    ecs_tet_mask = ecs_vertex_mask[connectivity].all(axis=1)
    ecs_tets_local = remap[connectivity[ecs_tet_mask]].astype(np.int64)
    ecs_points = points[ecs_vertex_idx].astype(np.float64)

    cells = np.column_stack(
        [np.full(len(ecs_tets_local), 4, dtype=np.int64), ecs_tets_local]
    ).ravel()
    celltypes = np.full(len(ecs_tets_local), pv.CellType.TETRA, dtype=np.uint8)
    pv_grid = pv.UnstructuredGrid(cells, celltypes, ecs_points)

    ng_mesh = pyvista_volume_to_netgen(pv_grid)
    mesh = ngs.Mesh(ng_mesh)
    return mesh, ecs_vertex_idx, ecs_tets_local, ecs_points


def build_vertex_dof_map(V, n_vertices):
    """Return an array `dof_for_vertex` of length n_vertices.

    For an order-1 H1 space NGSolve numbers vertex DOFs first and in vertex
    order, so this is almost always the identity; the explicit map is cheap
    insurance against future NGSolve renumberings.
    """
    out = np.empty(n_vertices, dtype=np.int64)
    for k in range(n_vertices):
        dofs = V.GetDofNrs(ngs.NodeId(ngs.VERTEX, k))
        out[k] = dofs[0]
    return out


def outer_box_boundary_vertices(ecs_points, mesh_box_min, mesh_box_max, eps):
    """Indices (into ecs_points) of vertices on any of the 6 axis-aligned box faces."""
    on_face = np.zeros(len(ecs_points), dtype=bool)
    for axis in range(3):
        on_face |= np.isclose(ecs_points[:, axis], mesh_box_min[axis], atol=eps)
        on_face |= np.isclose(ecs_points[:, axis], mesh_box_max[axis], atol=eps)
    return np.where(on_face)[0]


def mean_edge_length(ecs_tets_local, ecs_points):
    edges = np.concatenate([
        ecs_tets_local[:, [0, 1]], ecs_tets_local[:, [0, 2]],
        ecs_tets_local[:, [0, 3]], ecs_tets_local[:, [1, 2]],
        ecs_tets_local[:, [1, 3]], ecs_tets_local[:, [2, 3]],
    ], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    diffs = ecs_points[edges[:, 0]] - ecs_points[edges[:, 1]]
    return float(np.linalg.norm(diffs, axis=1).mean())


# ---------------------------------------------------------------------------
# Heat method
# ---------------------------------------------------------------------------

def assemble_heat_method(mesh, t_heat, pinned_dof):
    """Assemble M, L, the heat operator (M + t*L), and the pinned-L inverse.

    The Poisson step uses one globally-pinned DOF to remove the constant
    null-space of L; each synapse's phi is later shifted so its minimum is 0,
    which restores phi(anchor) ~= 0 regardless of which DOF was pinned.
    """
    V = ngs.H1(mesh, order=1)
    u, v = V.TnT()

    heat_form = ngs.BilinearForm(V)
    heat_form += (u * v + t_heat * ngs.grad(u) * ngs.grad(v)) * ngs.dx
    heat_form.Assemble()

    L_form = ngs.BilinearForm(V)
    L_form += ngs.grad(u) * ngs.grad(v) * ngs.dx
    L_form.Assemble()

    free_poisson = ngs.BitArray(V.FreeDofs())
    free_poisson.Clear(pinned_dof)

    heat_inv = heat_form.mat.Inverse(
        freedofs=V.FreeDofs(), inverse="sparsecholesky"
    )
    L_inv = L_form.mat.Inverse(
        freedofs=free_poisson, inverse="sparsecholesky"
    )
    return V, heat_inv, L_inv


def heat_method_phi(V, heat_inv, L_inv, anchor_dof, pinned_dof, dof_for_vertex):
    """Run one heat-method step from a Dirac source at `anchor_dof`.

    Returns phi_np of shape (n_ecs,) in local ECS vertex ordering, shifted so
    that phi_np.min() == 0.
    """
    rhs = ngs.GridFunction(V)
    rhs.vec[:] = 0.0
    rhs.vec[anchor_dof] = 1.0

    u_gf = ngs.GridFunction(V)
    u_gf.vec.data = heat_inv * rhs.vec

    grad_u = ngs.grad(u_gf)
    norm_grad = ngs.sqrt(grad_u * grad_u + 1e-30)
    X = -grad_u / norm_grad

    _, v_test = V.TnT()
    b_form = ngs.LinearForm(V)
    b_form += ngs.InnerProduct(ngs.grad(v_test), X) * ngs.dx
    b_form.Assemble()
    b_form.vec[pinned_dof] = 0.0  # consistent with the homogeneous Dirichlet pin

    phi_gf = ngs.GridFunction(V)
    phi_gf.vec.data = L_inv * b_form.vec

    phi_np = np.asarray(phi_gf.vec.FV().NumPy())[dof_for_vertex].copy()
    phi_np -= phi_np.min()
    return phi_np


# ---------------------------------------------------------------------------
# Marching tetrahedra: volume of {phi < r}
# ---------------------------------------------------------------------------

def _tet_volumes(P0, P1, P2, P3):
    return np.abs(np.einsum(
        "ti,ti->t", P1 - P0, np.cross(P2 - P0, P3 - P0)
    )) / 6.0


def below_volume(ecs_tets_local, ecs_points, phi, r):
    """Sum of vol({phi < r} intersect tet) over all ECS tets (piecewise linear)."""
    P = ecs_points[ecs_tets_local]  # (T, 4, 3)
    D = phi[ecs_tets_local]         # (T, 4)

    order = np.argsort(D, axis=1)
    D_sorted = np.take_along_axis(D, order, axis=1)
    P_sorted = np.take_along_axis(P, order[..., None], axis=1)

    d0, d1, d2, d3 = (D_sorted[:, k] for k in range(4))
    P0, P1, P2, P3 = (P_sorted[:, k] for k in range(4))

    V_tet = _tet_volumes(P0, P1, P2, P3)
    out = np.zeros(len(ecs_tets_local), dtype=np.float64)

    def safe_t(num, denom):
        return np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)

    # Case 1: only vertex 0 below
    case1 = (r > d0) & (r <= d1)
    if case1.any():
        t01 = safe_t(r - d0, d1 - d0)
        t02 = safe_t(r - d0, d2 - d0)
        t03 = safe_t(r - d0, d3 - d0)
        out[case1] = (V_tet * t01 * t02 * t03)[case1]

    # Case 2: vertices 0, 1 below (prism of 6 vertices)
    case2 = (r > d1) & (r <= d2)
    if case2.any():
        t02 = safe_t(r - d0, d2 - d0)
        t03 = safe_t(r - d0, d3 - d0)
        t12 = safe_t(r - d1, d2 - d1)
        t13 = safe_t(r - d1, d3 - d1)
        A02 = P0 + t02[:, None] * (P2 - P0)
        A03 = P0 + t03[:, None] * (P3 - P0)
        A12 = P1 + t12[:, None] * (P2 - P1)
        A13 = P1 + t13[:, None] * (P3 - P1)
        # Prism decomposed into 3 tets:
        #   T1 {P0, A02, A03, P1}, T2 {A02, A03, P1, A12}, T3 {A03, P1, A12, A13}
        v1 = _tet_volumes(P0, A02, A03, P1)
        v2 = _tet_volumes(A02, A03, P1, A12)
        v3 = _tet_volumes(A03, P1, A12, A13)
        out[case2] = (v1 + v2 + v3)[case2]

    # Case 3: vertices 0, 1, 2 below (small tet near P3 above)
    case3 = (r > d2) & (r <= d3)
    if case3.any():
        s30 = safe_t(d3 - r, d3 - d0)
        s31 = safe_t(d3 - r, d3 - d1)
        s32 = safe_t(d3 - r, d3 - d2)
        out[case3] = (V_tet * (1.0 - s30 * s31 * s32))[case3]

    # Case 4: all below
    case4 = r > d3
    out[case4] = V_tet[case4]

    return float(out.sum())


# ---------------------------------------------------------------------------
# Sphere ∩ box volume (analytic reference volume for normalizing v_local)
# ---------------------------------------------------------------------------
# The local ECS volume is normalized by the volume of a Euclidean ball clipped
# to the simulation box, vol(ball(r) ∩ box), rather than a full sphere: the box
# is a thin slab, so every ball pokes out of the z-faces and the full-sphere
# volume is the wrong reference. This is a closed-form primitive (a 1D integral
# over the slab axis of the circle–rectangle cross-section), so it does not
# touch the mesh.

def _disk_cap(rho, d):
    """Area of the disk of radius rho lying beyond the line {coord > d}."""
    if d >= rho:
        return 0.0
    if d <= -rho:
        return math.pi * rho * rho
    return rho * rho * math.acos(d / rho) - d * math.sqrt(max(rho * rho - d * d, 0.0))


def _disk_corner(rho, a, b):
    """Area of {x > a, y > b} ∩ disk(rho), for a, b >= 0 (origin inside rect)."""
    if a * a + b * b >= rho * rho:
        return 0.0

    def prim(x):  # antiderivative of sqrt(rho^2 - x^2) - b
        return 0.5 * (x * math.sqrt(max(rho * rho - x * x, 0.0))
                      + rho * rho * math.asin(min(max(x / rho, -1.0), 1.0))) - b * x

    return prim(math.sqrt(rho * rho - b * b)) - prim(a)


def _disk_rect_area(rho, xl, xh, yl, yh):
    """Area of disk(rho) centered at origin intersected with [xl,xh]x[yl,yh].

    Assumes the origin is inside the rectangle (xl < 0 < xh, yl < 0 < yh), which
    holds because each ball is centered on an in-domain synapse. Inclusion-
    exclusion over the four sides; opposite sides never co-occur so only the four
    adjacent-side corners contribute.
    """
    if rho <= 0:
        return 0.0
    area = math.pi * rho * rho
    area -= _disk_cap(rho, xh) + _disk_cap(rho, -xl)
    area -= _disk_cap(rho, yh) + _disk_cap(rho, -yl)
    area += (_disk_corner(rho, xh, yh) + _disk_corner(rho, xh, -yl)
             + _disk_corner(rho, -xl, yh) + _disk_corner(rho, -xl, -yl))
    return area


def sphere_box_volume(center, box_min, box_max, r):
    """Volume of the ball of radius r at `center` intersected with the box.

    Integrates the circle–rectangle cross-section along z. Breakpoints (where the
    shrinking cross-section touches a box edge/corner) are handed to the
    quadrature so it stays accurate and warning-free at the slab kinks.
    """
    xl, yl, zl = np.asarray(box_min) - center
    xh, yh, zh = np.asarray(box_max) - center
    z_lo = max(zl, -r)
    z_hi = min(zh, r)
    if z_lo >= z_hi:
        return 0.0

    dists = [abs(xl), abs(xh), abs(yl), abs(yh),
             math.hypot(xl, yl), math.hypot(xl, yh),
             math.hypot(xh, yl), math.hypot(xh, yh)]
    breaks = sorted({
        z for d in dists if d < r
        for z in (math.sqrt(r * r - d * d), -math.sqrt(r * r - d * d))
        if z_lo < z < z_hi
    })

    def cross_section(z):
        rho = math.sqrt(max(r * r - z * z, 0.0))
        return _disk_rect_area(rho, xl, xh, yl, yh)

    val, _ = quad(cross_section, z_lo, z_hi, points=breaks or None, limit=200)
    return float(val)


# ---------------------------------------------------------------------------
# Per-seed processing
# ---------------------------------------------------------------------------

def process_seed(result_path, radii, heat_m, eps_rel=1e-6):
    h5_path = os.path.join(result_path, "snapshot.h5")
    with h5py.File(h5_path, "r") as h5:
        points = h5["mesh/points"][:].astype(np.float64)
        connectivity = h5["mesh/connectivity"][:].astype(np.int64)
        ecs_indicator = h5["compartments/ecs"][:]
        synapse_centers = find_synapse_centers(h5)

    if len(synapse_centers) == 0:
        return []

    # Per-synapse local Ca trace; ordering matches find_synapse_centers.
    _, local_ca = compute_local_ca(result_path)
    min_ca = local_ca.min(axis=0)  # (n_synapses,)

    mesh, _, ecs_tets_local, ecs_points = build_ecs_ngs_mesh(
        points, connectivity, ecs_indicator
    )
    n_ecs = len(ecs_points)

    box_min = points.min(axis=0)
    box_max = points.max(axis=0)
    eps = eps_rel * float((box_max - box_min).max())
    boundary_local = outer_box_boundary_vertices(
        ecs_points, box_min, box_max, eps
    )

    h_bar = mean_edge_length(ecs_tets_local, ecs_points)
    t_heat = heat_m * h_bar * h_bar

    # Anchor each synapse at its nearest ECS vertex
    kd = cKDTree(ecs_points)
    _, anchor_local = kd.query(synapse_centers)

    pinned_dof = 0  # arbitrary; shift-by-min removes the additive constant
    V, heat_inv, L_inv = assemble_heat_method(mesh, t_heat, pinned_dof)
    dof_for_vertex = build_vertex_dof_map(V, n_ecs)

    rows = []
    for i, anchor in enumerate(anchor_local):
        anchor_dof = int(dof_for_vertex[anchor])
        phi_np = heat_method_phi(
            V, heat_inv, L_inv, anchor_dof, pinned_dof, dof_for_vertex
        )

        d_boundary = float(phi_np[boundary_local].min()) if len(boundary_local) else np.inf

        if len(anchor_local) > 1:
            other_anchors = np.delete(anchor_local, i)
            d_neighbor = float(phi_np[other_anchors].min())
        else:
            d_neighbor = np.inf

        v_local = [below_volume(ecs_tets_local, ecs_points, phi_np, r) for r in radii]

        # Box-clipped reference volume: vol(Euclidean ball(r) intersect box),
        # the normalization denominator for v_local. Closed-form primitive.
        v_sphere_box = [
            sphere_box_volume(synapse_centers[i], box_min, box_max, r)
            for r in radii
        ]

        x, y, z = synapse_centers[i]
        rows.append([i, float(x), float(y), float(z),
                     d_boundary, d_neighbor, float(min_ca[i]), *v_local,
                     *v_sphere_box])

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_seed_remote(result_path, radii, heat_m):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_synapse_distribution_spatial import process_seed

    return process_seed(result_path, radii, heat_m)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path", nargs="?", default=None,
        help="Path to sweep directory (default: latest auto-detected)",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Number of seeds to process (default: all available)",
    )
    p.add_argument(
        "--n-workers", type=int, default=4,
        help=f"Number of Dask workers (default: 4)",
    )
    p.add_argument(
        "--radii", default=",".join(f"{r}" for r in DEFAULT_RADII),
        help="Comma-separated radii (um) for v_local columns "
             f"(default: {','.join(str(r) for r in DEFAULT_RADII)})",
    )
    p.add_argument(
        "--heat-m", type=float, default=30,
        help="Multiplier on mean-edge-length^2 for the heat time step "
             f"(default: 30)",
    )
    p.add_argument(
        "--out", default=None,
        help="Output CSV path (default: <sweep_dir>/spatial_metrics.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    radii = [float(s) for s in args.radii.split(",") if s.strip()]
    if not radii:
        raise ValueError("--radii must contain at least one value")

    sweep_dir = args.path or find_latest_sweep_dir(RESULTS_ROOT)
    seed_dirs = find_seed_dirs(sweep_dir)
    print(f"Found {len(seed_dirs)} seed results in {sweep_dir}")

    if args.n is not None:
        if not 0 < args.n <= len(seed_dirs):
            raise ValueError(
                f"--n must satisfy 0 < n <= {len(seed_dirs)}, got {args.n}"
            )
        seed_dirs = seed_dirs[: args.n]

    out_path = args.out or os.path.join(sweep_dir, "spatial_metrics.csv")
    header = ["seed", "synapse_idx", "x", "y", "z",
              "d_boundary", "d_neighbor", "min_ca"]
    header += [f"v_local_r{r:g}" for r in radii]
    header += [f"v_sphere_box_r{r:g}" for r in radii]

    t_total = time.time()
    n_workers = min(args.n_workers, len(seed_dirs))
    with create_cluster("local", n_workers=n_workers) as cluster, \
         Client(cluster) as client, \
         open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        futures = {
            client.submit(_process_seed_remote, result_path, radii, args.heat_m): seed_idx
            for seed_idx, result_path in seed_dirs
        }
        for future in as_completed(futures):
            seed_idx = futures[future]
            try:
                rows = future.result()
            except Exception as e:
                print(f"  seed {seed_idx}: skipped ({type(e).__name__}: {e})")
                continue
            for r in rows:
                writer.writerow([seed_idx, *r])
            f.flush()
            print(f"  seed {seed_idx}: {len(rows)} synapses")

    print(f"Wrote {out_path} in {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
