"""Export per-synapse heat-method distance fields as VTK files.

For a single seed result, recompute the geodesic distance phi from each of the
first N synapse anchors (heat-method setup identical to
``evaluate_synapse_distribution_spatial.py``) and write each field to
``<seed_dir>/phi_synapse_<NNN>.vtk`` alongside the raw simulation snapshot.
"""

import argparse
import os

import h5py
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from evaluate_ecs_ratio import find_synapse_centers
from evaluate_synapse_distribution_spatial import (
    DEFAULT_HEAT_M,
    assemble_heat_method,
    build_ecs_ngs_mesh,
    build_vertex_dof_map,
    heat_method_phi,
    mean_edge_length,
)


def export_distance_maps(seed_dir, n_maps, heat_m=DEFAULT_HEAT_M):
    h5_path = os.path.join(seed_dir, "snapshot.h5")
    with h5py.File(h5_path, "r") as h5:
        points = h5["mesh/points"][:].astype(np.float64)
        connectivity = h5["mesh/connectivity"][:].astype(np.int64)
        ecs_indicator = h5["compartments/ecs"][:]
        synapse_centers = find_synapse_centers(h5)

    if len(synapse_centers) == 0:
        print(f"No synapses found in {seed_dir}")
        return

    n_export = min(n_maps, len(synapse_centers))
    print(
        f"Exporting {n_export} of {len(synapse_centers)} distance maps from {seed_dir}"
    )

    mesh, _, ecs_tets_local, ecs_points = build_ecs_ngs_mesh(
        points, connectivity, ecs_indicator
    )
    n_ecs = len(ecs_points)

    h_bar = mean_edge_length(ecs_tets_local, ecs_points)
    t_heat = heat_m * h_bar * h_bar

    kd = cKDTree(ecs_points)
    _, anchor_local = kd.query(synapse_centers[:n_export])

    pinned_dof = 0
    V, heat_inv, L_inv = assemble_heat_method(mesh, t_heat, pinned_dof)
    dof_for_vertex = build_vertex_dof_map(V, n_ecs)

    # Build the ECS UnstructuredGrid once; we just swap point_data per synapse.
    cells = np.column_stack(
        [np.full(len(ecs_tets_local), 4, dtype=np.int64), ecs_tets_local]
    ).ravel()
    celltypes = np.full(len(ecs_tets_local), pv.CellType.TETRA, dtype=np.uint8)

    for i, anchor in enumerate(anchor_local):
        anchor_dof = int(dof_for_vertex[anchor])
        phi_np = heat_method_phi(
            V, heat_inv, L_inv, anchor_dof, pinned_dof, dof_for_vertex
        )

        grid = pv.UnstructuredGrid(cells, celltypes, ecs_points)
        grid.point_data["phi"] = phi_np
        is_anchor = np.zeros(n_ecs, dtype=np.uint8)
        is_anchor[anchor] = 1
        grid.point_data["is_anchor"] = is_anchor

        out_path = os.path.join(seed_dir, f"phi_synapse_{i:03d}.vtk")
        grid.save(out_path)
        x, y, z = synapse_centers[i]
        print(f"  synapse {i:3d} @ ({x:.2f}, {y:.2f}, {z:.2f})  ->  {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        help="Seed result directory containing snapshot.h5 (e.g. "
             "results/synapse_distribution_ecs_25_*/tissue_kinetics_seed0_*)",
    )
    p.add_argument(
        "--n", type=int, default=5,
        help="Number of synapse distance maps to export (default: 5)",
    )
    p.add_argument(
        "--heat-m", type=float, default=DEFAULT_HEAT_M,
        help=f"Multiplier on mean-edge-length^2 for the heat time step "
             f"(default: {DEFAULT_HEAT_M})",
    )
    args = p.parse_args()

    if not os.path.isdir(args.path):
        raise SystemExit(f"Not a directory: {args.path}")
    if not os.path.isfile(os.path.join(args.path, "snapshot.h5")):
        raise SystemExit(f"No snapshot.h5 in: {args.path}")
    if args.n <= 0:
        raise SystemExit("--n must be positive")

    export_distance_maps(args.path, args.n, heat_m=args.heat_m)


if __name__ == "__main__":
    main()
