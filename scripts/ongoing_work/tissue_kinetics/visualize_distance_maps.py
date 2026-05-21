"""Export per-synapse heat-method distance fields as a single VTK file.

For a single seed result, recompute the geodesic distance phi from each of the
first N synapse anchors (heat-method setup identical to
``evaluate_synapse_distribution_spatial.py``) and write all fields, together
with the per-node minimum Ca concentration over the recorded time series, to
``<seed_dir>/distance_maps.vtk`` alongside the raw simulation snapshot.
"""

import argparse
import os

import h5py
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from evaluate_ecs_ratio import SPECIES_NAME, find_synapse_centers
from evaluate_synapse_distribution_spatial import (
    DEFAULT_HEAT_M,
    assemble_heat_method,
    build_ecs_ngs_mesh,
    build_vertex_dof_map,
    heat_method_phi,
    mean_edge_length,
)


def export_distance_maps(seed_dir, n_maps=None, heat_m=DEFAULT_HEAT_M):
    h5_path = os.path.join(seed_dir, "snapshot.h5")
    with h5py.File(h5_path, "r") as h5:
        points = h5["mesh/points"][:].astype(np.float64)
        connectivity = h5["mesh/connectivity"][:].astype(np.int64)
        ecs_indicator = h5["compartments/ecs"][:]
        synapse_centers = find_synapse_centers(h5)

        # Per-node minimum Ca over the recorded time series (full-mesh field).
        step_keys = sorted(h5[f"data/{SPECIES_NAME}"].keys())
        min_ca_full = np.full(len(points), np.inf, dtype=np.float32)
        for step_key in step_keys:
            ca_data = h5[f"data/{SPECIES_NAME}/{step_key}"][:]
            np.minimum(min_ca_full, ca_data, out=min_ca_full)

    if len(synapse_centers) == 0:
        print(f"No synapses found in {seed_dir}")
        return

    n_export = len(synapse_centers) if n_maps is None else min(n_maps, len(synapse_centers))
    print(
        f"Exporting {n_export} of {len(synapse_centers)} distance maps from {seed_dir}"
    )

    mesh, ecs_vertex_idx, ecs_tets_local, ecs_points = build_ecs_ngs_mesh(
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

    cells = np.column_stack(
        [np.full(len(ecs_tets_local), 4, dtype=np.int64), ecs_tets_local]
    ).ravel()
    celltypes = np.full(len(ecs_tets_local), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, ecs_points)

    # Min-Ca restricted to ECS vertices, in the local ECS ordering used by grid.
    grid.point_data["min_ca"] = min_ca_full[ecs_vertex_idx]

    # Single int field marking which synapse each anchor vertex belongs to
    # (-1 elsewhere) — lets ParaView show all anchors in one go.
    anchor_id = np.full(n_ecs, -1, dtype=np.int32)

    for i, anchor in enumerate(anchor_local):
        anchor_dof = int(dof_for_vertex[anchor])
        phi_np = heat_method_phi(
            V, heat_inv, L_inv, anchor_dof, pinned_dof, dof_for_vertex
        )
        grid.point_data[f"phi_synapse_{i:03d}"] = phi_np
        anchor_id[anchor] = i
        x, y, z = synapse_centers[i]
        print(f"  synapse {i:3d} @ ({x:.2f}, {y:.2f}, {z:.2f})")

    grid.point_data["anchor_id"] = anchor_id

    out_path = os.path.join(seed_dir, "distance_maps.vtk")
    grid.save(out_path)
    print(f"  -> {out_path}")

    # Separate point cloud of the true (un-snapped) synapse centers.
    synapse_cloud = pv.PolyData(np.asarray(synapse_centers[:n_export], dtype=np.float64))
    synapse_cloud.point_data["synapse_index"] = np.arange(n_export, dtype=np.int32)
    cloud_path = os.path.join(seed_dir, "synapses.vtk")
    synapse_cloud.save(cloud_path)
    print(f"  -> {cloud_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        help="Seed result directory containing snapshot.h5 (e.g. "
             "results/synapse_distribution_ecs_25_*/tissue_kinetics_seed0_*)",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Number of synapse distance maps to export (default: all)",
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
    if args.n is not None and args.n <= 0:
        raise SystemExit("--n must be positive")

    export_distance_maps(args.path, args.n, heat_m=args.heat_m)


if __name__ == "__main__":
    main()
