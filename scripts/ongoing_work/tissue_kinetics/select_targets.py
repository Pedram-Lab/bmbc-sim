"""Pick the evaluation synapses and the buffered half for the dose-response study.

From the no-buffer reference run: the local ECS volume fraction around every
synapse (heat-method geodesic balls, see evaluate_synapse_distribution_spatial.py),
then

* 2 *restricted* synapses (lowest vfrac) and 2 *connected* ones (highest vfrac),
  at least ``--min-boundary`` um from every box face, pairwise at least
  ``--min-separation`` um apart, each alone in its cell (so the cell's Ca content
  is that synapse's uptake);
* the *buffered half*: half of all active synapses, drawn at random except that
  it contains exactly one restricted and one connected target.

Indices are into ``synapse_centers.npy`` of the run (= ``ecm.local_target``).
Writes ``targets.json`` next to the run and prints the YAML line for the sweep.

    uv run scripts/ongoing_work/tissue_kinetics/select_targets.py <reference run or sweep dir>
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np

from analysis import load_synapse_centers, synapse_cells
from bmbcsim.simulation.result_io import find_run_dirs
from evaluate_synapse_distribution_spatial import DEFAULT_HEAT_M, process_seed


def pick_extremes(order, centers, n, min_sep):
    """First `n` indices along `order` that are pairwise at least `min_sep` apart."""
    picked = []
    for i in order:
        if all(np.linalg.norm(centers[i] - centers[j]) >= min_sep for j in picked):
            picked.append(int(i))
        if len(picked) == n:
            return picked
    raise SystemExit(f"could not find {n} synapses {min_sep} um apart")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="reference run directory (or a sweep dir holding exactly one run)")
    p.add_argument("--radius", type=float, default=1.0, help="geodesic radius for vfrac (um)")
    p.add_argument("--min-boundary", type=float, default=2.0, help="min distance to the box faces (um)")
    p.add_argument("--min-separation", type=float, default=5.0, help="min target separation (um)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the buffered half")
    args = p.parse_args()

    runs = find_run_dirs(args.path)
    if len(runs) != 1:
        raise SystemExit(f"expected exactly one run beneath {args.path}, found {len(runs)}")
    run = str(runs[0])

    rows = process_seed(run, [args.radius], DEFAULT_HEAT_M)
    idx = np.array([r[0] for r in rows])
    d_boundary = np.array([r[4] for r in rows])
    vfrac = np.array([r[7] / r[8] for r in rows])  # v_local / v_sphere_box
    with h5py.File(os.path.join(run, "snapshot.h5"), "r") as h5:
        centers = load_synapse_centers(run, h5)
        cells, n_in_cell = synapse_cells(h5, centers)
        box_min, box_max = h5["mesh/points"][:].min(axis=0), h5["mesh/points"][:].max(axis=0)
    assert np.array_equal(idx, np.arange(len(centers)))
    # Straight-line distance to the nearest box face (the Robin boundary), in all
    # three directions: the box is no longer a thin slab (named d_lateral in the outputs).
    d_lateral = np.minimum(centers - box_min, box_max - centers).min(axis=1)

    with open(os.path.join(run, "spatial_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "seed", "synapse_idx", "x", "y", "z", "d_boundary", "d_neighbor",
                    "min_ca", f"v_local_r{args.radius:g}", f"v_sphere_box_r{args.radius:g}",
                    "d_lateral", "cell", "n_in_cell"])
        for r, d, c, n in zip(rows, d_lateral, cells, n_in_cell):
            w.writerow([".", 0, *r, d, c, n])

    eligible = np.where((d_lateral >= args.min_boundary) & (n_in_cell == 1))[0]
    if len(eligible) < 4:
        raise SystemExit(f"only {len(eligible)} eligible synapses")
    by_vfrac = eligible[np.argsort(vfrac[eligible])]
    restricted = pick_extremes(by_vfrac, centers, 2, args.min_separation)
    connected = pick_extremes(
        [i for i in by_vfrac[::-1] if all(np.linalg.norm(centers[i] - centers[j]) >= args.min_separation
                                           for j in restricted)],
        centers, 2, args.min_separation)

    rng = np.random.default_rng(args.seed)
    targets = restricted + connected
    others = [i for i in range(len(centers)) if i not in targets]
    n_half = len(centers) // 2
    buffered = sorted([restricted[0], connected[0]]
                      + [int(i) for i in rng.choice(others, n_half - 2, replace=False)])

    out = {
        "reference_run": run,
        "radius_um": args.radius,
        "restricted": restricted,
        "connected": connected,
        "buffered_half": buffered,
        "vfrac": {int(i): float(vfrac[i]) for i in targets},
        "d_lateral": {int(i): float(d_lateral[i]) for i in targets},
        "cell": {int(i): int(cells[i]) for i in targets},
        "n_synapses": len(centers),
    }
    with open(os.path.join(run, "targets.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"{len(centers)} active synapses, {len(eligible)} eligible; vfrac(r={args.radius}) "
          f"range {vfrac.min():.3f}-{vfrac.max():.3f}, median {np.median(vfrac):.3f}")
    for label, group in (("restricted", restricted), ("connected", connected)):
        for i in group:
            print(f"  {label:10s} {i:3d}: vfrac {vfrac[i]:.3f}, d_lateral {d_lateral[i]:.2f} um, "
                  f"cell_{cells[i]}, buffered={i in buffered}")
    print(f"buffered half ({len(buffered)}): {buffered}")
    print(f"\nlocal_target: {buffered}")
    print(f"-> {run}/targets.json")


if __name__ == "__main__":
    main()
