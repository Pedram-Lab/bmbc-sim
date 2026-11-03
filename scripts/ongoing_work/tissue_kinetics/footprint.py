"""Depletion footprint of the active synapse(s) in a solitary-synapse reference run.

Ring-averaged ECS Ca deficit (ca_ecs - Ca) around each active synapse at the time its
own Ca is lowest, the radius where the deficit has fallen to 1/e of its value in the
innermost ring (the catchment, used as the buffer width of single_buffer_sweep) and
the far-field deficit (pool effect).

    uv run scripts/ongoing_work/tissue_kinetics/footprint.py <run dir> [<run dir> ...]
        [--rings 0.5,1,1.5,2,3,4,6] [--out footprint.csv]
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from analysis import load_synapse_centers

RINGS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def footprint(run, rings=RINGS, half_width=0.25):
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    ca0 = float(str(cfg["diffusion"]["ca_ecs"]).split()[0])
    active = cfg["synapse"]["active"]
    with h5py.File(os.path.join(run, "snapshot.h5"), "r") as h5:
        centers = load_synapse_centers(run, h5)
        points = h5["mesh/points"][:]
        ecs = np.where(h5["compartments/ecs"][:] > 0.5)[0]
        keys = sorted(h5["data/Ca"])
        times = h5["data/time"][:]
        tree = cKDTree(points[ecs])
        rows = []
        for s in (range(len(centers)) if active is None else active):
            node = ecs[tree.query(centers[s])[1]]
            series = np.array([h5[f"data/Ca/{k}"][node] for k in keys])
            k_min = int(series.argmin())
            deficit = ca0 - h5[f"data/Ca/{keys[k_min]}"][:][ecs]
            d = np.linalg.norm(points[ecs] - centers[s], axis=1)
            prof = np.array([deficit[(d >= r - half_width) & (d < r + half_width)].mean() for r in rings])
            far = prof[-1]
            local = prof - far  # synapse-specific part on top of the pool deficit
            target = local[0] / np.e
            below = np.where(local <= target)[0]
            sigma = np.interp(target, local[:below[0] + 1][::-1], rings[:below[0] + 1][::-1]) \
                if len(below) and below[0] > 0 else np.nan
            rows.append(dict(run=run, synapse=s, t_min_ms=float(times[k_min]), peak_deficit=ca0 - series.min(),
                             far_deficit=far, sigma_1e=sigma,
                             **{f"deficit_r{r:g}": v for r, v in zip(rings, prof)}))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--rings", default=",".join(map(str, RINGS)))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rings = [float(r) for r in args.rings.split(",")]
    df = pd.DataFrame([r for run in args.runs for r in footprint(run, rings)])
    pd.set_option("display.width", 200)
    print(df.drop(columns="run").round(3).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
