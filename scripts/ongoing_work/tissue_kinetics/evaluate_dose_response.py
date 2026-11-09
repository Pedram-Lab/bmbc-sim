"""Per-synapse depletion metrics for the buffer dose-response study -> one tidy CSV.

One row per (run, synapse) across all the sweep directories given, with the run's
parameters (dose, kinetics, buffer placement, stimulus), the synapse's role
(evaluation target class from targets.json, whether a local buffer is centered
on it, distance to the nearest buffered synapse) and the metrics:

* uptake_<k>   Ca taken up by the synapse's cell during spike k (10 ms window), amol
* uptake_cum   ... from the first pulse to the end of the run
* time_to_floor / frac_at_floor   synapse Ca <= depletion + eps, from the first pulse
* peak_deficit   ca_ecs - min(synapse Ca)
* recovery_half  time after the post-last-pulse minimum to recover half the deficit

Synapse Ca is the nearest ECS vertex (analysis.compute_local_ca); the cell content is
the lumped-mass integral over the cell, which is the cumulative uptake because the
cells have no efflux. Uptake metrics are NaN for synapses sharing a cell.

Effects are differences to the no-buffer reference with the same seed, stimulus and
active-synapse set (``effect_<metric>``); ``thr_<metric>`` is the IQR of the metric
across the reference synapses that carry flux, the significance threshold of the plan.
Solitary-synapse runs (``synapse.active``) add ``active`` (the index set), ``is_active``
and ``d_active`` (distance to the nearest active synapse); their buffer scenarios are
``on`` (buffer on the active synapse) and ``off`` (on a neighbour).

    uv run scripts/ongoing_work/tissue_kinetics/evaluate_dose_response.py \\
        --targets <ref run>/targets.json --out dose_response.csv results/dose-*
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import yaml

from analysis import compute_local_ca, load_synapse_centers, synapse_cells
from bmbcsim.simulation.result_io import find_run_dirs

FLOOR_EPS = 0.02  # mM above the depletion floor still counts as "at floor"
METRICS = ["uptake_cum", "time_to_floor", "frac_at_floor", "peak_deficit", "recovery_half"]


def _mm(q):
    """'0.3 mM' -> 0.3 (strip the unit of a config quantity string)."""
    return float(str(q).split()[0])


def cell_content(h5, cell_of_node, n_cells):
    """Ca content per cell (amol = mM um^3) at every record: lumped-mass integral."""
    points = h5["mesh/points"][:].astype(np.float64)
    conn = h5["mesh/connectivity"][:]
    p = points[conn]
    vol = np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                           np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6.0
    nodal_vol = np.bincount(conn.ravel(), weights=np.repeat(vol / 4.0, 4), minlength=len(points))
    in_cell = cell_of_node >= 0
    keys = sorted(h5["data/Ca"])
    content = np.empty((len(keys), n_cells))
    for t, k in enumerate(keys):
        ca = h5[f"data/Ca/{k}"][:]
        content[t] = np.bincount(cell_of_node[in_cell], weights=(nodal_vol * ca)[in_cell],
                                 minlength=n_cells)
    return content


def evaluate_run(run, targets):
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    ecm, syn, diff = cfg["ecm"], cfg["synapse"], cfg["diffusion"]
    pulses = np.array([_mm(t) for t in syn["pulse_times"]])
    ca0, floor = _mm(diff["ca_ecs"]), _mm(syn["depletion"]) + FLOOR_EPS
    local_width = None if ecm["local_width"] is None else _mm(ecm["local_width"])
    active = syn.get("active")  # None = every synapse carries flux
    if not ecm["enabled"]:
        scenario = "reference"
    elif local_width is None:
        scenario = "uniform"
    elif ecm["local_target"] is None:
        scenario = "all"
    elif active is None:
        scenario = "half"
    else:  # solitary-synapse runs: buffer on the active synapse or on a neighbour
        scenario = "on" if sorted(ecm["local_target"]) == sorted(active) else "off"

    times, local_ca = compute_local_ca(run)
    with h5py.File(os.path.join(run, "snapshot.h5"), "r") as h5:
        centers = load_synapse_centers(run, h5)
        cells, n_in_cell = synapse_cells(h5, centers)
        if active is not None:  # only synapses with flux count towards sharing a cell
            n_in_cell = np.array([np.sum(cells[active] == c) for c in cells])
        points = h5["mesh/points"][:]
        names = [n for n in h5["compartments"] if n.startswith("cell_")]
        cell_of_node = np.full(len(points), -1)
        for name in names:
            cell_of_node[h5[f"compartments/{name}"][:] > 0.5] = int(name.split("_")[1])
        content = cell_content(h5, cell_of_node, max(cell_of_node) + 1)
        if scenario != "reference":
            k0 = sorted(h5["data/ECM"])[0]
            ecm_tot0 = h5[f"data/ECM/{k0}"][:] + h5[f"data/ECM_Ca/{k0}"][:]
            ecs = np.where(h5["compartments/ecs"][:] > 0.5)[0]
            from scipy.spatial import cKDTree
            _, nearest = cKDTree(points[ecs]).query(centers)
            ecm_at_synapse = ecm_tot0[ecs[nearest]]
            ecm_peak = float(ecm_tot0[ecs].max())
        else:
            ecm_at_synapse = np.full(len(centers), np.nan)
            ecm_peak = np.nan

    n = len(centers)
    if scenario == "all":
        buffered = np.ones(n, bool)
    elif scenario in ("half", "on", "off"):
        buffered = np.zeros(n, bool)
        buffered[ecm["local_target"]] = True
    else:
        buffered = np.zeros(n, bool)
    if buffered.any():
        d_buffered = np.linalg.norm(centers[:, None] - centers[buffered][None], axis=2).min(axis=1)
    else:
        d_buffered = np.full(n, np.nan)
    is_active = np.ones(n, bool) if active is None else np.isin(np.arange(n), active)
    d_active = np.linalg.norm(centers[:, None] - centers[is_active][None], axis=2).min(axis=1)

    t_idx = lambda t: int(np.argmin(np.abs(times - t)))
    i0 = t_idx(pulses[0])
    rows = []
    for s in range(n):
        ca = local_ca[:, s]
        row = dict(
            run=run, seed=cfg["seed"], ecs_ratio=cfg["geometry"]["ecs_ratio"], scenario=scenario,
            ecm_total=_mm(ecm["ecm_total"]) if ecm["enabled"] else 0.0,
            ecm_kf=_mm(ecm["ecm_kf"]), ecm_kd=_mm(ecm["ecm_kd"]), local_width=local_width,
            n_buffered=int(buffered.sum()), n_pulses=len(pulses),
            stimulus=f"{len(pulses)}p_{_mm(syn['i_channel']):g}pA",
            active="all" if active is None else ",".join(map(str, sorted(active))),
            is_active=bool(is_active[s]), d_active=d_active[s],
            synapse=s, cell=int(cells[s]), n_in_cell=int(n_in_cell[s]),
            x=centers[s, 0], y=centers[s, 1], z=centers[s, 2],
            target_class=targets.get(s, ""), buffered=bool(buffered[s]),
            d_buffered=d_buffered[s], ecm_at_synapse=ecm_at_synapse[s], ecm_peak=ecm_peak,
        )
        c = content[:, cells[s]]
        for k, t in enumerate(pulses, 1):
            row[f"uptake_{k}"] = c[t_idx(t + 10)] - c[t_idx(t)]
        row["uptake_cum"] = c[-1] - c[i0]
        if n_in_cell[s] > 1:
            for key in [k for k in row if k.startswith("uptake")]:
                row[key] = np.nan
        at_floor = ca[i0:] <= floor
        row["time_to_floor"] = times[i0:][at_floor][0] - pulses[0] if at_floor.any() else np.nan
        row["frac_at_floor"] = at_floor.mean()
        row["peak_deficit"] = ca0 - ca.min()
        iL = t_idx(pulses[-1])
        imin = iL + int(np.argmin(ca[iL:]))
        half = ca[imin] + 0.5 * (ca0 - ca[imin])
        rec = np.where(ca[imin:] >= half)[0]
        row["recovery_half"] = times[imin + rec[0]] - times[imin] if len(rec) else np.nan
        rows.append(row)
    return rows


def add_effects(df):
    """effect_<m> = m - reference (same seed, ECS ratio, stimulus, active set, synapse); thr_<m> = reference IQR.

    The IQR is taken across the synapses that carry flux: all synapses of the full-stimulus
    reference, or the active synapse of each solitary-synapse reference (one per run)."""
    metrics = METRICS + sorted(c for c in df.columns if c.startswith("uptake_") and c[7:].isdigit())
    ref = df[df.scenario == "reference"]
    if ref.empty:
        raise SystemExit("no reference run (ecm.enabled=false) among the inputs")
    ref_idx = ref.set_index(["seed", "ecs_ratio", "stimulus", "active", "synapse"])
    key = pd.MultiIndex.from_frame(df[["seed", "ecs_ratio", "stimulus", "active", "synapse"]])
    ref_active = ref[ref.is_active]
    solitary = lambda d: d.active != "all"
    thr_key = pd.MultiIndex.from_arrays([df.seed, df.ecs_ratio, df.stimulus, solitary(df)])
    for m in metrics:
        df[f"effect_{m}"] = df[m].to_numpy() - ref_idx[m].reindex(key).to_numpy()
        iqr = ref_active.groupby([ref_active.seed, ref_active.ecs_ratio, ref_active.stimulus,
                                  solitary(ref_active)])[m].agg(
            lambda v: v.quantile(0.75) - v.quantile(0.25))
        df[f"thr_{m}"] = iqr.reindex(thr_key).to_numpy()
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="sweep directories (any depth) to collect runs from")
    p.add_argument("--targets", required=True, help="targets.json from select_targets.py")
    p.add_argument("--out", default="dose_response.csv")
    args = p.parse_args()

    tj = json.load(open(args.targets))
    targets = {**{i: "restricted" for i in tj["restricted"]}, **{i: "connected" for i in tj["connected"]}}
    runs = [r for path in args.paths for r in find_run_dirs(path)]
    print(f"{len(runs)} runs")
    rows = []
    for run in runs:
        try:
            rows += evaluate_run(str(run), targets)
        except Exception as e:  # a crashed run must not kill the collection
            print(f"  skipped {run}: {type(e).__name__}: {e}")
    df = add_effects(pd.DataFrame(rows))
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")
    ref = df[df.scenario == "reference"]
    for (ecs, stim, act), g in ref.groupby(["ecs_ratio", "stimulus", ref.active == "all"]):
        print(f"reference ecs {ecs} {stim} ({'all synapses' if act else 'solitary'}): " + ", ".join(
            f"{m} IQR {g[f'thr_{m}'].iloc[0]:.3g}" for m in METRICS))


if __name__ == "__main__":
    main()
