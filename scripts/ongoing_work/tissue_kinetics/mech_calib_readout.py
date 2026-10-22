"""Phase 1 readout of mech_calib_sweep: per run, peak ECS swelling, min J, s/step and the
uptake and depletion (peak deficit, onset and recovery rates of the nearest-ECS-vertex Ca) of the
active synapse (solitary) or median synapse (crowd) vs the mechanics-off twin.

Cell content uses the DEFORMED element volumes, so uptake is an amount even when the cells
are compressed by ECS swelling (the undeformed-volume integral in evaluate_dose_response
would read the compression as uptake).

local_vol_peak: ECS volume ratio within 2 um of the active synapse (whole ECS for the crowd).
total_vol_peak / ecs_frac_peak: whole-block volume ratio and ECS-fraction ratio (block inflation vs
ECS gained at the cells' expense). Concentration metrics include the dilution by swelling.

Usage: uv run scripts/ongoing_work/tissue_kinetics/mech_calib_readout.py results/mech-calib_* [--out=phase1.csv]
"""
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from analysis import compute_local_ca, load_synapse_centers, synapse_cells
from bmbcsim.simulation.result_io import find_run_dirs


def tet_vol(points, conn):
    p = points[conn]
    return np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                            np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6.0


def step_seconds(log, factor):
    """Wall seconds per time step from the first/last record timestamps (records every `factor` steps)."""
    lines = [l for l in open(log) if "Recording XDMF" in l]
    # a worker keeps the previous run's log handler, so the NEXT run's lines leak in: cut at its t=0
    t_sim = [float(l.split("at time")[1].split()[0]) for l in lines]
    end = next((i for i in range(1, len(t_sim)) if t_sim[i] < t_sim[i - 1]), len(t_sim))
    ts = [l[:23] for l in lines[:end]]
    if len(ts) < 2:
        return np.nan
    t0, t1 = (pd.Timestamp(t.replace(",", ".")) for t in (ts[0], ts[-1]))
    return (t1 - t0).total_seconds() / ((len(ts) - 1) * factor)


def readout(run):
    cfg = yaml.safe_load(open(f"{run}/config.yaml"))
    mech = cfg["mechanics"]
    k = float(str(mech["ecm_ca_condensation"]).split()[0]) if mech["enabled"] else 0.0
    active = cfg["synapse"].get("active")
    with h5py.File(f"{run}/snapshot.h5", "r") as h5:
        points = h5["mesh/points"][:].astype(np.float64)
        conn = h5["mesh/connectivity"][:]
        ecs_el = h5["compartments/ecs"][:][conn].mean(axis=1) > 0.5
        cell_of_node = np.full(len(points), -1)
        n_cells = sum(k_.startswith("cell_") for k_ in h5["compartments"])
        for c in range(n_cells):
            cell_of_node[h5[f"compartments/cell_{c}"][:] > 0.5] = c
        keys = sorted(h5["data/Ca"])
        vol0 = tet_vol(points, conn)
        centers = load_synapse_centers(run, h5)
        cells, n_in_cell = synapse_cells(h5, centers)
        c_local = centers[active[0]] if active else None
        near = ecs_el & (np.linalg.norm(points[conn].mean(axis=1) - c_local, axis=1) < 2.0) \
            if active else ecs_el
        content, ecs_vol, j_min, j_max, local_vol, tot_vol = [], [], [], [], [], []
        for key in keys:
            ca = h5[f"data/Ca/{key}"][:]
            pts = points + (h5[f"data/deformation/{key}"][:] if "deformation" in h5["data"] else 0)
            vol = tet_vol(pts, conn)
            J = vol / vol0
            nodal = np.bincount(conn.ravel(), weights=np.repeat(vol / 4, 4), minlength=len(points))
            m = cell_of_node >= 0
            content.append(np.bincount(cell_of_node[m], weights=(nodal * ca)[m], minlength=n_cells))
            ecs_vol.append(vol[ecs_el].sum() / vol0[ecs_el].sum())
            j_min.append(J.min())
            j_max.append(J[ecs_el].max())
            local_vol.append(vol[near].sum() / vol0[near].sum())
            tot_vol.append(vol.sum() / vol0.sum())
    content = np.array(content)
    ecs_frac = np.array(ecs_vol) / np.array(tot_vol)
    times, local_ca = compute_local_ca(run)  # ms
    ca0 = float(str(cfg["diffusion"]["ca_ecs"]).split()[0])
    pulses = [float(str(t).split()[0]) for t in cfg["synapse"]["pulse_times"]]  # ms
    sel = [active[0]] if active else np.where(n_in_cell == 1)[0]
    dep = {}
    for key, fn in (("peak_deficit", lambda ca: ca0 - ca.min()), ("t_min_ms", lambda ca: times[ca.argmin()]),
                    ("onset_mM_s", lambda ca: -1e3 * np.gradient(ca, times)[times <= pulses[0] + 10].min()),
                    ("recovery_mM_s", lambda ca: 1e3 * np.gradient(ca, times)[(times >= pulses[-1] + 12)
                                                                                  & (times <= pulses[-1] + 30)].mean())):
        dep[key] = float(np.median([fn(local_ca[:, s].astype(float)) for s in sel]))
    uptake = content[-1] - content[0]
    syn_uptake = uptake[cells]
    if active is None:
        up = np.nanmedian(np.where(n_in_cell == 1, syn_uptake, np.nan))
    else:
        up = syn_uptake[active[0]]
    return dict(run=run.split("/")[-2][:40], k=k, active="all" if active is None else active[0],
                local_width=cfg["ecm"].get("local_width"), ecs_vol_peak=max(ecs_vol),
                ecs_vol_end=ecs_vol[-1], local_vol_peak=max(local_vol), total_vol_peak=max(tot_vol),
                ecs_frac_peak=ecs_frac.max(), J_min=min(j_min), J_max_ecs=max(j_max), uptake_amol=up, **dep,
                s_per_step=step_seconds(f"{run}/simulation.log", cfg["record_interval_factor"]))


def main(paths, out=None):
    df = pd.DataFrame([readout(str(r)) for p in paths for r in find_run_dirs(p)])
    df["regime"] = np.where(df.active == "all", "crowd", "solitary")
    twin = df[df.k == 0].set_index("regime")
    for m in ("uptake_amol", "peak_deficit", "onset_mM_s", "recovery_mM_s"):
        ref = twin[m].reindex(df.regime).to_numpy()
        df[f"d_{m}_%"] = 100 * (df[m] - ref) / ref
    df = df.sort_values(["regime", "k"])
    pd.set_option("display.width", 200)
    if out:
        df.to_csv(out, index=False)
    print(df.drop(columns="run").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return df


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--out=")],
         next((a[6:] for a in sys.argv[1:] if a.startswith("--out=")), None))
