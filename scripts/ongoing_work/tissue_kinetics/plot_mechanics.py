"""Mechanics figures: swelling and depletion vs the condensation coefficient k, and the crowd
swelling time courses that show the dilution -> unbinding runaway.

    uv run scripts/ongoing_work/tissue_kinetics/plot_mechanics.py mechanics.csv \\
        --runs <crowd run dirs> --buffer dose_response.csv --out plots/

mech_k_sweep.png          peak swelling and change of the peak free-Ca deficit vs k (mech_calib_readout
                          CSV), solitary and crowd; dashed: the chemical buffer's effect at 1 mM from the
                          dose-response CSV (solitary: on-target sigma 1.5 um; crowd: uniform)
mech_crowd_timeseries.png crowd runs: ECS volume ratio, ECS mean free Ca and mean bound ECM-Ca vs time
"""
import argparse
import os

import h5py
import matplotlib
import numpy as np
import pandas as pd
import yaml

from mech_calib_readout import tet_vol

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {"solitary": "tab:red", "crowd": "tab:blue"}


def buffer_reference(csv):
    """Chemical buffer effect on the peak deficit at 1 mM, % of the reference deficit."""
    df = pd.read_csv(csv)
    df = df[df.stimulus == "5p_0.5pA"]
    ref = df[df.scenario == "reference"].set_index(["active", "synapse"]).peak_deficit
    df = df[(df.ecm_kf == 1e5) & (df.ecm_kd == 1.0)]  # the reference runs carry the config default kinetics
    out = {}
    sol = df[(df.scenario == "on") & (df.ecm_total == 1.0) & (df.local_width == 1.5) & df.is_active]
    crowd = df[(df.scenario == "uniform") & (df.ecm_total == 1.0) & (df.active == "all")]
    for name, g in (("solitary", sol), ("crowd", crowd)):
        r = ref.reindex(pd.MultiIndex.from_frame(g[["active", "synapse"]])).to_numpy()
        out[name] = float(np.nanmedian(100 * g.effect_peak_deficit.to_numpy() / r))
    return out


def fig_k_sweep(df, out, buffer_ref=None):
    df = df[df.k > 0].copy()
    df["swell_%"] = 100 * (np.where(df.regime == "solitary", df.local_vol_peak, df.ecs_vol_peak) - 1)
    df["finished"] = df.t_end_ms >= df.end_ms - 1e-6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for regime, g in df.groupby("regime"):
        c = COLOR[regime]
        for bs, gb in g.groupby("boundary_stiffness"):
            lab = regime + (" (solitary: within 2 um)" if regime == "solitary" else "") + \
                (" free block" if bs == 0 else "")
            ok = gb[gb.finished].sort_values("k")
            if len(ok):
                ax1.plot(ok.k, ok["swell_%"], "o-", color=c, mfc=c if bs else "w", label=lab)
                ax2.plot(ok.k, ok["d_peak_deficit_%"], "o-", color=c, mfc=c if bs else "w", label=lab)
            bad = gb[~gb.finished]
            ax1.plot(bad.k, bad["swell_%"], "x", color=c, ms=9, mew=2,
                     label="runaway, solver failed" if (regime, bs) == ("crowd", 1.0) else None)
        if buffer_ref and regime in buffer_ref:
            ax2.axhline(buffer_ref[regime], color=c, ls="--", lw=1,
                        label=f"{regime}: chemical buffer 1 mM")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("k = ECS swelling per mM ECM-Ca unbound (1/mM)")
        ax.axhline(0, color="0.7", lw=0.8)
    ax1.set_ylabel("peak ECS swelling (%)")
    ax2.set_ylabel("change of peak free-Ca deficit (%)")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.suptitle("Shape change: swelling and depletion vs k (mechanics-on minus mechanics-off twin)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mech_k_sweep.png"), dpi=130)


def timeseries(run):
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    mech = cfg["mechanics"]
    k = float(str(mech["ecm_ca_condensation"]).split()[0]) if mech["enabled"] else 0.0
    with h5py.File(os.path.join(run, "snapshot.h5"), "r") as h5:
        points = h5["mesh/points"][:].astype(np.float64)
        conn = h5["mesh/connectivity"][:]
        ecs_el = h5["compartments/ecs"][:][conn].mean(axis=1) > 0.5
        ecs_node = h5["compartments/ecs"][:] > 0.5
        vol0 = tet_vol(points, conn)[ecs_el].sum()
        keys = sorted(h5["data/Ca"])
        t = h5["data/time"][:] if "time" in h5["data"] else np.arange(len(keys)) * 2.0
        rows = []
        for key in keys:
            pts = points + (h5[f"data/deformation/{key}"][:] if "deformation" in h5["data"] else 0)
            vol = tet_vol(pts, conn)
            nodal = np.bincount(conn.ravel(), weights=np.repeat(vol / 4, 4), minlength=len(points))[ecs_node]
            ca = h5[f"data/Ca/{key}"][:][ecs_node]
            bound = h5[f"data/ECM_Ca/{key}"][:][ecs_node] if "ECM_Ca" in h5["data"] else np.full_like(ca, np.nan)
            rows.append((vol[ecs_el].sum() / vol0, (ca * nodal).sum() / nodal.sum(),
                         (bound * nodal).sum() / nodal.sum()))
    pulses = [float(str(p).split()[0]) for p in cfg["synapse"]["pulse_times"]]
    return dict(k=k, bs=mech.get("boundary_stiffness", 1.0), t=np.asarray(t[:len(rows)], float),
                data=np.array(rows), pulses=pulses, end=1e3 * float(str(cfg["end_time"]).split()[0]))


def fig_crowd_timeseries(runs, out):
    series = {}
    for run in runs:
        s = timeseries(run)
        key = (s["k"], s["bs"])
        if key not in series or len(s["t"]) > len(series[key]["t"]):  # keep the longest attempt
            series[key] = s
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cmap = plt.get_cmap("viridis")
    ks = sorted({k for k, _ in series})
    for (k, bs), s in sorted(series.items()):
        c = cmap(ks.index(k) / max(len(ks) - 1, 1))
        ls = "--" if bs == 0 else "-"
        lab = f"k {k:g}" + (" free block" if bs == 0 else "") + ("" if s["t"][-1] >= s["end"] - 1e-6 else " (failed)")
        for ax, col in zip(axes, range(3)):
            ax.plot(s["t"], s["data"][:, col], ls, color=c, marker="." if len(s["t"]) < 10 else None, label=lab)
    for ax, (yl, title) in zip(axes, (("ECS volume / initial", "ECS swelling"),
                                      ("ECS mean free Ca (mM)", "free Ca: synaptic uptake + dilution"),
                                      ("ECS mean bound ECM-Ca (mM)", "ECM-Ca unbinding drives the swelling"))):
        for p in next(iter(series.values()))["pulses"]:
            ax.axvline(p, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 50)
    axes[0].legend(fontsize=8)
    fig.suptitle("Crowd (all synapses firing): the swelling runaway at k >= 10")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mech_crowd_timeseries.png"), dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="mech_calib_readout CSV")
    p.add_argument("--runs", nargs="*", default=[], help="crowd run dirs for the time-series figure")
    p.add_argument("--buffer", help="dose_response.csv for the chemical-buffer reference lines")
    p.add_argument("--out", default=".")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_k_sweep(pd.read_csv(args.csv), args.out, buffer_reference(args.buffer) if args.buffer else None)
    if args.runs:
        fig_crowd_timeseries(args.runs, args.out)


if __name__ == "__main__":
    main()
