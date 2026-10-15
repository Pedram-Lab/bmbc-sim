"""Plots for the solitary-synapse experiment (single_reference_sweep + single_buffer_sweep).

Rows with ``active != "all"``: one synapse carries flux per run, the buffer is uniform,
on the active synapse (``on``) or on a neighbour (``off``) at the same total amount.

* single_effect.png      effect at the active synapse: uniform vs on vs off, per dose and target
* single_shielding.png   effect on the SILENT synapses vs distance to the active one (crosstalk)
* single_equivalent_dose.png / .csv   uniform dose with the same effect as the on-target buffer

    uv run scripts/ongoing_work/tissue_kinetics/plot_single_synapse.py dose_response.csv --out plots/
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_dose_response import KD0, KF0, LABEL, equivalent_dose  # noqa: E402

PLACEMENT_COLOR = {"uniform": "k", "on": "tab:red", "off": "tab:blue"}
METRICS = ["uptake_cum", "peak_deficit"]


def solitary(df):
    """Solitary-synapse rows at the default kinetics (reference rows have NaN kinetics)."""
    d = df[df.active != "all"]
    return d[(d.scenario == "reference") | ((d.ecm_kf == KF0) & (d.ecm_kd == KD0))]


def active_rows(df):
    """One row per run: the active synapse itself."""
    return solitary(df)[lambda d: d.is_active]


def fig_effect(df, out):
    act = active_rows(df)
    act = act[act.scenario != "reference"]
    syns = sorted(act.synapse.unique())
    fig, axes = plt.subplots(len(METRICS), len(syns), figsize=(3.2 * len(syns), 2.8 * len(METRICS)),
                             squeeze=False, sharex=True)
    for j, s in enumerate(syns):
        g0 = act[act.synapse == s]
        for i, m in enumerate(METRICS):
            ax = axes[i, j]
            for (scen, width), g in g0.groupby(["scenario", g0.local_width.fillna(0)]):
                g = g.sort_values("ecm_total")
                label = scen if scen == "uniform" else f"{scen}, sigma={width:g} um"
                ax.plot(g.ecm_total, g[f"effect_{m}"], "-o" if width != 1.0 or scen == "uniform" else "--s",
                        color=PLACEMENT_COLOR[scen], ms=4, label=label)
            thr = g0[f"thr_{m}"].iloc[0]
            if np.isfinite(thr) and thr > 0:
                ax.axhspan(-thr, thr, color="0.85", zorder=0)
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(f"active synapse {s} ({g0.target_class.iloc[0]})", fontsize=10)
            if j == 0:
                ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
            if i == len(METRICS) - 1:
                ax.set_xlabel("buffer dose, ECS mean (mM)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Solitary synapse: effect of the buffer at the active synapse (grey = reference IQR)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "single_effect.png"), dpi=130)
    plt.close(fig)


def fig_shielding(df, out):
    sol = solitary(df)
    sil = sol[~sol.is_active & (sol.scenario != "reference")]
    dose = sil.ecm_total.max()
    sil = sil[(sil.ecm_total == dose)]
    syns = sorted(sil.active.unique())
    fig, axes = plt.subplots(1, len(syns), figsize=(3.4 * len(syns), 3.2), squeeze=False, sharey=True)
    for j, a in enumerate(syns):
        ax = axes[0, j]
        g0 = sil[sil.active == a]
        for (scen, width), g in g0.groupby(["scenario", g0.local_width.fillna(0)]):
            g = g[g.d_active <= 6]
            label = scen if scen == "uniform" else f"{scen}, sigma={width:g} um"
            ax.plot(g.d_active, g.effect_peak_deficit, "o" if width != 1.0 or scen == "uniform" else "s",
                    color=PLACEMENT_COLOR[scen], ms=4, alpha=0.7, label=label)
            if scen == "off":
                ax.axvline(g.d_buffered.min() if g.d_buffered.notna().any() else np.nan, color="tab:blue",
                           lw=0.8, ls=":")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(f"active synapse {a}", fontsize=10)
        ax.set_xlabel("distance to the active synapse (um)")
        if j == 0:
            ax.set_ylabel("Δ peak deficit at silent synapses (mM)", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Shielding: buffer effect on the silent neighbours, dose {dose:g} mM "
                 "(dotted = position of the off-target buffer)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "single_shielding.png"), dpi=130)
    plt.close(fig)


def fig_equivalent_dose(df, out):
    act = active_rows(df)
    syns = sorted(act.synapse.unique())
    fig, axes = plt.subplots(len(METRICS), len(syns), figsize=(3.2 * len(syns), 2.8 * len(METRICS)),
                             squeeze=False, sharex=True)
    table = []
    for j, s in enumerate(syns):
        syn = act[act.synapse == s]
        for i, m in enumerate(METRICS):
            ax = axes[i, j]
            inv = equivalent_dose(syn[syn.scenario == "uniform"], f"effect_{m}")
            for width, g in syn[syn.scenario == "on"].groupby("local_width"):
                g = g.sort_values("ecm_total")
                n_eq = inv(g[f"effect_{m}"].to_numpy()) if callable(inv) else np.full(len(g), np.nan)
                ax.plot(g.ecm_total, n_eq / g.ecm_total, "-o", ms=4, label=f"sigma={width:g} um: N_eq/N")
                ax.plot(g.ecm_total, g.ecm_at_synapse / g.ecm_total, "-", alpha=0.4, lw=2,
                        label=f"sigma={width:g} um: local conc./N")
                for n, r, c in zip(g.ecm_total, n_eq / g.ecm_total, g.ecm_at_synapse / g.ecm_total):
                    table.append(dict(synapse=s, metric=m, local_width=width, ecm_total=n,
                                      n_eq_over_n=r, conc_factor=c))
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"active synapse {s}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{m}: equivalent uniform dose / N", fontsize=8)
            if i == len(METRICS) - 1:
                ax.set_xlabel("buffer dose N (mM)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Solitary synapse: equivalent uniform dose of the on-target buffer")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "single_equivalent_dose.png"), dpi=130)
    plt.close(fig)
    pd.DataFrame(table).to_csv(os.path.join(out, "single_equivalent_dose.csv"), index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--out", default="plots")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if solitary(df).empty:
        raise SystemExit("no solitary-synapse runs (synapse.active) in the CSV")
    fig_effect(df, args.out)
    fig_shielding(df, args.out)
    fig_equivalent_dose(df, args.out)
    print(f"plots -> {args.out}: {sorted(f for f in os.listdir(args.out) if f.startswith('single_'))}")


if __name__ == "__main__":
    main()
