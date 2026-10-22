"""Plots for the solitary-synapse experiment (single_reference_sweep + single_buffer_sweep).

Rows with ``active != "all"``: one synapse carries flux per run, the buffer is uniform,
on the active synapse (``on``) or on a neighbour (``off``) at the same total amount.

* single_effect.png      effect at the active synapse: uniform vs on vs off, per dose and target
* single_lollipop.png    the same at the top dose, one cluster per target (lollipops, IQR shaded)
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

from plot_dose_response import KD0, KF0, LABEL  # noqa: E402

PLACEMENT_COLOR = {"uniform": "k", "on": "tab:red", "off": "tab:blue"}
METRICS = ["uptake_cum", "peak_deficit"]
MIN_EFFECT = 1e-3  # uniform effect (amol / mM) below which no dose equivalence is extrapolated


def log_doses(ax, doses):
    ax.set_xscale("log")
    ax.set_xticks(doses, [f"{d:g}" for d in doses])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def power_law_dose(uniform, effect, min_effect=MIN_EFFECT):
    """Uniform dose with the same effect, power-law extrapolated from the uniform doses
    (the on-target effects lie far above the uniform range, so interpolation is impossible).
    None when the uniform effects are not of one sign or too small to fit."""
    u = uniform.sort_values("ecm_total")
    e, n = u[effect].to_numpy(), u.ecm_total.to_numpy()
    s = np.sign(e[-1]) if len(e) else 0
    if len(e) < 2 or abs(e[-1]) < min_effect or np.any(s * e <= 0):
        return None
    b, a = np.polyfit(np.log(n), np.log(s * e), 1)
    return lambda y: np.where(s * y > 0, np.exp((np.log(np.abs(y)) - a) / b), np.nan)


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
            if np.isfinite(thr):
                ax.text(0.03, 0.95, f"reference IQR {thr:.2g}", transform=ax.transAxes, fontsize=7, va="top")
            ax.axhline(0, color="0.5", lw=0.5)
            log_doses(ax, sorted(g0.ecm_total.unique()))
            if i == 0:
                ax.set_title(f"active synapse {s} ({g0.target_class.iloc[0]})", fontsize=10)
            if j == 0:
                ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
            if i == len(METRICS) - 1:
                ax.set_xlabel("buffer dose, ECS mean (mM)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Solitary synapse: effect of the buffer at the active synapse")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "single_effect.png"), dpi=130)
    plt.close(fig)


def fig_lollipop(df, out):
    """Top-dose effects at the active synapse as lollipops: one cluster per target, one stick per
    placement (same colours/markers as fig_effect); the reference IQR is shaded on the side the
    effects fall on."""
    act = active_rows(df)
    act = act[(act.scenario != "reference") & (act.ecm_total == act.ecm_total.max())]
    syns = sorted(act.synapse.unique())
    keys = sorted(set(zip(act.scenario, act.local_width.fillna(0))),
                  key=lambda k: ({"uniform": 0, "off": 1, "on": 2}[k[0]], -k[1]))
    offs = np.linspace(-0.3, 0.3, len(keys))
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(1.6 * len(syns) + 1, 2.6 * len(METRICS)),
                             sharex=True, squeeze=False)
    for i, m in enumerate(METRICS):
        ax = axes[i, 0]
        thr, sign = act[f"thr_{m}"].iloc[0], np.sign(act[f"effect_{m}"].median())
        if np.isfinite(thr):
            ax.axhspan(0, sign * thr, color="0.88", zorder=0, label="reference IQR")
        for (scen, width), o in zip(keys, offs):
            g = act[(act.scenario == scen) & (act.local_width.fillna(0) == width)].set_index("synapse")
            x = np.array([syns.index(s) for s in g.index]) + o
            y = g[f"effect_{m}"].to_numpy()
            marker = "s" if width == 1.0 and scen != "uniform" else "o"
            ax.vlines(x, 0, y, color=PLACEMENT_COLOR[scen], lw=1)
            ax.plot(x, y, marker, color=PLACEMENT_COLOR[scen], ms=5,
                    label=scen if scen == "uniform" else f"{scen}, sigma={width:g} um")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
    axes[-1, 0].set_xticks(range(len(syns)), [f"synapse {s}\n({act[act.synapse == s].target_class.iloc[0]})"
                                              for s in syns])
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Solitary synapse: buffer effect at the active synapse, dose {act.ecm_total.iloc[0]:g} mM")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "single_lollipop.png"), dpi=130)
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
            if scen == "off" and g.d_buffered.notna().any():  # the buffered neighbour's distance
                ax.axvline(g.loc[g.d_buffered.idxmin(), "d_active"], color="tab:blue", lw=0.8, ls=":")
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
    """Equivalent uniform dose of the on-target buffer (power-law extrapolation of the two
    uniform doses) against the local buffer concentration, both relative to the dose N."""
    act = active_rows(df)
    syns = sorted(act.synapse.unique())
    colors = dict(zip(syns, plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.4), squeeze=False)
    table = []
    for i, m in enumerate(METRICS):
        ax = axes[0, i]
        for s in syns:
            syn = act[act.synapse == s]
            inv = power_law_dose(syn[syn.scenario == "uniform"], f"effect_{m}")
            for width, g in syn[syn.scenario == "on"].groupby("local_width"):
                n_eq = inv(g[f"effect_{m}"].to_numpy()) if inv else np.full(len(g), np.nan)
                conc = g.ecm_at_synapse / g.ecm_total
                ax.plot(conc, n_eq / g.ecm_total, "o" if width != 1.0 else "s", color=colors[s], ms=5,
                        label=f"synapse {s}, sigma={width:g} um")
                for n, r, c in zip(g.ecm_total, n_eq / g.ecm_total, conc):
                    table.append(dict(synapse=s, metric=m, local_width=width, ecm_total=n,
                                      n_eq_over_n=r, conc_factor=c))
        lim = [1, max(10, (act.ecm_at_synapse / act.ecm_total).max() * 1.5)]
        ax.plot(lim, lim, "--", color="0.6", lw=0.8, label="N_eq/N = local conc./N")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("buffer concentration at the synapse / N")
        ax.set_ylabel(f"{m}: equivalent uniform dose / N", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Solitary synapse: equivalent uniform dose of the on-target buffer\n"
                 "(power-law extrapolation of the uniform doses; missing = uniform effect below "
                 f"{MIN_EFFECT:g})", fontsize=10)
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
    fig_lollipop(df, args.out)
    fig_shielding(df, args.out)
    fig_equivalent_dose(df, args.out)
    print(f"plots -> {args.out}: {sorted(f for f in os.listdir(args.out) if f.startswith('single_'))}")


if __name__ == "__main__":
    main()
