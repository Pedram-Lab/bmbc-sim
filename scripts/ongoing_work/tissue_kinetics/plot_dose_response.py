"""Plots for the buffer dose-response study, from evaluate_dose_response.py's CSV.

    uv run scripts/ongoing_work/tissue_kinetics/plot_dose_response.py dose_response.csv --out plots/

dose_response.png     effect vs dose per evaluation synapse, uniform vs local buffer
equivalent_dose.png   uniform dose giving the same effect as the local buffer, / dose
confinement.png       effect vs distance to the nearest buffered synapse (half-buffered)
robustness_<axis>.png effect vs k_on / Kd / stimulus for uniform vs half-buffered
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["uptake_cum", "frac_at_floor", "peak_deficit", "time_to_floor"]
LABEL = {"uptake_cum": "cumulative uptake (amol)", "frac_at_floor": "fraction of time at floor",
         "peak_deficit": "peak free-Ca deficit (mM)", "time_to_floor": "time to floor (ms)",
         "recovery_half": "recovery t1/2 (ms)", "uptake_1": "uptake, 1st spike (amol)"}
COLOR = {"uniform": "k", "all": "tab:blue", "half": "tab:red"}
DASH = {0.5: "--", 1.0: "-"}
KF0, KD0, STIM0 = 1e5, 1.0, "5p_0.5pA"


def core(df):
    """Rows with the default kinetics and stimulus (steps 1/2)."""
    return df[(df.ecm_kf == KF0) & (df.ecm_kd == KD0) & (df.stimulus == STIM0)]


def targets(df):
    t = df[df.target_class.notna()].drop_duplicates("synapse").sort_values(["target_class", "synapse"])
    return list(zip(t.synapse, t.target_class))


def series_label(scenario, width):
    return scenario if scenario == "uniform" else f"{scenario}, sigma={width:g} um"


def plot_lines(ax, sub, xcol, metric, logx=True):
    """One line per (scenario, sigma); filled markers where the synapse is buffered."""
    for (scen, width), g in sub.groupby(["scenario", sub.local_width.fillna(0)]):
        g = g.sort_values(xcol)
        ls = DASH.get(width, "-") if scen != "uniform" else "-"
        ax.plot(g[xcol], g[f"effect_{metric}"], ls, color=COLOR[scen], lw=1.2,
                label=series_label(scen, width))
        buffered = g.buffered.to_numpy()
        ax.plot(g[xcol][buffered], g[f"effect_{metric}"][buffered], "o", color=COLOR[scen], ms=5)
        ax.plot(g[xcol][~buffered], g[f"effect_{metric}"][~buffered], "o", mfc="white",
                color=COLOR[scen], ms=5)
    thr = sub[f"thr_{metric}"].iloc[0]
    if np.isfinite(thr):
        ax.axhspan(-thr, thr, color="0.85", zorder=0, label="±IQR(reference)")
    ax.axhline(0, color="0.5", lw=0.5)
    if logx:
        ax.set_xscale("log")


def fig_dose_response(df, out):
    sub, tg = core(df), targets(df)
    fig, axes = plt.subplots(len(METRICS), len(tg), figsize=(3.2 * len(tg), 2.6 * len(METRICS)),
                             squeeze=False, sharex=True)
    for j, (s, cls) in enumerate(tg):
        for i, m in enumerate(METRICS):
            ax = axes[i, j]
            plot_lines(ax, sub[(sub.synapse == s) & (sub.scenario != "reference")], "ecm_total", m)
            if i == 0:
                ax.set_title(f"synapse {s} ({cls})", fontsize=10)
            if j == 0:
                ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
            if i == len(METRICS) - 1:
                ax.set_xlabel("buffer dose, ECS mean (mM)")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Effect of the buffer relative to no buffer (filled = local buffer on this synapse)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dose_response.png"), dpi=130)
    plt.close(fig)


def equivalent_dose(uniform, effect):
    """Uniform dose with the same effect, interpolated in log dose (NaN outside the range)."""
    u = uniform.sort_values("ecm_total")
    e, ln = u[effect].to_numpy(), np.log(u.ecm_total.to_numpy())
    order = np.argsort(e)
    if len(e) < 2 or np.ptp(e) == 0:
        return np.nan
    return lambda y: np.exp(np.interp(y, e[order], ln[order], left=np.nan, right=np.nan))


def fig_equivalent_dose(df, out):
    sub, tg = core(df), targets(df)
    metrics = ["uptake_cum", "peak_deficit"]
    fig, axes = plt.subplots(len(metrics), len(tg), figsize=(3.2 * len(tg), 2.8 * len(metrics)),
                             squeeze=False, sharex=True)
    table = []
    for j, (s, cls) in enumerate(tg):
        syn = sub[sub.synapse == s]
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            inv = equivalent_dose(syn[syn.scenario == "uniform"], f"effect_{m}")
            for (scen, width), g in syn[syn.scenario.isin(["all", "half"]) & syn.buffered].groupby(
                    ["scenario", "local_width"]):
                g = g.sort_values("ecm_total")
                n_eq = inv(g[f"effect_{m}"].to_numpy()) if callable(inv) else np.full(len(g), np.nan)
                ax.plot(g.ecm_total, n_eq / g.ecm_total, DASH[width] + "o", color=COLOR[scen], ms=4,
                        label=f"{series_label(scen, width)}: N_eq/N")
                ax.plot(g.ecm_total, g.ecm_at_synapse / g.ecm_total, DASH[width], color=COLOR[scen],
                        alpha=0.4, lw=2, label=f"{series_label(scen, width)}: local conc./N")
                for n, r, c in zip(g.ecm_total, n_eq / g.ecm_total, g.ecm_at_synapse / g.ecm_total):
                    table.append(dict(synapse=s, target_class=cls, metric=m, scenario=scen,
                                      local_width=width, ecm_total=n, n_eq_over_n=r, conc_factor=c))
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"synapse {s} ({cls})", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{m}: equivalent uniform dose / N", fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("buffer dose N (mM)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Equivalent uniform dose vs the local concentration factor at the synapse")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "equivalent_dose.png"), dpi=130)
    plt.close(fig)
    pd.DataFrame(table).to_csv(os.path.join(out, "equivalent_dose.csv"), index=False)


def fig_confinement(df, out):
    sub = core(df)
    sub = sub[(sub.scenario == "half") & (sub.ecm_total == sub.ecm_total.max())]
    metrics = ["uptake_cum", "frac_at_floor", "peak_deficit"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2))
    for ax, m in zip(axes, metrics):
        for width, g in sub.groupby("local_width"):
            ax.plot(g.d_buffered, g[f"effect_{m}"], "o", ms=4, label=f"sigma={width:g} um",
                    alpha=0.7)
            t = g[g.target_class.notna()]
            ax.plot(t.d_buffered, t[f"effect_{m}"], "k+", ms=9, mew=1.5)
        thr = sub[f"thr_{m}"].iloc[0]
        if np.isfinite(thr):
            ax.axhspan(-thr, thr, color="0.85", zorder=0)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlabel("distance to nearest buffered synapse (um)")
        ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Half-buffered, dose {sub.ecm_total.max():g} mM, all synapses (+ = evaluation synapse)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confinement.png"), dpi=130)
    plt.close(fig)


def relative_effects(df, scenarios=("half",), metrics=("uptake_cum", "peak_deficit")):
    """Local-run effect divided by the uniform-run effect on the SAME synapse (same dose,
    kinetics, stimulus): removes synapse identity, so buffered and unbuffered synapses
    become comparable. Rows of the local runs with rel_<m> added."""
    key = ["seed", "stimulus", "active", "ecm_kf", "ecm_kd", "ecm_total", "synapse"]
    uni = df[df.scenario == "uniform"].drop_duplicates(key).set_index(key)
    # the Step 3 sweeps repeat the core grid point: keep one copy of each run
    loc = df[df.scenario.isin(scenarios)].drop_duplicates(key + ["scenario", "local_width"]).copy()
    idx = pd.MultiIndex.from_frame(loc[key])
    for m in metrics:
        loc[f"rel_{m}"] = loc[f"effect_{m}"].to_numpy() / uni[f"effect_{m}"].reindex(idx).to_numpy()
    return loc


def fig_placement(df, out):
    """On- vs off-target at the same amount: relative effect vs distance to the nearest
    buffered synapse, all synapses of the half-buffered core runs."""
    sub = relative_effects(core(df))
    doses = sorted(sub.ecm_total.unique())[-2:]
    metrics = ["uptake_cum", "peak_deficit"]
    fig, axes = plt.subplots(len(metrics), len(doses), figsize=(4 * len(doses), 3 * len(metrics)),
                             squeeze=False, sharex=True)
    for j, dose in enumerate(doses):
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            for width, g in sub[sub.ecm_total == dose].groupby("local_width"):
                ax.plot(g.d_buffered, g[f"rel_{m}"], "o", ms=4, alpha=0.6, label=f"sigma={width:g} um")
                t = g[g.target_class.notna()]
                ax.plot(t.d_buffered, t[f"rel_{m}"], "k+", ms=9, mew=1.5)
            ax.axhline(1, color="0.5", lw=0.8, label="uniform")
            ax.axhline(0, color="0.5", lw=0.5, ls=":")
            if i == 0:
                ax.set_title(f"dose {dose:g} mM", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Δ{m} / Δ{m} uniform (same synapse)", fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("distance to nearest buffered synapse (um)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Placement: effect relative to the uniform buffer, half-buffered runs (+ = evaluation synapse)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "placement.png"), dpi=130)
    plt.close(fig)


def fig_robustness(df, out, axis, xlabel, logx=True):
    """Effect vs `axis` (ecm_kf / ecm_kd), one column per dose, uniform vs half-buffered."""
    other = {"ecm_kf": KD0, "ecm_kd": KF0}
    fixed = "ecm_kd" if axis == "ecm_kf" else "ecm_kf"
    sub = df[(df[fixed] == other[axis]) & (df.stimulus == STIM0)
             & df.scenario.isin(["uniform", "half"]) & (df.local_width.fillna(1.0) == 1.0)]
    doses = sorted(sub[sub[axis] != (KF0 if axis == "ecm_kf" else KD0)].ecm_total.unique())
    sub = sub[sub.ecm_total.isin(doses)]
    if sub.empty:
        return
    tg = targets(df)
    fig, axes = plt.subplots(len(METRICS), len(doses), figsize=(4.2 * len(doses), 2.6 * len(METRICS)),
                             squeeze=False, sharex=True)
    markers = "osD^"
    for j, dose in enumerate(doses):
        for i, m in enumerate(METRICS):
            ax = axes[i, j]
            for k, (s, cls) in enumerate(tg):
                g = sub[(sub.ecm_total == dose) & (sub.synapse == s)]
                for scen, gg in g.groupby("scenario"):
                    gg = gg.sort_values(axis)
                    buffered = bool(gg.buffered.iloc[0])
                    ax.plot(gg[axis], gg[f"effect_{m}"], "-" + markers[k], color=COLOR[scen], ms=4,
                            mfc=COLOR[scen] if buffered else "white",
                            label=f"{s} ({cls}), {scen}" + (", buffered" if buffered else ""))
            thr = g[f"thr_{m}"].iloc[0] if len(g) else np.nan
            if np.isfinite(thr):
                ax.axhspan(-thr, thr, color="0.85", zorder=0)
            ax.axhline(0, color="0.5", lw=0.5)
            if logx:
                ax.set_xscale("log")
            if i == 0:
                ax.set_title(f"dose {dose:g} mM", fontsize=10)
            if j == 0:
                ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
            if i == len(METRICS) - 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"robustness_{axis}.png"), dpi=130)
    plt.close(fig)


def fig_stimulus(df, out):
    sub = df[(df.stimulus != STIM0) & (df.scenario != "reference")]
    if sub.empty:
        return
    tg = targets(df)
    metrics = ["uptake_1", "peak_deficit", "frac_at_floor", "recovery_half"]
    fig, axes = plt.subplots(len(metrics), len(tg), figsize=(3.2 * len(tg), 2.6 * len(metrics)),
                             squeeze=False, sharex=True)
    for j, (s, cls) in enumerate(tg):
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            plot_lines(ax, sub[sub.synapse == s], "ecm_total", m)
            if i == 0:
                ax.set_title(f"synapse {s} ({cls})", fontsize=10)
            if j == 0:
                ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("buffer dose (mM)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Sub-saturating stimulus ({sub.stimulus.iloc[0]})")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "robustness_stimulus.png"), dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--out", default="plots")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "active" not in df:
        df["active"] = "all"
    fig_dose_response(df, args.out)
    fig_equivalent_dose(df, args.out)
    fig_confinement(df, args.out)
    fig_placement(df, args.out)
    fig_robustness(df, args.out, "ecm_kf", "k_on (1/(mM s))")
    fig_robustness(df, args.out, "ecm_kd", "K_d (mM)")
    fig_stimulus(df, args.out)
    print(f"plots -> {args.out}: {sorted(os.listdir(args.out))}")


if __name__ == "__main__":
    main()
