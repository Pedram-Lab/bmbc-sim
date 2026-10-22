"""Plots for the buffer dose-response study, from evaluate_dose_response.py's CSV.

    uv run scripts/ongoing_work/tissue_kinetics/plot_dose_response.py dose_response.csv --out plots/

dose_response.png     effect vs dose per evaluation synapse, uniform vs local buffer
local_dose.png        effect vs buffer concentration at the synapse, all synapses, uniform as reference line
relative_effect.png   effect(local)/effect(uniform) vs local concentration/dose with power-law fit (+ .csv)
uptake_vs_deficit.png the two verdict metrics against each other, raw and as effects
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
MARKER = {0.5: "^", 1.0: "o"}
KF0, KD0, STIM0 = 1e5, 1.0, "5p_0.5pA"


def core(df):
    """Rows with the default kinetics and stimulus (steps 1/2)."""
    sub = df[(df.ecm_kf == KF0) & (df.ecm_kd == KD0) & (df.stimulus == STIM0) & (df.active == "all")]
    # the k_on / K_d sweeps repeat the default kinetics: one row per (scenario, dose, sigma, synapse)
    return sub.drop_duplicates(["scenario", "ecm_total", "local_width", "synapse"])


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


def local_dose(sub):
    """Buffer concentration at the synapse: the dose itself for the uniform runs."""
    return np.where(sub.scenario == "uniform", sub.ecm_total, sub.ecm_at_synapse)


def fig_local_dose(df, out, xmin=0.01):
    """Effect vs buffer concentration at the synapse, all synapses; uniform as median/IQR line."""
    sub = core(df)
    sub = sub[sub.scenario != "reference"].assign(local=local_dose)
    metrics = ["uptake_cum", "peak_deficit"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.8))
    for ax, m in zip(axes, metrics):
        for (scen, width), g in sub.groupby(["scenario", sub.local_width.fillna(0)]):
            if scen == "uniform":
                st = g.groupby("ecm_total")[f"effect_{m}"].quantile([0.25, 0.5, 0.75]).unstack()
                ax.plot(st.index, st[0.5], "k-", lw=1.5, label="uniform (median, IQR over synapses)")
                ax.fill_between(st.index, st[0.25], st[0.75], color="k", alpha=0.12)
                continue
            lo = g.local < xmin
            ax.plot(g.local[~lo], g[f"effect_{m}"][~lo], MARKER[width], color=COLOR[scen], ms=4,
                    alpha=0.5, mec="none", label=series_label(scen, width))
            ax.plot(np.full(lo.sum(), 0.7 * xmin), g[f"effect_{m}"][lo], "<", color=COLOR[scen],
                    ms=4, alpha=0.3, mec="none")
        thr = sub[f"thr_{m}"].iloc[0]
        if np.isfinite(thr):
            ax.axhspan(-thr, thr, color="0.88", zorder=0)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(f"buffer concentration at the synapse (mM);  <: below {xmin:g} mM")
        ax.set_ylabel("Δ " + LABEL[m], fontsize=9)
    axes[0].legend(fontsize=6)
    fig.suptitle("Effect vs buffer concentration at the synapse, all synapses and doses")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "local_dose.png"), dpi=130)
    plt.close(fig)


def fig_relative_effect(df, out, min_dose=1.0):
    """effect(local) / effect(uniform) vs local concentration / dose, log-log, with a power-law fit.

    Slope 1 = the effect follows the local concentration, 0 = only the total amount counts.
    Doses below `min_dose` are left out because the uniform effect is not resolved there."""
    sub = core(df)
    sub = sub[sub.scenario != "reference"].assign(local=local_dose)
    uni = sub[sub.scenario == "uniform"].set_index(["ecm_total", "synapse"])
    loc = sub[(sub.scenario != "uniform") & (sub.ecm_total >= min_dose)]
    key = pd.MultiIndex.from_frame(loc[["ecm_total", "synapse"]])
    metrics = ["uptake_cum", "peak_deficit"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    fits = []
    for ax, m in zip(axes, metrics):
        rel = loc[f"effect_{m}"].to_numpy() / uni[f"effect_{m}"].reindex(key).to_numpy()
        cf = (loc.local / loc.ecm_total).to_numpy()
        ok = np.isfinite(rel) & (rel > 0) & (cf > 1e-3)
        for (scen, width), idx in loc.groupby(["scenario", "local_width"]).groups.items():
            sel = loc.index.isin(idx) & ok
            sc = ax.scatter(cf[sel], rel[sel], marker=MARKER[width], c=np.log10(loc.ecm_total.to_numpy()[sel]),
                            cmap="viridis", vmin=np.log10(min_dose), vmax=np.log10(loc.ecm_total.max()),
                            edgecolor=COLOR[scen], lw=0.6, s=22, alpha=0.8, label=series_label(scen, width))
        slope, icpt = np.polyfit(np.log(cf[ok]), np.log(rel[ok]), 1)
        fits.append(dict(metric=m, slope=slope, prefactor=np.exp(icpt), n=int(ok.sum()), min_dose=min_dose))
        xx = np.geomspace(1e-3, 30, 50)
        ax.plot(xx, np.exp(icpt) * xx ** slope, "k--", lw=1, label=f"fit: slope {slope:.2f}")
        ax.plot(xx, xx, "k:", lw=1, label="effect ∝ local conc. (slope 1)")
        ax.axhline(1, color="0.5", lw=1, label="same as uniform")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(1e-3, 30), ax.set_ylim(1e-2, 30)
        ax.set_xlabel("buffer at the synapse / dose N")
        ax.set_ylabel(f"effect(local) / effect(uniform), {m}", fontsize=8)
        ax.text(0.03, 0.95, f"n = {ok.sum()} (synapse, run) pairs", transform=ax.transAxes, fontsize=7, va="top")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.colorbar(sc, ax=axes, label="log10 dose N (mM)", shrink=0.8)
    fig.suptitle(f"Does the local concentration buy a proportional effect? all synapses, doses >= {min_dose:g} mM")
    fig.savefig(os.path.join(out, "relative_effect.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(fits).to_csv(os.path.join(out, "relative_effect.csv"), index=False)


def fig_uptake_vs_deficit(df, out):
    """Do the two verdict metrics agree? uptake_cum vs peak_deficit, raw values (left) and effects (right)."""
    sub = core(df)  # reference rows carry the config-default kinetics, so core() drops them
    ref = df[(df.scenario == "reference") & (df.stimulus == STIM0) & (df.active == "all")]
    sub = pd.concat([sub, ref])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, pre in zip(axes, ["", "effect_"]):
        for (scen, width), g in sub.groupby(["scenario", sub.local_width.fillna(0)]):
            if scen == "reference":
                if pre:
                    continue
                ax.plot(g.peak_deficit, g.uptake_cum, "s", color="0.6", ms=4, mec="none", label="no buffer")
                continue
            ax.plot(g[f"{pre}peak_deficit"], g[f"{pre}uptake_cum"], MARKER[width] if width else "o",
                    color=COLOR[scen], ms=4, alpha=0.5, mec="none", label=series_label(scen, width))
        if pre:
            thr_x, thr_y = sub.thr_peak_deficit.iloc[0], sub.thr_uptake_cum.iloc[0]
            ax.axvspan(-thr_x, thr_x, color="0.88", zorder=0), ax.axhspan(-thr_y, thr_y, color="0.88", zorder=0)
            ax.axhline(0, color="0.5", lw=0.5), ax.axvline(0, color="0.5", lw=0.5)
        ax.set_xlabel(("Δ " if pre else "") + LABEL["peak_deficit"])
        ax.set_ylabel(("Δ " if pre else "") + LABEL["uptake_cum"])
    axes[0].legend(fontsize=6)
    axes[0].set_title("per synapse and run"), axes[1].set_title("effects (gray: IQR thresholds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "uptake_vs_deficit.png"), dpi=130)
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
    df = df[df.active == "all"]  # solitary-synapse runs: plot_single_synapse.py
    fig_dose_response(df, args.out)
    fig_local_dose(df, args.out)
    fig_relative_effect(df, args.out)
    fig_uptake_vs_deficit(df, args.out)
    fig_placement(df, args.out)
    fig_robustness(df, args.out, "ecm_kf", "k_on (1/(mM s))")
    fig_robustness(df, args.out, "ecm_kd", "K_d (mM)")
    fig_stimulus(df, args.out)
    print(f"plots -> {args.out}: {sorted(os.listdir(args.out))}")


if __name__ == "__main__":
    main()
