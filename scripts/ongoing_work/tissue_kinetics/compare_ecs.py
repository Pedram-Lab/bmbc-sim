"""ECS-ratio comparison of the dose-response study: the same synapses (the cell set is
cropped at reference_ecs_ratio, so indices match) at two ECS volume fractions.

    uv run scripts/ongoing_work/tissue_kinetics/compare_ecs.py a/dose_response.csv b/dose_response.csv --out plots/

* ecs_compare.png   crowd: per-synapse uptake effect at the largest dose, ecs A vs ecs B;
                    solitary: effect at the active synapse per placement at both ratios
* ecs_compare.csv   summary table (also returned by ``summary``)
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_dose_response import KD0, KF0, STIM0  # noqa: E402
from plot_single_synapse import PLACEMENT_COLOR  # noqa: E402


def load(csv, ecs=None):
    df = pd.read_csv(csv)
    if "ecs_ratio" not in df:  # CSVs written before the column existed
        df["ecs_ratio"] = ecs
    return df


def crowd(df):
    d = df[(df.active == "all") & (df.stimulus == STIM0)]
    return d[(d.scenario == "reference") | ((d.ecm_kf == KF0) & (d.ecm_kd == KD0) & (d.local_width.fillna(1) == 1))]


def solitary(df):
    d = df[(df.active != "all") & df.is_active]
    return d[(d.scenario == "reference") | ((d.ecm_kf == KF0) & (d.ecm_kd == KD0) & (d.ecm_total == d.ecm_total.max()))]


def summary(df):
    """One row per ECS ratio: pool depletion, uniform effect, placement gain (crowd) and the
    solitary on-target/uniform ratio, relative uptake change and shielding."""
    rows = []
    for ecs, g in df.groupby("ecs_ratio"):
        c, s = crowd(g), solitary(g)
        dose = c.ecm_total.max()
        ref = c[c.scenario == "reference"]
        half = c[(c.scenario == "half") & (c.ecm_total == dose)].set_index("synapse")
        uni = c[(c.scenario == "uniform") & (c.ecm_total == dose)].set_index("synapse")
        rel = (half.effect_uptake_cum / uni.effect_uptake_cum.reindex(half.index))
        sref = s[s.scenario == "reference"].set_index("synapse")
        s_uni = s[s.scenario == "uniform"].set_index("synapse")
        sigma = s[s.scenario == "on"].local_width.max()
        s_on = s[(s.scenario == "on") & (s.local_width == sigma)].set_index("synapse")
        ratio = s_on.effect_uptake_cum / s_uni.effect_uptake_cum.reindex(s_on.index)
        pct = 100 * s_on.effect_uptake_cum / sref.uptake_cum.reindex(s_on.index)
        sil = df[(df.ecs_ratio == ecs) & (df.active != "all") & ~df.is_active & (df.d_active <= 3)
                 & (df.ecm_total == s.ecm_total.max()) & (df.ecm_kf == KF0) & (df.local_width.fillna(0) != 1.0)]
        shield = sil.groupby("scenario").effect_peak_deficit.median()
        rows.append({
            "ecs_ratio": ecs,
            "crowd: ref peak deficit (mM, median)": ref.peak_deficit.median(),
            "crowd: ref IQR uptake (amol)": ref.thr_uptake_cum.iloc[0],
            f"crowd: uniform {dose:g} mM uptake effect (median)": uni.effect_uptake_cum.median(),
            "crowd: placement on-target / uniform": rel[half.buffered].median(),
            "crowd: placement off-target / uniform": rel[~half.buffered].median(),
            "solitary: sigma* (um)": sigma,
            "solitary: ref peak deficit (mM)": f"{sref.peak_deficit.min():.2f}-{sref.peak_deficit.max():.2f}",
            "solitary: on/uniform uptake effect": f"{ratio.min():.0f}-{ratio.max():.0f}",
            "solitary: on-target uptake % ref": f"{pct.min():.1f}-{pct.max():.1f}",
            "solitary: shielding <3 um, on (mM)": shield.get("on", np.nan),
            "solitary: shielding <3 um, uniform (mM)": shield.get("uniform", np.nan),
        })
    return pd.DataFrame(rows)


def fig(df, out):
    ecs = sorted(df.ecs_ratio.unique())
    f, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    c = crowd(df)
    dose = c.ecm_total.max()
    for scen, color in (("uniform", "k"), ("half", "tab:red")):
        w = c[(c.scenario == scen) & (c.ecm_total == dose)].pivot(index="synapse", columns="ecs_ratio",
                                                                  values="effect_uptake_cum").dropna()
        if len(ecs) == 2 and not w.empty:
            ax.plot(w[ecs[0]], w[ecs[1]], "o", ms=4, alpha=0.7, color=color, label=scen)
    lim = ax.get_xlim()
    ax.plot(lim, lim, "--", color="0.6", lw=0.8)
    ax.set_xlabel(f"Δ uptake at ecs {ecs[0]} (amol)")
    ax.set_ylabel(f"Δ uptake at ecs {ecs[-1]} (amol)")
    ax.set_title(f"Crowd, {dose:g} mM: per synapse", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    s = solitary(df)
    s = s[s.scenario != "reference"]
    syns = sorted(s.synapse.unique())
    x = np.arange(len(syns))
    for k, e in enumerate(ecs):
        g = s[s.ecs_ratio == e]
        off = (k - (len(ecs) - 1) / 2) * 0.3
        for (scen, w), gg in g.groupby(["scenario", g.local_width.fillna(0)]):
            sigma_star = gg.local_width.max() != 1.0 or scen == "uniform"
            y = gg.set_index("synapse").effect_uptake_cum.reindex(syns)
            ax.plot(x + off, y, "o" if sigma_star else "s", color=PLACEMENT_COLOR[scen], ms=5,
                    mfc=PLACEMENT_COLOR[scen] if k else "none",
                    label=f"ecs {e}: {scen}" + ("" if scen == "uniform" else f" σ={w:g}"))
    ax.set_xticks(x, [f"synapse {s}" for s in syns], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Δ uptake at the active synapse (amol)")
    ax.set_title(f"Solitary, {s.ecm_total.max():g} mM (open = ecs {ecs[0]}, filled = ecs {ecs[-1]})", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    f.suptitle("ECS ratio comparison, same synapses")
    f.tight_layout()
    f.savefig(os.path.join(out, "ecs_compare.png"), dpi=130)
    plt.close(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs=2)
    p.add_argument("--ecs", nargs=2, type=float, default=[0.1, 0.19],
                   help="ECS ratios of the two CSVs, used only where the CSV lacks the column")
    p.add_argument("--out", default="plots")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.concat([load(c, e) for c, e in zip(args.csvs, args.ecs)])
    fig(df, args.out)
    t = summary(df)
    t.to_csv(os.path.join(args.out, "ecs_compare.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.T.to_string())


if __name__ == "__main__":
    main()
