"""Compare per-synapse minimum [Ca] between two ECS regimes (KDE + significance).

For each synapse the depletion depth is the lowest point of its local-ECS Ca trace
over time -- the ``min_ca`` column written by evaluate_synapse_distribution_spatial.py
(= local_ca.min over time at the nearest ECS vertex). This script bins synapses by
their *local* ECS volume fraction at radius RADIUS
(``v_local_r<R> / v_sphere_box_r<R>``) into two regimes, then plots the min_ca
distribution of each regime as a Gaussian KDE (with a light histogram behind it)
and tests whether the two distributions differ.

The two-sided Mann-Whitney U test (non-parametric, no normality assumption) is the
primary test for a location shift; the two-sample Kolmogorov-Smirnov test is also
reported as a whole-distribution check. Effect size is the common-language
statistic P(min_ca[regime 1] > min_ca[regime 2]).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

# ============ Configuration ============
SWEEP_DIRS = [
    "results/synapse_distribution_ecs_10_2026-05-29-174131",
    "results/synapse_distribution_ecs_25_2026-05-29-185441",
]
CSV_NAME = "spatial_metrics.csv"   # written by evaluate_synapse_distribution_spatial.py
RADIUS = 0.4                       # um; selects the v_local_r<R>/v_sphere_box_r<R> columns
# Synapse regimes on the local ECS volume fraction: (label, center, half-width).
REGIMES = [
    ("7% ECS", 0.07, 0.02),
    ("25% ECS", 0.25, 0.02),
]
OUT_PATH = "results/min_ca_by_regime.png"
SHOW = True
# =======================================


def _vfrac(df, radius):
    """Local ECS volume fraction at `radius`: v_local / v_sphere_box."""
    vcol = f"v_local_r{radius:g}"
    bcol = f"v_sphere_box_r{radius:g}"
    for col in (vcol, bcol):
        if col not in df.columns:
            raise SystemExit(
                f"Error: column '{col}' missing from spatial_metrics CSV. "
                f"Re-run evaluate_synapse_distribution_spatial.py with "
                f"--radii including {radius:g}.")
    denom = df[bcol].replace(0.0, np.nan)
    return df[vcol] / denom


def load_metrics(sweep_dirs, csv_name):
    """Concatenate the per-sweep spatial_metrics CSVs, tagged by sweep dir."""
    frames = []
    for sweep in sweep_dirs:
        csv_path = os.path.join(sweep, csv_name)
        if not os.path.isfile(csv_path):
            raise SystemExit(
                f"Error: '{csv_path}' not found. Run "
                f"evaluate_synapse_distribution_spatial.py on '{sweep}' first.")
        df = pd.read_csv(csv_path)
        df["sweep_dir"] = sweep
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def regime_min_ca(df):
    """Return [(label, min_ca_array), ...], one entry per configured regime."""
    out = []
    for label, center, half in REGIMES:
        lo, hi = center - half, center + half
        sel = df[(df["vfrac"] >= lo) & (df["vfrac"] <= hi)]
        vals = sel["min_ca"].to_numpy()
        vals = vals[np.isfinite(vals)]
        # Clip unphysical negative depletion depths (solver undershoot in the
        # tightest synapses) to zero before any stats / KDE.
        vals = np.clip(vals, 0.0, None)
        print(f"  {label}: fraction in [{lo:.0%}, {hi:.0%}] -> {len(vals)} synapses, "
              f"min_ca median={np.median(vals):.3f} mean={np.mean(vals):.3f} mM")
        out.append((label, vals))
    return out


def main():
    df = load_metrics(SWEEP_DIRS, CSV_NAME)
    df["vfrac"] = _vfrac(df, RADIUS)
    print(f"Loaded {len(df)} synapses from {len(SWEEP_DIRS)} sweeps")

    groups = regime_min_ca(df)
    if len(groups) != 2:
        raise SystemExit("This analysis expects exactly two regimes.")
    (label_a, a), (label_b, b) = groups
    if len(a) < 2 or len(b) < 2:
        raise SystemExit("Each regime needs >= 2 synapses for a KDE / test.")

    # Significance: Mann-Whitney U (location shift) + KS (whole distribution).
    u_stat, p_mwu = mannwhitneyu(a, b, alternative="two-sided")
    ks_stat, p_ks = ks_2samp(a, b)
    cles = u_stat / (len(a) * len(b))  # P(a > b), ties counted as 0.5
    print(f"\nMann-Whitney U: U={u_stat:.0f}, p={p_mwu:.3e}")
    print(f"Kolmogorov-Smirnov: D={ks_stat:.3f}, p={p_ks:.3e}")
    print(f"Common-language effect size P({label_a} > {label_b}) = {cles:.3f}")

    # KDE over a shared grid spanning both samples (with a small margin).
    pooled = np.concatenate([a, b])
    margin = 0.05 * (pooled.max() - pooled.min())
    grid = np.linspace(pooled.min() - margin, pooled.max() + margin, 512)
    bins = np.linspace(pooled.min(), pooled.max(), 30)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, (label, vals) in enumerate(groups):
        color = colors[i % len(colors)]
        kde = gaussian_kde(vals)
        density = kde(grid)
        ax.hist(vals, bins=bins, density=True, alpha=0.2, color=color)
        ax.fill_between(grid, density, alpha=0.25, color=color)
        ax.plot(grid, density, linewidth=2, color=color,
                label=f"{label} (n={len(vals)}, median={np.median(vals):.2f} mM)")
        ax.axvline(np.median(vals), color=color, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Per-synapse minimum [Ca] (depletion depth, mM)")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"Depletion depth by ECS regime (local ECS fraction at r={RADIUS:g} um)")
    txt = (f"Mann-Whitney U: p = {p_mwu:.2e}\n"
           f"KS test: p = {p_ks:.2e}\n"
           f"P({label_a} > {label_b}) = {cles:.2f}")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nWrote {OUT_PATH}")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
