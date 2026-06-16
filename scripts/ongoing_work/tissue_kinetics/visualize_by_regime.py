"""Compare two ECS regimes for synapse sub-populations: depletion-depth or time trace.

Both analyses bin synapses by their *local* ECS volume fraction at radius RADIUS
(``v_local_r<R> / v_sphere_box_r<R>``, written by
evaluate_synapse_distribution_spatial.py into ``<sweep>/spatial_metrics.csv``) into
the regimes in REGIMES, pooled over all seeds of the configured sweeps. (That CSV
must therefore have been produced with RADIUS among its --radii.)

Two plot modes (``--plot``):

* ``min-ca`` (default): per-synapse depletion depth -- the ``min_ca`` column
  (= local_ca.min over time at the nearest ECS vertex) -- as a Gaussian KDE per
  regime, with a light histogram behind it. The two-sided Mann-Whitney U test
  (non-parametric, no normality assumption) is the primary test for a location
  shift; the two-sample Kolmogorov-Smirnov test is reported as a whole-distribution
  check. Effect size is the common-language statistic
  P(min_ca[regime 1] > min_ca[regime 2]). Negative depletion depths (solver
  undershoot in the tightest synapses) are clipped to zero before any stats / KDE.

* ``time-trace``: repeats the visualize_time_trace.py analysis but over the binned
  synapse sub-populations. Each selected synapse contributes its per-timestep
  local-ECS Ca trace -- the point value at the nearest ECS vertex, via
  evaluate_ecs_ratio.compute_local_ca; the (seed, synapse_idx) key joins the two
  pipelines, both of which order synapses by find_synapse_centers. Per regime we
  plot the mean (solid) and median (dashed) over the selected synapses, with the
  CENTILE..(100-CENTILE)% range shaded.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ecs_ratio import compute_local_ca
from evaluate_synapse_distribution_spatial import find_seed_dirs

# ============ Configuration ============
SWEEP_DIRS = [
    "results/synapse_distribution_ecs_10_2026-05-29-174131",
    "results/synapse_distribution_ecs_25_2026-05-29-185441",
]
CSV_NAME = "spatial_metrics.csv"   # written by evaluate_synapse_distribution_spatial.py
SPECIES_NAME = "Ca"                # species to plot (time-trace mode)
CENTILE = 5                        # shaded band is CENTILE..(100-CENTILE)% (time-trace mode)
RADIUS = 0.4                       # um; selects the v_local_r<R>/v_sphere_box_r<R> columns
# Synapse regimes on the local ECS volume fraction: (label, center, half-width).
REGIMES = [
    ("7% ECS", 0.07, 0.02),
    ("25% ECS", 0.25, 0.02),
]
OUT_PATHS = {                      # default output per --plot mode
    "min-ca": "results/min_ca_by_regime.png",
    "time-trace": "results/time_trace_by_regime.png",
}
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


def select_regime(df, center, half):
    """Rows whose local ECS fraction (column 'vfrac') falls within center +/- half."""
    lo, hi = center - half, center + half
    return lo, hi, df[(df["vfrac"] >= lo) & (df["vfrac"] <= hi)]


# ---------------------------------------------------------------------------
# min-ca mode: depletion-depth distribution + significance
# ---------------------------------------------------------------------------
def regime_min_ca(df):
    """Return [(label, min_ca_array), ...], one entry per configured regime."""
    out = []
    for label, center, half in REGIMES:
        lo, hi, sel = select_regime(df, center, half)
        vals = sel["min_ca"].to_numpy()
        vals = vals[np.isfinite(vals)]
        # Clip unphysical negative depletion depths (solver undershoot in the
        # tightest synapses) to zero before any stats / KDE.
        vals = np.clip(vals, 0.0, None)
        print(f"  {label}: fraction in [{lo:.0%}, {hi:.0%}] -> {len(vals)} synapses, "
              f"min_ca median={np.median(vals):.3f} mean={np.mean(vals):.3f} mM")
        out.append((label, vals))
    return out


def plot_min_ca(df, ax):
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


# ---------------------------------------------------------------------------
# time-trace mode: pooled local-Ca trace per regime
# ---------------------------------------------------------------------------
class TraceProvider:
    """Lazily map (sweep, seed) -> local-Ca traces, caching per seed."""

    def __init__(self):
        self._seed_paths = {}   # sweep -> {seed_int: path}
        self._traces = {}       # (sweep, seed) -> (times, local_ca)

    def _path(self, sweep, seed):
        if sweep not in self._seed_paths:
            self._seed_paths[sweep] = dict(find_seed_dirs(sweep))
        return self._seed_paths[sweep][seed]

    def get(self, sweep, seed):
        key = (sweep, seed)
        if key not in self._traces:
            self._traces[key] = compute_local_ca(self._path(sweep, seed),
                                                  species=SPECIES_NAME)
        return self._traces[key]


def collect_regime_traces(df_regime, provider):
    """Pool local-Ca traces for the selected synapses into (n_times, n_synapses).

    Groups by (sweep, seed) so compute_local_ca runs once per seed, then picks
    the selected synapse columns. Returns (times, traces).
    """
    times_ref = None
    columns = []
    for (sweep, seed), grp in df_regime.groupby(["sweep_dir", "seed"]):
        times, local_ca = provider.get(sweep, int(seed))
        if times_ref is None:
            times_ref = times
        elif len(times) != len(times_ref):
            raise SystemExit(
                f"Error: seed {seed} in {sweep} has {len(times)} timesteps, "
                f"expected {len(times_ref)}; cannot pool traces.")
        idx = grp["synapse_idx"].to_numpy()
        columns.append(local_ca[:, idx])
    if not columns:
        return times_ref, np.empty((0, 0))
    return times_ref, np.concatenate(columns, axis=1)


def plot_time_trace(df, ax):
    provider = TraceProvider()
    colors = plt.cm.tab10.colors

    for i, (label, center, half) in enumerate(REGIMES):
        lo, hi, sel = select_regime(df, center, half)
        print(f"  {label}: fraction in [{lo:.0%}, {hi:.0%}] -> {len(sel)} synapses")
        if sel.empty:
            continue

        times, traces = collect_regime_traces(sel, provider)
        lows = np.nanquantile(traces, CENTILE / 100.0, axis=1)
        medians = np.nanmedian(traces, axis=1)
        means = np.nanmean(traces, axis=1)
        highs = np.nanquantile(traces, 1 - CENTILE / 100.0, axis=1)

        color = colors[i % len(colors)]
        ax.fill_between(times, lows, highs, alpha=0.2, color=color)
        ax.plot(times, means, linewidth=2, linestyle="-", color=color,
                label=f"{label} mean (n={traces.shape[1]})")
        ax.plot(times, medians, linewidth=2, linestyle="--", color=color,
                label=f"{label} median")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"Local ECS [{SPECIES_NAME}] near synapses (mM)")
    ax.set_title(
        f"Local [{SPECIES_NAME}] at synapses by ECS regime "
        f"(r={RADIUS:g} um, {CENTILE}-{100 - CENTILE}% range shaded)")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
PLOTTERS = {"min-ca": plot_min_ca, "time-trace": plot_time_trace}


def main(plot_kind, out_path, show):
    df = load_metrics(SWEEP_DIRS, CSV_NAME)
    df["vfrac"] = _vfrac(df, RADIUS)
    n_valid = int(df["vfrac"].notna().sum())
    print(f"Loaded {len(df)} synapses ({n_valid} with finite r={RADIUS:g} fraction) "
          f"from {len(SWEEP_DIRS)} sweeps")

    fig, ax = plt.subplots(figsize=(10, 6))
    PLOTTERS[plot_kind](df, ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--plot", choices=list(PLOTTERS), default="min-ca",
        help="Which analysis to plot (default: min-ca).")
    parser.add_argument(
        "--out", default=None,
        help="Output PNG path (default: per-mode entry in OUT_PATHS).")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure without opening an interactive window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = args.out or OUT_PATHS[args.plot]
    main(args.plot, out, SHOW and not args.no_show)
