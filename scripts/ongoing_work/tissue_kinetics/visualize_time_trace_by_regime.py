"""Plot local ECS [Ca] over time for synapse sub-populations binned by ECS regime.

This repeats the visualize_time_trace.py analysis (a species concentration in ECS
over time, with mean/median lines and a shaded quantile band) but the population is
no longer every ECS vertex of a single simulation. Instead it is the set of
synapses whose *local* ECS volume fraction at radius RADIUS falls into a given
regime, pooled over all seeds of the configured sweeps.

The local ECS fraction is ``v_local_r<R> / v_sphere_box_r<R>`` as written by
evaluate_synapse_distribution_spatial.py into ``<sweep>/spatial_metrics.csv`` (so
that CSV must have been produced with RADIUS among its --radii). Each selected
synapse contributes its per-timestep local-ECS Ca trace -- the point value at the
nearest ECS vertex, via evaluate_ecs_ratio.compute_local_ca -- and the (seed,
synapse_idx) key joins the two pipelines, both of which order synapses by
find_synapse_centers.

For each regime we plot the mean (solid) and median (dashed) over the selected
synapses, with the CENTILE..(100-CENTILE)% range shaded.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ecs_ratio import compute_local_ca
from evaluate_synapse_distribution_spatial import find_seed_dirs

# ============ Configuration ============
SWEEP_DIRS = [
    "results/synapse_distribution_ecs_10_2026-05-29-174131",
    "results/synapse_distribution_ecs_25_2026-05-29-185441",
]
CSV_NAME = "spatial_metrics.csv"   # written by evaluate_synapse_distribution_spatial.py
SPECIES_NAME = "Ca"                # species to plot
CENTILE = 5                        # shaded band is CENTILE..(100-CENTILE)%
RADIUS = 0.4                       # um; selects the v_local_r<R>/v_sphere_box_r<R> columns
# Synapse regimes on the local ECS volume fraction: (label, center, half-width).
REGIMES = [
    ("7% ECS", 0.07, 0.02),
    ("25% ECS", 0.25, 0.02),
]
OUT_PATH = "results/time_trace_by_regime.png"
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


def main():
    df = load_metrics(SWEEP_DIRS, CSV_NAME)
    df["vfrac"] = _vfrac(df, RADIUS)
    n_valid = int(df["vfrac"].notna().sum())
    print(f"Loaded {len(df)} synapses ({n_valid} with finite r={RADIUS:g} fraction) "
          f"from {len(SWEEP_DIRS)} sweeps")

    provider = TraceProvider()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (label, center, half) in enumerate(REGIMES):
        lo, hi = center - half, center + half
        sel = df[(df["vfrac"] >= lo) & (df["vfrac"] <= hi)]
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
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
