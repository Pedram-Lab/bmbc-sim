"""Plot the per-synapse difference in local ECS Ca between two ECS ratios over time.

The two configured sweeps run the SAME synapses (same cell set, same seeded
placement) at two different ECS ratios -- this is what the reference-shrink fix in
simulation.py guarantees, so (seed, synapse_idx) identifies one physical synapse in
both sweeps (verified: idx-matched centers are the spatial nearest neighbour ~98% of
the time, median displacement ~0.2 um from the cell-size scaling alone).

For every shared (seed, synapse_idx) we form the trace

    d[Ca](t) = [Ca]_HIGH(t) - [Ca]_LOW(t)

at the synapse's nearest ECS vertex (via analysis.compute_local_ca, whose
synapse ordering matches across sweeps). Pairs where either trace dips negative
anywhere (solver undershoot in the tightest synapses) are dropped outright. Two plot modes (``--plot``):

* ``trace`` (default): pooling over the surviving synapses and seeds, the mean (solid)
  and median (dashed) difference with the CENTILE..(100-CENTILE)% range shaded, plus a
  faint subsample of the individual difference traces.

* ``vs-ecs``: one point per synapse -- its peak difference (the signed d[Ca] at the
  time of largest |d[Ca]|) against how much local ECS that synapse GAINED between the
  two runs, d_vfrac = vfrac_HIGH - vfrac_LOW, where vfrac is the local ECS volume
  fraction at radius RADIUS (``v_local_r<R> / v_sphere_box_r<R>`` from
  ``<sweep>/spatial_metrics.csv``, written by evaluate_synapse_distribution_spatial.py;
  both sweeps need that CSV with RADIUS among its --radii). A binned median over
  deciles of d_vfrac is drawn on top.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import compute_local_ca
from evaluate_synapse_distribution_spatial import find_seed_dirs
from visualize_by_regime import _vfrac

# ============ Configuration ============
# (sweep dir, label). HIGH - LOW is the plotted difference; the suffix in each dir
# name is int(100*(ecs_ratio+0.06)), i.e. ecs_ratio 0.04 -> "10", 0.19 -> "25".
LOW = ("results/synapse_distribution_ecs_10_2026-06-17-142107", "10% ECS")
HIGH = ("results/synapse_distribution_ecs_25_2026-06-17-142107", "25% ECS")
SPECIES_NAME = "Ca"
CENTILE = 5                  # shaded band is CENTILE..(100-CENTILE)%
SHOW_INDIVIDUAL = True       # overlay a faint subsample of per-synapse traces
MAX_INDIVIDUAL = 400         # cap on how many individual traces to draw
RADIUS = 0.4                 # um; selects the v_local_r<R>/v_sphere_box_r<R> columns
N_BINS = 10                  # quantile bins for the binned median (vs-ecs)
OUT_PATHS = {                # default output per --plot mode
    "trace": "results/ca_difference_by_synapse.png",
    "vs-ecs": "results/ca_difference_vs_ecs_volume.png",
}
SHOW = True
# =======================================


def collect_differences(low_sweep, high_sweep):
    """Pool d[Ca] = ca_high - ca_low over every shared (seed, synapse_idx).

    Returns (times, diffs, pair_seeds, pair_synapse_idx). diffs has shape
    (n_timesteps, n_pairs); pair_seeds/pair_synapse_idx (length n_pairs) give the
    originating seed and synapse index of each column, so a specific extreme value
    in diffs can be traced back to a (seed, synapse_idx) to investigate further.
    Seeds present in only one sweep, or with mismatched synapse count / timestep
    count, are skipped with a warning. Synapse pairs where either the LOW or HIGH
    trace dips negative anywhere (solver undershoot) are dropped before differencing.
    """
    low_paths = dict(find_seed_dirs(low_sweep))
    high_paths = dict(find_seed_dirs(high_sweep))
    common = sorted(set(low_paths) & set(high_paths))
    print(f"Seeds: low={len(low_paths)} high={len(high_paths)} common={len(common)}")

    times_ref = None
    columns = []
    pair_seeds = []
    pair_synapse_idx = []
    n_skipped = 0
    n_dropped = 0
    for seed in common:
        times_lo, ca_lo = compute_local_ca(low_paths[seed], species=SPECIES_NAME)
        times_hi, ca_hi = compute_local_ca(high_paths[seed], species=SPECIES_NAME)

        if ca_lo.shape != ca_hi.shape:
            print(f"  seed {seed}: shape mismatch low={ca_lo.shape} "
                  f"high={ca_hi.shape}; skipped")
            n_skipped += 1
            continue
        if times_ref is None:
            times_ref = times_hi
        elif len(times_hi) != len(times_ref):
            print(f"  seed {seed}: {len(times_hi)} timesteps, expected "
                  f"{len(times_ref)}; skipped")
            n_skipped += 1
            continue

        # Keep only pairs whose LOW and HIGH traces are both non-negative throughout.
        keep = (ca_lo.min(axis=0) >= 0.0) & (ca_hi.min(axis=0) >= 0.0)
        n_dropped += int((~keep).sum())
        columns.append((ca_hi - ca_lo)[:, keep])
        synapse_idx = np.where(keep)[0]
        pair_seeds.extend([seed] * len(synapse_idx))
        pair_synapse_idx.extend(synapse_idx.tolist())

    if not columns:
        raise SystemExit("No matched synapses found across the two sweeps.")
    diffs = np.concatenate(columns, axis=1)
    if diffs.shape[1] == 0:
        raise SystemExit("All synapse pairs dropped for negative values; nothing to plot.")
    print(f"Pooled {diffs.shape[1]} synapse pairs from {len(common) - n_skipped} "
          f"seeds ({n_skipped} seeds skipped; {n_dropped} pairs dropped for negatives)")
    return times_ref, diffs, np.array(pair_seeds), np.array(pair_synapse_idx)


def plot_differences(times, diffs, low_label, high_label, ax):
    lows = np.nanquantile(diffs, CENTILE / 100.0, axis=1)
    medians = np.nanmedian(diffs, axis=1)
    means = np.nanmean(diffs, axis=1)
    highs = np.nanquantile(diffs, 1 - CENTILE / 100.0, axis=1)

    color = plt.cm.tab10.colors[0]

    if SHOW_INDIVIDUAL:
        n = diffs.shape[1]
        if n > MAX_INDIVIDUAL:
            sample = np.random.default_rng(0).choice(n, MAX_INDIVIDUAL, replace=False)
        else:
            sample = np.arange(n)
        ax.plot(times, diffs[:, sample], color="gray", alpha=0.04, linewidth=0.5)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(times, lows, highs, alpha=0.2, color=color)
    ax.plot(times, means, linewidth=2, linestyle="-", color=color,
            label=f"mean (n={diffs.shape[1]})")
    ax.plot(times, medians, linewidth=2, linestyle="--", color=color, label="median")

    # Scale the y-axis to the shaded band, not any residual single-trace outliers.
    y_lo = min(0.0, float(np.min(lows)))
    y_hi = max(0.0, float(np.max(highs)))
    margin = 0.15 * (y_hi - y_lo)
    ax.set_ylim(y_lo - margin, y_hi + margin)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"d[{SPECIES_NAME}] = [{high_label}] - [{low_label}] (mM)")
    ax.set_title(
        f"Per-synapse local [{SPECIES_NAME}] difference between ECS ratios "
        f"({CENTILE}-{100 - CENTILE}% range shaded)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def peak_differences(times, diffs):
    """Per-synapse signed d[Ca] at its own time of largest |d[Ca]|, and that time."""
    t_idx = np.nanargmax(np.abs(diffs), axis=0)
    return diffs[t_idx, np.arange(diffs.shape[1])], times[t_idx]


def load_vfrac(sweep, pair_seeds, pair_synapse_idx):
    """Local ECS volume fraction at RADIUS for each (seed, synapse_idx) pair.

    Missing pairs (seeds the spatial evaluation skipped or never covered) come back
    as NaN, so the caller can just drop them.
    """
    csv_path = os.path.join(sweep, "spatial_metrics.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(
            f"Error: '{csv_path}' not found. Run\n"
            f"  uv run python evaluate_synapse_distribution_spatial.py {sweep} "
            f"--radii {RADIUS:g}\nfirst.")
    df = pd.read_csv(csv_path)
    df["vfrac"] = _vfrac(df, RADIUS)
    lookup = df.set_index(["seed", "synapse_idx"])["vfrac"]
    lookup = lookup[~lookup.index.duplicated()]
    return lookup.reindex(pd.MultiIndex.from_arrays(
        [pair_seeds, pair_synapse_idx])).to_numpy()


def plot_peak_vs_vfrac(times, diffs, pair_seeds, pair_synapse_idx,
                       low_label, high_label, ax):
    peaks, peak_times = peak_differences(times, diffs)
    # x axis is the per-synapse GAIN in local ECS between the two runs, not either
    # run's fraction on its own.
    vfrac = (load_vfrac(HIGH[0], pair_seeds, pair_synapse_idx)
             - load_vfrac(LOW[0], pair_seeds, pair_synapse_idx))

    keep = np.isfinite(vfrac) & np.isfinite(peaks)
    print(f"{keep.sum()} of {len(peaks)} synapse pairs have a finite r={RADIUS:g} "
          f"fraction in both sweeps")
    if keep.sum() < 2:
        raise SystemExit("Not enough synapses with a local ECS fraction to plot.")
    vfrac, peaks, peak_times = vfrac[keep], peaks[keep], peak_times[keep]

    color = plt.cm.tab10.colors[0]
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.scatter(vfrac, peaks, s=8, alpha=0.3, color=color, edgecolors="none")

    # Binned median over quantile bins of the fraction (equal counts per bin).
    edges = np.nanquantile(vfrac, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(edges)
    bin_idx = np.clip(np.digitize(vfrac, edges[1:-1]), 0, len(edges) - 2)
    centers = np.array([np.median(vfrac[bin_idx == b]) for b in range(len(edges) - 1)])
    medians = np.array([np.median(peaks[bin_idx == b]) for b in range(len(edges) - 1)])
    ax.plot(centers, medians, "o-", color="black", linewidth=2, markersize=5,
            label=f"binned median ({N_BINS} quantile bins)")

    ax.set_xlabel(f"Local ECS volume fraction gain at r={RADIUS:g} um: "
                  f"{high_label} - {low_label}")
    ax.set_ylabel(f"Peak d[{SPECIES_NAME}] = [{high_label}] - [{low_label}] (mM)")
    ax.set_title(
        f"Per-synapse peak local [{SPECIES_NAME}] difference vs local ECS gain "
        f"(n={len(peaks)}, peak at t={np.median(peak_times):.0f} ms median)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main(plot_kind, out_path, show):
    low_sweep, low_label = LOW
    high_sweep, high_label = HIGH
    times, diffs, pair_seeds, pair_synapse_idx = collect_differences(low_sweep, high_sweep)

    peak = times[np.nanargmax(np.abs(np.nanmean(diffs, axis=1)))]
    print(f"Largest mean |d[Ca]| at t={peak:.1f} ms: "
          f"{np.nanmean(diffs, axis=1)[np.nanargmax(np.abs(np.nanmean(diffs, axis=1)))]:.3f} mM")

    for extreme, argfunc in [("Lowest", np.nanargmin), ("Highest", np.nanargmax)]:
        t_idx, pair_idx = np.unravel_index(argfunc(diffs), diffs.shape)
        print(f"{extreme} d[Ca]={diffs[t_idx, pair_idx]:.3f} mM at t={times[t_idx]:.1f} ms: "
              f"seed={pair_seeds[pair_idx]}, synapse_idx={pair_synapse_idx[pair_idx]}")

    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_kind == "trace":
        plot_differences(times, diffs, low_label, high_label, ax)
    else:
        plot_peak_vs_vfrac(times, diffs, pair_seeds, pair_synapse_idx,
                           low_label, high_label, ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", choices=list(OUT_PATHS), default="trace",
                        help="Which analysis to plot (default: trace).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: per-mode entry in OUT_PATHS).")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the figure without opening a window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.plot, args.out or OUT_PATHS[args.plot], SHOW and not args.no_show)
