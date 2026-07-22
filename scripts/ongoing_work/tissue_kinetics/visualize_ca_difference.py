"""Plot the per-synapse difference in local ECS Ca between two ECS ratios over time.

The two configured sweeps run the SAME synapses (same cell set, same seeded
placement) at two different ECS ratios -- this is what the reference-shrink fix in
simulation.py guarantees, so (seed, synapse_idx) identifies one physical synapse in
both sweeps (verified: idx-matched centers are the spatial nearest neighbour ~98% of
the time, median displacement ~0.2 um from the cell-size scaling alone).

For every shared (seed, synapse_idx) we form the trace

    d[Ca](t) = [Ca]_HIGH(t) - [Ca]_LOW(t)

at the synapse's nearest ECS vertex (via evaluate_ecs_ratio.compute_local_ca, whose
synapse ordering matches across sweeps). Pairs where either trace dips negative
anywhere (solver undershoot in the tightest synapses) are dropped outright. Pooling
over the surviving synapses and seeds, we plot the mean (solid) and median (dashed)
difference with the CENTILE..(100-CENTILE)% range shaded, plus a faint subsample of
the individual difference traces.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ecs_ratio import compute_local_ca
from evaluate_synapse_distribution_spatial import find_seed_dirs

# ============ Configuration ============
# (sweep dir, label). HIGH - LOW is the plotted difference; the suffix in each dir
# name is int(100*(ecs_ratio+0.06)), i.e. ecs_ratio 0.04 -> "10", 0.19 -> "25".
LOW = ("results/synapse_distribution_ecs_10_2026-06-17-142107", "10% ECS")
HIGH = ("results/synapse_distribution_ecs_25_2026-06-17-142107", "25% ECS")
SPECIES_NAME = "Ca"
CENTILE = 5                  # shaded band is CENTILE..(100-CENTILE)%
SHOW_INDIVIDUAL = True       # overlay a faint subsample of per-synapse traces
MAX_INDIVIDUAL = 400         # cap on how many individual traces to draw
OUT_PATH = "results/ca_difference_by_synapse.png"
SHOW = True
# =======================================


def collect_differences(low_sweep, high_sweep):
    """Pool d[Ca] = ca_high - ca_low over every shared (seed, synapse_idx).

    Returns (times, diffs) with diffs shape (n_timesteps, n_pairs). Seeds present
    in only one sweep, or with mismatched synapse count / timestep count, are
    skipped with a warning. Synapse pairs where either the LOW or HIGH trace dips
    negative anywhere (solver undershoot) are dropped before differencing.
    """
    low_paths = dict(find_seed_dirs(low_sweep))
    high_paths = dict(find_seed_dirs(high_sweep))
    common = sorted(set(low_paths) & set(high_paths))
    print(f"Seeds: low={len(low_paths)} high={len(high_paths)} common={len(common)}")

    times_ref = None
    columns = []
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

    if not columns:
        raise SystemExit("No matched synapses found across the two sweeps.")
    diffs = np.concatenate(columns, axis=1)
    if diffs.shape[1] == 0:
        raise SystemExit("All synapse pairs dropped for negative values; nothing to plot.")
    print(f"Pooled {diffs.shape[1]} synapse pairs from {len(common) - n_skipped} "
          f"seeds ({n_skipped} seeds skipped; {n_dropped} pairs dropped for negatives)")
    return times_ref, diffs


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


def main(out_path, show):
    low_sweep, low_label = LOW
    high_sweep, high_label = HIGH
    times, diffs = collect_differences(low_sweep, high_sweep)

    peak = times[np.nanargmax(np.abs(np.nanmean(diffs, axis=1)))]
    print(f"Largest mean |d[Ca]| at t={peak:.1f} ms: "
          f"{np.nanmean(diffs, axis=1)[np.nanargmax(np.abs(np.nanmean(diffs, axis=1)))]:.3f} mM")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_differences(times, diffs, low_label, high_label, ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=OUT_PATH, help="Output PNG path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the figure without opening a window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.out, SHOW and not args.no_show)
