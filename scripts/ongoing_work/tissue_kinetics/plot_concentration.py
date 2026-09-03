"""Plot ECS calcium against time, one curve per swept value, over any sweep.

Replaces ``evaluate_ecs_ratio.py``, ``evaluate_synapse_distribution.py`` and
``visualize_time_trace.py``, which were the same plot three times over with the axis
hardcoded: one could only group by ECS ratio, one could only pool seeds, one needed
every condition's directory named by hand. The sweep axis is now an argument, so this
works on the ECM-affinity, kinetics, diffusivity and contraction sweeps too.

Runs sharing a value of ``--group-by`` are one curve: the mean over their pooled
samples, with a spread band. What counts as a sample is ``--metric``:

    region-mean   the ECS volume average, one sample per run    (band: SD over runs)
    per-synapse   Ca at each synapse's nearest ECS vertex       (band: SD over synapses)
    ecs-points    Ca at every ECS mesh vertex                   (band: 5-95%, + median)

    # one curve per ECS fraction, labelled by the *measured* fraction
    uv run .../plot_concentration.py results/ecs-ratio-sweep_<stamp> \
        --group-by ecs_ratio --actual-ecs
    # mean +/- SD across 100 seeds
    uv run .../plot_concentration.py results/synapse-distribution-sweep_<stamp>
    # any other sweep axis
    uv run .../plot_concentration.py results/buffer-capacity-sweep_<stamp> --group-by ecm_kd
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from bmbcsim.simulation.result_io import ResultLoader, latest_sweep_dir
from analysis import SPECIES_NAME, compute_local_ca, group_runs

# Above this many curves a legend stops being readable, so numeric labels switch to a
# colorbar (which also shows the spacing of the swept values).
MAX_LEGEND_ENTRIES = 8
PERCENTILE = 5  # ecs-points band: PERCENTILE .. 100-PERCENTILE


def samples_region_mean(run):
    """One sample per snapshot: the ECS volume-average concentration."""
    concentration = ResultLoader(str(run)).load_concentration("ecs", SPECIES_NAME)
    return concentration.coords["time"].values, concentration.values[:, None]


def samples_per_synapse(run):
    """One sample per synapse per snapshot, at each synapse's nearest ECS vertex."""
    return compute_local_ca(str(run))


def samples_ecs_points(run):
    """One sample per ECS mesh vertex per snapshot."""
    loader = ResultLoader(str(run))
    ecs_mask = np.array(loader.load_regions()["ecs"], dtype=bool)
    times, values = [], []
    for step in range(len(loader)):
        times.append(float(loader.load_total_substance(step)["time"]))
        values.append(np.array(loader.load_snapshot(step)[SPECIES_NAME])[ecs_mask])
    return np.array(times), np.array(values)


# metric -> (sampler, band kind). The band kind follows from what is being pooled:
# a handful of run-level means is described by an SD, a cloud of point values by a
# percentile range.
METRICS = {
    "region-mean": (samples_region_mean, "sd"),
    "per-synapse": (samples_per_synapse, "sd"),
    "ecs-points": (samples_ecs_points, "percentile"),
}


def load_group(runs, sampler):
    """Pool one group's samples across runs.

    :returns: (times, samples) with samples of shape (n_snapshots, n_pooled_samples).
    :raises SystemExit: If the runs disagree on the number of snapshots, which would
        silently misalign the curves.
    """
    times, pooled = None, []
    for run in runs:
        run_times, run_samples = sampler(run)
        if times is None:
            times = run_times
        elif len(run_times) != len(times):
            raise SystemExit(
                f"{run} has {len(run_times)} snapshots, expected {len(times)}; "
                "these runs cannot share a time axis"
            )
        pooled.append(run_samples)
        print(f"    {run.name}: {run_samples.shape[1]} sample(s) per snapshot")
    return times, np.concatenate(pooled, axis=1)


def measured_ecs_fraction(runs):
    """Mean measured ECS volume fraction over `runs` (the geometry, not the config)."""
    fractions = []
    for run in runs:
        sizes = ResultLoader(str(run)).compute_region_sizes()
        fractions.append(sizes["ecs"] / sum(sizes.values()))
    return float(np.mean(fractions))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path", nargs="?", type=Path, default=None,
        help="Sweep directory, or any subtree of one (default: latest under results/)",
    )
    parser.add_argument(
        "--group-by", default=None, metavar="AXIS",
        help="Swept axis to draw one curve per value of, as it appears in the result "
             "path (e.g. ecs_ratio, ecm_kd). Default: pool every run into one curve.",
    )
    parser.add_argument(
        "--metric", choices=sorted(METRICS), default="region-mean",
        help="What to sample from each run (default: region-mean)",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, metavar="VALUE",
        help="Restrict to these values of the grouping axis (default: all)",
    )
    parser.add_argument(
        "--actual-ecs", action="store_true",
        help="Label curves by the measured ECS volume fraction instead of the "
             "configured value (they differ: the mesher does not hit the target exactly)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write the figure here instead of showing it interactively",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sweep_dir = args.path or latest_sweep_dir("results")
    groups = group_runs(sweep_dir, args.group_by)
    if args.only is not None:
        missing = set(args.only) - set(groups)
        if missing:
            raise SystemExit(f"no such {args.group_by} value(s): {sorted(missing)}; "
                             f"have {sorted(groups)}")
        groups = {label: runs for label, runs in groups.items() if label in args.only}

    n_runs = sum(len(runs) for runs in groups.values())
    print(f"{sweep_dir}: {n_runs} run(s) in {len(groups)} group(s)")

    sampler, band = METRICS[args.metric]
    curves = []  # (label, sort_value, times, samples)
    for label, runs in groups.items():
        print(f"  {args.group_by}={label}" if label else "  (all runs pooled)")
        times, samples = load_group(runs, sampler)
        if args.actual_ecs:
            label = f"{100 * measured_ecs_fraction(runs):.0f}%"
        curves.append((label, times, samples))

    # A colorbar instead of a legend once there are too many numeric labels to read.
    numeric = []
    for label, _, _ in curves:
        try:
            numeric.append(float(label.rstrip("%")))
        except ValueError:
            numeric = []
            break
    use_colorbar = bool(numeric) and len(curves) > MAX_LEGEND_ENTRIES

    fig, ax = plt.subplots(figsize=(10, 6))
    if use_colorbar:
        # The affinity and kinetics sweeps step by decades, where a linear scale would
        # give every value below the largest the same colour.
        low, high = min(numeric), max(numeric)
        log_scaled = low > 0 and high / low >= 100
        norm = LogNorm(vmin=low, vmax=high) if log_scaled else Normalize(vmin=low, vmax=high)
        cmap = plt.cm.coolwarm
        colors = [cmap(norm(value)) for value in numeric]
    else:
        colors = [plt.cm.tab10.colors[i % 10] for i in range(len(curves))]

    for (label, times, samples), color in zip(curves, colors):
        mean = samples.mean(axis=1)
        line, = ax.plot(times, mean, linewidth=1.5, color=color,
                        label=f"{args.group_by}={label}" if label else "mean")
        if samples.shape[1] < 2:
            continue  # a single sample per snapshot has no spread to show
        if band == "sd":
            spread = samples.std(axis=1)
            low, high = mean - spread, mean + spread
        else:
            low = np.percentile(samples, PERCENTILE, axis=1)
            high = np.percentile(samples, 100 - PERCENTILE, axis=1)
            ax.plot(times, np.median(samples, axis=1), linewidth=1.5,
                    linestyle="--", color=line.get_color())
        ax.fill_between(times, low, high, alpha=0.2, color=line.get_color())

    if use_colorbar:
        colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
        colorbar.set_label("Measured ECS fraction (%)" if args.actual_ecs else args.group_by)
    else:
        ax.legend()

    spread_label = f"{PERCENTILE}-{100 - PERCENTILE}%" if band == "percentile" else "1 SD"
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{SPECIES_NAME} in ECS (mM)")
    ax.set_title(f"ECS [{SPECIES_NAME}], {args.metric} "
                 f"(mean, {spread_label} shaded; N={n_runs} runs)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
