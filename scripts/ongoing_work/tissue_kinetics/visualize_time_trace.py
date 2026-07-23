"""Plot species concentration statistics in ECS over time across multiple simulations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bmbcsim.simulation.result_io import ResultLoader

SPECIES_NAME = "Ca"  # Species to plot
CENTILE = 5


def find_result_dirs(path):
    """Return leaf result dirs under `path` (or [path] if it holds a result directly)."""
    if (path / "snapshot.pvd").exists() or (path / "snapshot.h5").exists():
        return [path]
    dirs = sorted(
        d for d in path.rglob("*")
        if d.is_dir() and ((d / "snapshot.pvd").exists() or (d / "snapshot.h5").exists())
    )
    if not dirs:
        raise FileNotFoundError(f"No simulation results found under {path}")
    return dirs


def load_condition(path):
    """Pool ECS point values for every seed found under `path`.

    Returns (times, values) where values[step] is a 1-D array pooling the ECS
    point values from every seed at that snapshot.
    """
    result_dirs = find_result_dirs(path)
    print(f"{path}: found {len(result_dirs)} result director{'y' if len(result_dirs) == 1 else 'ies'}")

    times = None
    pooled = None
    for result_dir in result_dirs:
        loader = ResultLoader(str(result_dir))
        ecs_mask = np.array(loader.load_regions()["ecs"], dtype=bool)

        t, seed_values = [], []
        for step in range(len(loader)):
            mesh = loader.load_snapshot(step)
            time_series = loader.load_total_substance(step)
            t.append(float(time_series["time"]))
            seed_values.append(np.array(mesh[SPECIES_NAME])[ecs_mask])

        if times is None:
            times = t
            pooled = [list(v) for v in seed_values]
        elif len(t) != len(times):
            raise ValueError(f"{result_dir} has {len(t)} snapshots, expected {len(times)}")
        else:
            for step_values, extra in zip(pooled, seed_values):
                step_values.extend(extra)

    return np.array(times), [np.array(v) for v in pooled]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+",
        help="Simulation result directories to compare (one per condition); each may "
             "itself be a folder containing multiple seed result directories",
    )
    args = parser.parse_args()
    for path in args.paths:
        if not path.exists():
            parser.error(f"No such path: {path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, path in enumerate(args.paths):
        times, values = load_condition(path)

        lows = [np.quantile(v, CENTILE / 100.0) for v in values]
        medians = [np.median(v) for v in values]
        means = [np.mean(v) for v in values]
        highs = [np.quantile(v, 1 - CENTILE / 100.0) for v in values]

        color = colors[i % len(colors)]
        label = path.name

        # Shaded area for low-high range
        ax.fill_between(times, lows, highs, alpha=0.2, color=color)

        # Mean as solid line, median as dashed
        ax.plot(times, means, linewidth=2, linestyle="-", color=color, label=f"{label} mean")
        ax.plot(times, medians, linewidth=2, linestyle="--", color=color, label=f"{label} median")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{SPECIES_NAME} in ECS (mM)")
    ax.set_title(f"{SPECIES_NAME} concentration in ECS over time ({CENTILE}-{100-CENTILE}% range shaded)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
