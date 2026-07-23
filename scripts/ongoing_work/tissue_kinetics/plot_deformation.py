"""Plot deformation magnitude over time for mechanics simulations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bmbcsim.simulation.result_io import ResultLoader

MAX_TRAJECTORIES = 1000  # Limit number of individual trajectories to plot


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


def load_deformation(result_dir):
    """Return (times, deformation_magnitude[n_snapshots, n_points]) for one run."""
    loader = ResultLoader(str(result_dir))
    times, magnitudes = [], []
    for step in range(len(loader)):
        mesh = loader.load_snapshot(step)
        ts = loader.load_total_substance(step)
        times.append(float(ts.coords['time'].values[0]))
        magnitudes.append(np.linalg.norm(mesh.point_data['deformation'], axis=1))
    return np.array(times), np.array(magnitudes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path,
        help="Simulation result directory, or a folder containing multiple seed result directories",
    )
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"No such path: {args.path}")

    result_dirs = find_result_dirs(args.path)
    print(f"Found {len(result_dirs)} result director{'y' if len(result_dirs) == 1 else 'ies'}")

    times = None
    all_magnitudes = []
    for result_dir in result_dirs:
        print(f"Loading {result_dir} ...")
        t, magnitudes = load_deformation(result_dir)
        if times is None:
            times = t
        elif len(t) != len(times):
            raise ValueError(f"{result_dir} has {len(t)} snapshots, expected {len(times)}")
        all_magnitudes.append(magnitudes)

    # Combine seeds by stacking their points together (all share the same times)
    deformation_magnitudes = np.concatenate(all_magnitudes, axis=1)
    print(f"Deformation data shape: {deformation_magnitudes.shape}")

    n_points = deformation_magnitudes.shape[1]
    if n_points > MAX_TRAJECTORIES:
        rng = np.random.default_rng(seed=42)
        sample_indices = rng.choice(n_points, size=MAX_TRAJECTORIES, replace=False)
    else:
        sample_indices = np.arange(n_points)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in sample_indices:
        ax.plot(times, deformation_magnitudes[:, idx], color='gray', alpha=0.05, linewidth=0.5)

    mean_deformation = np.mean(deformation_magnitudes, axis=1)
    ax.plot(times, mean_deformation, color='blue', linewidth=2, label='Mean')

    p5 = np.percentile(deformation_magnitudes, 5, axis=1)
    p95 = np.percentile(deformation_magnitudes, 95, axis=1)
    ax.fill_between(times, p5, p95, color='blue', alpha=0.2, label='5-95% range')

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Deformation magnitude (µm)")
    ax.set_title(f"Mesh deformation over time - {args.path.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
