"""Plot average ECS calcium concentration with error bars across synapse distribution seeds."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from bmbcsim.simulation.result_io import ResultLoader

# ============ Configuration ============
RESULTS_DIR = None  # Set to a specific path, or None to auto-detect latest
RESULTS_ROOT = "results"
SPECIES_NAME = "Ca"
# =======================================


def find_latest_sweep_dir(results_root):
    """Find the latest synapse_distribution_* directory."""
    pattern = re.compile(r"synapse_distribution_\d{4}-\d{2}-\d{2}-\d{6}$")
    dirs = [
        d for d in os.listdir(results_root)
        if pattern.match(d) and os.path.isdir(os.path.join(results_root, d))
    ]
    if not dirs:
        raise RuntimeError("No synapse_distribution_* directories found")
    return os.path.join(results_root, sorted(dirs)[-1])


def find_seed_dirs(sweep_dir):
    """Find all tissue_kinetics_seed* result directories."""
    pattern = re.compile(r"tissue_kinetics_seed(\d+)_\d{4}-\d{2}-\d{2}-\d{6}$")
    dirs = []
    for d in sorted(os.listdir(sweep_dir)):
        m = pattern.match(d)
        if m and os.path.isdir(os.path.join(sweep_dir, d)):
            dirs.append((int(m.group(1)), os.path.join(sweep_dir, d)))
    return dirs


if __name__ == "__main__":
    sweep_dir = RESULTS_DIR or find_latest_sweep_dir(RESULTS_ROOT)
    seed_dirs = find_seed_dirs(sweep_dir)
    print(f"Found {len(seed_dirs)} seed results in {sweep_dir}")

    # Load per-seed ECS average concentration time series
    all_concentrations = []
    times = None

    for seed_idx, result_path in seed_dirs:
        loader = ResultLoader(result_path)
        region_sizes = loader.compute_region_sizes()
        ecs_volume = region_sizes["ecs"]

        total_substance = xr.concat(
            [loader.load_total_substance(i) for i in range(len(loader))],
            dim="time",
        )
        ecs_substance = total_substance.sel(region="ecs", species=SPECIES_NAME)
        ecs_conc = ecs_substance / ecs_volume

        if times is None:
            times = total_substance.coords["time"].values

        all_concentrations.append(ecs_conc.values)
        print(f"  Loaded seed {seed_idx}")

    # Compute statistics across seeds
    all_concentrations = np.array(all_concentrations)  # (n_seeds, n_timesteps)
    mean_conc = np.mean(all_concentrations, axis=0)
    std_conc = np.std(all_concentrations, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, mean_conc, linewidth=2, color="tab:blue",
            label=f"Mean [{SPECIES_NAME}] in ECS")
    ax.fill_between(
        times,
        mean_conc - std_conc,
        mean_conc + std_conc,
        alpha=0.3, color="tab:blue",
        label=r"$\pm$ 1 SD across seeds",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{SPECIES_NAME} in ECS (mM)")
    ax.set_title(
        f"ECS [{SPECIES_NAME}] across synapse distribution seeds (N={len(seed_dirs)})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
