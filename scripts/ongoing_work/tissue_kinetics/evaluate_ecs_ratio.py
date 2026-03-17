"""Plot average ECS calcium concentration for different ECS volume fractions."""

import os
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr

from bmbcsim.simulation.result_io import ResultLoader

# ============ Configuration ============
RESULTS_DIR = None  # Set to a specific path, or None to auto-detect latest
RESULTS_ROOT = "results"
SPECIES_NAME = "Ca"
# =======================================


def find_latest_sweep_dir(results_root):
    """Find the latest ecs_ratio_* directory."""
    pattern = re.compile(r"ecs_ratio_\d{4}-\d{2}-\d{2}-\d{6}$")
    dirs = [
        d for d in os.listdir(results_root)
        if pattern.match(d) and os.path.isdir(os.path.join(results_root, d))
    ]
    if not dirs:
        raise RuntimeError("No ecs_ratio_* directories found")
    return os.path.join(results_root, sorted(dirs)[-1])


def find_ratio_dirs(sweep_dir):
    """Find all tissue_kinetics_ecs* result directories."""
    pattern = re.compile(r"tissue_kinetics_ecs(\d+)_\d{4}-\d{2}-\d{2}-\d{6}$")
    dirs = []
    for d in sorted(os.listdir(sweep_dir)):
        m = pattern.match(d)
        if m and os.path.isdir(os.path.join(sweep_dir, d)):
            dirs.append((int(m.group(1)), os.path.join(sweep_dir, d)))
    return dirs


if __name__ == "__main__":
    sweep_dir = RESULTS_DIR or find_latest_sweep_dir(RESULTS_ROOT)
    ratio_dirs = find_ratio_dirs(sweep_dir)
    print(f"Found {len(ratio_dirs)} ECS ratio results in {sweep_dir}")

    # Load per-ratio ECS average concentration time series
    actual_fractions = []
    all_concentrations = []
    times = None

    for input_pct, result_path in ratio_dirs:
        loader = ResultLoader(result_path)
        region_sizes = loader.compute_region_sizes()
        ecs_volume = region_sizes["ecs"]
        total_volume = sum(region_sizes.values())
        actual_fraction = ecs_volume / total_volume

        total_substance = xr.concat(
            [loader.load_total_substance(i) for i in range(len(loader))],
            dim="time",
        )
        ecs_substance = total_substance.sel(region="ecs", species=SPECIES_NAME)
        ecs_conc = ecs_substance / ecs_volume

        if times is None:
            times = total_substance.coords["time"].values

        actual_fractions.append(actual_fraction)
        all_concentrations.append(ecs_conc.values)
        print(f"  Loaded ecs_ratio input={input_pct}%, actual={actual_fraction*100:.1f}%")

    # Plot with colormap based on actual ECS fraction
    actual_fractions = np.array(actual_fractions)
    norm = plt.Normalize(vmin=actual_fractions.min() * 100, vmax=actual_fractions.max() * 100)
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (fraction, conc) in enumerate(zip(actual_fractions, all_concentrations)):
        color = cmap(norm(fraction * 100))
        ax.plot(times, conc, linewidth=1.5, color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Actual ECS fraction (%)")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{SPECIES_NAME} in ECS (mM)")
    ax.set_title(f"ECS [{SPECIES_NAME}] across ECS volume fractions (N={len(ratio_dirs)})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
