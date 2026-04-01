"""Plot average ECS calcium concentration for different ECS volume fractions."""

import os
import re

import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from bmbcsim.simulation.result_io import ResultLoader

# ============ Configuration ============
RESULTS_DIR = None  # Set to a specific path, or None to auto-detect latest
RESULTS_ROOT = "results"
SPECIES_NAME = "Ca"
FLUX_FIELD = "Ca_ProportionalFlux_flux_value"
FLUX_THRESHOLD_FRACTION = 0.1  # fraction of max flux to detect peaks
CLUSTER_MIN_DIST = 0.25  # µm, minimum distance between synapse centers
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


def find_synapse_centers(h5, flux_field=FLUX_FIELD,
                         threshold_frac=FLUX_THRESHOLD_FRACTION,
                         min_dist=CLUSTER_MIN_DIST):
    """Find synapse center coordinates from surface flux coefficient peaks.

    Uses threshold + greedy clustering to support multiple peaks per membrane.

    :param h5: Open h5py.File for the simulation snapshot.
    :param flux_field: Name of the flux dataset in surface_coefficients.
    :param threshold_frac: Fraction of max flux used as detection threshold.
    :param min_dist: Minimum distance between cluster centers (µm).
    :returns: Array of synapse center coordinates, shape (n_synapses, 3).
    """
    centers = []
    if "surface_coefficients" not in h5:
        return np.empty((0, 3), dtype=np.float32)

    for membrane_name in sorted(h5["surface_coefficients"]):
        if not membrane_name.startswith("membrane_"):
            continue
        coeff_grp = h5[f"surface_coefficients/{membrane_name}"]
        if flux_field not in coeff_grp:
            continue

        flux = coeff_grp[flux_field][:]
        pts = h5[f"surface_mesh/{membrane_name}/points"][:]
        threshold = threshold_frac * flux.max()
        above = np.where(flux > threshold)[0]

        # Greedy clustering by descending flux value
        peak_coords = pts[above]
        peak_fluxes = flux[above]
        order = np.argsort(-peak_fluxes)
        peak_coords = peak_coords[order]
        used = np.zeros(len(peak_coords), dtype=bool)

        for i in range(len(peak_coords)):
            if used[i]:
                continue
            centers.append(peak_coords[i])
            dists = np.linalg.norm(peak_coords - peak_coords[i], axis=1)
            used[dists < min_dist] = True

    if not centers:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(centers)


def compute_local_ca(result_path, species=SPECIES_NAME):
    """Compute per-synapse local ECS Ca concentration time series.

    For each synapse, samples the concentration at the nearest ECS vertex
    to the synapse center (point value, no spatial averaging).

    :param result_path: Path to a simulation result directory.
    :param species: Species name to evaluate.
    :returns: (times, local_ca) where local_ca has shape (n_timesteps, n_synapses).
    """
    h5_path = os.path.join(result_path, "snapshot.h5")
    with h5py.File(h5_path, "r") as h5:
        synapse_centers = find_synapse_centers(h5)
        n_synapses = len(synapse_centers)
        if n_synapses == 0:
            raise RuntimeError(f"No synapses found in {result_path}")

        # Find nearest ECS vertex for each synapse center
        points = h5["mesh/points"][:]
        ecs_indicator = h5["compartments/ecs"][:]
        ecs_mask = ecs_indicator > 0.5
        ecs_indices = np.where(ecs_mask)[0]
        ecs_tree = cKDTree(points[ecs_indices])

        _, nearest = ecs_tree.query(synapse_centers)
        sample_indices = ecs_indices[nearest]

        # Load time series
        step_keys = sorted(h5[f"data/{species}"].keys())
        times = h5["data/time"][:]
        local_ca = np.empty((len(step_keys), n_synapses), dtype=np.float32)

        for t_idx, step_key in enumerate(step_keys):
            ca_data = h5[f"data/{species}/{step_key}"][:]
            local_ca[t_idx] = ca_data[sample_indices]

    return times, local_ca


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

    # ============ Local Ca near synapses ============
    # Select 3 cases: lowest, midpoint, highest ECS fraction
    idx_low = np.argmin(actual_fractions)
    idx_high = np.argmax(actual_fractions)
    idx_mid = np.argmin(np.abs(actual_fractions - np.median(actual_fractions)))
    selected = [
        (idx_low, "Low", "#1f77b4"),
        (idx_mid, "Mid", "#2ca02c"),
        (idx_high, "High", "#d62728"),
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for idx, label, color in selected:
        _, result_path = ratio_dirs[idx]
        frac = actual_fractions[idx]
        print(f"  Computing local Ca for {label} ECS={frac*100:.1f}% ...")
        t, local_ca = compute_local_ca(result_path)
        mean_ca = np.nanmean(local_ca, axis=1)
        std_ca = np.nanstd(local_ca, axis=1)
        ax2.plot(t, mean_ca, color=color, linewidth=1.5,
                 label=f"ECS {frac*100:.1f}% ({label})")
        ax2.fill_between(t, mean_ca - std_ca, mean_ca + std_ca,
                         alpha=0.2, color=color)

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel(f"Local ECS [{SPECIES_NAME}] near synapses (mM)")
    ax2.set_title(
        f"Local [{SPECIES_NAME}] at synapse locations (mean ± std over synapses)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
