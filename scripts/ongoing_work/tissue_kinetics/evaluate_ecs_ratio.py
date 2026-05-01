"""Plot average ECS calcium concentration for different ECS volume fractions."""

import argparse
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
RESULTS_ROOT = "results"
SPECIES_NAME = "Ca"
FLUX_FIELD = "Ca_ProportionalFlux_flux_value"
FLUX_THRESHOLD_FRACTION = 0.1  # fraction of max flux to detect peaks
CLUSTER_MIN_DIST = 0.25  # µm, minimum distance between synapse centers
# Select which sweep variant to load (mechanics implies ECM)
WITH_ECM = False
WITH_MECHANICS = False
# =======================================


def _variant_suffix(with_ecm, with_mechanics):
    if with_mechanics:
        return "_mechanics"
    if with_ecm:
        return "_ecm"
    return ""


def find_latest_sweep_dir(results_root, with_ecm=False, with_mechanics=False):
    """Find the latest ecs_ratio[_variant]_<timestamp> directory."""
    suffix = _variant_suffix(with_ecm, with_mechanics)
    pattern = re.compile(rf"ecs_ratio{suffix}_\d{{4}}-\d{{2}}-\d{{2}}-\d{{6}}$")
    dirs = [
        d for d in os.listdir(results_root)
        if pattern.match(d) and os.path.isdir(os.path.join(results_root, d))
    ]
    if not dirs:
        raise RuntimeError(f"No ecs_ratio{suffix}_* directories found")
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


def find_ratio_dirs(sweep_dir, with_ecm=False, with_mechanics=False):
    """Find all tissue_kinetics[_variant]_ecs* result directories."""
    suffix = _variant_suffix(with_ecm, with_mechanics)
    pattern = re.compile(
        rf"tissue_kinetics{suffix}_ecs(\d+)_\d{{4}}-\d{{2}}-\d{{2}}-\d{{6}}$"
    )
    dirs = []
    for d in sorted(os.listdir(sweep_dir)):
        m = pattern.match(d)
        if m and os.path.isdir(os.path.join(sweep_dir, d)):
            dirs.append((int(m.group(1)), os.path.join(sweep_dir, d)))
    return dirs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", default=None,
        help="Path to sweep directory (default: latest auto-detected)",
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Indices into the sorted-by-ECS-fraction list to plot (default: all)",
    )
    parser.add_argument(
        "--plot", choices=["average", "synapses"], default="average",
        help="Plot average ECS concentration or local concentration near synapses",
    )
    return parser.parse_args()


def compute_fractions(ratio_dirs):
    """Compute actual ECS volume fractions (cheap; geometry only)."""
    fractions = []
    ecs_volumes = []
    for _, result_path in ratio_dirs:
        region_sizes = ResultLoader(result_path).compute_region_sizes()
        ecs_volume = region_sizes["ecs"]
        total_volume = sum(region_sizes.values())
        fractions.append(ecs_volume / total_volume)
        ecs_volumes.append(ecs_volume)
    return np.array(fractions), ecs_volumes


def load_concentrations(ratio_dirs, ecs_volumes, fractions):
    """Load average ECS concentration time series for the given dirs."""
    all_concentrations = []
    times = None
    for (input_pct, result_path), ecs_volume, frac in zip(
        ratio_dirs, ecs_volumes, fractions
    ):
        loader = ResultLoader(result_path)
        total_substance = xr.concat(
            [loader.load_total_substance(i) for i in range(len(loader))],
            dim="time",
        )
        ecs_substance = total_substance.sel(region="ecs", species=SPECIES_NAME)
        ecs_conc = ecs_substance / ecs_volume

        if times is None:
            times = total_substance.coords["time"].values

        all_concentrations.append(ecs_conc.values)
        print(f"  Loaded ecs_ratio input={input_pct}%, actual={frac*100:.0f}%")

    return all_concentrations, times


def plot_average(times, fractions, concentrations, n_total, full_selection):
    fig, ax = plt.subplots(figsize=(10, 6))

    if full_selection:
        norm = plt.Normalize(vmin=fractions.min() * 100, vmax=fractions.max() * 100)
        cmap = cm.coolwarm
        for fraction, conc in zip(fractions, concentrations):
            ax.plot(times, conc, linewidth=1.5, color=cmap(norm(fraction * 100)))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Actual ECS fraction (%)")
    else:
        for fraction, conc in zip(fractions, concentrations):
            ax.plot(times, conc, linewidth=1.5, label=f"ECS {fraction*100:.0f}%")
        ax.legend()

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{SPECIES_NAME} in ECS (mM)")
    ax.set_title(f"ECS [{SPECIES_NAME}] across ECS volume fractions (N={n_total})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_synapses(ratio_dirs, fractions, full_selection):
    fig, ax = plt.subplots(figsize=(10, 6))

    if full_selection:
        norm = plt.Normalize(vmin=fractions.min() * 100, vmax=fractions.max() * 100)
        cmap = cm.coolwarm
        colors = [cmap(norm(f * 100)) for f in fractions]
    else:
        colors = [None] * len(fractions)

    for i, ((_, result_path), frac) in enumerate(zip(ratio_dirs, fractions)):
        print(f"  Computing local Ca for ECS={frac*100:.0f}% ...")
        t, local_ca = compute_local_ca(result_path)
        mean_ca = np.nanmean(local_ca, axis=1)
        std_ca = np.nanstd(local_ca, axis=1)
        line, = ax.plot(t, mean_ca, color=colors[i], linewidth=1.5,
                        label=f"ECS {frac*100:.0f}%")
        ax.fill_between(t, mean_ca - std_ca, mean_ca + std_ca,
                        alpha=0.2, color=line.get_color())

    if full_selection:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Actual ECS fraction (%)")
    else:
        ax.legend()

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"Local ECS [{SPECIES_NAME}] near synapses (mM)")
    ax.set_title(
        f"Local [{SPECIES_NAME}] at synapse locations (mean ± std over synapses)"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    args = parse_args()

    sweep_dir = args.path or find_latest_sweep_dir(
        RESULTS_ROOT, with_ecm=WITH_ECM, with_mechanics=WITH_MECHANICS
    )
    ratio_dirs = find_ratio_dirs(
        sweep_dir, with_ecm=WITH_ECM, with_mechanics=WITH_MECHANICS
    )
    print(f"Found {len(ratio_dirs)} ECS ratio results in {sweep_dir}")

    fractions, ecs_volumes = compute_fractions(ratio_dirs)

    # Sort everything by actual ECS fraction
    order = np.argsort(fractions)
    fractions = fractions[order]
    ecs_volumes = [ecs_volumes[i] for i in order]
    ratio_dirs = [ratio_dirs[i] for i in order]

    n_total = len(ratio_dirs)
    if args.indices is None:
        full_selection = True
    else:
        full_selection = False
        fractions = fractions[args.indices]
        ecs_volumes = [ecs_volumes[i] for i in args.indices]
        ratio_dirs = [ratio_dirs[i] for i in args.indices]

    if args.plot == "average":
        concentrations, times = load_concentrations(ratio_dirs, ecs_volumes, fractions)
        plot_average(times, fractions, concentrations, n_total, full_selection)
    else:
        plot_synapses(ratio_dirs, fractions, full_selection)

    plt.show()
