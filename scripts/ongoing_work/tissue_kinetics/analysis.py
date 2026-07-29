"""Shared analysis primitives for the tissue-kinetics results.

The pieces every evaluation script here needs, in one place instead of imported out
of whichever plotting script happened to define them first:

* :func:`group_runs` -- turn a sweep directory into ``{axis value: [runs]}``. This is
  what makes a script work on *any* sweep: the axis is a parameter, not a hardcoded
  assumption, and replicates of one grid point group together.
* :func:`find_synapse_centers` / :func:`compute_local_ca` -- locate the synapse
  patches from the recorded surface flux and sample Ca next to them.

The thresholds below are tuned to this experiment's synapse geometry, which is why
they live here rather than in ``bmbcsim``.
"""
import os
import re

import h5py
import numpy as np
from scipy.spatial import cKDTree

from bmbcsim.simulation.result_io import find_run_dirs, run_labels

SPECIES_NAME = "Ca"
FLUX_FIELD = "Ca_ProportionalFlux_flux_value"
FLUX_THRESHOLD_FRACTION = 0.1  # fraction of max flux to detect peaks
CLUSTER_MIN_DIST = 0.25  # µm, minimum distance between synapse centers

# Legacy layout: the ECS percentage used to be baked into the run name
# ("tissue_kinetics_ecs04_<timestamp>"). A sweep now puts every swept value in an
# "<axis>=<value>" directory above the run; this keeps archived results groupable.
_LEGACY_ECS_PATTERN = re.compile(r"_ecs(\d+)_")


def _sort_key(label):
    """Sort labels numerically when they are numbers, alphabetically otherwise."""
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def group_runs(sweep_dir, group_by=None):
    """Group the runs beneath `sweep_dir` by the value of one swept axis.

    :param sweep_dir: A sweep root, or any subtree of one.
    :param group_by: Axis name as it appears in the path (e.g. ``"ecs_ratio"``,
        ``"ecm_kd"``). ``None`` puts every run in one group, which is what pools all
        replicates of a single-point sweep together.
    :returns: ``{label: [run_dir, ...]}``, ordered by label; the label is ``""`` when
        not grouping.
    :raises SystemExit: If `group_by` names an axis that this sweep did not vary.
    """
    groups = {}
    for run in find_run_dirs(sweep_dir):
        labels = run_labels(run, sweep_dir)
        if group_by is None:
            key = ""
        elif group_by in labels:
            key = labels[group_by]
        elif group_by == "ecs_ratio" and (match := _LEGACY_ECS_PATTERN.search(run.name)):
            key = f"{int(match.group(1)) / 100:g}"
        else:
            continue
        groups.setdefault(key, []).append(run)

    if group_by is not None and not groups:
        axes = sorted({
            axis for run in find_run_dirs(sweep_dir) for axis in run_labels(run, sweep_dir)
        })
        raise SystemExit(
            f"no '{group_by}=<value>' directories under {sweep_dir}; "
            f"axes this sweep varied: {axes or ['(none)']}"
        )
    return dict(sorted(groups.items(), key=lambda item: _sort_key(item[0])))


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


def _self_check():
    """Check group_runs against a synthetic sweep tree: uv run analysis.py"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for kd in ("0.1-mM", "1.3-mM"):
            for ecs in ("0.19", "0.04"):  # deliberately unsorted on disk
                for seed in (0, 1, 10):
                    run = root / f"ecm_kd={kd}/ecs_ratio={ecs}/2026-07-26-150039_sim_seed{seed}"
                    run.mkdir(parents=True)
                    (run / "snapshot.h5").touch()

        by_kd = group_runs(root, "ecm_kd")
        assert list(by_kd) == ["0.1-mM", "1.3-mM"], by_kd
        assert all(len(runs) == 6 for runs in by_kd.values()), by_kd

        by_ecs = group_runs(root, "ecs_ratio")
        assert list(by_ecs) == ["0.04", "0.19"], by_ecs  # numeric order, not on-disk order

        pooled = group_runs(root)
        assert list(pooled) == [""] and len(pooled[""]) == 12, pooled

        try:
            group_runs(root, "diffusivity_ecs")
        except SystemExit as exc:
            assert "ecm_kd" in str(exc) and "ecs_ratio" in str(exc), exc
        else:
            raise AssertionError("grouping by an axis the sweep never varied must fail")

        legacy = root / "old/tissue_kinetics_ecs04_2026-05-29-185441"
        legacy.mkdir(parents=True)
        (legacy / "snapshot.pvd").touch()
        assert list(group_runs(root / "old", "ecs_ratio")) == ["0.04"]

    print("analysis.py self-check passed")


if __name__ == "__main__":
    _self_check()
