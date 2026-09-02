"""Collect a sensor Kd sweep into one xarray dataset for ``parameter_sweep_evaluation.py``.

This is the second half of the old ``parameter_sweep.py``: ``sweep.py`` now runs
the grid, this reads it back. Runs are found by content (a snapshot file) rather
than by name, and each run's Kd pair is read from the resolved config it wrote,
so nothing here has to reconstruct the sweep's directory naming.

    uv run scripts/ongoing_work/sensor_simulation/collect_kd_sweep.py
    uv run scripts/ongoing_work/sensor_simulation/collect_kd_sweep.py results/<stamp>_sensor-kd-sweep
"""
import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
import xarray as xr
import yaml

from bmbcsim import ResultLoader
from bmbcsim.simulation.result_io import find_run_dirs

# Where sweep.py's configs/kd_sweep.yaml puts its runs (timestamp appended; the
# glob also catches older runs that had it prepended).
SWEEP_GLOB = "*sensor-kd-sweep*"
DEFAULT_OUT = "results/sensor_parameter_sweep.zarr"
KD_UNIT = u.mmol / u.L
REGION = "cube:sphere"  # the sensor sphere


def latest_kd_sweep(results_root="results"):
    """Newest sweep tree written by sweep.py (by mtime, so either naming order works)."""
    candidates = sorted(
        Path(results_root).glob(SWEEP_GLOB), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise SystemExit(
            f"No {SWEEP_GLOB} directory under {results_root}; run sweep.py first, "
            "or pass the sweep directory explicitly."
        )
    return candidates[-1]


def run_kds(run):
    """The (buffer Kd, sensor Kd) pair of one run, in mM, from its own config."""
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    return tuple(
        u.Quantity(cfg[binder]["kd"]).to_value(KD_UNIT) for binder in ("buffer", "sensor")
    )


def sphere_substance(run):
    """Free and total Ca substance in the sensor sphere over time."""
    loader = ResultLoader(str(run))
    total_substance = xr.concat(
        [loader.load_total_substance(i) for i in range(len(loader))], dim="time"
    ).sel(region=REGION)
    free_ca = total_substance.sel(species="ca")
    total_ca = (
        free_ca
        + total_substance.sel(species="ca_sensor")
        + total_substance.sel(species="ca_buffer")
    )
    return free_ca, total_ca


def collect(sweep_dir):
    """Build the (time, buffer_kd, sensor_kd, channel) dataset for `sweep_dir`."""
    runs = find_run_dirs(sweep_dir)
    if not runs:
        raise SystemExit(f"No runs found under {sweep_dir}")
    kds = {run: run_kds(run) for run in runs}
    buffer_kds = sorted({buffer_kd for buffer_kd, _ in kds.values()})
    sensor_kds = sorted({sensor_kd for _, sensor_kd in kds.values()})
    print(f"{sweep_dir}: {len(runs)} run(s), "
          f"{len(buffer_kds)} buffer Kd x {len(sensor_kds)} sensor Kd")

    data, times = None, None
    for run, (buffer_kd, sensor_kd) in kds.items():
        free_ca, total_ca = sphere_substance(run)
        if data is None:
            times = free_ca.coords["time"].values
            # NaN, not zeros: a run that failed must not read as "no calcium".
            data = np.full(
                (len(times), len(buffer_kds), len(sensor_kds), 2), np.nan
            )
        elif len(free_ca) != len(times):
            raise SystemExit(
                f"{run} has {len(free_ca)} snapshots, expected {len(times)}; "
                "these runs cannot share a time axis"
            )
        i, j = buffer_kds.index(buffer_kd), sensor_kds.index(sensor_kd)
        data[:, i, j, :] = np.array([free_ca.values, total_ca.values]).T
        print(f"  buffer_kd={buffer_kd:g} mM, sensor_kd={sensor_kd:g} mM: {run.name}")

    return xr.Dataset(
        data_vars={"parameter_sweep": (("time", "buffer_kd", "sensor_kd", "channel"), data)},
        coords={
            "time": times,
            "buffer_kd": buffer_kds,
            "sensor_kd": sensor_kds,
            "channel": ["free_ca", "total_ca"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=None,
                        help=f"Sweep directory (default: latest results/{SWEEP_GLOB})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Zarr output (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    results = collect(args.path or latest_kd_sweep())
    results.to_zarr(args.out, mode="w")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
