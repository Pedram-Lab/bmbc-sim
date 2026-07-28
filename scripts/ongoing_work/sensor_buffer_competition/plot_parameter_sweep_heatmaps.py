"""Heatmaps of estimated / actual Ca2+ over the buffer concentration x Kd sweep.

The sensor reports Ca indirectly, as Kd_sensor * complex / free. This compares
that estimate against the true free Ca in each compartment, so a ratio of 1 means
the sensor is telling the truth and anything else is buffer interference.

Reads whatever ``sweep.py`` produced: the runs are found by content and their
axis values come from the resolved config each run wrote, so the grid does not
have to be repeated here (the old version hardcoded it and rebuilt each run's
directory name).

    uv run scripts/ongoing_work/sensor_buffer_competition/plot_parameter_sweep_heatmaps.py
    uv run .../plot_parameter_sweep_heatmaps.py results/<stamp>_buffer-competition-sweep
"""
import argparse
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bmbcsim import ResultLoader
from bmbcsim.simulation.result_io import find_run_dirs

from simulation import Config

# Where sweep.py's configs/buffer_sweep.yaml puts its runs (timestamp prepended).
SWEEP_GLOB = "*_buffer-competition-sweep"
CONCENTRATION_UNIT = u.mmol / u.L
COMPARTMENTS = ("top", "bottom")


def latest_sweep(results_root="results"):
    """Newest sweep tree written by sweep.py (timestamps sort chronologically)."""
    candidates = sorted(Path(results_root).glob(SWEEP_GLOB))
    if not candidates:
        raise SystemExit(
            f"No {SWEEP_GLOB} directory under {results_root}; run sweep.py first, "
            "or pass the sweep directory explicitly."
        )
    return candidates[-1]


def buffer_axes(run):
    """The (concentration, Kd) grid point of one run, in mM, from its own config."""
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    return tuple(
        u.Quantity(cfg["buffer"][field]).to_value(CONCENTRATION_UNIT)
        for field in ("concentration", "kd")
    )


def sensor_ratio(run, sensor_kd):
    """Estimated / actual free Ca at the last snapshot, per compartment."""
    loader = ResultLoader(str(run))
    last = loader.load_total_substance(-1)
    region_sizes = loader.compute_region_sizes()
    # Substance ratios need no volume correction; the actual concentration does.
    estimated = sensor_kd * last.sel(species="sensor_complex") / last.sel(species="sensor")
    actual = last.sel(species="ca")
    return {
        name: float(
            (estimated.sel(region=name)
             / (actual.sel(region=name) / region_sizes[name])).squeeze().values
        )
        for name in COMPARTMENTS
    }


def collect(sweep_dir, sensor_kd):
    """Build one ratio grid per compartment, indexed [kd, concentration]."""
    runs = find_run_dirs(sweep_dir)
    if not runs:
        raise SystemExit(f"No runs found under {sweep_dir}")
    axes = {run: buffer_axes(run) for run in runs}
    concentrations = sorted({conc for conc, _ in axes.values()})
    kds = sorted({kd for _, kd in axes.values()})
    print(f"{sweep_dir}: {len(runs)} run(s), "
          f"{len(concentrations)} concentration(s) x {len(kds)} Kd(s)")

    # NaN, not zeros: a run that failed must not plot as a ratio of 0.
    ratios = {name: np.full((len(kds), len(concentrations)), np.nan) for name in COMPARTMENTS}
    for run, (concentration, kd) in axes.items():
        i, j = kds.index(kd), concentrations.index(concentration)
        try:
            for name, ratio in sensor_ratio(run, sensor_kd).items():
                ratios[name][i, j] = ratio
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  skipping {run}: {exc}")
    return concentrations, kds, ratios


def plot_heatmap(data, concentrations, kds, ax, title):
    """Create a heatmap using pcolormesh."""
    x, y = np.meshgrid(np.arange(len(concentrations)), np.arange(len(kds)))
    im = ax.pcolormesh(x, y, data, cmap="viridis", shading="nearest")

    ax.set_xticks(np.arange(len(concentrations)))
    ax.set_yticks(np.arange(len(kds)))
    ax.set_xticklabels([f"{conc:.0e}" for conc in concentrations])
    ax.set_yticklabels([f"{kd:.0e}" for kd in kds])

    plt.colorbar(im, ax=ax, label="Estimated/Actual Ca²⁺ ratio")
    ax.set_xlabel("Buffer concentration (mM)")
    ax.set_ylabel("Buffer Kd (mM)")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=None,
                        help=f"Sweep directory (default: latest results/{SWEEP_GLOB})")
    parser.add_argument("--out", type=Path, default=Path("parameter_sweep_heatmap.pdf"))
    args = parser.parse_args()

    # The sensor is not swept, so its Kd comes from the simulation's own default.
    sensor_kd = Config().sensor.kd.to_value(CONCENTRATION_UNIT)
    concentrations, kds, ratios = collect(args.path or latest_sweep(), sensor_kd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, COMPARTMENTS):
        plot_heatmap(ratios[name], concentrations, kds, ax, f"{name.capitalize()} compartment")

    plt.suptitle("Ratio of estimated to actual Ca²⁺ concentration")
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
