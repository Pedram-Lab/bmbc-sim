"""Plot the neuron-bath sweep: intracellular Ca2+ over time and its rise against bath Ca2+.

Left: the cell's mean Ca2+ rise, one curve per bath concentration. Right: the peak
rise against the bath concentration, next to the "expected" line of an
uptake proportional to the bath concentration (through the highest bath).

    uv run scripts/ongoing_work/neuron_bath/visualize.py [sweep_dir] [--out fig.png]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import units as u

from bmbcsim.simulation.result_io import ResultLoader, find_run_dirs, latest_sweep_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=None,
                        help="Sweep directory (default: latest under results/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the figure here instead of showing it")
    args = parser.parse_args()
    sweep_dir = args.path or latest_sweep_dir("results")

    runs = []  # (bath Ca in mM, times in s, cell Ca rise in mM, config)
    for run in find_run_dirs(sweep_dir):
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        bath = u.Quantity(cfg["diffusion"]["ca_ecs"]).to_value("mM")
        ca = ResultLoader(str(run)).load_concentration("cell", "Ca")
        runs.append((bath, ca.coords["time"].values / 1e3, ca.values - ca.values[0], cfg))
        print(f"  {run.name}: bath {bath:g} mM, peak cell Ca rise {runs[-1][2].max():.4g} mM")
    if not runs:
        raise SystemExit(f"no runs under {sweep_dir}")
    runs.sort(key=lambda r: -r[0])

    fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[3, 2])
    colors = plt.cm.Blues(np.linspace(0.95, 0.35, len(runs)))
    for (bath, times, rise, _), color in zip(runs, colors):
        ax_t.plot(times, rise, color=color, label=f"{bath:g} mM")
        if bath < runs[0][0]:  # expected peak: proportional to the bath concentration
            ax_t.axhline(runs[0][2].max() * bath / runs[0][0], color=color, ls="--")
    upt = runs[0][3]["uptake"]
    start = u.Quantity(upt["first_pulse"]).to_value(u.s)
    stop = start + upt["n_pulses"] / u.Quantity(upt["frequency"]).to_value(u.Hz)
    ax_t.axvspan(start, stop, color="0.9", zorder=0, label=f"{upt['n_pulses']} AP")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel(r"$\Delta$[Ca$^{2+}$]$_i$, cell mean (mM)")
    ax_t.legend(title=r"[Ca$^{2+}$]$_e$")

    baths = np.array([r[0] for r in runs])
    rises = np.array([r[2].max() for r in runs])
    ax_d.plot([0, baths[0]], [0, rises[0]], "k--", label="expected")
    ax_d.plot(baths, rises, "ko-", label="simulated")
    ax_d.set_xlabel(r"[Ca$^{2+}$]$_e$ (mM)")
    ax_d.set_ylabel(r"$\Delta$[Ca$^{2+}$]$_i$ peak (mM)")
    ax_d.set_xlim(0, None)
    ax_d.set_ylim(0, None)
    ax_d.legend()
    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
