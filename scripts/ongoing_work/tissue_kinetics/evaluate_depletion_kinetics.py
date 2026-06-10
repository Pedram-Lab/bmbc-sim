"""Characterize per-synapse Ca depletion/replenishment across a parameter sweep.

For every parameter value in a sweep directory we pool the local-ECS Ca traces
of all synapses (aggregating over any ECS-ratio sub-levels and over all seeds)
and reduce each trace to three metrics:

  * t_depletion       - time (ms) at maximum depletion (the trace minimum)
  * depletion         - the Ca value (mM) at that minimum
  * replenishment_rate- 1/tau (s^-1) of a single-exponential fit to the
                        *recovery phase* (t >= t_min):
                            C(t) = C0 - D * exp(-(t - t_min) / tau)

Why not the double exponential ``C0 - a*(exp(-(t-t0)/tau1) - exp(-(t-t0)/tau2))``?
On this data it is degenerate: the best fit always collapses to tau1 == tau2
(an alpha-function pulse with a single timescale), because the depletion onset
bottoms out within ~2-3 samples at dt = 10 ms and is not separately identifiable
from the recovery. The three metrics above keep the same reported quantities
(when, how deep, how fast it refills) but are each well-conditioned and
independent; the recovery rate cleanly tracks the swept parameter (e.g. it rises
monotonically with diffusivity).

The sweep layout is auto-detected: a "parameter" is an immediate child directory
of the sweep root (excluding processed-data/ and plots/), and every
``tissue_kinetics_seed*`` directory found anywhere beneath it is pooled.

Output, per sweep: a stacked 3-panel box plot at
``<sweep>/plots/depletion_kinetics.png`` and a tidy per-synapse CSV at
``<sweep>/processed-data/depletion_kinetics.csv``.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ecs_ratio import compute_local_ca

# ============ Fixed depletion-protocol constants ============
C0 = 1.3      # baseline Ca (mM); the recovery asymptote
T0 = 300.0    # stimulus onset (ms); traces are flat at C0 before this
# ============================================================

_SEED_PATTERN = re.compile(r"tissue_kinetics(?:_\w+?)?_seed\d+_\d{4}-\d{2}-\d{2}-\d{6}$")
_RESERVED_DIRS = {"processed-data", "plots"}

METRICS = [
    # (csv/key name, axis label)
    ("t_depletion", "Time to max\ndepletion after\nstimulus (ms)"),
    ("depletion", "Depletion\n[Ca] (mM)"),
    ("recovery_tau", "Recovery time\nconstant $\\tau$ (ms)"),
]


# ---------------------------------------------------------------------------
# Sweep / seed discovery
# ---------------------------------------------------------------------------

def find_seed_dirs(root):
    """All tissue_kinetics_seed* directories anywhere beneath `root`."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        if _SEED_PATTERN.match(os.path.basename(dirpath)):
            out.append(dirpath)
            dirnames[:] = []  # don't descend into a seed dir
    return sorted(out)


def discover_parameters(sweep_dir):
    """Return [(param_label, [seed_dir, ...]), ...] for a sweep directory.

    A parameter is an immediate child directory (excluding processed-data/ and
    plots/) that contains at least one seed directory beneath it.
    """
    params = []
    for name in sorted(os.listdir(sweep_dir)):
        if name in _RESERVED_DIRS:
            continue
        child = os.path.join(sweep_dir, name)
        if not os.path.isdir(child):
            continue
        seeds = find_seed_dirs(child)
        if seeds:
            params.append((name, seeds))
    return params


def _natural_key(label):
    """Sort key splitting a label into (text, number) chunks for natural order."""
    parts = re.split(r"(-?\d+\.?\d*)", label)
    key = []
    for p in parts:
        try:
            key.append((1, float(p)))
        except ValueError:
            key.append((0, p))
    return key


# ---------------------------------------------------------------------------
# Per-trace metrics
# ---------------------------------------------------------------------------

def _parabola_min(t, c, i):
    """Sub-sample (t, value) of the minimum near integer index `i`.

    Fits a parabola through the three points around `i`; falls back to the
    sampled point at domain boundaries or when the parabola is not convex.
    """
    if i <= 0 or i >= len(t) - 1:
        return float(t[i]), float(c[i])
    x0, x1, x2 = t[i - 1], t[i], t[i + 1]
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return float(t[i]), float(c[i])
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a <= 0:  # not a minimum
        return float(t[i]), float(c[i])
    xv = -b / (2 * a)
    if not (x0 <= xv <= x2):
        return float(t[i]), float(c[i])
    yv = (y0 * (xv - x1) * (xv - x2) / ((x0 - x1) * (x0 - x2))
          + y1 * (xv - x0) * (xv - x2) / ((x1 - x0) * (x1 - x2))
          + y2 * (xv - x0) * (xv - x1) / ((x2 - x0) * (x2 - x1)))
    return float(xv), float(yv)


def _recovery_model(t, D, tau, t_min):
    return C0 - D * np.exp(-(t - t_min) / tau)


def trace_metrics(times, ca):
    """Reduce one synapse trace to (t_depletion, depletion, recovery_tau).

    `t_depletion` is the latency (ms) of the trace minimum after the stimulus at
    T0; `recovery_tau` is the time constant (ms) of the single-exponential
    recovery fit. Returns NaNs for any metric that cannot be estimated (e.g. too
    few recovery samples or a failed recovery fit).
    """
    mask = times >= T0
    tt, cc = times[mask], ca[mask]
    i = int(np.argmin(cc))
    t_dep, depletion = _parabola_min(tt, cc, i)
    t_dep_after_stim = t_dep - T0

    # Recovery phase: from the discrete minimum onward.
    tr, cr = tt[i:], cc[i:]
    tau = np.nan
    if len(tr) >= 4:
        D0 = max(C0 - depletion, 1e-3)
        try:
            (_, tau), _ = curve_fit(
                lambda t, D, tau: _recovery_model(t, D, tau, tr[0]),
                tr, cr, p0=[D0, 80.0],
                bounds=([0.0, 1.0], [2.0, 5000.0]), maxfev=20000,
            )
        except Exception:
            tau = np.nan
    return t_dep_after_stim, depletion, tau


# ---------------------------------------------------------------------------
# Per-seed processing
# ---------------------------------------------------------------------------

def process_seed(result_path):
    """Return per-synapse metric rows for one seed result directory."""
    times, local_ca = compute_local_ca(result_path)
    rows = []
    for s in range(local_ca.shape[1]):
        t_dep, depletion, tau = trace_metrics(times, local_ca[:, s])
        rows.append((s, t_dep, depletion, tau))
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(sweep_name, param_labels, data, out_path, n_synapses):
    """Stacked box plots: one panel per metric, one box per parameter."""
    n = len(param_labels)
    fig_w = max(7.0, 0.55 * n + 2.5)
    fig, axes = plt.subplots(
        len(METRICS), 1, sharex=True, figsize=(fig_w, 8.0)
    )
    positions = np.arange(n) + 1

    for ax, (key, ylabel) in zip(axes, METRICS):
        series = [data[label][key] for label in param_labels]
        series = [arr[np.isfinite(arr)] for arr in series]
        bp = ax.boxplot(
            series, positions=positions, widths=0.6, patch_artist=True,
            showfliers=False, medianprops=dict(color="black"),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor="#4C72B0", alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(x=0.02)
        # Prune top/bottom y ticks so adjacent (touching) panels don't collide.
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="both"))

    axes[-1].set_xticks(positions)
    rot = 90 if n > 6 else 45
    ha = "center" if rot == 90 else "right"
    axes[-1].set_xticklabels(
        param_labels, rotation=rot, ha=ha, rotation_mode="anchor",
        fontsize=8 if n > 12 else 9,
    )
    axes[0].set_title(
        f"{sweep_name}\n"
        f"synapse Ca depletion kinetics ({n_synapses} traces, pooled over ECS ratios & seeds)",
        fontsize=10,
    )

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.0, left=0.13, right=0.98, top=0.94,
                        bottom=0.2 if rot == 90 else 0.14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="One or more sweep directories")
    return p.parse_args()


def run_sweep(sweep_dir):
    sweep_dir = os.path.abspath(sweep_dir)
    sweep_name = os.path.basename(sweep_dir.rstrip("/"))
    params = discover_parameters(sweep_dir)
    if not params:
        print(f"  No parameters with seed results found in {sweep_dir}; skipping.")
        return
    param_labels = sorted((lbl for lbl, _ in params), key=_natural_key)
    print(f"  {len(param_labels)} parameters: {', '.join(param_labels)}")

    # tidy rows for CSV, plus arrays for plotting
    csv_rows = []
    collected = {label: {key: [] for key, _ in METRICS} for label, _ in params}
    for label, seed_dirs in params:
        for sd in seed_dirs:
            try:
                rows = process_seed(sd)
            except Exception as e:
                print(f"    {label}/{os.path.basename(sd)}: skipped ({type(e).__name__}: {e})")
                continue
            for syn, t_dep, depletion, tau in rows:
                csv_rows.append((label, os.path.basename(sd), syn, t_dep, depletion, tau))
                collected[label]["t_depletion"].append(t_dep)
                collected[label]["depletion"].append(depletion)
                collected[label]["recovery_tau"].append(tau)

    data = {
        label: {key: np.asarray(vals, dtype=float) for key, vals in metrics.items()}
        for label, metrics in collected.items()
    }
    n_synapses = len(csv_rows)

    processed_dir = os.path.join(sweep_dir, "processed-data")
    plots_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(processed_dir, "depletion_kinetics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "seed_dir", "synapse_idx",
                    "t_depletion_after_stim_ms", "depletion_mM", "recovery_tau_ms"])
        w.writerows(csv_rows)

    plot_path = os.path.join(plots_dir, "depletion_kinetics.png")
    plot_sweep(sweep_name, param_labels, data, plot_path, n_synapses)
    print(f"  Wrote {csv_path}")
    print(f"  Wrote {plot_path}")


def main():
    args = parse_args()
    for sweep_dir in args.sweeps:
        print(f"==> {sweep_dir}")
        run_sweep(sweep_dir)


if __name__ == "__main__":
    main()
