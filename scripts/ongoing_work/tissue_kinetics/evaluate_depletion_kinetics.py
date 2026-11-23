"""Characterize per-synapse Ca depletion/replenishment across a parameter sweep.

For every parameter value in a sweep directory we pool the local-ECS Ca traces
of all synapses (aggregating over any ECS-ratio sub-levels and over all seeds)
and reduce each trace to three metrics. Each trace starts at the baseline C0
(the run's ``diffusion.ca_ecs``) and dips to a minimum C_min after the stimulus
at T0 (the run's first ``synapse.pulse_times`` entry); both are read from the
run's dumped ``config.yaml``:

  * depletion          - the Ca value (mM) at the minimum (the depletion depth),
                         always shown in the middle panel.

plus one depletion-side and one replenishment-side kinetics metric selected with
``--metric`` (all four families are always written to the CSV; the switch only
picks the plotted pair). Depletion metrics are measured from T0 to the minimum,
replenishment metrics from the minimum onwards:

  time (default)   t_95_*: time (ms) to the first crossing of 95% of the depth,
                   i.e. C0 - 0.95*(C0 - C_min) on the way down and
                   C_min + 0.95*(C0 - C_min) on the way up, linearly interpolated
                   between samples. Depth-independent, so best for comparing the
                   *shape* of the transient across parameters.
  mean-rate        *_rate: 0.95*(C0 - C_min) / t_95_* (mM/ms), the mean slope over
                   those intervals. Mixes depth and speed.
  max-rate         max_*_rate: steepest finite-difference slope |dC/dt| (mM/ms)
                   within the interval. Well resolved on the recovery (6-7 samples),
                   but the drop bottoms out within 2-3 samples so the depletion value
                   is dominated by the recording interval.
  time-constant    tau_*: tau (ms) of a one-parameter exponential fit with the
                   asymptotes pinned to the data, C_min + (C0 - C_min) exp(-(t-T0)/tau)
                   on the way down and C0 - (C0 - C_min) exp(-(t-t_min)/tau) on the
                   way up. The recovery is close to exponential; the stimulus-driven
                   drop is not, so tau_depletion is a rough summary only.

The sweep layout is auto-detected: a "parameter" is an immediate child directory
of the sweep root (excluding processed-data/ and plots/), and every simulation run
found anywhere beneath it is pooled -- so with more than one swept axis, the outer
axis is the parameter and the inner ones are pooled into it. ``--aggregate-level N``
makes the first N directory levels the parameter instead (e.g. 2 for one x-axis item
per ``ecm_total=.../local_width=...`` pair). Parameters are shown in
natural (numeric) order; ``--order`` takes a permutation of their positions to override it.
``--split-ecs`` keeps the pooling over seeds but draws one box per ECS ratio (the run's
``geometry.ecs_ratio``, read from its config, so it works for any directory layout)
side by side within each parameter.

Output, per sweep: a stacked 3-panel box plot at
``<sweep>/plots/depletion_kinetics[_<metric>][_by-ecs].png`` (no metric suffix for
``time``) and a tidy per-synapse CSV with all metrics and the ECS ratio at
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
import yaml
from scipy.optimize import curve_fit

from bmbcsim.simulation.result_io import find_run_dirs
from analysis import compute_local_ca
from simulation import Config


def protocol(result_path):
    """(C0, T0, ECS ratio) of a run: baseline Ca (mM), stimulus onset (ms) and
    ``geometry.ecs_ratio``, from its config.yaml."""
    with open(os.path.join(result_path, "config.yaml"), encoding="utf-8") as f:
        cfg = Config.model_validate(yaml.safe_load(f))
    c0 = float(cfg.diffusion.ca_ecs.to("mM").value)
    t0 = float(min(t.to("ms").value for t in cfg.synapse.pulse_times))
    return c0, t0, cfg.geometry.ecs_ratio


# csv/key name -> axis label, in CSV column order
LABELS = {
    "t_95_depletion": "Time to 95%\ndepletion after\nstimulus (ms)",
    "depletion_rate": "Mean depletion\nrate (mM/ms)",
    "max_depletion_rate": "Max depletion\nrate (mM/ms)",
    "tau_depletion": "Depletion time\nconstant (ms)",
    "depletion": "Minimum\n[Ca] (mM)",
    "t_95_replenishment": "Time to 95%\nreplenishment (ms)",
    "replenishment_rate": "Mean replenishment\nrate (mM/ms)",
    "max_replenishment_rate": "Max replenishment\nrate (mM/ms)",
    "tau_replenishment": "Replenishment time\nconstant (ms)",
}
# --metric family -> (depletion-side key, replenishment-side key) plotted around "depletion"
FAMILIES = {
    "time": ("t_95_depletion", "t_95_replenishment"),
    "mean-rate": ("depletion_rate", "replenishment_rate"),
    "max-rate": ("max_depletion_rate", "max_replenishment_rate"),
    "time-constant": ("tau_depletion", "tau_replenishment"),
}


def panels(metric):
    """[(key, axis label), ...] for the three plotted panels of one metric family."""
    dep, rep = FAMILIES[metric]
    return [(k, LABELS[k]) for k in (dep, "depletion", rep)]


# ---------------------------------------------------------------------------
# Sweep / seed discovery
# ---------------------------------------------------------------------------

def discover_parameters(sweep_dir, level=1):
    """Return [(param_label, [seed_dir, ...]), ...] for a sweep directory.

    A parameter is the first `level` directory levels below the sweep root (joined
    with "/"); every run beneath it (processed-data/ and plots/ excluded) is pooled.
    """
    params = {}
    for run in find_run_dirs(sweep_dir):
        label = "/".join(run.relative_to(sweep_dir).parts[:level])
        params.setdefault(label, []).append(str(run))
    return list(params.items())


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
    if not x0 <= xv <= x2:
        return float(t[i]), float(c[i])
    yv = (y0 * (xv - x1) * (xv - x2) / ((x0 - x1) * (x0 - x2))
          + y1 * (xv - x0) * (xv - x2) / ((x1 - x0) * (x1 - x2))
          + y2 * (xv - x0) * (xv - x1) / ((x2 - x0) * (x2 - x1)))
    return float(xv), float(yv)


def _crossing_time(t, c, i_start, level, descending):
    """First (interpolated) time at/after index `i_start` where `c` hits `level`.

    `descending=True` looks for `c` falling to/below `level`; `False` for `c`
    rising to/above it. Linearly interpolates between the bracketing samples for
    sub-sample resolution. Returns NaN if `level` is never reached within the
    window.
    """
    for k in range(i_start, len(t)):
        hit = c[k] <= level if descending else c[k] >= level
        if not hit:
            continue
        if k == i_start:
            return float(t[k])
        c0, c1 = c[k - 1], c[k]
        if c1 == c0:
            return float(t[k])
        frac = (level - c0) / (c1 - c0)
        return float(t[k - 1] + frac * (t[k] - t[k - 1]))
    return np.nan


def _max_slope(t, c):
    """Steepest finite-difference |dC/dt| over the samples; NaN with fewer than two."""
    return float(np.max(np.abs(np.diff(c) / np.diff(t)))) if len(t) > 1 else np.nan


def _tau_fit(t, c, t_origin, c_start, c_end):
    """tau of c(t) = c_end + (c_start - c_end) * exp(-(t - t_origin) / tau); NaN if unfittable."""
    if len(t) < 3 or c_start == c_end:
        return np.nan

    def model(x, tau):
        return c_end + (c_start - c_end) * np.exp(-(x - t_origin) / tau)

    try:
        (tau,), _ = curve_fit(model, t, c, p0=[max(t[-1] - t_origin, 1.0) / 3], bounds=(1e-6, np.inf))
    except RuntimeError:
        return np.nan
    return float(tau)


def trace_metrics(times, ca, c0, t0):
    """Reduce one synapse trace to a dict of the LABELS keys.

    `t_95_depletion` is the time (ms, relative to the stimulus `t0`) of the first
    crossing of 95% of the depletion depth on the way down; `depletion` is the Ca
    value (mM) at the minimum; `t_95_replenishment` is the time (ms, relative to
    the minimum) of the first crossing of 95% recovery toward the baseline `c0`.
    The mean rates are 0.95*(c0 - c_min) / t over those intervals, the max rates the
    steepest sampled slope within them, and the time constants one-parameter
    exponential fits (see the module docstring). Returns NaN for a time/rate metric
    whose level is never reached within the window.
    """
    mask = times >= t0
    tt, cc = times[mask], ca[mask]
    i = int(np.argmin(cc))
    t_min, c_min = _parabola_min(tt, cc, i)
    # A flat bottom followed by a sharp rise makes the parabola overshoot the data
    # (by up to ~40%), which puts the 95% level below every sample; never go deeper
    # than the sampled minimum.
    c_min = max(c_min, float(cc[i]))
    drop = c0 - c_min
    if drop <= 0:
        return dict.fromkeys(LABELS, np.nan) | {"depletion": c_min}

    # Depletion: first downward crossing of the 95%-of-depth level (from t0).
    level_dep = c0 - 0.95 * drop
    t_dep = _crossing_time(tt, cc, 0, level_dep, descending=True)
    t_95_depletion = t_dep - t0 if np.isfinite(t_dep) else np.nan

    # Replenishment: first upward crossing of the 95%-recovered level (from min).
    level_rep = c_min + 0.95 * drop
    t_rep = _crossing_time(tt, cc, i, level_rep, descending=False)
    t_95_replenishment = t_rep - t_min if np.isfinite(t_rep) else np.nan

    # Sampled segments on the way down (t0 .. min) and up (min ..) for slopes and fits.
    down, up = slice(0, i + 1), slice(i, None)
    return {
        "t_95_depletion": t_95_depletion,
        "depletion_rate": 0.95 * drop / t_95_depletion,
        "max_depletion_rate": _max_slope(tt[down], cc[down]),
        "tau_depletion": _tau_fit(tt[down], cc[down], t0, c0, cc[i]),
        "depletion": c_min,
        "t_95_replenishment": t_95_replenishment,
        "replenishment_rate": 0.95 * drop / t_95_replenishment,
        "max_replenishment_rate": _max_slope(tt[up], cc[up]),
        "tau_replenishment": _tau_fit(tt[up], cc[up], tt[i], cc[i], c0),
    }


# ---------------------------------------------------------------------------
# Per-seed processing
# ---------------------------------------------------------------------------

def process_seed(result_path):
    """Return (ECS ratio, per-synapse metric rows) for one seed result directory."""
    c0, t0, ecs_ratio = protocol(result_path)
    times, local_ca = compute_local_ca(result_path)
    return ecs_ratio, [trace_metrics(times, local_ca[:, s], c0, t0) for s in range(local_ca.shape[1])]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_SLOT_INCHES = 1.0  # horizontal space per parameter
_BOX_WIDTH = 0.12   # box width, in parameter slots (i.e. in inches at the scale above)


def plot_sweep(sweep_name, param_labels, data, out_path, n_synapses, metrics, split_ecs=False):
    """Stacked box plots: one panel per (key, label) in `metrics`, one box per parameter
    (with `split_ecs`: one box per ECS ratio, side by side within each parameter)."""
    n = len(param_labels)
    # Fixed x scale (inches per parameter slot) so a box has the same size on paper for
    # any number of parameters and groups; few parameters just leave room at the sides.
    fig_w = max(4.5, _SLOT_INCHES * n + 2.0)
    fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(fig_w, 8.0))
    positions = np.arange(n) + 1

    # One box per group within each parameter; a single group (None) pools everything.
    if split_ecs:
        groups = sorted({e for label in param_labels for e in data[label]["ecs_ratio"]})
    else:
        groups = [None]
    colors = [f"C{g}" for g in range(len(groups))]  # default colour cycle, in ECS order

    for ax, (key, ylabel) in zip(axes, metrics):
        handles = []
        for g, (ecs, color) in enumerate(zip(groups, colors)):
            series = [
                data[label][key] if ecs is None
                else data[label][key][data[label]["ecs_ratio"] == ecs]
                for label in param_labels
            ]
            series = [arr[np.isfinite(arr)] for arr in series]
            bp = ax.boxplot(
                series, positions=positions + (g - (len(groups) - 1) / 2) * 1.4 * _BOX_WIDTH,
                widths=_BOX_WIDTH, patch_artist=True,
                showfliers=False, medianprops=dict(color="black"),
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=0.7)
            handles.append(bp["boxes"][0])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        # Prune top/bottom y ticks so adjacent (touching) panels don't collide.
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="both"))

    axes[-1].set_xticks(positions)
    half_span = (0.98 - 0.13) * fig_w / _SLOT_INCHES / 2  # axes width in parameter slots
    axes[-1].set_xlim((n + 1) / 2 - half_span, (n + 1) / 2 + half_span)
    rot = 90 if n > 6 else 45
    ha = "center" if rot == 90 else "right"
    axes[-1].set_xticklabels(
        param_labels, rotation=rot, ha=ha, rotation_mode="anchor",
        fontsize=8 if n > 12 else 9,
    )
    pooling = "split by ECS ratio, pooled over seeds" if split_ecs else "pooled over ECS ratios & seeds"
    axes[0].set_title(
        f"{sweep_name}\nsynapse Ca depletion kinetics ({n_synapses} traces, {pooling})",
        fontsize=10,
    )
    if split_ecs:
        axes[0].legend(handles, [f"{e:g}" for e in groups], title="ECS ratio",
                       loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8)

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.0, left=0.13, right=0.98, top=0.94,
                        bottom=0.2 if rot == 90 else 0.14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="One or more sweep directories")
    p.add_argument("--order", type=int, nargs="+", metavar="POS",
                   help="Permutation of the parameter positions after natural sorting (0-based), "
                        "e.g. '--order 2 0 1' puts the third parameter first")
    p.add_argument("--metric", choices=FAMILIES, default="time",
                   help="Kinetics metric family to plot around the minimum-[Ca] panel "
                        "(the CSV always holds all of them). Default: %(default)s")
    p.add_argument("--split-ecs", action="store_true",
                   help="One box per ECS ratio within each parameter instead of pooling them")
    p.add_argument("--aggregate-level", type=int, default=1, metavar="N",
                   help="Directory levels below the sweep root that make up one parameter "
                        "(x-axis item); everything deeper is pooled. Default: %(default)s")
    return p.parse_args()


def run_sweep(sweep_dir, metric, order=None, split_ecs=False, level=1):
    sweep_dir = os.path.abspath(sweep_dir)
    sweep_name = os.path.basename(sweep_dir.rstrip("/"))
    params = discover_parameters(sweep_dir, level)
    if not params:
        print(f"  No parameters with seed results found in {sweep_dir}; skipping.")
        return
    param_labels = sorted((lbl for lbl, _ in params), key=_natural_key)
    if order:
        if sorted(order) != list(range(len(param_labels))):
            raise SystemExit(f"--order must be a permutation of 0..{len(param_labels) - 1}, got {order}")
        param_labels = [param_labels[i] for i in order]
    print(f"  {len(param_labels)} parameters: {', '.join(param_labels)}")

    # tidy rows for CSV, plus arrays for plotting
    csv_rows = []
    collected = {label: {key: [] for key in (*LABELS, "ecs_ratio")} for label, _ in params}
    for label, seed_dirs in params:
        for sd in seed_dirs:
            try:
                ecs_ratio, rows = process_seed(sd)
            except Exception as e:
                print(f"    {label}/{os.path.basename(sd)}: skipped ({type(e).__name__}: {e})")
                continue
            for syn, m in enumerate(rows):
                csv_rows.append((label, ecs_ratio, os.path.basename(sd), syn, *(m[k] for k in LABELS)))
                collected[label]["ecs_ratio"].append(ecs_ratio)
                for k in LABELS:
                    collected[label][k].append(m[k])

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
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "ecs_ratio", "seed_dir", "synapse_idx", *LABELS])
        w.writerows(csv_rows)

    suffix = ("" if metric == "time" else f"_{metric}") + ("_by-ecs" if split_ecs else "")
    plot_path = os.path.join(plots_dir, f"depletion_kinetics{suffix}.png")
    plot_sweep(sweep_name, param_labels, data, plot_path, n_synapses, panels(metric), split_ecs)
    print(f"  Wrote {csv_path}")
    print(f"  Wrote {plot_path}")


def main():
    args = parse_args()
    for sweep_dir in args.sweeps:
        print(f"==> {sweep_dir}")
        run_sweep(sweep_dir, args.metric, args.order, args.split_ecs, args.aggregate_level)


if __name__ == "__main__":
    main()
