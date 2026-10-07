"""Plot the per-synapse difference in local ECS Ca between two ECS ratios over time.

The two configured sweeps run the SAME synapses (same cell set, same seeded
placement) at two different ECS ratios -- this is what the reference-shrink fix in
simulation.py guarantees, so (seed, synapse_idx) identifies one physical synapse in
both sweeps (verified: idx-matched centers are the spatial nearest neighbour ~98% of
the time, median displacement ~0.2 um from the cell-size scaling alone).

For every shared (seed, synapse_idx) we form the trace

    d[Ca](t) = [Ca]_HIGH(t) - [Ca]_LOW(t)

at the synapse's nearest ECS vertex (via analysis.compute_local_ca, whose
synapse ordering matches across sweeps). Pairs where either trace dips negative
anywhere (solver undershoot in the tightest synapses) are dropped outright. Two plot modes (``--plot``):

* ``trace`` (default): pooling over the surviving synapses and seeds, the mean (solid)
  and median (dashed) difference with the CENTILE..(100-CENTILE)% range shaded, plus a
  faint subsample of the individual difference traces.

* ``vs-ecs``: one point per synapse -- its peak difference (the signed d[Ca] at the
  time of largest |d[Ca]|) against how much local ECS that synapse GAINED between the
  two runs, d_vfrac = vfrac_HIGH - vfrac_LOW, where vfrac is the local ECS volume
  fraction at radius RADIUS (``v_local_r<R> / v_sphere_box_r<R>`` from the pooled
  ``<sweep>/spatial_metrics.csv`` written by running
  evaluate_synapse_distribution_spatial.py on the sweep root with RADIUS among its
  --radii; rows are matched to each ECS ratio via the ``group`` column). A binned
  median over deciles of d_vfrac is drawn on top.
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import compute_local_ca, find_synapse_centers
from bmbcsim.simulation.result_io import ResultLoader
from evaluate_synapse_distribution_spatial import find_seed_dirs
from visualize_by_regime import _vfrac

EXPLORE_RADIUS = 3.0  # um, half-width of the pyvista crop box around a synapse center

# CLI defaults. The names below are rebound from the parsed arguments in __main__,
# so every function can keep reading them as module globals.
SPECIES_NAME = "Ca"
CENTILE = 5                  # shaded band is CENTILE..(100-CENTILE)%
SHOW_INDIVIDUAL = True       # overlay a faint subsample of per-synapse traces
MAX_INDIVIDUAL = 400         # cap on how many individual traces to draw
RADIUS = 0.4                 # um; selects the v_local_r<R>/v_sphere_box_r<R> columns
N_BINS = 10                  # quantile bins for the binned median (vs-ecs)
OUT_NAMES = {                # default output file per --plot mode, inside the sweep dir
    "trace": "ca_difference_by_synapse.png",
    "vs-ecs": "ca_difference_vs_ecs_volume.png",
}
LOW = HIGH = None            # (dir, label); set from the sweep dir in __main__


def find_ratio_dirs(sweep_root):
    """The two ``ecs_ratio=*`` sublevels of `sweep_root` as (dir, label), low first.

    The label converts the nominal ratio to the effective ECS percentage, +6% from
    the cell-size scaling: ecs_ratio 0.04 -> "10% ECS", 0.19 -> "25% ECS".
    """
    subs = sorted(Path(sweep_root).glob("ecs_ratio=*"),
                  key=lambda p: float(p.name.split("=")[1]))
    if len(subs) != 2:
        raise SystemExit(f"Expected exactly 2 ecs_ratio=* dirs under "
                         f"'{sweep_root}', found {len(subs)}.")
    return [(str(p), f"{100 * (float(p.name.split('=')[1]) + 0.06):.0f}% ECS")
            for p in subs]


def collect_differences(low_sweep, high_sweep):
    """Pool d[Ca] = ca_high - ca_low over every shared (seed, synapse_idx).

    Returns (times, diffs, pair_seeds, pair_synapse_idx). diffs has shape
    (n_timesteps, n_pairs); pair_seeds/pair_synapse_idx (length n_pairs) give the
    originating seed and synapse index of each column, so a specific extreme value
    in diffs can be traced back to a (seed, synapse_idx) to investigate further.
    Seeds present in only one sweep, or with mismatched synapse count / timestep
    count, are skipped with a warning. Synapse pairs where either the LOW or HIGH
    trace dips negative anywhere (solver undershoot) are dropped before differencing.
    """
    low_paths = dict(find_seed_dirs(low_sweep))
    high_paths = dict(find_seed_dirs(high_sweep))
    common = sorted(set(low_paths) & set(high_paths))
    print(f"Seeds: low={len(low_paths)} high={len(high_paths)} common={len(common)}")

    times_ref = None
    columns = []
    pair_seeds = []
    pair_synapse_idx = []
    n_skipped = 0
    n_dropped = 0
    for seed in common:
        times_lo, ca_lo = compute_local_ca(low_paths[seed], species=SPECIES_NAME)
        times_hi, ca_hi = compute_local_ca(high_paths[seed], species=SPECIES_NAME)

        if ca_lo.shape != ca_hi.shape:
            print(f"  seed {seed}: shape mismatch low={ca_lo.shape} "
                  f"high={ca_hi.shape}; skipped")
            n_skipped += 1
            continue
        if times_ref is None:
            times_ref = times_hi
        elif len(times_hi) != len(times_ref):
            print(f"  seed {seed}: {len(times_hi)} timesteps, expected "
                  f"{len(times_ref)}; skipped")
            n_skipped += 1
            continue

        # Keep only pairs whose LOW and HIGH traces are both non-negative throughout.
        keep = (ca_lo.min(axis=0) >= 0.0) & (ca_hi.min(axis=0) >= 0.0)
        n_dropped += int((~keep).sum())
        columns.append((ca_hi - ca_lo)[:, keep])
        synapse_idx = np.where(keep)[0]
        pair_seeds.extend([seed] * len(synapse_idx))
        pair_synapse_idx.extend(synapse_idx.tolist())

    if not columns:
        raise SystemExit("No matched synapses found across the two sweeps.")
    diffs = np.concatenate(columns, axis=1)
    if diffs.shape[1] == 0:
        raise SystemExit("All synapse pairs dropped for negative values; nothing to plot.")
    print(f"Pooled {diffs.shape[1]} synapse pairs from {len(common) - n_skipped} "
          f"seeds ({n_skipped} seeds skipped; {n_dropped} pairs dropped for negatives)")
    return times_ref, diffs, np.array(pair_seeds), np.array(pair_synapse_idx)


def plot_differences(times, diffs, low_label, high_label, ax):
    lows = np.nanquantile(diffs, CENTILE / 100.0, axis=1)
    medians = np.nanmedian(diffs, axis=1)
    means = np.nanmean(diffs, axis=1)
    highs = np.nanquantile(diffs, 1 - CENTILE / 100.0, axis=1)

    color = plt.cm.tab10.colors[0]

    if SHOW_INDIVIDUAL:
        n = diffs.shape[1]
        if n > MAX_INDIVIDUAL:
            sample = np.random.default_rng(0).choice(n, MAX_INDIVIDUAL, replace=False)
        else:
            sample = np.arange(n)
        ax.plot(times, diffs[:, sample], color="gray", alpha=0.04, linewidth=0.5)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(times, lows, highs, alpha=0.2, color=color)
    ax.plot(times, means, linewidth=2, linestyle="-", color=color,
            label=f"mean (n={diffs.shape[1]})")
    ax.plot(times, medians, linewidth=2, linestyle="--", color=color, label="median")

    # Scale the y-axis to the shaded band, not any residual single-trace outliers.
    y_lo = min(0.0, float(np.min(lows)))
    y_hi = max(0.0, float(np.max(highs)))
    margin = 0.15 * (y_hi - y_lo)
    ax.set_ylim(y_lo - margin, y_hi + margin)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"d[{SPECIES_NAME}] = [{high_label}] - [{low_label}] (mM)")
    ax.set_title(
        f"Per-synapse local [{SPECIES_NAME}] difference between ECS ratios "
        f"({CENTILE}-{100 - CENTILE}% range shaded)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def peak_differences(times, diffs):
    """Per-synapse signed d[Ca] at its own time of largest |d[Ca]|, and that time."""
    t_idx = np.nanargmax(np.abs(diffs), axis=0)
    return diffs[t_idx, np.arange(diffs.shape[1])], times[t_idx]


def load_vfrac(ratio_dir, pair_seeds, pair_synapse_idx):
    """Local ECS volume fraction at RADIUS for each (seed, synapse_idx) pair of one
    ``ecs_ratio=*`` sublevel, read from the pooled CSV at the sweep root (written by
    running evaluate_synapse_distribution_spatial.py on the root) via its ``group``
    column.

    Missing pairs (seeds the spatial evaluation skipped or never covered) come back
    as NaN, so the caller can just drop them.
    """
    root = os.path.dirname(ratio_dir)
    csv_path = os.path.join(root, "spatial_metrics.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(
            f"Error: '{csv_path}' not found. Run\n"
            f"  uv run python evaluate_synapse_distribution_spatial.py {root} "
            f"--radii {RADIUS:g}\nfirst.")
    df = pd.read_csv(csv_path)
    group = os.path.basename(ratio_dir)
    df = df[df["group"] == group]
    if df.empty:
        raise SystemExit(f"Error: no rows with group='{group}' in '{csv_path}'.")
    df = df.copy()
    df["vfrac"] = _vfrac(df, RADIUS)
    lookup = df.set_index(["seed", "synapse_idx"])["vfrac"]
    lookup = lookup[~lookup.index.duplicated()]
    return lookup.reindex(pd.MultiIndex.from_arrays(
        [pair_seeds, pair_synapse_idx])).to_numpy()


MARKER_RADIUS = 0.15  # um, synapse-location sphere
TITLE_POSITION = 2  # CornerAnnotation index for the default "upper_left" add_text position


def crop_to_synapse(grid, center, radius):
    """Restrict a snapshot grid to a box of half-width `radius` around `center`.

    `crinkle=True` keeps whole original cells instead of cutting through them at
    the box face -- a plain clip interpolates new points at the cut, which have
    no entry in `_orig_idx` and would corrupt the fast per-step field swap below.
    """
    cx, cy, cz = center
    bounds = (cx - radius, cx + radius, cy - radius, cy + radius, cz - radius, cz + radius)
    return grid.clip_box(bounds, invert=False, crinkle=True)


def _load_ecs_panel(path, step, center, radius):
    """ECS-only, cropped snapshot grid, plus what's needed to re-slice by step fast.

    Loading a full snapshot (mesh + field) via ResultLoader is not cheap, so the
    slider below does not call it again per step: it re-reads just the `Ca` array
    from the h5 file and re-indexes it through `_orig_idx`, the panel's point
    indices into the *full* (pre-crop, pre-ECS-mask) mesh, which is invariant
    across steps because this geometry does not deform over time.
    """
    loader = ResultLoader(path)
    grid = loader.load_snapshot(step)
    grid.point_data["ecs"] = loader.load_regions().point_data["ecs"]
    grid.point_data["_orig_idx"] = np.arange(grid.n_points)
    ecs_grid = grid.threshold(0.5, scalars="ecs")
    return crop_to_synapse(ecs_grid, center, radius)


def explore(panels, radius=EXPLORE_RADIUS, step=-1, species=SPECIES_NAME):
    """Show ECS-only snapshots side by side in linked pyvista views, one per
    ``(run_dir, center, other_center, label)`` in `panels`, each cropped to a box
    around its own synapse center, with a slider to scrub the recorded timepoints.
    Blocks until the window is closed, so the picker below runs it in a spawned
    process rather than call it directly (a second GUI event loop can't share the
    process with matplotlib's). `species` is a parameter, not the module global,
    because the spawned child re-imports this module and would only see the default.

    Each panel gets an opaque marker sphere at its own synapse center and, if
    `other_center` is not None, a translucent one at the other geometry's -- for
    LOW/HIGH the two coordinates are close but not identical (see module docstring),
    and this makes that offset visible instead of quietly plotting one as a stand-in
    for the other.
    """
    with h5py.File(os.path.join(panels[0][0], "snapshot.h5")) as h5:
        times = np.array(h5["data/time"])
    if step < 0:
        step += len(times)

    plotter = pv.Plotter(shape=(1, len(panels)))
    views = []
    for i, (path, own_center, other_center, label) in enumerate(panels):
        cropped = _load_ecs_panel(path, step, own_center, radius)
        plotter.subplot(0, i)
        # Semi-transparent: at this crop radius the synapse marker sphere is usually
        # inside a fold of the ECS surface, and an opaque mesh would hide it.
        actor = plotter.add_mesh(cropped, scalars=species, cmap="viridis", opacity=0.5)
        plotter.add_mesh(pv.Sphere(radius=MARKER_RADIUS, center=own_center), color="red")
        if other_center is not None:
            plotter.add_mesh(pv.Sphere(radius=MARKER_RADIUS, center=other_center),
                             color="red", opacity=0.5)
        title = plotter.add_text(f"{label}: {Path(path).name}", font_size=8)
        views.append((os.path.join(path, "snapshot.h5"), cropped, actor, title, label, path))

    def set_step(value):
        idx = int(round(value))
        fields = []
        for h5_path, cropped, actor, title, label, path in views:
            with h5py.File(h5_path) as h5:
                field = np.array(h5[f"data/{species}/step_{idx:05d}"])
            cropped.point_data[species] = field[cropped.point_data["_orig_idx"]]
            title.SetText(TITLE_POSITION,
                          f"{label}: {Path(path).name}\nt={times[idx]:.0f} ms")
            fields.append(cropped.point_data[species])

        # Shared color scale across both panels, so LOW and HIGH stay comparable
        # instead of each panel re-normalizing to its own range.
        combined = np.concatenate(fields)
        scalar_range = (float(combined.min()), float(combined.max()))
        for _, _, actor, _, _, _ in views:
            actor.mapper.scalar_range = scalar_range
        plotter.render()

    set_step(step)
    plotter.subplot(0, 0)
    plotter.add_slider_widget(set_step, [0, len(times) - 1], value=step,
                              title="step", fmt="%.0f")
    plotter.link_views()
    plotter.show()


def _connect_point_picker(ax, vfrac, peaks, peak_times, pair_seeds, pair_synapse_idx):
    """Click a scatter point to open that synapse's LOW/HIGH volumes in pyvista.

    Spawns `explore()` in a fresh process (see there for why; "spawn" rather than
    fork, so the child doesn't inherit matplotlib's live GUI state), passing each
    sweep's own synapse center -- idx-matched but not identical, see module
    docstring. Also prints the identifying info in case the pyvista window isn't
    wanted.
    """
    low_paths = dict(find_seed_dirs(LOW[0]))
    high_paths = dict(find_seed_dirs(HIGH[0]))

    def on_pick(event):
        for i in event.ind:
            seed, idx = int(pair_seeds[i]), int(pair_synapse_idx[i])
            low_dir, high_dir = low_paths[seed], high_paths[seed]
            print(f"\nseed={seed} synapse_idx={idx}  d_vfrac={vfrac[i]:.4f}  "
                  f"peak_dCa={peaks[i]:.3f} mM  t={peak_times[i]:.0f} ms")
            with h5py.File(os.path.join(low_dir, "snapshot.h5")) as h5:
                low_center = find_synapse_centers(h5)[idx]
            with h5py.File(os.path.join(high_dir, "snapshot.h5")) as h5:
                high_center = find_synapse_centers(h5)[idx]
            print(f"  LOW  center (um): ({low_center[0]:.2f}, {low_center[1]:.2f}, "
                  f"{low_center[2]:.2f})  dir: {low_dir}")
            print(f"  HIGH center (um): ({high_center[0]:.2f}, {high_center[1]:.2f}, "
                  f"{high_center[2]:.2f})  dir: {high_dir}")
            multiprocessing.get_context("spawn").Process(
                target=explore,
                args=([(low_dir, low_center, high_center, "LOW"),
                       (high_dir, high_center, low_center, "HIGH")],),
                kwargs={"species": SPECIES_NAME}).start()

    ax.figure.canvas.mpl_connect("pick_event", on_pick)


def plot_peak_vs_vfrac(times, diffs, pair_seeds, pair_synapse_idx,
                       low_label, high_label, ax):
    peaks, peak_times = peak_differences(times, diffs)
    # x axis is the per-synapse GAIN in local ECS between the two runs, not either
    # run's fraction on its own.
    vfrac = (load_vfrac(HIGH[0], pair_seeds, pair_synapse_idx)
             - load_vfrac(LOW[0], pair_seeds, pair_synapse_idx))

    keep = np.isfinite(vfrac) & np.isfinite(peaks)
    print(f"{keep.sum()} of {len(peaks)} synapse pairs have a finite r={RADIUS:g} "
          f"fraction in both sweeps")
    if keep.sum() < 2:
        raise SystemExit("Not enough synapses with a local ECS fraction to plot.")
    vfrac, peaks, peak_times = vfrac[keep], peaks[keep], peak_times[keep]
    pair_seeds, pair_synapse_idx = pair_seeds[keep], pair_synapse_idx[keep]

    color = plt.cm.tab10.colors[0]
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.scatter(vfrac, peaks, s=8, alpha=0.3, color=color, edgecolors="none",
               picker=True, pickradius=5)
    _connect_point_picker(ax, vfrac, peaks, peak_times, pair_seeds, pair_synapse_idx)

    # Binned median over quantile bins of the fraction (equal counts per bin).
    edges = np.nanquantile(vfrac, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(edges)
    bin_idx = np.clip(np.digitize(vfrac, edges[1:-1]), 0, len(edges) - 2)
    centers = np.array([np.median(vfrac[bin_idx == b]) for b in range(len(edges) - 1)])
    medians = np.array([np.median(peaks[bin_idx == b]) for b in range(len(edges) - 1)])
    ax.plot(centers, medians, "o-", color="black", linewidth=2, markersize=5,
            label=f"binned median ({N_BINS} quantile bins)")

    ax.set_xlabel(f"Local ECS volume fraction gain at r={RADIUS:g} um: "
                  f"{high_label} - {low_label}")
    ax.set_ylabel(f"Peak d[{SPECIES_NAME}] = [{high_label}] - [{low_label}] (mM)")
    ax.set_title(
        f"Per-synapse peak local [{SPECIES_NAME}] difference vs local ECS gain "
        f"(n={len(peaks)}, peak at t={np.median(peak_times):.0f} ms median)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main(plot_kind, out_path, show):
    low_sweep, low_label = LOW
    high_sweep, high_label = HIGH
    times, diffs, pair_seeds, pair_synapse_idx = collect_differences(low_sweep, high_sweep)

    peak = times[np.nanargmax(np.abs(np.nanmean(diffs, axis=1)))]
    print(f"Largest mean |d[Ca]| at t={peak:.1f} ms: "
          f"{np.nanmean(diffs, axis=1)[np.nanargmax(np.abs(np.nanmean(diffs, axis=1)))]:.3f} mM")

    for extreme, argfunc in [("Lowest", np.nanargmin), ("Highest", np.nanargmax)]:
        t_idx, pair_idx = np.unravel_index(argfunc(diffs), diffs.shape)
        print(f"{extreme} d[Ca]={diffs[t_idx, pair_idx]:.3f} mM at t={times[t_idx]:.1f} ms: "
              f"seed={pair_seeds[pair_idx]}, synapse_idx={pair_synapse_idx[pair_idx]}")

    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_kind == "trace":
        plot_differences(times, diffs, low_label, high_label, ax)
    else:
        plot_peak_vs_vfrac(times, diffs, pair_seeds, pair_synapse_idx,
                           low_label, high_label, ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep",
                        help="Sweep directory containing two ecs_ratio=* sublevels.")
    parser.add_argument("--plot", choices=list(OUT_NAMES), default="trace",
                        help="Which analysis to plot (default: trace).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: per-mode entry in OUT_NAMES, "
                             "inside the sweep directory).")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the figure without opening a window.")
    parser.add_argument("--species", default=SPECIES_NAME,
                        help=f"Species to compare (default: {SPECIES_NAME}).")
    parser.add_argument("--centile", type=float, default=CENTILE,
                        help=f"Shaded band is CENTILE..(100-CENTILE)%% "
                             f"(default: {CENTILE}).")
    parser.add_argument("--hide-individual", action="store_true",
                        help="Don't overlay individual per-synapse traces.")
    parser.add_argument("--max-individual", type=int, default=MAX_INDIVIDUAL,
                        help=f"Cap on individual traces drawn (default: {MAX_INDIVIDUAL}).")
    parser.add_argument("--radius", type=float, default=RADIUS,
                        help=f"Local ECS fraction radius in um, selects the "
                             f"v_local_r<R>/v_sphere_box_r<R> columns (default: {RADIUS:g}).")
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help=f"Quantile bins for the binned median in vs-ecs "
                             f"(default: {N_BINS}).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    LOW, HIGH = find_ratio_dirs(args.sweep)
    SPECIES_NAME = args.species
    CENTILE = args.centile
    SHOW_INDIVIDUAL = not args.hide_individual
    MAX_INDIVIDUAL = args.max_individual
    RADIUS = args.radius
    N_BINS = args.bins
    main(args.plot, args.out or os.path.join(args.sweep, OUT_NAMES[args.plot]),
         not args.no_show)
