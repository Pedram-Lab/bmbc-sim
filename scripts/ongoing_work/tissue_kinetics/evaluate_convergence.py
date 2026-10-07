"""Space/time convergence of ECS Ca for a `convergence_sweep` run.

Per run, per recorded time: the volume-weighted distribution of Ca over the ECS
(tet-mean Ca, bins 0-1.3 mM in 0.05 mM steps), plus the mean and minimum ECS Ca.
Levels are compared by the L1 distance between histograms (max over time) and by the
max-norm difference of the mean/min traces; the observed order is log2(e_coarse/e_fine).

    uv run scripts/ongoing_work/tissue_kinetics/evaluate_convergence.py results/convergence-sweep_<stamp>
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy import units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bmbcsim.simulation.result_io import find_run_dirs

BINS = np.arange(0.0, 1.3 + 0.025, 0.05)  # 26 bins, 0-1.3 mM
AXES = ("n_refine", "time_step")


def ecs_stats(run_dir):
    """(times, hist[t, bin], mean[t], min[t]) over ECS tets, volume weighted."""
    with h5py.File(Path(run_dir) / "snapshot.h5", "r") as h5:
        pts = h5["mesh/points"][:].astype(np.float64)
        tets = h5["mesh/connectivity"][:].astype(np.int64)
        ecs = (h5["compartments/ecs"][:] > 0.5)[tets].all(axis=1)
        tets = tets[ecs]
        a, b, c, d = (pts[tets[:, i]] for i in range(4))
        vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
        times = h5["data/time"][:]
        steps = sorted(h5["data/Ca"].keys())
        hist = np.empty((len(steps), len(BINS) - 1))
        mean = np.empty(len(steps))
        mn = np.empty(len(steps))
        for i, step in enumerate(steps):
            ca = h5[f"data/Ca/{step}"][:].astype(np.float64)[tets].mean(axis=1)
            hist[i] = np.histogram(np.clip(ca, BINS[0], BINS[-1] - 1e-9), BINS, weights=vol)[0]
            mean[i] = (ca * vol).sum()
            mn[i] = ca.min()
    return times, hist / vol.sum(), mean / vol.sum(), mn


def load_sweep(sweep_dir):
    """{(ecm_enabled, n_refine, time_step_ms): stats}, from each run's config.yaml."""
    runs = {}
    for run in find_run_dirs(sweep_dir):
        if not (run / "snapshot.h5").exists():
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        key = (bool(cfg["ecm"]["enabled"]), int(cfg["geometry"]["n_refine"]),
               float(u.Quantity(cfg["time_step"]).to_value(u.ms)))
        runs[key] = ecs_stats(run)
    return runs


def l1_hist(s1, s2):
    return np.abs(s1[1] - s2[1]).sum(axis=1).max()


def linf(s1, s2, idx):
    return np.abs(s1[idx] - s2[idx]).max()


METRICS = (("hist L1", l1_hist),
           ("mean Ca Linf", lambda a, b: linf(a, b, 2)),
           ("min Ca Linf", lambda a, b: linf(a, b, 3)))


def report(runs, ecm, out):
    refines = sorted({k[1] for k in runs if k[0] == ecm})
    dts = sorted({k[2] for k in runs if k[0] == ecm}, reverse=True)
    lines = [f"\n=== ECM {ecm} ===", "metric | fixed | e(level0->1)  e(level1->2)  order"]
    for name, err in METRICS:
        for dt in dts:  # spatial, dt fixed
            ks = [(ecm, r, dt) for r in refines]
            if len(ks) < 2 or not all(k in runs for k in ks):
                continue
            e = [err(runs[ks[i]], runs[ks[i + 1]]) for i in range(len(ks) - 1)]
            lines.append(f"{name:13s}| h, dt={dt:<5g} | " + "  ".join(f"{x:.3e}" for x in e)
                         + (f"  p={np.log2(e[0] / e[1]):.2f}" if len(e) == 2 and e[1] > 0 else ""))
        for r in refines:  # temporal, h fixed
            ks = [(ecm, r, dt) for dt in dts]
            if len(ks) < 2 or not all(k in runs for k in ks):
                continue
            e = [err(runs[ks[i]], runs[ks[i + 1]]) for i in range(len(ks) - 1)]
            lines.append(f"{name:13s}| dt, refine={r} | " + "  ".join(f"{x:.3e}" for x in e)
                         + (f"  p={np.log2(e[0] / e[1]):.2f}" if len(e) == 2 and e[1] > 0 else ""))
    text = "\n".join(lines)
    print(text)
    (out / f"convergence_ecm-{ecm}.txt").write_text(text)

    # Error plot: successive-level differences vs. the coarser level's h (rel.) or dt,
    # log-log, one column per metric, with order-1/2 guide lines.
    fig, axs = plt.subplots(2, len(METRICS), figsize=(4 * len(METRICS), 7))
    for col, (name, err) in enumerate(METRICS):
        for fixed_dt in dts:  # h-refinement
            ks = [(ecm, r, fixed_dt) for r in refines]
            if all(k in runs for k in ks):
                e = [err(runs[ks[i]], runs[ks[i + 1]]) for i in range(len(ks) - 1)]
                axs[0, col].loglog([0.5 ** r for r in refines[:-1]], e, "o-", label=f"dt={fixed_dt:g} ms")
        for fixed_r in refines:  # dt-refinement
            ks = [(ecm, fixed_r, dt) for dt in dts]
            if all(k in runs for k in ks):
                e = [err(runs[ks[i]], runs[ks[i + 1]]) for i in range(len(ks) - 1)]
                axs[1, col].loglog(dts[:-1], e, "o-", label=f"refine={fixed_r}")
        for row, xlabel in enumerate(("h / h0 (coarser level)", "dt [ms] (coarser level)")):
            ax = axs[row, col]
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            xs = np.array([x0, x1])
            for p_, ls in ((1, ":"), (2, "--")):
                ax.loglog(xs, y1 * (xs / x1) ** p_, "k" + ls, lw=0.8, label=f"order {p_}")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_xlabel(xlabel); ax.set_title(name); ax.legend(fontsize=7)
    axs[0, 0].set_ylabel("|level k - level k+1|,  h-refinement")
    axs[1, 0].set_ylabel("|level k - level k+1|,  dt-refinement")
    fig.suptitle(f"Convergence, ECM {ecm}")
    fig.tight_layout()
    fig.savefig(out / f"errors_ecm-{ecm}.png", dpi=150)

    # Histogram overlays at peak depletion (min of mean Ca on the finest run) and at the end
    ref = runs[(ecm, refines[-1], dts[-1])]
    i_peak = int(np.argmin(ref[2]))
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    for row, (title, series) in enumerate((
        (f"dt = {dts[-1]:g} ms, h varies", [(ecm, r, dts[-1]) for r in refines]),
        (f"refine = {refines[-1]}, dt varies", [(ecm, refines[-1], dt) for dt in dts]),
    )):
        for col, (tlabel, idx) in enumerate((("peak depletion", i_peak), ("end", -1))):
            ax = axs[row, col]
            for k in series:
                if k in runs:
                    ax.step(centers, runs[k][1][idx], where="mid",
                            label=f"refine={k[1]}, dt={k[2]:g} ms")
            ax.set_title(f"{title}; {tlabel} (t = {ref[0][idx]:.0f} ms)")
            ax.set_xlabel("ECS Ca [mM]"); ax.set_ylabel("volume fraction")
            ax.legend(fontsize=8)
    fig.suptitle(f"ECS Ca distribution, ECM {ecm}")
    fig.tight_layout()
    fig.savefig(out / f"histograms_ecm-{ecm}.png", dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for k in sorted(k for k in runs if k[0] == ecm):
        t, _, mean, mn = runs[k]
        style = dict(label=f"refine={k[1]}, dt={k[2]:g} ms", lw=1)
        axs[0].plot(t, mean, **style); axs[1].plot(t, mn, **style)
    axs[0].set_title("mean ECS Ca [mM]"); axs[1].set_title("min ECS Ca (tet mean) [mM]")
    for ax in axs: ax.set_xlabel("t [ms]")
    axs[1].legend(fontsize=7)
    fig.suptitle(f"ECM {ecm}")
    fig.tight_layout()
    fig.savefig(out / f"traces_ecm-{ecm}.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <sweep_dir>/plots")
    args = ap.parse_args()
    out = args.out or args.sweep_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_sweep(args.sweep_dir)
    print(f"{len(runs)} runs: " + ", ".join(f"ecm={k[0]},refine={k[1]},dt={k[2]:g}" for k in sorted(runs)))
    for ecm in sorted({k[0] for k in runs}):
        report(runs, ecm, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
