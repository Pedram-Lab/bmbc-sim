"""Analyze the buffered-diffusion experiment.

Loads the two runs produced by ``simulation.py`` (no buffer / with buffer),
samples [Ca] along the long (y) axis of the box at every recorded snapshot,
tracks the half-maximum front position y_half(t), and fits y_half^2 vs t to
extract an effective diffusivity for each condition.

Ca enters at y = 0 as a constant flux. For constant-flux diffusion into a
semi-infinite medium the concentration profile, normalized by its surface
value, is self-similar in eta = y / (2*sqrt(D t)):

    C(y,t) / C(0,t) = exp(-eta^2) - sqrt(pi)*eta*erfc(eta).

The half-maximum front (C = 0.5*C(0,t)) therefore sits at a fixed eta_half, so
y_half = 2*eta_half*sqrt(D t), i.e. y_half^2 = (4*eta_half^2) * D * t. Fitting
the slope of y_half^2 vs t gives D_eff = slope / (4*eta_half^2). For the
buffered (nonlinear) case this is an *apparent* effective diffusivity -- a
directly comparable measure of how fast the front advances.
"""
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bmbcsim import ResultLoader
from bmbcsim.units import to_simulation_units
from simulation import BOX_LENGTH_Y, BOX_HEIGHT_Z

# Geometry constants (taken from simulation.py). The source face is at y = 0,
# so distance from the source equals y.
RESULT_ROOT = "results"
BOX_LENGTH = to_simulation_units(BOX_LENGTH_Y, "length")  # um, along the long axis
MID_X = 0.0
MID_Z = to_simulation_units(BOX_HEIGHT_Z, "length") / 2.0  # mid-height
FREE_DIFFUSIVITY = 0.7  # um^2/ms, for reference

N_SAMPLE_POINTS = 120
FRONT_FRACTION = 0.5  # front = where C drops to this fraction of the surface value


def _constant_flux_profile(eta):
    """Self-similar normalized profile C(y,t)/C(0,t) for constant-flux diffusion."""
    return math.exp(-eta**2) - math.sqrt(math.pi) * eta * math.erfc(eta)


def front_eta(fraction):
    """Similarity variable eta where the normalized profile equals `fraction`."""
    lo, hi = 0.0, 10.0
    for _ in range(100):  # bisection (profile is monotonically decreasing in eta)
        mid = 0.5 * (lo + hi)
        if _constant_flux_profile(mid) > fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# y_half^2 = FRONT_COEFF * D * t  with FRONT_COEFF = 4 * eta_half^2
ETA_HALF = front_eta(FRONT_FRACTION)
FRONT_COEFF = 4.0 * ETA_HALF**2

CONDITIONS = {
    "no buffer": "buffered_diffusion_nobuffer",
    "with buffer": "buffered_diffusion_buffer",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
DATA_DIR = os.path.join(SCRIPT_DIR, "processed-data")


def load_kymograph(simulation_name):
    """Return (times_ms, y_from_source_um, ca_mM[n_times, n_points])."""
    loader = ResultLoader.find(
        simulation_name=simulation_name, results_root=RESULT_ROOT
    )

    # Sample line down the long axis. The first point sits essentially on the
    # source face (y ~ 0) so that cs = C(0,t): referencing the surface at an
    # inset point would bias D_eff high by a few percent, since the half-max
    # threshold 0.5*cs would then be taken against a value below the true C(0,t).
    # Pass a list of [x, y, z] lists (ResultLoader.load_point_values wraps a
    # bare 2-D ndarray incorrectly).
    y = np.linspace(0.05, BOX_LENGTH - 0.5, N_SAMPLE_POINTS)
    points = [[MID_X, float(yi), MID_Z] for yi in y]
    y_from_source = y  # source face is at y = 0

    times, profiles = [], []
    for step in range(len(loader)):
        ds = loader.load_point_values(step, points)
        times.append(float(ds.coords["time"].values[0]))
        profiles.append(ds.sel(species="Ca").values[0])

    return np.array(times), y_from_source, np.array(profiles)


def half_max_position(y_from_source, ca_profile, threshold):
    """First position (from the source) where the profile drops below threshold."""
    below = ca_profile < threshold
    if not below.any() or below[0]:
        return np.nan
    i = int(np.argmax(below))  # first index that is below threshold
    y0, y1 = y_from_source[i - 1], y_from_source[i]
    c0, c1 = ca_profile[i - 1], ca_profile[i]
    if c1 == c0:
        return y1
    return y0 + (threshold - c0) * (y1 - y0) / (c1 - c0)


def track_front(times, y_from_source, ca):
    """Return (Cs[t], y_half[t]) arrays for the source value and front position."""
    cs = ca[:, 0]  # concentration at the source-most sample point
    y_half = np.array([
        half_max_position(y_from_source, ca[i], FRONT_FRACTION * cs[i])
        for i in range(len(times))
    ])
    return cs, y_half


def fit_effective_diffusivity(times, y_half):
    """Fit y_half^2 vs t over the established-front window; return (D_eff, mask)."""
    # Constant-flux diffusion is self-similar for all t>0; restrict the fit to
    # where the front is resolved (> 3 um past the first sample) and has not yet
    # reached the far wall (< 0.6 * box length).
    mask = np.isfinite(y_half) & (y_half > 3.0) & (y_half < 0.6 * BOX_LENGTH)
    if mask.sum() < 2:
        return np.nan, mask
    slope, _ = np.polyfit(times[mask], y_half[mask] ** 2, 1)
    return slope / FRONT_COEFF, mask


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    results = {}
    for label, name in CONDITIONS.items():
        times, y_from_source, ca = load_kymograph(name)
        cs, y_half = track_front(times, y_from_source, ca)
        d_eff, mask = fit_effective_diffusivity(times, y_half)
        results[label] = dict(
            times=times, y=y_from_source, ca=ca,
            cs=cs, y_half=y_half, d_eff=d_eff, mask=mask,
        )

    # --- Report -----------------------------------------------------
    print(f"\n(front at eta_half={ETA_HALF:.4f}, y_half^2 = {FRONT_COEFF:.4f} * D * t)")
    print("\n=== Effective diffusivity (apparent) ===")
    for label in CONDITIONS:
        print(f"  {label:>12}: D_eff = {results[label]['d_eff']:.3f} um^2/ms")
    d_free = results["no buffer"]["d_eff"]
    d_buf = results["with buffer"]["d_eff"]
    print(f"\n  free-diffusion reference   : {FREE_DIFFUSIVITY:.3f} um^2/ms")
    print(f"  no-buffer / reference      : {d_free / FREE_DIFFUSIVITY:.2f}")
    if np.isfinite(d_buf) and d_buf > 0:
        print(f"  buffer slows front by      : {d_free / d_buf:.2f}x "
              f"(D_eff ratio buffer/no-buffer = {d_buf / d_free:.2f})")

    # --- CSV --------------------------------------------------------
    csv_path = os.path.join(DATA_DIR, "buffered_diffusion_front.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("condition,time_ms,Cs_mM,y_half_um\n")
        for label in CONDITIONS:
            r = results[label]
            for t, cs, yh in zip(r["times"], r["cs"], r["y_half"]):
                f.write(f"{label},{t:.4f},{cs:.6f},{yh:.6f}\n")
    print(f"\nWrote {csv_path}")

    # --- Figure -----------------------------------------------------
    fig, (ax_prof, ax_fit) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"no buffer": "tab:blue", "with buffer": "tab:red"}
    styles = {"no buffer": "-", "with buffer": "--"}

    # (a) concentration profiles at a few snapshot times
    snapshot_times = [100.0, 300.0, 600.0, 1000.0]  # ms
    for label in CONDITIONS:
        r = results[label]
        for tt in snapshot_times:
            idx = int(np.argmin(np.abs(r["times"] - tt)))
            ax_prof.plot(
                r["y"], r["ca"][idx],
                styles[label], color=colors[label], alpha=0.4 + 0.5 * tt / 1000.0,
                label=f"{label}, t={r['times'][idx]:.0f} ms",
            )
    ax_prof.set_xlabel("distance from source (um)")
    ax_prof.set_ylabel("[Ca] (mM)")
    ax_prof.set_title("Concentration profiles along the box")
    ax_prof.legend(fontsize=7, ncol=2)
    ax_prof.grid(alpha=0.3)

    # (b) y_half^2 vs t with fits
    for label in CONDITIONS:
        r = results[label]
        m = r["mask"]
        ax_fit.plot(r["times"], r["y_half"] ** 2, "o", ms=3,
                    color=colors[label], alpha=0.5, label=f"{label} (data)")
        if np.isfinite(r["d_eff"]):
            tline = r["times"][m]
            slope = r["d_eff"] * FRONT_COEFF
            offset = np.mean(r["y_half"][m] ** 2 - slope * tline)
            ax_fit.plot(tline, slope * tline + offset, styles[label],
                        color=colors[label], lw=2,
                        label=f"{label}: D_eff={r['d_eff']:.3f} um^2/ms")
    ax_fit.set_xlabel("time (ms)")
    ax_fit.set_ylabel(r"$y_{1/2}^2$ (um$^2$)")
    ax_fit.set_title("Front position: $y_{1/2}^2 \\propto D_{eff}\\, t$")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(alpha=0.3)

    fig.suptitle("Buffering slows Ca$^{2+}$ diffusion through an elongated box")
    fig.tight_layout()
    fig_path = os.path.join(PLOT_DIR, "buffered_diffusion.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
