"""Analyze the buffered-diffusion experiment.

Loads every scenario ``simulation.py`` has results for (missing ones are
skipped), samples [Ca] along the long (y) axis of the box at every recorded
snapshot, tracks the half-maximum front position y_half(t), and fits y_half^2
vs t to extract an effective diffusivity per scenario.

Ca crosses y = 0 as a constant flux. For constant-flux diffusion into a
semi-infinite medium the *excess* over the initial concentration, normalized by
its surface value, is self-similar in eta = y / (2*sqrt(D t)):

    S(y,t) / S(0,t) = exp(-eta^2) - sqrt(pi)*eta*erfc(eta),   S = |C - C_init|.

The half-maximum front (S = 0.5*S(0,t)) therefore sits at a fixed eta_half, so
y_half = 2*eta_half*sqrt(D t), i.e. y_half^2 = (4*eta_half^2) * D * t. Fitting
the slope of y_half^2 vs t gives D_eff = slope / (4*eta_half^2). For the
buffered (nonlinear) scenarios this is an *apparent* effective diffusivity -- a
directly comparable measure of how fast the front advances.

Front-tracking the excess magnitude S rather than C itself is what makes the
scenarios comparable: "saturated"/"replenishment" start from a uniform
C_init = Kd, against which a threshold on C carries no information, and
"replenishment" draws Ca *out*, so its excess is negative. S decreases
monotonically away from the source in all four scenarios, and for C_init = 0
with an influx it is just C, i.e. the criterion used before the scenarios
existed. The single "nobuffer" run is a valid reference for all of them: pure
diffusion is linear, so its excess profile is unaffected by a baseline offset and
an efflux would only mirror its sign. Caveat: the buffer's differential capacity
total*Kd/(C+Kd)^2 falls as
C rises, so the front accelerates in "depleted"/"saturated" and decelerates in
"replenishment" -- a single fitted slope averages over that curvature, and the
value depends somewhat on the fit window.
"""
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bmbcsim import ResultLoader
from bmbcsim.units import to_simulation_units
from simulation import Config

# Geometry constants, read off simulation.py's config defaults. The source face is
# at y = 0, so distance from the source equals y.
_DEFAULTS = Config()
RESULT_ROOT = "results"
BOX_LENGTH = to_simulation_units(_DEFAULTS.box.length_y, "length")  # um, long axis
MID_X = 0.0
MID_Z = to_simulation_units(_DEFAULTS.box.height_z, "length") / 2.0  # mid-height
FREE_DIFFUSIVITY = to_simulation_units(_DEFAULTS.diffusivity, "diffusivity")  # reference

N_SAMPLE_POINTS = 120
FRONT_FRACTION = 0.5  # front = where the excess drops to this fraction of its surface value


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

BASELINE = "nobuffer"  # the free-diffusion reference every other scenario is scaled by
SCENARIOS = [BASELINE, "depleted", "saturated", "replenishment"]
COLORS = {"nobuffer": "tab:blue", "depleted": "tab:red",
          "saturated": "tab:orange", "replenishment": "tab:green"}
STYLES = {"nobuffer": "-", "depleted": "--", "saturated": "-.", "replenishment": ":"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "processed-data")


def load_kymograph(simulation_name):
    """Return (loader, times_ms, y_from_source_um, ca_mM[n_times, n_points])."""
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

    return loader, np.array(times), y_from_source, np.array(profiles)


def half_max_position(y_from_source, profile, threshold):
    """First position (from the source) where the profile drops below threshold."""
    below = profile < threshold
    if not below.any() or below[0]:
        return np.nan
    i = int(np.argmax(below))  # first index that is below threshold
    y0, y1 = y_from_source[i - 1], y_from_source[i]
    c0, c1 = profile[i - 1], profile[i]
    if c1 == c0:
        return y1
    return y0 + (threshold - c0) * (y1 - y0) / (c1 - c0)


def track_front(times, y_from_source, ca, ca_init=0.0):
    """Return (S[t], y_half[t]): excess magnitude at the source, and front position.

    The tracked signal is S = |C - ca_init|, not C: see the module docstring. For
    ca_init = 0 and an influx it is C, so nothing changes for those scenarios.
    """
    excess = np.abs(ca - ca_init)
    cs = excess[:, 0]  # excess at the source-most sample point
    y_half = np.array([
        half_max_position(y_from_source, excess[i], FRONT_FRACTION * cs[i])
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


def analyze_run(simulation_name, ca_init=0.0):
    """Load a run and front-track it. Returns a dict with times, y, ca, cs,
    y_half, d_eff, mask, loader -- everything main()'s report/CSV/plot code needs,
    and what other scripts need to pull the fitted diffusivity back out.

    :param ca_init: The run's uniform initial [Ca] (mM), i.e. ``Config.initial_ca``;
        the front is tracked on the excess over it.
    """
    loader, times, y, ca = load_kymograph(simulation_name)
    cs, y_half = track_front(times, y, ca, ca_init)
    d_eff, mask = fit_effective_diffusivity(times, y_half)
    return dict(times=times, y=y, ca=ca, cs=cs, y_half=y_half, d_eff=d_eff, mask=mask,
                loader=loader)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    results = {}
    for scenario in SCENARIOS:
        # The initial [Ca] the front is measured against is the scenario's, not a
        # property of the recorded data; take it from the same config that ran it.
        ca_init = to_simulation_units(Config(scenario=scenario).initial_ca)
        try:
            results[scenario] = analyze_run(f"buffered_diffusion_{scenario}", ca_init)
        except RuntimeError as exc:  # not every batch runs all four scenarios
            print(f"-- skipping '{scenario}': {exc}")
    if BASELINE not in results:
        raise RuntimeError(
            f"No '{BASELINE}' run in {RESULT_ROOT}/: it is the reference the other "
            "scenarios are scaled by. Run simulation.py with scenario=nobuffer."
        )

    # --- Report -----------------------------------------------------
    print(f"\n(front at eta_half={ETA_HALF:.4f}, y_half^2 = {FRONT_COEFF:.4f} * D * t)")
    print("\n=== Effective diffusivity (apparent) ===")
    for scenario, r in results.items():
        print(f"  {scenario:>14}: D_eff = {r['d_eff']:.3f} um^2/ms")
    d_free = results[BASELINE]["d_eff"]
    print(f"\n  free-diffusion reference   : {FREE_DIFFUSIVITY:.3f} um^2/ms")
    print(f"  {BASELINE} / reference      : {d_free / FREE_DIFFUSIVITY:.2f}")
    for scenario, r in results.items():
        if scenario == BASELINE or not (np.isfinite(r["d_eff"]) and r["d_eff"] > 0):
            continue
        print(f"  {scenario:>14}: front slowed by {d_free / r['d_eff']:.2f}x "
              f"(D_eff ratio to {BASELINE} = {r['d_eff'] / d_free:.2f})")

    # --- CSV --------------------------------------------------------
    csv_path = os.path.join(DATA_DIR, "buffered_diffusion_front.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scenario,time_ms,excess_at_source_mM,y_half_um\n")
        for scenario, r in results.items():
            for t, cs, yh in zip(r["times"], r["cs"], r["y_half"]):
                f.write(f"{scenario},{t:.4f},{cs:.6f},{yh:.6f}\n")
    print(f"\nWrote {csv_path}")

    # --- Figure -----------------------------------------------------
    fig, (ax_prof, ax_fit) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) concentration profiles at a few snapshot times (raw [Ca], so that the
    # scenarios' different baselines and flux directions are visible)
    snapshot_times = [100.0, 300.0, 600.0, 1000.0]  # ms
    for scenario, r in results.items():
        for tt in snapshot_times:
            idx = int(np.argmin(np.abs(r["times"] - tt)))
            ax_prof.plot(
                r["y"], r["ca"][idx],
                STYLES[scenario], color=COLORS[scenario], alpha=0.4 + 0.5 * tt / 1000.0,
                label=f"{scenario}, t={r['times'][idx]:.0f} ms",
            )
    ax_prof.set_xlabel("distance from source (um)")
    ax_prof.set_ylabel("[Ca] (mM)")
    ax_prof.set_title("Concentration profiles along the box")
    # Upper right: the far end of the box sits at the baseline, so nothing is plotted
    # there, while the upper left is where the saturated profiles peak.
    ax_prof.legend(fontsize=7, ncol=len(results), loc="upper right")
    ax_prof.grid(alpha=0.3)

    # (b) y_half^2 vs t with fits
    for scenario, r in results.items():
        m = r["mask"]
        ax_fit.plot(r["times"], r["y_half"] ** 2, "o", ms=3,
                    color=COLORS[scenario], alpha=0.5, label=f"{scenario} (data)")
        if np.isfinite(r["d_eff"]):
            tline = r["times"][m]
            slope = r["d_eff"] * FRONT_COEFF
            offset = np.mean(r["y_half"][m] ** 2 - slope * tline)
            ax_fit.plot(tline, slope * tline + offset, STYLES[scenario],
                        color=COLORS[scenario], lw=2,
                        label=f"{scenario}: D_eff={r['d_eff']:.3f} um^2/ms")
    ax_fit.set_xlabel("time (ms)")
    ax_fit.set_ylabel(r"$y_{1/2}^2$ (um$^2$)")
    ax_fit.set_title("Front position: $y_{1/2}^2 \\propto D_{eff}\\, t$")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(alpha=0.3)

    fig.suptitle("Buffering slows Ca$^{2+}$ diffusion through an elongated box")
    fig.tight_layout()
    # The comparison lives with a buffered run: those are the ones being
    # characterized. Only one run's plots/ is asked for, so the others are not
    # created empty.
    owner = next((s for s in results if s != BASELINE), BASELINE)
    fig_path = os.path.join(results[owner]["loader"].plot_dir, "buffered_diffusion.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
