"""Check that evaluate.py's front criterion is scenario-independent.

Builds analytic constant-flux profiles: the same shape, placed on each
scenario's baseline and flux direction. Runs them through ``track_front``
and ``fit_effective_diffusivity``; each must recover the diffusivity used to
build it. Without the |C - C_init| generalization, offset scenarios return
NaN (nothing drops below 0.5*C(0,t)), and the reversed scenario also returns
NaN (its profile rises with y instead).

    uv run scripts/ongoing_work/buffered_diffusion/test_front_criterion.py
"""
import math

import numpy as np

import evaluate
from simulation import Config

D = 0.7  # um^2/ms, the diffusivity the synthetic profiles are built with
Q = 0.03  # mM um/ms, flux density; only sets the amplitude, not the front position


def analytic_profiles(times, y, ca_init, sign):
    """C(y,t) = ca_init + sign * 2 q sqrt(t)/sqrt(pi D) * f(y / (2 sqrt(D t)))."""
    profile = np.vectorize(evaluate._constant_flux_profile)
    surface = 2 * Q * np.sqrt(times) / math.sqrt(math.pi * D)
    eta = y[None, :] / (2 * np.sqrt(D * times)[:, None])
    return ca_init + sign * surface[:, None] * profile(eta)


def main():
    times = np.arange(10.0, 1001.0, 10.0)  # ms, as recorded
    y = np.linspace(0.05, 59.5, evaluate.N_SAMPLE_POINTS)

    tracked = {}
    for scenario in evaluate.SCENARIOS:
        ca_init = float(Config(scenario=scenario).initial_ca.to_value("mM"))
        sign = -1.0 if scenario == "replenishment" else 1.0
        ca = analytic_profiles(times, y, ca_init, sign)

        cs, y_half = evaluate.track_front(times, y, ca, ca_init)
        d_eff, mask = evaluate.fit_effective_diffusivity(times, y_half)
        tracked[scenario] = (cs, y_half)

        assert np.all(np.isfinite(y_half)), f"{scenario}: front lost"
        assert mask.sum() > 10, f"{scenario}: fit window nearly empty ({mask.sum()} points)"
        assert abs(d_eff - D) < 0.01 * D, f"{scenario}: D_eff={d_eff:.4f}, expected {D}"
        print(f"  {scenario:>14}: D_eff = {d_eff:.4f} um^2/ms (built with {D})")

    # Same underlying profile in every scenario. Baseline and flux direction
    # must drop out of both tracked quantities: the reversed scenario must
    # report a positive excess amplitude, not a negative one.
    for scenario, (cs, y_half) in tracked.items():
        assert np.allclose(cs, tracked[evaluate.BASELINE][0]), f"{scenario}: excess differs"
        assert np.allclose(y_half, tracked[evaluate.BASELINE][1]), f"{scenario}: front differs"

    print("front criterion is scenario-independent: ok")


if __name__ == "__main__":
    main()
