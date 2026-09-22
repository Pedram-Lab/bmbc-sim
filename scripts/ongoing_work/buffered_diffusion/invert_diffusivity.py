"""Invert the buffered-diffusion forward model.

``simulation.run(Config(scenario=..., diffusivity=D, ...))`` builds a front.
``evaluate.py`` fits it to an effective diffusivity D_eff(D). D_eff has no
closed form; we measure it from the simulation. So we invert the pipeline as
a black box, with a bracketed line search (``scipy.optimize.brentq`` on
``D_eff(D) - target``). D_eff(D) rises with D, so the search always
converges. Any buffered scenario works (``--scenario``): buffering slows the
front the same way in all of them.

Each D_eff(D) evaluation is a full FEM run. Expect about ten runs per target.

This calibrates ``scripts/ongoing_work/tissue_kinetics/simulation.py``. Run
it once with ``ecm.enabled=false`` (diffusivity_ecs=TISSUE_DIFFUSIVITY_ECS),
then once with ``ecm.enabled=true`` using the diffusivity this script
returns. That matches the buffered run's effective ECS diffusivity to the
unbuffered baseline. The (kf, Kd) constants below are pinned by hand, not
read from ``simulation.py``'s defaults: this script is the ground truth for
"the tissue values", and the tissue config has since drifted (its Kd default
is now 10 mM; its contraction sweeps use kf = 769.23 / (mM s)). Check the
constants against the tissue config you actually want to match.
"""
import argparse

import astropy.units as u
import numpy as np
from scipy.optimize import brentq

import evaluate
from bmbcsim.units import to_simulation_units
from simulation import Config, run

RESULT_ROOT = "results"
# The tissue sim starts its ECS at ca_ecs, with the ECM at equilibrium. Its
# synapses are Ca sinks, so the ECS sees a depletion wave off that baseline:
# "replenishment". The other scenarios also invert, but only match a tissue
# run that starts from the same buffer state and drains it the same way.
DEFAULT_SCENARIO = "replenishment"

# Mirrors scripts/ongoing_work/tissue_kinetics/simulation.py's defaults.
TISSUE_DIFFUSIVITY_ECS = 0.7  # um^2/ms, the "diffusion only" baseline to match
TISSUE_CA_ECS = "1.3 mM"
TISSUE_ECM_TOTAL = "1.3 mM"
TISSUE_KD = "1.0 mM"
TISSUE_ECM_KF = "10.0 / (mM s)"


def measure_d_eff(diffusivity, *, scenario=DEFAULT_SCENARIO, result_root=RESULT_ROOT,
                  **overrides):
    """Run the buffered simulation at `diffusivity` (um^2/ms) and return the
    measured effective diffusivity (um^2/ms), using evaluate.py's own fit.

    :param scenario: Which buffered scenario to invert; see ``simulation.Config``.
    :param overrides: Extra ``Config`` fields, e.g. ``end_time`` or ``box``.
    """
    cfg = Config(
        result_root=result_root,
        scenario=scenario,
        ca_source=TISSUE_CA_ECS,
        diffusivity=diffusivity * u.um**2 / u.ms,
        buffer={
            "ecm_total": TISSUE_ECM_TOTAL,
            "ecm_kf": TISSUE_ECM_KF,
            "kd": TISSUE_KD,
        },
        **overrides,
    )
    run(cfg)
    evaluate.RESULT_ROOT = result_root
    # The front is measured against the scenario's initial [Ca], not 0.
    # Equilibrated scenarios start at Kd; a plain [Ca] threshold would find
    # no front there, and the fit below would fail.
    d_eff = evaluate.analyze_run(
        cfg.run_name, to_simulation_units(cfg.initial_ca)
    )["d_eff"]
    if not np.isfinite(d_eff):
        raise RuntimeError(
            f"Could not fit D_eff at diffusivity={diffusivity:.4g} um^2/ms: "
            "the front never crossed the fit window."
        )
    return d_eff


def find_diffusivity(target_d_eff, *, scenario=DEFAULT_SCENARIO, result_root=RESULT_ROOT,
                     xtol=1e-3, **overrides):
    """Line search: return the free diffusivity whose measured D_eff matches
    `target_d_eff` (um^2/ms), to within `xtol`."""
    cache = {}

    def residual(d):
        if d not in cache:
            cache[d] = measure_d_eff(d, scenario=scenario, result_root=result_root,
                                     **overrides)
            print(f"  D={d:.4f} um^2/ms  ->  D_eff={cache[d]:.4f} um^2/ms")
        return cache[d] - target_d_eff

    # Buffering only slows the front (D_eff(D) < D for any D). So
    # target_d_eff is a safe lower bracket. Expand the upper bracket until
    # the residual turns positive.
    lo, hi = target_d_eff, target_d_eff * 2.0
    while residual(hi) < 0:
        hi *= 2.0

    d_solution = brentq(residual, lo, hi, xtol=xtol)
    assert abs(residual(d_solution)) < max(10 * xtol, 1e-2), (
        "brentq root does not reproduce the target D_eff -- forward map may not be monotonic"
    )
    print(
        f"\nD = {d_solution:.4f} um^2/ms  ->  D_eff = {cache[d_solution]:.4f} um^2/ms "
        f"(target {target_d_eff:.4f})"
    )
    return d_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target_d_eff", type=float, nargs="?", default=TISSUE_DIFFUSIVITY_ECS,
        help=f"target effective diffusivity, um^2/ms (default: {TISSUE_DIFFUSIVITY_ECS}, "
             "tissue_kinetics's no-ECM diffusivity_ecs)",
    )
    # "nobuffer" is not offered: with no buffer, D_eff == D, so inversion
    # is the identity.
    parser.add_argument(
        "--scenario", default=DEFAULT_SCENARIO,
        choices=["depleted", "saturated", "replenishment"],
        help=f"buffered scenario to invert (default: {DEFAULT_SCENARIO}, the one that "
             "matches tissue_kinetics: an equilibrated ECS drained by its synapses)",
    )
    parser.add_argument("--xtol", type=float, default=1e-3, help="um^2/ms")
    parser.add_argument("--result-root", default=RESULT_ROOT)
    parser.add_argument("--mesh-size", type=float, default=None, help="um; coarser = faster search")
    parser.add_argument("--end-time", type=float, default=None, help="ms; shorter = faster search")
    parser.add_argument("--n-threads", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.mesh_size is not None:
        overrides["box"] = {"mesh_size": args.mesh_size * u.um}
    if args.end_time is not None:
        overrides["end_time"] = args.end_time * u.ms
    if args.n_threads is not None:
        overrides["n_threads"] = args.n_threads

    find_diffusivity(args.target_d_eff, scenario=args.scenario,
                     result_root=args.result_root, xtol=args.xtol, **overrides)
