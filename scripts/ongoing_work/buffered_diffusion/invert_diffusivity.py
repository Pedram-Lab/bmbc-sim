"""Invert the buffered-diffusion forward model.

``simulation.run(Config(buffer={"enabled": True}, diffusivity=D, ...))`` produces a
front that ``evaluate.py`` fits to an *effective* diffusivity D_eff(D). There
is no closed form for D_eff here -- it's measured from the simulated front,
not a rapid-buffering-approximation formula -- so we treat the pipeline as a
black-box forward map and invert it with a bracketed line search:
D_eff(D) only increases with D (more free diffusivity never slows the
front), so ``scipy.optimize.brentq`` on ``residual(D) = D_eff(D) - target``
converges reliably.

Each evaluation of D_eff(D) is a full FEM simulation, so this is slow --
expect on the order of ten simulation runs (bracket expansion plus
bisection) per target.

This is meant to calibrate ``scripts/ongoing_work/tissue_kinetics/simulation.py``:
run it once with ``ecm.enabled=false`` (pure diffusion, diffusivity_ecs=
TISSUE_DIFFUSIVITY_ECS) and once with ``ecm.enabled=true`` using the diffusivity
this script returns, so the buffered run's *effective* ECS diffusivity
matches the unbuffered baseline. The buffer/reservoir parameters below are
copied from that script's defaults so the calibration reaction matches; both
configs are parametrized by (kf, Kd), so they transfer directly.
"""
import argparse

import astropy.units as u
import numpy as np
from scipy.optimize import brentq

import evaluate
from simulation import Config, run

RESULT_ROOT = "results"

# Mirrors scripts/ongoing_work/tissue_kinetics/simulation.py's defaults.
TISSUE_DIFFUSIVITY_ECS = 0.7  # um^2/ms, the "diffusion only" baseline to match
TISSUE_CA_ECS = "1.3 mM"
TISSUE_ECM_TOTAL = "2.0 mM"
TISSUE_KD = "1.3 mM"
TISSUE_ECM_KF = "10.0 / (mM s)"


def measure_d_eff(diffusivity, *, result_root=RESULT_ROOT, **overrides):
    """Run the buffered simulation at `diffusivity` (um^2/ms) and return the
    measured effective diffusivity (um^2/ms), using evaluate.py's own fit.

    :param overrides: Extra ``Config`` fields, e.g. ``end_time`` or ``box``.
    """
    run(Config(
        result_root=result_root,
        ca_source=TISSUE_CA_ECS,
        diffusivity=diffusivity * u.um**2 / u.ms,
        buffer={
            "enabled": True,
            "ecm_total": TISSUE_ECM_TOTAL,
            "ecm_kf": TISSUE_ECM_KF,
            "kd": TISSUE_KD,
        },
        **overrides,
    ))
    evaluate.RESULT_ROOT = result_root
    d_eff = evaluate.analyze_run("buffered_diffusion_buffer")["d_eff"]
    if not np.isfinite(d_eff):
        raise RuntimeError(
            f"Could not fit D_eff at diffusivity={diffusivity:.4g} um^2/ms: "
            "the front never crossed the fit window."
        )
    return d_eff


def find_diffusivity(target_d_eff, *, result_root=RESULT_ROOT, xtol=1e-3, **overrides):
    """Line search: return the free diffusivity whose measured D_eff matches
    `target_d_eff` (um^2/ms), to within `xtol`."""
    cache = {}

    def residual(d):
        if d not in cache:
            cache[d] = measure_d_eff(d, result_root=result_root, **overrides)
            print(f"  D={d:.4f} um^2/ms  ->  D_eff={cache[d]:.4f} um^2/ms")
        return cache[d] - target_d_eff

    # Buffering only ever slows the front (D_eff(D) < D for any D), so
    # target_d_eff itself is a safe lower bracket; expand the upper bracket
    # until the residual turns positive.
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

    find_diffusivity(args.target_d_eff, result_root=args.result_root, xtol=args.xtol, **overrides)
