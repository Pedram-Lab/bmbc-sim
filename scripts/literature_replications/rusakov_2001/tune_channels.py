"""Bisection search for the m50 / J50 parameter that yields 50% Ca²⁺ depletion
(0.65 mM minimum) at the cleft center.

Uses an interval halving method:
  - lower bound starts at 0, upper bound at the current M50/J50 value
  - each iteration runs the midpoint, evaluates min [Ca²⁺] at (0,0,0)
  - adjusts bounds based on whether depletion is too strong or too weak
  - stops when interval length is 1, reports the closer value
"""

import numpy as np
import xarray as xr

import bmbcsim

from simulation import run_simulation, M50, J50

TARGET_MIN = 0.65  # mM — 50% depletion of 1.3 mM
RESULT_ROOT = "results/rusakov_channel_tuning"
CENTER_POINT = [[0, 0, 0]]


def run_and_evaluate(channel_param, pre_or_post):
    """Run a simulation with the given channel parameter and return the min
    [Ca²⁺] at the cleft center in mM."""

    sim_name = f"rusakov_m{channel_param}"

    kwargs = {
        "simulation_name": sim_name,
        "result_root": RESULT_ROOT,
        "pre_or_post_synaptic": pre_or_post,
    }
    if pre_or_post == "pre":
        kwargs["m50"] = channel_param
    elif pre_or_post == "post":
        kwargs["j50"] = channel_param

    run_simulation(**kwargs)

    # Evaluate: load point values at center and find minimum
    loader = bmbcsim.ResultLoader.find(
        simulation_name=sim_name, results_root=RESULT_ROOT,
    )
    point_values = xr.concat(
        [loader.load_point_values(i, points=CENTER_POINT) for i in range(len(loader))],
        dim="time",
    )
    ca_center = point_values.sel(species="Ca").isel(point=0).values
    return float(np.min(ca_center))


def bisect(pre_or_post="pre"):
    """Run bisection to find the channel parameter yielding TARGET_MIN."""

    if pre_or_post == "pre":
        upper_start = M50
        param_name = "m50"
    elif pre_or_post == "post":
        upper_start = int(J50.value)
        param_name = "J50"
    else:
        raise ValueError(f"Invalid mode: {pre_or_post}")

    lo = 0
    hi = upper_start
    results = {}  # channel_param -> min_ca

    print(f"Tuning {param_name} for {pre_or_post}synaptic simulation")
    print(f"Target minimum [Ca²⁺]: {TARGET_MIN} mM")
    print(f"Initial interval: [{lo}, {hi}]")
    print()

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid in results:
            min_ca = results[mid]
            print(f"  {param_name}={mid}: min [Ca²⁺] = {min_ca:.4f} mM (cached)")
        else:
            print(f"  Running simulation with {param_name}={mid}...")
            min_ca = run_and_evaluate(mid, pre_or_post)
            results[mid] = min_ca
            print(f"  {param_name}={mid}: min [Ca²⁺] = {min_ca:.4f} mM")

        # If min_ca > target, depletion is insufficient → need more channels → raise lower bound
        # If min_ca < target, depletion is too strong → need fewer channels → lower upper bound
        if min_ca > TARGET_MIN:
            lo = mid
        else:
            hi = mid

        print(f"  Interval: [{lo}, {hi}]")
        print()

    # Evaluate both endpoints if not already known
    for val in [lo, hi]:
        if val not in results:
            print(f"  Running final simulation with {param_name}={val}...")
            min_ca = run_and_evaluate(val, pre_or_post)
            results[val] = min_ca
            print(f"  {param_name}={val}: min [Ca²⁺] = {min_ca:.4f} mM")

    # Pick the closer one
    lo_err = abs(results.get(lo, float("inf")) - TARGET_MIN)
    hi_err = abs(results.get(hi, float("inf")) - TARGET_MIN)
    best = lo if lo_err <= hi_err else hi

    print("=" * 50)
    print(f"Result: {param_name} = {best}")
    print(f"  min [Ca²⁺] = {results[best]:.4f} mM (target: {TARGET_MIN} mM)")
    print(f"  error: {abs(results[best] - TARGET_MIN):.4f} mM")
    print()
    print("All evaluated points:")
    for val in sorted(results):
        marker = " <-- best" if val == best else ""
        print(f"  {param_name}={val}: {results[val]:.4f} mM{marker}")


if __name__ == "__main__":
    bisect(pre_or_post="pre")
