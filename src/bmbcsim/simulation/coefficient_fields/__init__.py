"""Coefficient fields for spatially-varying simulation parameters.

This module provides coefficient field classes for specifying initial conditions,
diffusion coefficients, and other simulation parameters with spatial variation.

Usage:
    import bmbcsim.simulation.coefficient_fields as cf
    cf.Constant(1.0 * u.mM)
    cf.SmoothRandom(seed=42, value_range=(0.1 * u.mM, 1.0 * u.mM), correlation_length=2.0 * u.um)
"""

from bmbcsim.simulation.coefficient_fields.coefficient_field import (
    Coefficient,
    Constant,
    PiecewiseConstant,
    NodalNoise,
    SmoothRandom,
    LocalizedPeaks,
)

__all__ = [
    "Coefficient",
    "Constant",
    "PiecewiseConstant",
    "NodalNoise",
    "SmoothRandom",
    "LocalizedPeaks",
]
