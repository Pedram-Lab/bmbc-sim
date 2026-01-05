"""Coefficient fields for spatially-varying simulation parameters.

This module provides coefficient field classes for specifying initial conditions,
diffusion coefficients, and other simulation parameters with spatial variation.
"""

from bmbcsim.simulation.coefficient_fields.coefficient_field import (
    CoefficientField,
    ConstantField,
    NodalNoiseField,
    SmoothRandomField,
    LocalizedPeaksField,
)

__all__ = [
    "CoefficientField",
    "ConstantField",
    "NodalNoiseField",
    "SmoothRandomField",
    "LocalizedPeaksField",
]
