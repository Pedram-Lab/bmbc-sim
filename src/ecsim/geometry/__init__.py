"""
Geometry module; contains functions for creating and working with geometries for
simulations.
"""
from .simple_geometries import create_ca_depletion_mesh, create_dish_geometry, create_sensor_geometry
from .rusakov_geometry import create_rusakov_geometry
from .evaluators import LineEvaluator, PointEvaluator
from .tissue_geometry import TissueGeometry
from ._utils import create_mesh
# TODO: remove evaluators

__all__ = [
    "create_ca_depletion_mesh",
    "create_dish_geometry",
    "create_rusakov_geometry",
    "TissueGeometry",
    "LineEvaluator",
    "PointEvaluator",
    "create_mesh"
]
