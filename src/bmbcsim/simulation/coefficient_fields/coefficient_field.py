"""Coefficient fields for spatially-varying simulation parameters.

This module provides a unified abstraction for coefficient fields that can be used
for initial conditions, diffusion coefficients, and other simulation parameters.
"""

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
import ngsolve as ngs

from bmbcsim.units import to_simulation_units


class CoefficientField(ABC):
    """Abstract base class for coefficient fields.

    Coefficient fields generate spatially-varying values that can be used as
    initial conditions, diffusion coefficients, or other simulation parameters.
    Subclasses implement specific field generation strategies.
    """

    @abstractmethod
    def to_coefficient_function(
        self,
        mesh: ngs.Mesh,
        fes: ngs.FESpace,
        unit_name: str,
    ) -> ngs.CoefficientFunction:
        """Generate an NGSolve coefficient function.

        :param mesh: The NGSolve mesh object.
        :param fes: The finite element space for this compartment.
        :param unit_name: The physical unit name for conversion (e.g., 'molar concentration').
        :returns: An NGSolve CoefficientFunction representing the field.
        """
        pass


class ConstantField(CoefficientField):
    """Uniform constant value across the domain.

    This field type captures the existing functionality of specifying a single
    scalar value for the entire domain.
    """

    def __init__(self, value: u.Quantity):
        """Initialize a constant field.

        :param value: The constant value with units (e.g., 1.0 * u.mM).
        """
        self._value = value

    def to_coefficient_function(
        self,
        mesh: ngs.Mesh,
        fes: ngs.FESpace,
        unit_name: str,
    ) -> ngs.CoefficientFunction:
        """Generate a constant coefficient function.

        :param mesh: The NGSolve mesh object (unused for constant fields).
        :param fes: The finite element space (unused for constant fields).
        :param unit_name: The physical unit name for conversion.
        :returns: A constant NGSolve CoefficientFunction.
        """
        return ngs.CoefficientFunction(to_simulation_units(self._value, unit_name))

    @property
    def value(self) -> u.Quantity:
        """Return the constant value."""
        return self._value

    def __repr__(self) -> str:
        return f"ConstantField(value={self._value})"


class NodalNoiseField(CoefficientField):
    """Uncorrelated random values at each mesh node.

    Each mesh node receives an independent random value uniformly distributed
    within the specified range. This produces a highly irregular, noisy field.
    """

    def __init__(
        self,
        seed: int,
        value_range: tuple[u.Quantity, u.Quantity],
    ):
        """Initialize a nodal noise field.

        :param seed: Random seed for reproducibility (required).
        :param value_range: (min_value, max_value) tuple with units.
        """
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer")
        self._seed = seed
        self._value_range = value_range

    def to_coefficient_function(
        self,
        mesh: ngs.Mesh,
        fes: ngs.FESpace,
        unit_name: str,
    ) -> ngs.CoefficientFunction:
        """Generate uncorrelated random nodal values.

        :param mesh: The NGSolve mesh object.
        :param fes: The finite element space for this compartment.
        :param unit_name: The physical unit name for conversion.
        :returns: A GridFunction with random nodal values.
        """
        rng = np.random.default_rng(self._seed)
        min_val = to_simulation_units(self._value_range[0], unit_name)
        max_val = to_simulation_units(self._value_range[1], unit_name)

        gf = ngs.GridFunction(fes)
        gf.vec.FV().NumPy()[:] = rng.uniform(min_val, max_val, size=fes.ndof)
        return gf

    @property
    def seed(self) -> int:
        """Return the random seed."""
        return self._seed

    def __repr__(self) -> str:
        return f"NodalNoiseField(seed={self._seed}, value_range={self._value_range})"


class SmoothRandomField(CoefficientField):
    """Smooth random field with configurable correlation length.

    Uses radial basis function (RBF) interpolation from random values at
    control points spaced by the correlation length. This produces a smooth,
    Perlin-like noise field where the correlation length determines the
    spatial scale of variations.
    """

    def __init__(
        self,
        seed: int,
        value_range: tuple[u.Quantity, u.Quantity],
        correlation_length: u.Quantity,
    ):
        """Initialize a smooth random field.

        :param seed: Random seed for reproducibility (required).
        :param value_range: (min_value, max_value) tuple with units.
        :param correlation_length: Spatial scale of variations (e.g., 5 * u.um).
        """
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer")
        self._seed = seed
        self._value_range = value_range
        self._correlation_length = correlation_length

    def to_coefficient_function(
        self,
        mesh: ngs.Mesh,
        fes: ngs.FESpace,
        unit_name: str,
    ) -> ngs.CoefficientFunction:
        """Generate a smooth random field via RBF interpolation.

        :param mesh: The NGSolve mesh object.
        :param fes: The finite element space for this compartment.
        :param unit_name: The physical unit name for conversion.
        :returns: A GridFunction with smoothly varying random values.
        """
        from scipy.interpolate import RBFInterpolator

        rng = np.random.default_rng(self._seed)
        min_val = to_simulation_units(self._value_range[0], unit_name)
        max_val = to_simulation_units(self._value_range[1], unit_name)
        corr_len = to_simulation_units(self._correlation_length, 'length')

        # Get mesh coordinates and bounding box
        coords = np.array(mesh.ngmesh.Coordinates())
        bbox_min, bbox_max = coords.min(axis=0), coords.max(axis=0)

        # Create control point grid with spacing = correlation_length
        grid_points = []
        for dim in range(3):
            n_pts = max(2, int((bbox_max[dim] - bbox_min[dim]) / corr_len) + 2)
            grid_points.append(np.linspace(
                bbox_min[dim] - corr_len,
                bbox_max[dim] + corr_len,
                n_pts
            ))
        xx, yy, zz = np.meshgrid(*grid_points, indexing='ij')
        control_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Generate random values at control points and interpolate
        control_vals = rng.uniform(min_val, max_val, size=len(control_pts))
        rbf = RBFInterpolator(
            control_pts,
            control_vals,
            kernel='gaussian',
            epsilon=1.0 / corr_len,
        )

        # Interpolate to mesh nodes
        gf = ngs.GridFunction(fes)
        interpolated = np.clip(rbf(coords[:fes.ndof]), min_val, max_val)
        gf.vec.FV().NumPy()[:] = interpolated
        return gf

    @property
    def seed(self) -> int:
        """Return the random seed."""
        return self._seed

    @property
    def correlation_length(self) -> u.Quantity:
        """Return the correlation length."""
        return self._correlation_length

    def __repr__(self) -> str:
        return (f"SmoothRandomField(seed={self._seed}, "
                f"value_range={self._value_range}, "
                f"correlation_length={self._correlation_length})")


class LocalizedPeaksField(CoefficientField):
    """Random field with localized Gaussian peaks.

    Creates a background value with a specified number of Gaussian peaks
    centered at randomly selected mesh nodes. This is useful for modeling
    localized concentrations or "hot spots".
    """

    def __init__(
        self,
        seed: int,
        num_peaks: int,
        peak_value: u.Quantity,
        background_value: u.Quantity,
        peak_width: u.Quantity,
    ):
        """Initialize a localized peaks field.

        :param seed: Random seed for reproducibility (required).
        :param num_peaks: Number of Gaussian peaks to place.
        :param peak_value: Maximum value at peak centers.
        :param background_value: Baseline value away from peaks.
        :param peak_width: Standard deviation (width) of Gaussian peaks.
        """
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer")
        self._seed = seed
        self._num_peaks = num_peaks
        self._peak_value = peak_value
        self._background_value = background_value
        self._peak_width = peak_width
        self._peak_node_indices: np.ndarray | None = None

    def to_coefficient_function(
        self,
        mesh: ngs.Mesh,
        fes: ngs.FESpace,
        unit_name: str,
    ) -> ngs.CoefficientFunction:
        """Generate a field with localized Gaussian peaks.

        :param mesh: The NGSolve mesh object.
        :param fes: The finite element space for this compartment.
        :param unit_name: The physical unit name for conversion.
        :returns: A GridFunction with Gaussian peaks at random mesh nodes.
        """
        rng = np.random.default_rng(self._seed)

        peak_val = to_simulation_units(self._peak_value, unit_name)
        bg_val = to_simulation_units(self._background_value, unit_name)
        width = to_simulation_units(self._peak_width, 'length')
        height = peak_val - bg_val

        coords = np.array(mesh.ngmesh.Coordinates())

        # Select random mesh nodes as peak centers
        n_nodes = len(coords)
        actual_num_peaks = min(self._num_peaks, n_nodes)
        self._peak_node_indices = rng.choice(
            n_nodes, size=actual_num_peaks, replace=False
        )
        peak_centers = coords[self._peak_node_indices]

        # Compute field values at all nodes
        gf = ngs.GridFunction(fes)
        values = np.full(fes.ndof, bg_val)

        for center in peak_centers:
            dist_sq = np.sum((coords[:fes.ndof] - center) ** 2, axis=1)
            values += height * np.exp(-dist_sq / (2 * width ** 2))

        gf.vec.FV().NumPy()[:] = values
        return gf

    @property
    def seed(self) -> int:
        """Return the random seed."""
        return self._seed

    @property
    def num_peaks(self) -> int:
        """Return the number of peaks."""
        return self._num_peaks

    @property
    def peak_node_indices(self) -> np.ndarray | None:
        """Return indices of mesh nodes used as peak centers (available after generation)."""
        return self._peak_node_indices

    def __repr__(self) -> str:
        return (f"LocalizedPeaksField(seed={self._seed}, "
                f"num_peaks={self._num_peaks}, "
                f"peak_value={self._peak_value}, "
                f"background_value={self._background_value}, "
                f"peak_width={self._peak_width})")
