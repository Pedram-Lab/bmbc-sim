"""Tests for coefficient field classes."""

import numpy as np
import pytest
import astropy.units as u
import ngsolve as ngs
from netgen import occ

import bmbcsim
from bmbcsim.geometry import create_sphere_geometry
from bmbcsim.units import mM
import bmbcsim.simulation.coefficient_fields as cf


@pytest.fixture
def simple_mesh():
    """Create a simple sphere mesh for testing."""
    return create_sphere_geometry(radius=5 * u.um, mesh_size=1 * u.um)


@pytest.fixture
def fes(simple_mesh):
    """Create FE space for testing."""
    return ngs.H1(simple_mesh, order=1)


# cf.Constant tests
def test_constant_field_produces_uniform_values(simple_mesh, fes):
    """cf.Constant produces uniform values across all nodes."""
    field = cf.Constant(value=1.5 * mM)
    coeff_func = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    np.testing.assert_array_almost_equal(values, 1.5)


def test_constant_field_value_property():
    """cf.Constant exposes its value via property."""
    field = cf.Constant(value=2.0 * mM)
    assert field.value == 2.0 * mM


# cf.PiecewiseConstant tests
@pytest.fixture
def two_region_mesh():
    """Create a mesh with two regions for testing piecewise fields."""
    left = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    right = occ.Box(occ.Pnt(1, 0, 0), occ.Pnt(2, 1, 1))
    left.mat("cell:left")
    right.mat("cell:right")
    geo = occ.OCCGeometry(occ.Glue([left, right]))
    return ngs.Mesh(geo.GenerateMesh(maxh=0.3))


@pytest.fixture
def two_region_fes(two_region_mesh):
    """Create FE space for two-region mesh."""
    return ngs.Compress(ngs.H1(two_region_mesh, order=1, definedon="cell:left|cell:right"))


def test_piecewise_constant_produces_different_values_per_region(two_region_mesh, two_region_fes):
    """cf.PiecewiseConstant produces different values in different regions."""
    field = cf.PiecewiseConstant(
        region_values={"left": 1.0 * mM, "right": 2.0 * mM},
        region_full_names={"left": "cell:left", "right": "cell:right"},
    )
    coeff_func = field.to_coefficient_function(
        two_region_mesh, two_region_fes, "molar concentration"
    )

    gf = ngs.GridFunction(two_region_fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    # Values should be either 1.0 or 2.0 (not all the same)
    assert np.min(values) == pytest.approx(1.0)
    assert np.max(values) == pytest.approx(2.0)


def test_piecewise_constant_region_values_property():
    """cf.PiecewiseConstant exposes its region_values via property."""
    region_vals = {"left": 1.0 * mM, "right": 2.0 * mM}
    field = cf.PiecewiseConstant(
        region_values=region_vals,
        region_full_names={"left": "cell:left", "right": "cell:right"},
    )
    assert field.region_values == region_vals


# cf.NodalNoise tests
def test_nodal_noise_seed_reproducibility(simple_mesh, fes):
    """Same seed produces identical results."""
    field1 = cf.NodalNoise(seed=42, value_range=(0 * mM, 1 * mM))
    field2 = cf.NodalNoise(seed=42, value_range=(0 * mM, 1 * mM))

    coeff_func1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_nodal_noise_different_seeds_differ(simple_mesh, fes):
    """Different seeds produce different results."""
    field1 = cf.NodalNoise(seed=42, value_range=(0 * mM, 1 * mM))
    field2 = cf.NodalNoise(seed=43, value_range=(0 * mM, 1 * mM))

    coeff_func1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_nodal_noise_values_in_range(simple_mesh, fes):
    """All generated values are within specified range."""
    field = cf.NodalNoise(seed=42, value_range=(0.5 * mM, 1.5 * mM))
    coeff_func = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.all(values >= 0.5)
    assert np.all(values <= 1.5)


# cf.SmoothRandom tests
def test_smooth_field_seed_reproducibility(simple_mesh, fes):
    """Same seed produces identical results."""
    field1 = cf.SmoothRandom(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )
    field2 = cf.SmoothRandom(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )

    coeff_func1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_smooth_field_different_seeds_differ(simple_mesh, fes):
    """Different seeds produce different results."""
    field1 = cf.SmoothRandom(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )
    field2 = cf.SmoothRandom(
        seed=43,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )

    coeff_func1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_smooth_field_values_in_range(simple_mesh, fes):
    """All generated values are within specified range."""
    field = cf.SmoothRandom(
        seed=42,
        value_range=(0.5 * mM, 1.5 * mM),
        correlation_length=2.0 * u.um,
    )
    coeff_func = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.all(values >= 0.5)
    assert np.all(values <= 1.5)


# cf.LocalizedPeaks tests
def test_peaks_field_seed_reproducibility(simple_mesh, fes):
    """Same seed produces identical results."""
    field1 = cf.LocalizedPeaks(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    field2 = cf.LocalizedPeaks(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )

    coeff_func1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_peaks_field_different_seeds_differ(simple_mesh, fes):
    """Different seeds produce different results."""
    field1 = cf.LocalizedPeaks(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    field2 = cf.LocalizedPeaks(
        seed=43,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )

    coeff_func1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_peaks_field_peak_count(simple_mesh, fes):
    """Correct number of peaks generated."""
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=5,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    _ = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    assert field.peak_node_indices is not None
    assert len(field.peak_node_indices) == 5


def test_peaks_field_background_value_present(simple_mesh, fes):
    """Background value is the minimum value far from peaks."""
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=1,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=0.5 * u.um,
    )
    coeff_func = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.min(values) >= 0.1 - 1e-10
    assert np.min(values) < 0.2


def test_peaks_field_peak_value_reached(simple_mesh, fes):
    """Peak value is approximately reached at peak centers."""
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=1,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    coeff_func = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.max(values) >= 9.0


# Integration tests
def test_nodal_noise_in_simulation(simple_mesh, tmp_path):
    """cf.NodalNoise works in a full simulation context."""
    sim = bmbcsim.Simulation("test_noise", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    noise_ic = cf.NodalNoise(seed=42, value_range=(0.1 * mM, 1.0 * mM))
    cell.initialize_species(ca, noise_ic)
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_smooth_field_in_simulation(simple_mesh, tmp_path):
    """cf.SmoothRandom works in a full simulation context."""
    sim = bmbcsim.Simulation("test_smooth", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    smooth_ic = cf.SmoothRandom(
        seed=42,
        value_range=(0.1 * mM, 1.0 * mM),
        correlation_length=2.0 * u.um,
    )
    cell.initialize_species(ca, smooth_ic)
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_localized_peaks_in_simulation(simple_mesh, tmp_path):
    """cf.LocalizedPeaks works in a full simulation context."""
    sim = bmbcsim.Simulation("test_peaks", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    peaks_ic = cf.LocalizedPeaks(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    cell.initialize_species(ca, peaks_ic)
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_random_diffusion_coefficient(simple_mesh, tmp_path):
    """Random fields can be used for diffusion coefficients."""
    sim = bmbcsim.Simulation("test_diff", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)

    random_diff = cf.SmoothRandom(
        seed=42,
        value_range=(0.1 * u.um ** 2 / u.ms, 0.5 * u.um ** 2 / u.ms),
        correlation_length=2.0 * u.um,
    )
    cell.add_diffusion(ca, random_diff)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)
