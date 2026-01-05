"""Tests for coefficient field classes."""

import numpy as np
import pytest
import astropy.units as u
import ngsolve as ngs

import bmbcsim
from bmbcsim.geometry import create_sphere_geometry
from bmbcsim.units import mM
from bmbcsim.simulation.coefficient_fields import (
    ConstantField,
    PiecewiseConstantField,
    NodalNoiseField,
    SmoothRandomField,
    LocalizedPeaksField,
)


@pytest.fixture
def simple_mesh():
    """Create a simple sphere mesh for testing."""
    return create_sphere_geometry(radius=5 * u.um, mesh_size=1 * u.um)


@pytest.fixture
def fes(simple_mesh):
    """Create FE space for testing."""
    return ngs.H1(simple_mesh, order=1)


# ConstantField tests
def test_constant_field_produces_uniform_values(simple_mesh, fes):
    """ConstantField produces uniform values across all nodes."""
    field = ConstantField(value=1.5 * mM)
    cf = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(cf)
    values = gf.vec.FV().NumPy()

    np.testing.assert_array_almost_equal(values, 1.5)


def test_constant_field_value_property():
    """ConstantField exposes its value via property."""
    field = ConstantField(value=2.0 * mM)
    assert field.value == 2.0 * mM


# PiecewiseConstantField tests
@pytest.fixture
def two_region_mesh():
    """Create a mesh with two regions for testing piecewise fields."""
    from netgen import occ
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
    """PiecewiseConstantField produces different values in different regions."""
    field = PiecewiseConstantField(
        region_values={"left": 1.0 * mM, "right": 2.0 * mM},
        region_full_names={"left": "cell:left", "right": "cell:right"},
    )
    cf = field.to_coefficient_function(two_region_mesh, two_region_fes, 'molar concentration')

    gf = ngs.GridFunction(two_region_fes)
    gf.Set(cf)
    values = gf.vec.FV().NumPy()

    # Values should be either 1.0 or 2.0 (not all the same)
    assert np.min(values) == pytest.approx(1.0)
    assert np.max(values) == pytest.approx(2.0)


def test_piecewise_constant_region_values_property():
    """PiecewiseConstantField exposes its region_values via property."""
    region_vals = {"left": 1.0 * mM, "right": 2.0 * mM}
    field = PiecewiseConstantField(
        region_values=region_vals,
        region_full_names={"left": "cell:left", "right": "cell:right"},
    )
    assert field.region_values == region_vals


# NodalNoiseField tests
def test_nodal_noise_seed_reproducibility(simple_mesh, fes):
    """Same seed produces identical results."""
    field1 = NodalNoiseField(seed=42, value_range=(0 * mM, 1 * mM))
    field2 = NodalNoiseField(seed=42, value_range=(0 * mM, 1 * mM))

    cf1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    cf2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(cf1)
    gf2.Set(cf2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_nodal_noise_different_seeds_differ(simple_mesh, fes):
    """Different seeds produce different results."""
    field1 = NodalNoiseField(seed=42, value_range=(0 * mM, 1 * mM))
    field2 = NodalNoiseField(seed=43, value_range=(0 * mM, 1 * mM))

    cf1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    cf2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(cf1)
    gf2.Set(cf2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_nodal_noise_values_in_range(simple_mesh, fes):
    """All generated values are within specified range."""
    field = NodalNoiseField(seed=42, value_range=(0.5 * mM, 1.5 * mM))
    cf = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(cf)
    values = gf.vec.FV().NumPy()

    assert np.all(values >= 0.5)
    assert np.all(values <= 1.5)


# SmoothRandomField tests
def test_smooth_field_seed_reproducibility(simple_mesh, fes):
    """Same seed produces identical results."""
    field1 = SmoothRandomField(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )
    field2 = SmoothRandomField(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )

    cf1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    cf2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(cf1)
    gf2.Set(cf2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_smooth_field_different_seeds_differ(simple_mesh, fes):
    """Different seeds produce different results."""
    field1 = SmoothRandomField(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )
    field2 = SmoothRandomField(
        seed=43,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )

    cf1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    cf2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(cf1)
    gf2.Set(cf2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_smooth_field_values_in_range(simple_mesh, fes):
    """All generated values are within specified range."""
    field = SmoothRandomField(
        seed=42,
        value_range=(0.5 * mM, 1.5 * mM),
        correlation_length=2.0 * u.um,
    )
    cf = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(cf)
    values = gf.vec.FV().NumPy()

    assert np.all(values >= 0.5)
    assert np.all(values <= 1.5)


# LocalizedPeaksField tests
def test_peaks_field_seed_reproducibility(simple_mesh, fes):
    """Same seed produces identical results."""
    field1 = LocalizedPeaksField(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    field2 = LocalizedPeaksField(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )

    cf1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    cf2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(cf1)
    gf2.Set(cf2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_peaks_field_different_seeds_differ(simple_mesh, fes):
    """Different seeds produce different results."""
    field1 = LocalizedPeaksField(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    field2 = LocalizedPeaksField(
        seed=43,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )

    cf1 = field1.to_coefficient_function(simple_mesh, fes, 'molar concentration')
    cf2 = field2.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(cf1)
    gf2.Set(cf2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_peaks_field_peak_count(simple_mesh, fes):
    """Correct number of peaks generated."""
    field = LocalizedPeaksField(
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
    field = LocalizedPeaksField(
        seed=42,
        num_peaks=1,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=0.5 * u.um,
    )
    cf = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(cf)
    values = gf.vec.FV().NumPy()

    assert np.min(values) >= 0.1 - 1e-10
    assert np.min(values) < 0.2


def test_peaks_field_peak_value_reached(simple_mesh, fes):
    """Peak value is approximately reached at peak centers."""
    field = LocalizedPeaksField(
        seed=42,
        num_peaks=1,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    cf = field.to_coefficient_function(simple_mesh, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(cf)
    values = gf.vec.FV().NumPy()

    assert np.max(values) >= 9.0


# Integration tests
def test_nodal_noise_in_simulation(simple_mesh, tmp_path):
    """NodalNoiseField works in a full simulation context."""
    sim = bmbcsim.Simulation("test_noise", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    noise_ic = NodalNoiseField(seed=42, value_range=(0.1 * mM, 1.0 * mM))
    cell.initialize_species(ca, noise_ic)
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_smooth_field_in_simulation(simple_mesh, tmp_path):
    """SmoothRandomField works in a full simulation context."""
    sim = bmbcsim.Simulation("test_smooth", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    smooth_ic = SmoothRandomField(
        seed=42,
        value_range=(0.1 * mM, 1.0 * mM),
        correlation_length=2.0 * u.um,
    )
    cell.initialize_species(ca, smooth_ic)
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_localized_peaks_in_simulation(simple_mesh, tmp_path):
    """LocalizedPeaksField works in a full simulation context."""
    sim = bmbcsim.Simulation("test_peaks", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    peaks_ic = LocalizedPeaksField(
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

    random_diff = SmoothRandomField(
        seed=42,
        value_range=(0.1 * u.um ** 2 / u.ms, 0.5 * u.um ** 2 / u.ms),
        correlation_length=2.0 * u.um,
    )
    cell.add_diffusion(ca, random_diff)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)
