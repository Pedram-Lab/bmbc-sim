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
def volume_region(simple_mesh):
    """Create a volume region from the simple mesh."""
    return simple_mesh.Materials("sphere")


@pytest.fixture
def fes(simple_mesh):
    """Create FE space for testing."""
    return ngs.H1(simple_mesh, order=1)


# cf.Constant tests
def test_constant_field_produces_uniform_values(volume_region, fes):
    """cf.Constant produces uniform values across all nodes."""
    field = cf.Constant(value=1.5 * mM)
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

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


@pytest.fixture
def two_region_volume(two_region_mesh):
    """Create a volume region from the two-region mesh."""
    return two_region_mesh.Materials("cell:left|cell:right")


def test_piecewise_constant_produces_different_values_per_region(two_region_volume, two_region_fes):
    """cf.PiecewiseConstant produces different values in different regions."""
    field = cf.PiecewiseConstant(
        region_values={"left": 1.0 * mM, "right": 2.0 * mM},
        region_full_names={"left": "cell:left", "right": "cell:right"},
    )
    coeff_func = field.to_coefficient_function(
        two_region_volume, two_region_fes, "molar concentration"
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
def test_nodal_noise_seed_reproducibility(volume_region, fes):
    """Same seed produces identical results."""
    field1 = cf.NodalNoise(seed=42, value_range=(0 * mM, 1 * mM))
    field2 = cf.NodalNoise(seed=42, value_range=(0 * mM, 1 * mM))

    coeff_func1 = field1.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_nodal_noise_different_seeds_differ(volume_region, fes):
    """Different seeds produce different results."""
    field1 = cf.NodalNoise(seed=42, value_range=(0 * mM, 1 * mM))
    field2 = cf.NodalNoise(seed=43, value_range=(0 * mM, 1 * mM))

    coeff_func1 = field1.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_nodal_noise_values_in_range(volume_region, fes):
    """All generated values are within specified range."""
    field = cf.NodalNoise(seed=42, value_range=(0.5 * mM, 1.5 * mM))
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.all(values >= 0.5)
    assert np.all(values <= 1.5)


# cf.SmoothNoise tests
def test_smooth_field_seed_reproducibility(volume_region, fes):
    """Same seed produces identical results."""
    field1 = cf.SmoothNoise(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )
    field2 = cf.SmoothNoise(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )

    coeff_func1 = field1.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_smooth_field_different_seeds_differ(volume_region, fes):
    """Different seeds produce different results."""
    field1 = cf.SmoothNoise(
        seed=42,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )
    field2 = cf.SmoothNoise(
        seed=43,
        value_range=(0 * mM, 1 * mM),
        correlation_length=2.0 * u.um,
    )

    coeff_func1 = field1.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_smooth_field_values_in_range(volume_region, fes):
    """All generated values are within specified range."""
    field = cf.SmoothNoise(
        seed=42,
        value_range=(0.5 * mM, 1.5 * mM),
        correlation_length=2.0 * u.um,
    )
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.all(values >= 0.5)
    assert np.all(values <= 1.5)


# cf.LocalizedPeaks tests
def test_peaks_field_seed_reproducibility(volume_region, fes):
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

    coeff_func1 = field1.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    np.testing.assert_array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_peaks_field_different_seeds_differ(volume_region, fes):
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

    coeff_func1 = field1.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_func2 = field2.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff_func1)
    gf2.Set(coeff_func2)

    assert not np.array_equal(
        gf1.vec.FV().NumPy(),
        gf2.vec.FV().NumPy()
    )


def test_peaks_field_peak_count(volume_region, fes):
    """Correct number of peaks generated."""
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=5,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    _ = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    assert field.peak_node_indices is not None
    assert len(field.peak_node_indices) == 5


def test_peaks_field_background_value_present(volume_region, fes):
    """Background value is the minimum value far from peaks."""
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=1,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=0.5 * u.um,
    )
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf = ngs.GridFunction(fes)
    gf.Set(coeff_func)
    values = gf.vec.FV().NumPy()

    assert np.min(values) >= 0.1 - 1e-10
    assert np.min(values) < 0.2


def test_peaks_field_peak_value_reached(volume_region, fes):
    """Peak value is approximately reached at peak centers."""
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=1,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
    )
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

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
    """cf.SmoothNoise works in a full simulation context."""
    sim = bmbcsim.Simulation("test_smooth", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]

    ca = sim.add_species("ca")
    smooth_ic = cf.SmoothNoise(
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

    random_diff = cf.SmoothNoise(
        seed=42,
        value_range=(0.1 * u.um ** 2 / u.ms, 0.5 * u.um ** 2 / u.ms),
        correlation_length=2.0 * u.um,
    )
    cell.add_diffusion(ca, random_diff)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


# Surface coefficient tests
import bmbcsim.simulation.transport as transport


def test_surface_coefficient_in_transport(simple_mesh, tmp_path):
    """cf.NodalNoise works as a surface coefficient in transport mechanisms."""
    sim = bmbcsim.Simulation("test_surface", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]
    membrane = sim.simulation_geometry.membranes["boundary"]

    ca = sim.add_species("ca")
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    # Use spatial field for flux on the membrane surface
    flux = cf.NodalNoise(seed=42, value_range=(0.1 * u.mmol / u.ms, 0.5 * u.mmol / u.ms))
    membrane.add_transport(ca, transport.GeneralFlux(flux=flux), source=None, target=cell)

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_surface_coefficient_with_temporal_modulation(simple_mesh, tmp_path):
    """Surface coefficients can have temporal modulation."""
    sim = bmbcsim.Simulation("test_temporal", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]
    membrane = sim.simulation_geometry.membranes["boundary"]

    ca = sim.add_species("ca")
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    # Use spatial field with temporal modulation (active for first 0.05 ms)
    flux = cf.NodalNoise(seed=42, value_range=(0.1 * u.mmol / u.ms, 0.5 * u.mmol / u.ms))
    spike = lambda t: 1.0 if t < 0.05 * u.ms else 0.0
    membrane.add_transport(
        ca,
        transport.GeneralFlux(flux=flux, temporal=spike),
        source=None,
        target=cell
    )

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


def test_smooth_surface_coefficient_in_passive_transport(simple_mesh, tmp_path):
    """cf.SmoothNoise works as permeability in passive transport."""
    sim = bmbcsim.Simulation("test_passive_surface", simple_mesh, result_root=tmp_path)
    cell = sim.simulation_geometry.compartments["sphere"]
    membrane = sim.simulation_geometry.membranes["boundary"]

    ca = sim.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.2 * u.um ** 2 / u.ms)

    # Use spatial field for permeability
    perm = cf.SmoothNoise(
        seed=42,
        value_range=(0.1 * u.um ** 3 / u.ms, 0.5 * u.um ** 3 / u.ms),
        correlation_length=2.0 * u.um,
    )
    membrane.add_transport(
        ca,
        transport.Passive(permeability=perm, outside_concentration=0.5 * mM),
        source=cell,
        target=None
    )

    sim.run(end_time=0.1 * u.ms, time_step=0.01 * u.ms)


# Normalization tests
def test_nodal_noise_total_normalization(volume_region, fes):
    """NodalNoise with total parameter normalizes to target integral."""
    target_total = 100 * u.amol
    field = cf.NodalNoise(
        seed=42,
        value_range=(0.1 * mM, 1.0 * mM),
        total=target_total,
    )
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    # Integrate over region to verify total
    integral = ngs.Integrate(coeff_func, volume_region.mesh, definedon=volume_region)
    assert integral == pytest.approx(100.0, rel=1e-10)


def test_smooth_noise_total_normalization(volume_region, fes):
    """SmoothNoise with total parameter normalizes to target integral."""
    target_total = 50 * u.amol
    field = cf.SmoothNoise(
        seed=42,
        value_range=(0.1 * mM, 1.0 * mM),
        correlation_length=2.0 * u.um,
        total=target_total,
    )
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    integral = ngs.Integrate(coeff_func, volume_region.mesh, definedon=volume_region)
    assert integral == pytest.approx(50.0, rel=1e-10)


def test_localized_peaks_total_normalization(volume_region, fes):
    """LocalizedPeaks with total parameter normalizes to target integral."""
    target_total = 75 * u.amol
    field = cf.LocalizedPeaks(
        seed=42,
        num_peaks=3,
        peak_value=10.0 * mM,
        background_value=0.1 * mM,
        peak_width=1.0 * u.um,
        total=target_total,
    )
    coeff_func = field.to_coefficient_function(volume_region, fes, 'molar concentration')

    integral = ngs.Integrate(coeff_func, volume_region.mesh, definedon=volume_region)
    assert integral == pytest.approx(75.0, rel=1e-10)


def test_normalization_preserves_spatial_pattern(volume_region, fes):
    """Normalization preserves the relative spatial distribution."""
    # Create two fields with same seed - one normalized, one not
    field_unnorm = cf.SmoothNoise(
        seed=42,
        value_range=(0.1 * mM, 1.0 * mM),
        correlation_length=2.0 * u.um,
    )
    field_norm = cf.SmoothNoise(
        seed=42,
        value_range=(0.1 * mM, 1.0 * mM),
        correlation_length=2.0 * u.um,
        total=50 * u.amol,
    )

    coeff_unnorm = field_unnorm.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff_norm = field_norm.to_coefficient_function(volume_region, fes, 'molar concentration')

    # Get values at DOFs
    gf_unnorm = ngs.GridFunction(fes)
    gf_norm = ngs.GridFunction(fes)
    gf_unnorm.Set(coeff_unnorm)
    gf_norm.Set(coeff_norm)

    vals_unnorm = gf_unnorm.vec.FV().NumPy()
    vals_norm = gf_norm.vec.FV().NumPy()

    # Ratios should be constant (spatial pattern preserved)
    ratios = vals_norm / vals_unnorm
    np.testing.assert_array_almost_equal(ratios, ratios[0], decimal=10)


def test_normalization_without_total_unchanged(volume_region, fes):
    """Without total parameter, field values match original behavior."""
    field_no_total = cf.NodalNoise(seed=42, value_range=(0.1 * mM, 1.0 * mM))
    field_none_total = cf.NodalNoise(seed=42, value_range=(0.1 * mM, 1.0 * mM), total=None)

    coeff1 = field_no_total.to_coefficient_function(volume_region, fes, 'molar concentration')
    coeff2 = field_none_total.to_coefficient_function(volume_region, fes, 'molar concentration')

    gf1 = ngs.GridFunction(fes)
    gf2 = ngs.GridFunction(fes)
    gf1.Set(coeff1)
    gf2.Set(coeff2)

    np.testing.assert_array_equal(gf1.vec.FV().NumPy(), gf2.vec.FV().NumPy())
