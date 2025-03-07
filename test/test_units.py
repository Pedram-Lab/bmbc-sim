import astropy.units as u
import pytest

from ecsim.units import to_simulation_units


def test_base_unit_conversion_is_correct():
    """Test that conversion of a base unit is correct (cm to um)."""
    length = 1 * u.cm

    length_sim = to_simulation_units(length)

    assert length_sim == 10000.0


def test_derived_unit_conversion_is_correct():
    """Test that conversion of a derived unit is correct (m/s^2 to um/ms^2)."""
    acceleration = 9.81 * u.m / u.s**2

    acceleration_sim = to_simulation_units(acceleration, 'acceleration')

    assert acceleration_sim == 9.81


def test_named_unit_conversion_is_correct():
    """Test that conversion of a named unit is correct (nM to amol/um^3 = mM)."""
    concentration = 3 * u.nmol / u.L

    concentration_sim = to_simulation_units(concentration, physical_name='molar concentration')

    assert concentration_sim == 3e-6


def test_invalid_physical_quantity_raises_error():
    """Test that an error is raised when the physical quantity is invalid."""
    concentration = 3 * u.nmol / u.L

    with pytest.raises(ValueError):
        to_simulation_units(concentration, physical_name='invalid_physical_quantity')
