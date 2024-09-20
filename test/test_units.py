import astropy.units as u
import pytest

from ecsim.units import convert


def test_output_has_correct_type():
    actual = convert(1 * u.m, u.cm)
    assert isinstance(actual, float)


def test_conversion_is_correct():
    actual = convert(1 * u.m, u.cm)
    expected = 100
    assert actual == expected


def test_converting_plain_number_raises_error():
    with pytest.raises(ValueError):
        convert(1, u.cm)


def test_converting_non_number_raises_error():
    with pytest.raises(ValueError):
        convert('1', u.cm)


def test_converting_incompatible_units_raises_error():
    with pytest.raises(ValueError):
        convert(1 * u.m, u.s)