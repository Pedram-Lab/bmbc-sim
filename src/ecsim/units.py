"""
Defines the SI-base units used in the simulations. The choice of units is based
on the typical values of the quantities in the simulation (e.g., concentration
in mM, length in um, etc.).
"""
import astropy.units as u


# Define base units
MASS: u.Unit = u.ng
TIME: u.Unit = u.ms
LENGTH: u.Unit = u.um
CURRENT: u.Unit = u.pA
SUBSTANCE: u.Unit = u.amol
TEMPERATURE: u.Unit = u.K
LUMINOUS_INTENSITY: u.Unit = u.cd
ANGLE: u.Unit = u.rad

# Define derived units
AREA = LENGTH ** 2
VOLUME = LENGTH ** 3

DIFFUSIVITY = AREA / TIME
CONCENTRATION = SUBSTANCE / VOLUME
FORWARD_RATE = (CONCENTRATION * TIME) ** (-1)
REVERSE_RATE = TIME ** (-1)
FLUX_RATE = LENGTH / TIME

FORCE = MASS * LENGTH / TIME ** 2
PRESSURE = FORCE / LENGTH ** 2

CHARGE = CURRENT * TIME
ENERGY = FORCE * LENGTH
POTENTIAL = ENERGY / CHARGE
CAPACITANCE = CHARGE / POTENTIAL
PERMITTIVITY = CAPACITANCE / LENGTH


def convert(
        quantity: u.Quantity,
        unit: u.UnitBase
) -> float:
    """
    Convert a quantity to a different unit.
    :param quantity: The quantity to convert.
    :param unit: The unit to convert to.
    :return: The raw value of the converted quantity.
    :raises ValueError: If the conversion is not possible.
    """
    try:
        value = quantity.to(unit).value
    except u.UnitConversionError:
        raise ValueError(f"Cannot convert {quantity.unit} to {unit}") from None
    except AttributeError:
        raise ValueError(f"Cannot convert {quantity} to {unit}") from None

    return value
