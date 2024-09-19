import astropy.units as u


# Define the SI-base units used in the simulation
# The choice of units is based on the typical values of the quantities in the simulation
# (e.g., concentration in mM, length in um, etc.)
MASS = u.ng
TIME = u.ms
LENGTH = u.um
CURRENT = u.pA
SUBSTANCE = u.amol
TEMPERATURE = u.K
LUMINOUS_INTENSITY = u.cd

# Define derived units
AREA = LENGTH ** 2
VOLUME = LENGTH ** 3

DIFFUSIVITY = AREA / TIME
CONCENTRATION = SUBSTANCE / VOLUME
FORWARD_RATE = (CONCENTRATION * TIME) ** (-1)
REVERSE_RATE = TIME ** (-1)
FLUX = CONCENTRATION / TIME


def convert(quantity, unit):
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