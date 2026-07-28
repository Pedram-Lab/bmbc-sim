"""
Defines the SI-base units used in the simulations. The choice of units is based
on the typical values of the quantities in the simulation (e.g., concentration
in mM, length in um, etc.).
The bases units are:
- length: micrometers (um)
- time: milliseconds (ms)
- mass: nanograms (ng)
- amount of substance: attomoles (amol)
- electrical current: picoamperes (pA)
- temperature: Kelvin (K)
- luminous intensity: candela (cd)
- angle: radians (rad)
"""
import astropy.units as u


# Base units for all simulations
# These are all SI units, but with different prefixes. They are chosen such
# that, e.g., concentration is measured in millimolar (mM), length is measured
# in micrometers (um), etc.
BASE_UNITS = {
    'length': u.um,
    'time': u.ms,
    'mass': u.ng,
    'amount of substance': u.amol,
    'electrical current': u.pA,
    'temperature': u.K,
    'luminous intensity': u.cd,
    'angle': u.rad
}


# Define molar and derived
M = u.mol / u.liter
mM = M / 1000
uM = M / 1e6
nM = M / 1e9

# Teach astropy's *string* parser the same units (M, mM, uM, nM, ...), so that
# "1.3 mM" round-trips: configs are written as unit strings and read back by
# every script that loads a run's config.yaml. Registered here rather than in
# bmbcsim.config so that reading does not depend on importing the config module.
_molar_ns: dict[str, u.UnitBase] = {}
u.def_unit("M", M, prefixes=True, namespace=_molar_ns)
u.add_enabled_units(list(_molar_ns.values()))


def to_simulation_units(value: u.Quantity, physical_name: str = None) -> float:
    """Convert a value to simulation units.

    :param value: The value to convert.
    :param physical_name: The physical quantity to convert. If not None, the
        value is checked to be of the correct physical quantity.
    :raises ValueError: The argument `physical_name` was given and the quantity
        could not be converted to ththe given unit.
    """
    # Check if the value is of the correct physical quantity
    if physical_name is not None:
        actual_names = str(u.get_physical_type(value)).split("/")
        if physical_name not in actual_names:
            raise ValueError(
                f"Actual physical quantity is {actual_names}, but expected {physical_name}.")

    # Convert the value to simulation units
    conversion_factor = value.unit.decompose(bases=BASE_UNITS.values()).scale
    return float(value.value * conversion_factor)
