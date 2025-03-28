import abc

import ngsolve as ngs
import astropy.units as u

from ecsim.units import to_simulation_units


class Transport(abc.ABC):
    """Abstract base class for transport mechanisms in a membrane.
    """
    @abc.abstractmethod
    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction
    ) -> ngs.CoefficientFunction:
        """Compute the boundary flux of the transport mechanism.
        """


class Linear(Transport):
    """Linear transport mechanism that computes the flux as a linear function
    of the concentration difference across the membrane.
    """
    def __init__(self, permeability: u.Quantity, outside_concentration: u.Quantity = None):
        """Create a new linear transport mechanism with the given permeability.

        :param permeability: Permeability of the membrane (units: area/time).
        :param outside_concentration: Concentration of the substance outside of
            the domain if one of the compartments is a boundary.
        """
        self.permeability = permeability
        self.outside_concentration = outside_concentration


    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction
    ) -> ngs.CoefficientFunction:
        source = self._constant_if_none(source)
        target = self._constant_if_none(target)

        # TODO: find correct unit for channel flux!
        p = to_simulation_units(self.permeability, 'velocity')
        return p * (source - target)


    def _constant_if_none(self, cf):
        """Return a constant coefficient function if the input is None.
        """
        if cf is None:
            if self.outside_concentration is None:
                raise ValueError("No outside concentration specified.")
            outside_concentration = to_simulation_units(self.outside_concentration,
                                                        'molar concentration')
            return ngs.CoefficientFunction(outside_concentration)
        return cf
