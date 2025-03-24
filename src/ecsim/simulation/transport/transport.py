import abc

import ngsolve as ngs
import astropy.units as u

from ecsim.units import to_simulation_units


class AbstractTransport(abc.ABC):
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


class Linear(AbstractTransport):
    """Linear transport mechanism that computes the flux as a linear function
    of the concentration difference across the membrane.
    """
    def __init__(self, permeability: u.Quantity):
        """Create a new linear transport mechanism with the given permeability.

        :param permeability: Permeability of the membrane (units: area/time).
        """
        self.permeability = permeability


    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction
    ) -> ngs.CoefficientFunction:
        # TODO: find correct unit for channel flux!
        p = to_simulation_units(self.permeability, 'reaction rate')
        return p * (source - target)
