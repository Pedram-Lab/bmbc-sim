import abc
from typing import Callable

import ngsolve as ngs
import astropy.units as u

from ecsim.units import to_simulation_units


Coefficient = u.Quantity | Callable[[u.Quantity], u.Quantity]


class Transport(abc.ABC):
    """Abstract base class for transport mechanisms in a membrane.
    """
    def __init__(self):
        self._mutable_coefficients = []


    @abc.abstractmethod
    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction,
    ) -> ngs.CoefficientFunction:
        """Compute the boundary flux of the transport mechanism. The flux is
        assumed to be the total flux across the membrane. In case a flux
        density is readily available, it should be multiplied by :code:`membrane.area`.

        :param source: Coefficient function representing the concentration in
            the source compartment.
        :param target: Coefficient function representing the concentration in
            the target compartment.
        :return: Coefficient function representing the flux across the membrane
            from source to target
        """


    def update_flux(self, t: u.Quantity) -> None:
        """Update the transport parameters based on the time t.
        This method can be used to modify the transport properties dynamically
        during the simulation, e.g., to implement time-dependent transport rates.

        :param t: The current time in the simulation.
        """
        # Update all mutable coefficients based on the current time t
        # Changes made here are visible in the flux expression
        for coeff, parameter, physical_name in self._mutable_coefficients:
            new_value = to_simulation_units(coeff(t), physical_name)
            parameter.Set(new_value)


    def _register_coefficient(
            self,
            coeff: Coefficient,
            physical_name: str = None
    ) -> ngs.CoefficientFunction:
        """Register a coefficient as a time-dependent coefficient function that
        can be used in the definition of the flux.

        :param coeff: The coefficient to register. Can be a constant or a callable.
        :param physical_name: The physical quantity of the coefficient. If provided,
            it will be used to check if the coefficient is of the correct type.
        :return: The registered coefficient function.
        """
        if isinstance(coeff, u.Quantity):
            # Convert the quantity to simulation units
            value = to_simulation_units(coeff, physical_name)
            return ngs.CoefficientFunction(value)
        elif callable(coeff):
            # If it's a callable, we need to wrap it in a ngs.Parameter and remember
            # it for later updates.
            value = to_simulation_units(coeff(0.0 * u.s), physical_name)
            parameter = ngs.Parameter(value)
            self._mutable_coefficients.append((coeff, parameter, physical_name))
            return parameter
        else:
            raise ValueError("Coefficient must be either a Quantity or a callable.")


class Linear(Transport):
    """Linear transport mechanism that computes the flux as a linear function
    of the concentration difference across the membrane.
    """
    def __init__(
            self,
            permeability: Coefficient,
            outside_concentration: u.Quantity = None
    ):
        """Create a new linear transport mechanism with the given permeability.

        :param permeability: Permeability of the membrane (units: area/time).
        :param outside_concentration: Concentration of the substance outside of
            the domain if one of the compartments is a boundary.
        """
        super().__init__()
        self.permeability = self._register_coefficient(permeability, 'velocity')
        self.outside_concentration = None if outside_concentration is None else \
            self._register_coefficient(outside_concentration, 'molar concentration')


    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction
    ) -> ngs.CoefficientFunction:
        source = self._constant_if_none(source)
        target = self._constant_if_none(target)

        # TODO: find correct unit for channel flux!
        return self.permeability * (source - target)


    def _constant_if_none(self, cf):
        """Return a constant coefficient function if the input is None.
        """
        if cf is None:
            if self.outside_concentration is None:
                raise ValueError("No outside concentration specified.")
            return ngs.CoefficientFunction(self.outside_concentration)
        return cf


class MichaelisMenten(Transport):
    """Michaelis-Menten transport mechanism that computes the flux based on
    the source concentration and a maximum rate.
    """
    def __init__(
            self,
            v_max: Coefficient,
            km: Coefficient
    ):
        """Create a new Michaelis-Menten transport mechanism.

        :param v_max: Maximum rate of the transport (units: concentration/time).
        :param km: Michaelis constant (units: concentration).
        """
        super().__init__()
        self.v_max = self._register_coefficient(v_max)
        self.km = self._register_coefficient(km, 'molar concentration')


    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction
    ) -> ngs.CoefficientFunction:
        # Only the source concentration contributes to the (stationary) flux
        del target

        # Compute the flux using the Michaelis-Menten equation
        return self.v_max * source / (self.km + source)


class Channel(Transport):
    """Channel transport mechanism that allows for a constant flux across the
    membrane, independent of the concentration difference.
    """
    def __init__(
            self,
            flux: Coefficient
    ):
        """Create a new channel transport mechanism.

        :param flux: The constant flux across the membrane (units: amount/time).
        """
        super().__init__()
        self.flux_value = self._register_coefficient(flux, 'catalytic activity')


    def flux(
            self,
            source: ngs.CoefficientFunction,
            target: ngs.CoefficientFunction
    ) -> ngs.CoefficientFunction:
        # Channel flux is independent of the concentrations
        return self.flux_value
