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


    def flux_lhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None,
            src_test: ngs.comp.ProxyFunction | None,
            trg_test: ngs.comp.ProxyFunction | None
    ) -> ngs.CoefficientFunction:
        """Compute the lhs-version of the boundary flux of the transport
        mechanism. The flux is assumed to be the total flux across the membrane.
        In case a flux density is readily available, it should be multiplied by
        :code:`membrane.area`.
        All of the arguments can be None, in which case the flux is assumed to
        connect the domain to the outside.
        The following term is added to left-hand side of the PDE:
        :math:`\\int_{\\partial \\Omega} J \\, (v_t - v_s) \\, ds`, where :math:`J` is
        the flux density that's implemented by this method in terms of test
        functions and concentrations of the source and target compartments.

        :param source: Coefficient function representing the concentration in
            the source compartment.
        :param target: Coefficient function representing the concentration in
            the target compartment.
        :param src_test: Test function for the source compartment.
        :param trg_test: Test function for the target compartment.
        :return: Coefficient function representing the flux across the membrane
            from source to target
        """
        del source, target, src_test, trg_test  # Unused in default implementation
        return None

    def flux_rhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None
    ) -> ngs.CoefficientFunction:
        """Compute the rhs-version of the boundary flux of the transport
        mechanism. The flux is assumed to be the total flux across the membrane.
        In case a flux density is readily available, it should be multiplied by
        :code:`membrane.area`.
        All of the arguments can be None, in which case the flux is assumed to
        connect the domain to the outside.
        The following term is added to right-hand side of the PDE:
        :math:`\\int_{\\partial \\Omega} J \\, (v_t - v_s) \\, ds`, where :math:`J` is
        the flux density that's implemented by this method in terms of the
        concentrations of the source and target compartments.

        :param source: Coefficient function representing the concentration in
            the source compartment.
        :param target: Coefficient function representing the concentration in
            the target compartment.
        :return: Coefficient function representing the flux across the membrane
            from source to target
        """
        del source, target  # Unused in default implementation
        return None


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


class Passive(Transport):
    """Passive transport mechanism that computes the flux as a linear function
    of the concentration difference across the membrane.
    """
    def __init__(
            self,
            permeability: Coefficient,
            outside_concentration: u.Quantity = None
    ):
        """Create a new passive transport mechanism with the given permeability.

        :param permeability: Permeability of the membrane (units: area * length/time).
        :param outside_concentration: Concentration of the substance outside of
            the domain if one of the compartments is a boundary.
        """
        super().__init__()
        self.permeability = self._register_coefficient(permeability, 'volumetric flow rate')
        self.outside_concentration = None if outside_concentration is None else \
            self._register_coefficient(outside_concentration, 'molar concentration')


    def flux_lhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None,
            src_test: ngs.comp.ProxyFunction | None,
            trg_test: ngs.comp.ProxyFunction | None
    ) -> ngs.CoefficientFunction:
        del source, target  # Unused
        if src_test is None and trg_test is None:
            raise ValueError("Both source and target cannot be None.")
        if (src_test is None or trg_test is None) and self.outside_concentration is None:
            raise ValueError("No outside concentration specified for outside flux.")

        if trg_test is None:
            return self.permeability * src_test
        if src_test is None:
            return -self.permeability * trg_test

        return self.permeability * (src_test - trg_test)


    def flux_rhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None
    ) -> ngs.CoefficientFunction:
        if source is None:
            return self.permeability * self.outside_concentration
        if target is None:
            return -self.permeability * self.outside_concentration


class Active(Transport):
    """Active transport mechanism that computes the flux based on
    Michaelis-Menten kinetics for the source concentration.
    """
    def __init__(
            self,
            v_max: Coefficient,
            km: Coefficient
    ):
        """Create a new active transport mechanism based on Michaelis-Menten kinetics.

        :param v_max: Maximum rate of the transport (units: substance/time).
        :param km: Michaelis constant (units: concentration).
        """
        super().__init__()
        self.v_max = self._register_coefficient(v_max, 'catalytic activity')
        self.km = self._register_coefficient(km, 'molar concentration')


    def flux_lhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None,
            src_test: ngs.comp.ProxyFunction | None,
            trg_test: ngs.comp.ProxyFunction | None
    ) -> ngs.CoefficientFunction:
        # Only the source concentration contributes to the flux
        del target, trg_test  # Unused
        if src_test is None:
            raise ValueError("Source test function cannot be None in active transport.")

        # Compute the flux using the Michaelis-Menten equation
        return self.v_max * src_test / (self.km + source)


class GeneralFlux(Transport):
    """General transport mechanism that allows for a constant or time dependent
    flux across the membrane, independent of the concentration difference.
    """
    def __init__(
            self,
            flux: Coefficient
    ):
        """Create a new general transport mechanism.

        :param flux: The constant flux across the membrane (units: substance/time).
        """
        super().__init__()
        self.flux_value = self._register_coefficient(flux, 'catalytic activity')


    def flux_rhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None
    ) -> ngs.CoefficientFunction:
        # Flux is independent of the concentrations
        del source, target
        return self.flux_value


class Transparent(Transport):
    """Transport mechanism that aims to make the membrane as transparent as
    possible by mimicking the diffusive speed on both sides of the membrane.
    """
    def __init__(
            self,
            source_diffusivity: Coefficient,
            target_diffusivity: Coefficient,
            source_porosity: float = None,
            target_porosity: float = None,
            outside_concentration: u.Quantity = None
    ):
        """Create a new transparent transport mechanism with the given
        diffusivities (and optionally porosities) for the source and target
        compartments.

        :param source_diffusivity: Diffusivity of the substance in the source
            compartment
        :param target_diffusivity: Diffusivity of the substance in the target
            compartment
        :param source_porosity: Porosity of the source compartment (optional)
        :param target_porosity: Porosity of the target compartment (optional)
        :param outside_concentration: Concentration of the substance outside of
            the domain if one of the compartments is a boundary.
        """
        super().__init__()
        self.src_diffusivity = self._register_coefficient(
            source_diffusivity, "diffusivity"
        )
        self.tgt_diffusivity = self._register_coefficient(
            target_diffusivity, "diffusivity"
        )
        self.src_porosity = source_porosity
        self.tgt_porosity = target_porosity
        if outside_concentration is not None:
            self.outside_concentration = self._register_coefficient(
                outside_concentration, "molar concentration"
            )
        else:
            self.outside_concentration = None

    def flux_lhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None,
            src_test: ngs.comp.ProxyFunction | None,
            trg_test: ngs.comp.ProxyFunction | None
    ) -> ngs.CoefficientFunction:
        del source, target  # Unused
        if src_test is None and trg_test is None:
            raise ValueError("Both source and target cannot be None.")
        if (src_test is None or trg_test is None) and self.outside_concentration is None:
            raise ValueError("No outside concentration specified for outside flux.")

        permeability = 2 * self.src_diffusivity * self.tgt_diffusivity / (
            self.src_diffusivity + self.tgt_diffusivity
        )

        if trg_test is None:
            return permeability * src_test
        if src_test is None:
            return -permeability * trg_test

        return permeability * (src_test - trg_test)


    def flux_rhs(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None
    ) -> ngs.CoefficientFunction:
        permeability = 2 * self.src_diffusivity * self.tgt_diffusivity / (
            self.src_diffusivity + self.tgt_diffusivity
        )

        if source is None:
            return permeability * self.outside_concentration
        if target is None:
            return -permeability * self.outside_concentration
