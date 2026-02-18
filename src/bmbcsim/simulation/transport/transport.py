import abc
from typing import Callable

import ngsolve as ngs
import astropy.units as u

import bmbcsim.simulation.coefficient_fields as cf


TransportCoefficient = u.Quantity | cf.Coefficient
_CoefficientSpec = tuple[cf.Coefficient, str | None, Callable | None]


class Transport(abc.ABC):
    """Abstract base class for transport mechanisms in a membrane.
    """
    def __init__(self):
        self._mutable_coefficients: list[tuple[Callable, ngs.Parameter]] = []
        self._coefficient_specs: dict[str, _CoefficientSpec] = {}


    def flux(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None
    ) -> ngs.CoefficientFunction | None:
        """Compute the flux density through the membrane evaluated at current
        concentrations. The flux is assumed to be the total flux across the
        membrane. In case a flux density is readily available, it should be
        multiplied by :code:`membrane.area`.
        All of the arguments can be None, in which case the flux is assumed to
        connect the domain to the outside.

        :param source: Coefficient function representing the concentration in
            the source compartment.
        :param target: Coefficient function representing the concentration in
            the target compartment.
        :return: Coefficient function representing the flux across the membrane
            from source to target, or None if no flux contribution.
        """
        del source, target  # Unused in default implementation
        return None


    def update_flux(self, t: u.Quantity) -> None:
        """Update the transport parameters based on the time t.
        This method can be used to modify the transport properties dynamically
        during the simulation, e.g., to implement time-dependent transport rates.

        :param t: The current time in the simulation.
        """
        # Update all mutable coefficients (temporal modulation factors)
        # Changes made here are visible in the flux expression
        for temporal_func, parameter in self._mutable_coefficients:
            new_value = temporal_func(t)
            parameter.Set(new_value)


    def finalize_coefficients(self, region: ngs.Region, fes: ngs.FESpace) -> None:
        """Convert deferred spatial coefficients to CoefficientFunctions.

        Called during simulation setup after membrane FE spaces are created.
        This converts the coefficient specs stored by _register_coefficient
        into actual NGSolve CoefficientFunctions as instance attributes.

        :param region: The NGSolve region (typically a boundary region).
        :param fes: The finite element space for this membrane.
        """
        for attr_name, spec in list(self._coefficient_specs.items()):
            field, physical_name, temporal = spec
            spatial_cf = field.to_coefficient_function(region, fes, physical_name)

            if temporal is not None:
                # Multiply by time-dependent parameter
                time_param = ngs.Parameter(temporal(0.0 * u.s))
                self._mutable_coefficients.append((temporal, time_param))
                setattr(self, attr_name, spatial_cf * time_param)
            else:
                setattr(self, attr_name, spatial_cf)

            del self._coefficient_specs[attr_name]


    def _register_coefficient(
            self,
            attr_name: str,
            coeff: TransportCoefficient,
            physical_name: str = None,
            temporal: Callable[[u.Quantity], float] = None
    ) -> None:
        """Register a coefficient for deferred conversion.

        :param attr_name: The attribute name to set after finalization.
        :param coeff: The coefficient to register. Can be a Quantity or
            cf.Coefficient for spatially-varying values.
        :param physical_name: The physical quantity of the coefficient for unit
            conversion.
        :param temporal: Optional time modulation factor. The final value is:
            spatial_value(x) * temporal(t).
        """
        if isinstance(coeff, u.Quantity):
            coeff = cf.Constant(coeff)
        if not isinstance(coeff, cf.Coefficient):
            raise ValueError("Coefficient must be Quantity or cf.Coefficient.")
        self._coefficient_specs[attr_name] = (coeff, physical_name, temporal)


class Passive(Transport):
    """Passive transport mechanism that computes the flux as a linear function
    of the concentration difference across the membrane.
    """
    permeability: ngs.CoefficientFunction
    outside_concentration: ngs.CoefficientFunction | None = None

    def __init__(
            self,
            permeability: TransportCoefficient,
            outside_concentration: u.Quantity = None,
            temporal: Callable[[u.Quantity], float] = None
    ):
        """Create a new passive transport mechanism with the given permeability.

        :param permeability: Permeability of the membrane (units: area * length/time).
        :param outside_concentration: Concentration of the substance outside of
            the domain if one of the compartments is a boundary.
        :param temporal: Optional time modulation factor for spatial coefficients.
            The final value is: spatial_field(x) * temporal(t)
        """
        super().__init__()
        self._register_coefficient(
            "permeability", permeability, "volumetric flow rate", temporal
        )
        if outside_concentration is not None:
            self._register_coefficient(
                "outside_concentration", outside_concentration, "molar concentration"
            )


    def flux(self, source, target):
        s = source if source is not None else self.outside_concentration
        t = target if target is not None else self.outside_concentration
        return self.permeability * (s - t)


class Active(Transport):
    """Active transport mechanism that computes the flux based on
    Michaelis-Menten kinetics for the source concentration.
    """
    v_max: ngs.CoefficientFunction
    km: ngs.CoefficientFunction

    def __init__(
            self,
            v_max: TransportCoefficient,
            km: TransportCoefficient,
            temporal: Callable[[u.Quantity], float] = None
    ):
        """Create a new active transport mechanism based on Michaelis-Menten kinetics.

        :param v_max: Maximum rate of the transport (units: substance/time).
        :param km: Michaelis constant (units: concentration).
        :param temporal: Optional time modulation factor for spatial coefficients.
            The final value is: spatial_field(x) * temporal(t)
        """
        super().__init__()
        self._register_coefficient("v_max", v_max, "catalytic activity", temporal)
        self._register_coefficient("km", km, "molar concentration")


    def flux(self, source, target):
        del target  # Unused
        return self.v_max * source / (self.km + source)


class GeneralFlux(Transport):
    """General transport mechanism that allows for a constant or time dependent
    flux across the membrane, independent of the concentration difference.
    """
    flux_value: ngs.CoefficientFunction

    def __init__(
            self,
            flux: TransportCoefficient,
            temporal: Callable[[u.Quantity], float] = None
    ):
        """Create a new general transport mechanism.

        :param flux: The constant flux across the membrane (units: substance/time).
        :param temporal: Optional time modulation factor for spatial coefficients.
            The final value is: spatial_field(x) * temporal(t)
        """
        super().__init__()
        self._register_coefficient("flux_value", flux, "catalytic activity", temporal)


    def flux(self, source, target):
        del source, target  # Flux is independent of the concentrations
        return self.flux_value


class Transparent(Transport):
    """Transport mechanism that aims to make the membrane as transparent as
    possible by mimicking the diffusive speed on both sides of the membrane.
    """
    src_diffusivity: ngs.CoefficientFunction
    tgt_diffusivity: ngs.CoefficientFunction
    outside_concentration: ngs.CoefficientFunction | None = None

    def __init__(
            self,
            source_diffusivity: TransportCoefficient,
            target_diffusivity: TransportCoefficient,
            outside_concentration: u.Quantity = None,
            temporal: Callable[[u.Quantity], float] = None
    ):
        """Create a new transparent transport mechanism with the given
        diffusivities for the source and target compartments.

        :param source_diffusivity: Diffusivity of the substance in the source
            compartment
        :param target_diffusivity: Diffusivity of the substance in the target
            compartment
        :param outside_concentration: Concentration of the substance outside of
            the domain if one of the compartments is a boundary.
        :param temporal: Optional time modulation factor for spatial coefficients.
            The final value is: spatial_field(x) * temporal(t)
        """
        super().__init__()
        self._register_coefficient(
            "src_diffusivity", source_diffusivity, "diffusivity", temporal
        )
        self._register_coefficient(
            "tgt_diffusivity", target_diffusivity, "diffusivity", temporal
        )
        if outside_concentration is not None:
            self._register_coefficient(
                "outside_concentration", outside_concentration, "molar concentration"
            )

    def flux(self, source, target):
        s = source if source is not None else self.outside_concentration
        t = target if target is not None else self.outside_concentration
        permeability = 2 * self.src_diffusivity * self.tgt_diffusivity / (
            self.src_diffusivity + self.tgt_diffusivity
        )
        return permeability * (s - t)
