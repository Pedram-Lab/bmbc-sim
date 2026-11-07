import abc
from typing import Callable

import ngsolve as ngs
import astropy.units as u

import bmbcsim.simulation.coefficient_fields as cf
from bmbcsim.units import to_simulation_units


TransportCoefficient = u.Quantity | cf.Coefficient
_CoefficientSpec = tuple[cf.Coefficient, str | None, Callable | None, bool]


class Transport(abc.ABC):
    """Abstract base class for transport mechanisms in a membrane.
    """
    def __init__(self):
        self._mutable_coefficients: list[tuple[Callable, ngs.Parameter]] = []
        self._coefficient_specs: dict[str, _CoefficientSpec] = {}
        self._finalized_coefficients: list[str] = []
        self._spatial_coefficients: dict[str, ngs.CoefficientFunction] = {}


    def linearize(
            self,
            source: ngs.CoefficientFunction | None,
            target: ngs.CoefficientFunction | None
    ) -> tuple[ngs.CoefficientFunction | None, ...]:
        """Linearize the flux density through the membrane for an implicit step.

        The flux from source to target is written as
        :code:`J = a_s * c_s - a_t * c_t + b`, where :code:`c_s` and :code:`c_t`
        are the (new) concentrations in the source and target compartments.
        The coefficients may depend on the current concentrations (lagged), but
        :code:`a_s` and :code:`a_t` must be nonnegative so that the implicit
        step preserves positivity. The flux is the total flux across the
        membrane; area normalization is handled in :meth:`finalize_coefficients`.
        Either compartment can be None, in which case the membrane connects
        the domain to the outside.

        :param source: Current concentration in the source compartment.
        :param target: Current concentration in the target compartment.
        :return: Tuple :code:`(a_s, a_t, b)`; terms that vanish are None.
        """
        del source, target  # Unused in default implementation
        return None, None, None


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


    def finalize_coefficients(
            self,
            region: ngs.Region,
            fes: ngs.FESpace,
            membrane_area: u.Quantity
    ) -> None:
        """Convert deferred spatial coefficients to CoefficientFunctions.

        Called during simulation setup after membrane FE spaces are created.
        This converts the coefficient specs stored by _register_coefficient
        into actual NGSolve CoefficientFunctions as instance attributes.

        :param region: The NGSolve region (typically a boundary region).
        :param fes: The finite element space for this membrane.
        :param membrane_area: The area of the membrane for flux normalization.
        """
        area = to_simulation_units(membrane_area, 'area')

        for attr_name, spec in list(self._coefficient_specs.items()):
            field, physical_name, temporal, normalize_by_area = spec
            spatial_cf = field.to_coefficient_function(region, fes, physical_name)

            if normalize_by_area and field.needs_area_normalization:
                spatial_cf = spatial_cf / area

            self._spatial_coefficients[attr_name] = spatial_cf

            if temporal is not None:
                # Multiply by time-dependent parameter
                time_param = ngs.Parameter(temporal(0.0 * u.s))
                self._mutable_coefficients.append((temporal, time_param))
                setattr(self, attr_name, spatial_cf * time_param)
            else:
                setattr(self, attr_name, spatial_cf)

            self._finalized_coefficients.append(attr_name)
            del self._coefficient_specs[attr_name]


    def _register_coefficient(
            self,
            attr_name: str,
            coeff: TransportCoefficient,
            physical_name: str = None,
            temporal: Callable[[u.Quantity], float] = None,
            normalize_by_area: bool = False
    ) -> None:
        """Register a coefficient for deferred conversion.

        :param attr_name: The attribute name to set after finalization.
        :param coeff: The coefficient to register. Can be a Quantity or
            cf.Coefficient for spatially-varying values.
        :param physical_name: The physical quantity of the coefficient for unit
            conversion.
        :param temporal: Optional time modulation factor. The final value is:
            spatial_value(x) * temporal(t).
        :param normalize_by_area: Whether to divide by membrane area during
            finalization. Use for coefficients that represent total fluxes.
        """
        if isinstance(coeff, u.Quantity):
            coeff = cf.Constant(coeff)
        if not isinstance(coeff, cf.Coefficient):
            raise ValueError("Coefficient must be Quantity or cf.Coefficient.")
        self._coefficient_specs[attr_name] = (coeff, physical_name, temporal, normalize_by_area)


def _linear_exchange(permeability, outside_concentration, source, target):
    """Linearization of :code:`J = P * (c_s - c_t)`, where a missing side is
    replaced by the (fixed) outside concentration."""
    if source is None:
        return None, permeability, permeability * outside_concentration
    if target is None:
        return permeability, None, -permeability * outside_concentration
    return permeability, permeability, None


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
            "permeability", permeability, "volumetric flow rate", temporal,
            normalize_by_area=True
        )
        if outside_concentration is not None:
            self._register_coefficient(
                "outside_concentration", outside_concentration, "molar concentration"
            )


    def linearize(self, source, target):
        return _linear_exchange(self.permeability, self.outside_concentration, source, target)


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
        self._register_coefficient("v_max", v_max, "catalytic activity", temporal,
                                    normalize_by_area=True)
        self._register_coefficient("km", km, "molar concentration")


    def linearize(self, source, target):
        # Picard linearization v_max * c^{n+1} / (km + c^n) keeps a_s >= 0
        del target  # Unused
        if source is None:
            raise ValueError("Active transport requires a source compartment.")
        return self.v_max / (self.km + source), None, None


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
        self._register_coefficient("flux_value", flux, "catalytic activity", temporal,
                                    normalize_by_area=True)


    def linearize(self, source, target):
        del source, target  # Flux is independent of the concentrations
        return None, None, self.flux_value


class ProportionalFlux(Transport):
    """Transport with flux gated by source concentration.

    Scales flux by a clamped linear ramp of source concentration:
    - c >= saturation: full flux (scale = 1)
    - c <= depletion: zero flux (scale = 0)
    - depletion < c < saturation: linear interpolation
    """
    flux_value: ngs.CoefficientFunction
    saturation: ngs.CoefficientFunction
    depletion: ngs.CoefficientFunction

    def __init__(
            self,
            flux: TransportCoefficient,
            saturation: TransportCoefficient,
            depletion: TransportCoefficient,
            temporal: Callable[[u.Quantity], float] = None
    ):
        """Create a new proportional transport mechanism.

        :param flux: The maximum flux across the membrane (units: substance/time).
        :param saturation: Concentration above which full flux is attained.
        :param depletion: Concentration below which flux is zero.
        :param temporal: Optional time modulation factor for spatial coefficients.
            The final value is: spatial_field(x) * temporal(t)
        """
        super().__init__()
        self._register_coefficient("flux_value", flux, "catalytic activity", temporal,
                                    normalize_by_area=True)
        self._register_coefficient("saturation", saturation, "molar concentration")
        self._register_coefficient("depletion", depletion, "molar concentration")


    def linearize(self, source, target):
        raise NotImplementedError(
            "ProportionalFlux is treated by the DiffusionSolver directly (lagged clipping)."
        )


class Transparent(Transport):
    """Transport mechanism that aims to make the membrane as transparent as
    possible by mimicking the diffusive speed on both sides of the membrane.
    """
    permeability: ngs.CoefficientFunction
    outside_concentration: ngs.CoefficientFunction | None = None

    def __init__(
            self,
            source_diffusivity: u.Quantity,
            target_diffusivity: u.Quantity,
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
        permeability = 2 * source_diffusivity * target_diffusivity / (
            source_diffusivity + target_diffusivity
        )
        self._register_coefficient(
            "permeability", permeability, temporal=temporal, normalize_by_area=True
        )
        if outside_concentration is not None:
            self._register_coefficient(
                "outside_concentration", outside_concentration, "molar concentration"
            )

    def linearize(self, source, target):
        return _linear_exchange(self.permeability, self.outside_concentration, source, target)
