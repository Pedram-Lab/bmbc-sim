from dataclasses import dataclass, field
import numbers

import astropy.units as u
import astropy.constants as const
import ngsolve as ngs

from bmbcsim.simulation.simulation_agents import ChemicalSpecies
from bmbcsim.simulation.coefficient_fields.coefficient_field import (
    CoefficientField,
    ConstantField,
    PiecewiseConstantField,
)
from bmbcsim.units import to_simulation_units, BASE_UNITS


# Define type aliases to shorten type annotations
S = ChemicalSpecies
C = CoefficientField


class Region:
    """A region represents a part of the simulation geometry that is resolved in
    the mesh. It has a name and a volume.
    """
    def __init__(self, name: str, volume: float):
        self.name = name
        self._volume_parameter = ngs.Parameter(volume)

    @property
    def volume(self) -> u.Quantity:
        """Get the volume of the region."""
        return self._volume_parameter.Get() * BASE_UNITS['length'] ** 3

    def __eq__(self, other):
        if not isinstance(other, Region):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Compartment:
    """A compartment is a collection of regions that share the same
    bio-checmical agents. A compartment is made up of one or more regions and
    can aggregate diffusion and reaction events that occur within it.
    """
    def __init__(
            self,
            name: str,
            mesh: ngs.Mesh,
            regions: list[Region]
    ):
        """Create a new compartment.

        :param name: Name of the compartment.
        :param mesh: NGSolve mesh object that represents the geometry.
        :param regions: List of regions that make up the compartment.
        """
        self._mesh = mesh
        self.name = name
        self.regions = regions
        self.coefficients = SimulationDetails()


    @property
    def volume(self) -> u.Quantity:
        """Get the volume of the compartment aggregated over all regions.

        :return: Volume of the compartment
        """
        return sum(region.volume for region in self.regions)


    def get_region_names(self, *, full_names=False) -> list[str]:
        """Get the names of all regions of the compartment.

        :param full_names: Whether to return the full names of the regions or
            the names without the compartments.
        :returns regions: The regions of the given compartment.
        """
        if full_names and len(self.regions) > 1:
            return [full_name(self.name, region.name) for region in self.regions]
        else:
            return [region.name for region in self.regions]


    def _to_coefficient_field(
            self,
            value: u.Quantity | dict[str, u.Quantity] | CoefficientField
    ) -> CoefficientField:
        """Convert a value to a CoefficientField.

        :param value: The value to convert (scalar, dict, or CoefficientField).
        :returns: A CoefficientField instance.
        """
        if isinstance(value, CoefficientField):
            return value
        elif isinstance(value, u.Quantity):
            return ConstantField(value)
        else:
            # Dictionary of region -> value
            regions = set(value.keys())
            all_regions = set(self.get_region_names())
            if not regions.issubset(all_regions):
                raise ValueError(f"Regions {regions - all_regions} do not exist")
            full_names = {
                name: full_name_str
                for name, full_name_str in zip(
                    self.get_region_names(),
                    self.get_region_names(full_names=True)
                )
            }
            return PiecewiseConstantField(value, full_names)

    def initialize_species(
            self,
            species: S,
            value: u.Quantity | dict[str, u.Quantity] | CoefficientField
    ) -> None:
        """Set the initial concentration of a species in the compartment.

        :param species: Chemical species to initialize.
        :param value: Initial concentration value. It can be given as:
            - A scalar u.Quantity (same value in all regions)
            - A dictionary mapping region names to u.Quantity values
            - A CoefficientField object for spatially-varying fields
        :raises ValueError: If the species is already initialized.
        """
        if species in self.coefficients.initial_conditions:
            raise ValueError(f"Species {species} already initialized")
        self.coefficients.initial_conditions[species] = self._to_coefficient_field(value)


    def add_diffusion(
            self,
            species: S,
            diffusivity: u.Quantity | dict[str, u.Quantity] | CoefficientField
    ) -> None:
        """Add a diffusion event to the compartment.

        :param species: Chemical species that diffuses
        :param diffusivity: Coefficient for Fickian diffusion. It can be given as:
            - A scalar u.Quantity (same value in all regions)
            - A dictionary mapping region names to u.Quantity values
            - A CoefficientField object for spatially-varying fields
        :raises ValueError: If diffusion for the species is already defined.
        """
        if species in self.coefficients.diffusion:
            raise ValueError(f"Diffusion for {species} already defined")
        self.coefficients.diffusion[species] = self._to_coefficient_field(diffusivity)


    def add_relative_permittivity(
            self,
            relative_permittivity: float | dict[str, float]
    ) -> None:
        """Add a relative permittivity for electric fields in the compartment.

        :param relative_permittivity: Relative permittivity value for the compartment. The
            value is multiplied by the vacuum permittivity to get the actual
            permittivity for electric fields in the compartment.
        :raises ValueError: If permittivity for the compartment is already defined.
        """
        if self.coefficients.permittivity is not None:
            raise ValueError(f"Permittivity already defined for compartment '{self.name}'")
        if isinstance(relative_permittivity, numbers.Real):
            eps = relative_permittivity * const.eps0
        else:
            eps = {region: value * const.eps0 for region, value in relative_permittivity.items()}

        self.coefficients.permittivity = self._to_coefficient_field(eps)


    def add_porosity(
            self,
            porosity: float,
    ) -> None:
        """Add a porosity value 0 < alpha <= 1, which means that only the
        fraction alpha of the compartment is filled with the species.

        :param porosity: Porosity value for the compartment. The value must be
            between 0 and 1.
        :raises ValueError: If porosity for the compartment is already defined or
            if the value is not in the valid range.
        """
        if self.coefficients.porosity is not None:
            raise ValueError(f"Porosity already defined for compartment '{self.name}'")
        if not 0 < porosity <= 1:
            raise ValueError(f"Invalid porosity value {porosity}. Must be between 0 and 1.")

        self.coefficients.porosity = porosity


    def add_elasticity(
            self,
            youngs_modulus: u.Quantity | dict[str, u.Quantity],
            poisson_ratio: float | dict[str, float] = 0.3,
    ) -> None:
        """Add elastic material parameters for the compartment.

        :param youngs_modulus: Young's modulus (stiffness) of the material.
            Can be a single value or a dictionary mapping region names to values.
        :param poisson_ratio: Poisson's ratio of the material (default 0.3).
            Can be a single value or a dictionary mapping region names to values.
        :raises ValueError: If elasticity for the compartment is already defined.
        """
        if self.coefficients.elasticity is not None:
            raise ValueError(f"Elasticity already defined for compartment '{self.name}'")

        # Store raw values (converted to simulation units) for MechanicSolver to build MaterialCF
        if isinstance(youngs_modulus, dict):
            ym = {k: to_simulation_units(v, 'pressure') for k, v in youngs_modulus.items()}
        else:
            ym = to_simulation_units(youngs_modulus, 'pressure')

        self.coefficients.elasticity = (ym, poisson_ratio)


    def add_driving_species(
            self,
            species: S,
            coupling_strength: u.Quantity,
    ) -> None:
        """Set the species whose concentration drives mechanical contraction.

        The coupling strength represents the pressure generated per unit concentration.
        For example, a value of 1 kPa/mM means that 1 mM of the species generates
        1 kPa of chemical pressure driving contraction.

        :param species: Chemical species that drives contraction.
        :param coupling_strength: Pressure generated per unit concentration (e.g., kPa/mM).
        :raises ValueError: If a driving species is already defined for this compartment.
        """
        if self.coefficients.driving_species is not None:
            raise ValueError(f"Driving species already defined for compartment '{self.name}'")

        strength = to_simulation_units(coupling_strength, None)
        self.coefficients.driving_species = (species, strength)


    def add_reaction(
            self,
            reactants: list[S],
            products: list[S],
            k_f: u.Quantity | dict[str, u.Quantity] | CoefficientField,
            k_r: u.Quantity | dict[str, u.Quantity] | CoefficientField
    ) -> None:
        """Add a reaction event to the compartment.

        :param reactants: List of reactant species.
        :param products: List of product species.
        :param k_f: Forward reaction rate constant.
        :param k_r: Reverse reaction rate constant.
        :raises ValueError: If the reaction is already defined.
        """
        reaction_key = (tuple(reactants), tuple(products))
        if reaction_key in self.coefficients.reactions:
            raise ValueError(f"Reaction {reactants} -> {products} already defined")

        self.coefficients.reactions[reaction_key] = (
            self._to_coefficient_field(k_f),
            self._to_coefficient_field(k_r),
        )


    def __str__(self) -> str:
        return f"Compartment '{self.name}' with regions {self.get_region_names()}"


    def __repr__(self) -> str:
        return f"Compartment(name={self.name}, regions={self.regions}, volume={self.volume})"


    def __eq__(self, value):
        if not isinstance(value, Compartment):
            return False
        return self.name == value.name and self.regions == value.regions


    def __hash__(self):
        return hash((self.name, tuple(self.regions)))


def full_name(compartment: str, region: str) -> str:
    """Get the full name of a region within a compartment.
    
    :param compartment: The compartment name.
    :param region: The region name.
    :returns: The full name of the region.
    """
    return f'{compartment}:{region}'


@dataclass
class SimulationDetails:
    """A container for simulation details about a compartment.

    Coefficient fields are stored as CoefficientField objects and converted to
    NGSolve CoefficientFunctions during simulation setup.
    """
    initial_conditions: dict[S, C] = field(default_factory=dict)
    diffusion: dict[S, C] = field(default_factory=dict)
    reactions: dict[tuple[tuple[S, ...], tuple[S, ...]], tuple[C, C]] = field(default_factory=dict)
    permittivity: C | None = field(default=None)
    porosity: float | None = field(default=None)
    elasticity: tuple[float, float] | None = field(default=None)  # (E, nu)
    driving_species: tuple[S, float] | None = field(default=None)  # (species, strength)
