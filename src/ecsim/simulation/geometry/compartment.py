from dataclasses import dataclass, field

import astropy.units as u
import ngsolve as ngs

from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.units import to_simulation_units


# Define type aliases to shorten type annotations
S = ChemicalSpecies
C = ngs.CoefficientFunction


@dataclass(frozen=True)
class Region:
    """A region represents a part of the simulation geometry that is resolved in
    the mesh. It has a name and a volume.
    """
    name: str
    volume: u.Quantity


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


    def initialize_species(
            self,
            species: S,
            value: u.Quantity | dict[str, u.Quantity]
    ) -> None:
        """Set the initial concentration of a species in the compartment.

        :param species: Chemical species to initialize.
        :param value: Initial concentration value. It can be given as a single
            scalar, in which case it is assumed to be the same in all regions, or
            as a dictionary mapping region names to initial concentrations.
        :raises ValueError: If the species is already initialized or if the
            specified regions do not exist in the compartment.
        """
        if species in self.coefficients.initial_conditions:
            raise ValueError(f"Species {species} already initialized")
        self.coefficients.initial_conditions[species] = \
            self._to_coefficient_function(value, 'molar concentration')


    def add_diffusion(
            self,
            species: S,
            diffusivity: u.Quantity | dict[str, u.Quantity]
    ) -> None:
        """Add a diffusion event to the compartment.

        :param species: Chemical species that diffuses
        :param diffusivity: Coefficient for Fickian diffusion. It can be given
            as a single scalar, in which case it is assumed to be the same in all
            regions, or as a dictionary mapping region names to diffusivities.
        :raises ValueError: If diffusion for the species is already defined or if
            the specified regions do not exist in the compartment.
        """
        if species in self.coefficients.diffusion:
            raise ValueError(f"Diffusion for {species} already defined")
        self.coefficients.diffusion[species] = \
            self._to_coefficient_function(diffusivity, 'diffusivity')


    def add_reaction(
            self,
            reactants: list[S],
            products: list[S],
            k_on: u.Quantity | dict[str, u.Quantity],
            k_off: u.Quantity | dict[str, u.Quantity]
    ) -> None:
        """Add a reaction event to the compartment.

        :param reaction: Reaction term for use in the symbolic NGSolve
            reaction-diffusion equation
        """
        if (reactants, products) in self.coefficients.reactions:
            raise ValueError(f"Reaction {reactants} -> {products} already defined")

        k_on = self._to_coefficient_function(k_on, 'reaction rate')
        k_off = self._to_coefficient_function(k_off, 'reaction rate')

        self.coefficients.reactions[(reactants, products)] = (k_on, k_off)


    def _to_coefficient_function(
            self,
            value: u.Quantity | dict[str, u.Quantity],
            unit_name: str
    ) -> ngs.CoefficientFunction:
        """Convert a value or a dictionary of values to a NGSolve CoefficientFunction.
        """
        if isinstance(value, u.Quantity):
            return ngs.CoefficientFunction(to_simulation_units(value, unit_name))
        else:
            return self._create_piecewise_constant(value, unit_name)


    def _create_piecewise_constant(
            self,
            region_to_value: dict[str, u.Quantity],
            unit_name: str
        ):
        """Create a piecewise constant coefficient function from a dictionary of values.
        The given regions are checked against the list of all regions in the compartment.
        """
        # Check that all given regions exist in the compartment
        regions = set(region_to_value.keys())
        all_regions = set(self.get_region_names())
        if not regions.issubset(all_regions):
            raise ValueError(f"Regions {regions - all_regions} do not exist")

        full_names = {name: full_name for name, full_name in
                      zip(self.get_region_names(), self.get_region_names(full_names=True))}
        # Create a piecewise constant coefficient function
        coeff = {full_names[region]: to_simulation_units(value, unit_name)
                    for region, value in region_to_value.items()}
        return self._mesh.MaterialCF(coeff)


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
    """A container for simulation details about a compartment."""
    initial_conditions: dict[S, C] = field(default_factory=dict)
    diffusion: dict[S, C] = field(default_factory=dict)
    reactions: dict[tuple[list[S], list[S]], tuple[C, C]] = field(default_factory=dict)
