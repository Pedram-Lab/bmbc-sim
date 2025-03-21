from dataclasses import dataclass

import astropy.units as u
import ngsolve as ngs

from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.units import to_simulation_units


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
        self._diffusion = {}
        self._reactions = {}


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


    def add_diffusion(
            self,
            species: ChemicalSpecies,
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
        if species in self._diffusion:
            raise ValueError(f"Diffusion for {species} already defined")

        if isinstance(diffusivity, u.Quantity):
            coeff = to_simulation_units(diffusivity, 'diffusivity')
            self._diffusion[species] = ngs.CoefficientFunction(coeff)
        else:
            self._diffusion[species] = self._create_piecewise_constant(diffusivity, 'diffusivity')


    def add_reaction(
            self,
            reactants: list[ChemicalSpecies],
            products: list[ChemicalSpecies],
            k_on: u.Quantity | dict[str, u.Quantity],
            k_off: u.Quantity | dict[str, u.Quantity]
    ) -> None:
        """Add a reaction event to the compartment.

        :param reaction: Reaction term for use in the symbolic NGSolve
            reaction-diffusion equation
        """
        if (reactants, products) in self._reactions:
            raise ValueError(f"Reaction {reactants} -> {products} already defined")

        k_on = to_simulation_units(k_on, 'reaction rate') if isinstance(k_on, u.Quantity) \
            else self._create_piecewise_constant(k_on, 'reaction rate')

        k_off = to_simulation_units(k_off, 'reaction rate') if isinstance(k_off, u.Quantity) \
            else self._create_piecewise_constant(k_off, 'reaction rate')

        self._reactions[(reactants, products)] = (k_on, k_off)


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
            raise ValueError(f"Regions {regions - set(all_regions)} do not exist")

        # Create a piecewise constant coefficient function
        coeff = {region: to_simulation_units(value, unit_name)
                    for region, value in region_to_value.items()}
        return self._mesh.MaterialCF(coeff)


    def __str__(self) -> str:
        return f"Compartment {self.name} with regions {self.get_region_names()}"


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
