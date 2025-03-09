import logging

import astropy.units as u
import ngsolve as ngs

from ecsim.simulation.geometry_description import GeometryDescription, full_name
from ecsim.units import to_simulation_units
from .simulation_agents import ChemicalSpecies


logger = logging.getLogger(__name__)


class Simulation:
    """Build and execute a simulation of a reaction-diffusion system.
    """
    def __init__(
            self,
            geometry_description: GeometryDescription,
    ):
        self.geometry_description = geometry_description
        self.species: list[ChemicalSpecies] = []

        # Set up the finite element spaces
        logger.info("Setting up finite element spaces...")
        mesh = geometry_description.mesh
        self._compartment_fes = {}
        for compartment in self.geometry_description.compartments:
            regions = '|'.join(self.geometry_description.get_regions(compartment, full_names=True))
            fes = ngs.Compress(ngs.H1(mesh, order=1, definedon=regions))
            self._compartment_fes[compartment] = fes
            logger.info("Compartment %s has %d degrees of freedom.", compartment, fes.ndof)

        # Note that the order of the compartment spaces is the same as the order of compartments
        self._rd_fes = ngs.FESpace([self._compartment_fes[compartment]
                                    for compartment in self.geometry_description.compartments])
        logger.info("Total number of degrees of freedom for reaction-diffusion: %d.",
                    self._rd_fes.ndof)

        # Set up empty containers for simulation data
        self._blf = {}
        self._rhs = {}
        self._concentrations = {}
        self._test_and_trial = {}


    def add_species(
            self,
            species: ChemicalSpecies,
    ) -> ChemicalSpecies:
        """
        Add a new :class:`ChemicalSpecies` to the simulation.
        
        :param species: The :class:`ChemicalSpecies` to add.
        :returns: The added :class:`ChemicalSpecies`.
        :raises ValueError: If the species already exists in the simulation.
        """
        if species in self.species:
            raise ValueError(f"Species {species.name} already exists.") from None
        self.species.append(species)
        logger.debug("Add species %s to simulation.", species)

        # Set up finite element structures for the species
        self._blf[species] = ngs.BilinearForm(self._rd_fes)
        self._rhs[species] = ngs.LinearForm(self._rd_fes)
        self._concentrations[species] = ngs.GridFunction(self._rd_fes)
        self._test_and_trial[species] = {
            self.geometry_description.compartments[i]: (test, trial)
            for i, (test, trial) in enumerate(zip(*self._rd_fes.TnT()))
        }

        return species


    def add_diffusion(
            self,
            species: ChemicalSpecies,
            compartment: str,
            diffusivity: u.Quantity | dict[str, u.Quantity],
    ) -> None:
        """Add diffusion for a species in a compartment.

        :param species: The :class:`ChemicalSpecies` to specify diffusivity for.
        :param compartment: The name of the compartment for which to specify
            diffusivity.
        :param diffusivity: The diffusivity of the species in the compartment.
            Either a single value or a dictionary of values for each region in
            the compartment. Regions that are not specified will be set to zero
            diffusivity.
        :raises ValueError: If a given species, compartment, or region does not
            exist in the simulation, or the quantity is not a valid diffusivity.
        """
        if species not in self.species:
            raise ValueError(f"Species {species.name} does not exist.")
        if compartment not in self.geometry_description.compartments:
            raise ValueError(f"Compartment {compartment} does not exist.")

        if isinstance(diffusivity, u.Quantity):
            diffusivity = to_simulation_units(diffusivity, 'diffusivity')
        else:
            existing_regions = self.geometry_description.get_regions(compartment)
            if not all(region in existing_regions for region in diffusivity.keys()):
                raise ValueError(f"Some regions of {diffusivity.keys()}"
                                 f"do not exist in compartment {compartment}.")

            mesh = self.geometry_description.mesh
            coeffs = {
                full_name(compartment, region): to_simulation_units(quantity, 'diffusivity')
                for region, quantity in diffusivity.items()
            }
            diffusivity = mesh.MaterialCF(coeffs, default=0.0)

        blf = self._blf[species]
        test, trial = self._test_and_trial[species][compartment]
        blf += diffusivity * ngs.grad(trial) * ngs.grad(test) * ngs.dx
