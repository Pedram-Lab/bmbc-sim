import logging
from dataclasses import dataclass

import astropy.units as u
import ngsolve as ngs

from ecsim.simulation.geometry.simulation_geometry import SimulationGeometry
from ecsim.units import to_simulation_units
from .simulation_agents import ChemicalSpecies


logger = logging.getLogger(__name__)


class Simulation:
    """Build and execute a simulation of a reaction-diffusion system.
    """
    def __init__(
            self,
            simulation_geometry: SimulationGeometry,
    ):
        self.simulation_geometry = simulation_geometry
        self.species: list[ChemicalSpecies] = []

        # Set up the finite element spaces
        logger.info("Setting up finite element spaces...")
        mesh = simulation_geometry.mesh
        compartments = simulation_geometry.compartments.values()
        self._compartment_fes = {}
        for compartment in compartments:
            regions = '|'.join(compartment.get_region_names(full_names=True))
            fes = ngs.Compress(ngs.H1(mesh, order=1, definedon=regions))
            self._compartment_fes[compartment] = fes
            logger.debug("%s has %d degrees of freedom.", compartment, fes.ndof)

        # Note that the order of the compartment spaces is the same as the order of compartments
        self._rd_fes = ngs.FESpace([self._compartment_fes[compartment]
                                    for compartment in compartments])
        logger.info("Total number of degrees of freedom for reaction-diffusion: %d.",
                    self._rd_fes.ndof)

        # Set up empty containers for simulation data
        self._fem_setup: dict[str, FemSetup] = {}
        self._fem_matrices: dict[str, FemMatrices] = {}


    def add_species(
            self,
            name: str,
            *,
            valence: int = 0,
    ) -> ChemicalSpecies:
        """
        Add a new :class:`ChemicalSpecies` to the simulation.
        
        :param name: The name of the species.
        :param valence: The valence (i.e., unit charge) of the species (default is 0).
        :returns: The added :class:`ChemicalSpecies`.
        :raises ValueError: If the species already exists in the simulation.
        """
        species = ChemicalSpecies(name, valence=valence)
        if species in self.species:
            raise ValueError(f"Species {species.name} already exists.") from None

        self.species.append(species)
        logger.debug("Add species %s to simulation.", species)

        # Set up finite element structures for the species
        self._fem_setup[species] = FemSetup(self._rd_fes,
                                            self.simulation_geometry.compartment_names)

        return species


    def setup(self) -> None:
        """Set up the simulation by initializing the finite element matrices."""
        logger.info("Setting up simulation...")
        logger.debug("Initializing concentrations for species %s.", self.species)
        compartments = self.simulation_geometry.compartments.values()

        # Initialize the concentrations for each species in each compartment
        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients

            for species in self.species:
                if species in coefficients.initial_conditions:
                    concentration = self._fem_setup[species].concentration.components[i]
                    concentration.Set(coefficients.initial_conditions[species])


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
        if compartment not in self.simulation_geometry.compartment_names:
            raise ValueError(f"Compartment {compartment} does not exist.")

        if isinstance(diffusivity, u.Quantity):
            diffusivity = to_simulation_units(diffusivity, 'diffusivity')
        else:
            existing_regions = self.simulation_geometry.get_regions(compartment)
            if not all(region in existing_regions for region in diffusivity.keys()):
                raise ValueError(f"Some regions of {diffusivity.keys()}"
                                 f"do not exist in compartment {compartment}.")

            mesh = self.simulation_geometry.mesh
            coeffs = {
                full_name(compartment, region): to_simulation_units(quantity, 'diffusivity')
                for region, quantity in diffusivity.items()
            }
            diffusivity = mesh.MaterialCF(coeffs, default=0.0)

        blf = self._fem_setup[species].blf
        test, trial = self._fem_setup[species].test_and_trial[compartment]
        blf += diffusivity * ngs.grad(trial) * ngs.grad(test) * ngs.dx


    def simulate_until(
            self,
            *,
            time_step: u.Quantity,
            end_time: u.Quantity,
            start_time: u.Quantity = 0 * u.s,
    ) -> None:
        """Run the simulation until a given end time.

        :param time_step: The time step to use for the simulation.
        :param end_time: The end time of the simulation.
        :param start_time: The start time of the simulation.
        :raises ValueError: If the end time is not greater than the start time.
        """
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time.")

        n_steps = int((end_time - start_time) / time_step)
        self.simulate_for(time_step=time_step, n_steps=n_steps, start_time=start_time)


    def simulate_for(
            self,
            *,
            time_step: u.Quantity,
            n_steps: int,
            start_time: u.Quantity = 0 * u.s,
    ) -> None:
        """Run the simulation for a given number of time steps.

        :param time_step: The time step to use for the simulation.
        :param n_steps: The number of time steps to run the simulation for.
        :param start_time: The start time of the simulation.
        :raises ValueError: If the number of steps is less than 1 or time step is not positive.
        """
        if n_steps < 1:
            raise ValueError("Number of steps must be at least 1.")
        if time_step <= 0 * u.s:
            raise ValueError("Time step must be positive.")

        for species in self.species:
            logger.info("Setting up simulation for species %s.", species.name)
            self._fem_matrices[species] = self._fem_setup[species].assemble(
                dt=to_simulation_units(time_step, 'time'),
                simulation_geometry=self.simulation_geometry,
            )

        logger.info("Running simulation for %d steps of size %s.", n_steps, time_step)

        for i in range(n_steps):
            residual = {}

            # Solve the potential equation
            for name, fem_setup in self._fem_setup.items():
                fem_setup.rsh.Assemble()
                a = self._fem_matrices[name].stiffness
                u = fem_setup.concentrations[name]
                residual[name] = self._time_step_size * (f.vec - a.mat * u.vec)

            for name, u in self.concentrations.items():
                u.vec.data += self._time_stepping_matrix[name] * residual[name]


class FemSetup():
    """Gathers symbolic expressions for the finite element setup.
    """
    def __init__(self, fes: ngs.FESpace, compartments: list[str]):
        self.blf = ngs.BilinearForm(fes)
        self.rhs = ngs.LinearForm(fes)
        self.active_dofs = ngs.BitArray(fes.ndof)
        self.test_and_trial = {
            compartments[i]: (test, trial)
            for i, (test, trial) in enumerate(zip(*fes.TnT()))
        }
        self.concentration = ngs.GridFunction(fes)


    def assemble(
            self,
            dt: float,
            simulation_geometry: SimulationGeometry
    ) -> 'FemMatrices':
        """Assemble the bilinear and linear forms for the simulation.
        """
        fes = self.blf.space

        # Assemble the stiffness matrix
        self.blf.Assemble()
        stiffness = self.blf.mat

        # Set up the mass matrix
        mass = ngs.BilinearForm(fes)
        for compartment in simulation_geometry.compartment_names:
            test, trial = self.test_and_trial[compartment]
            mass += test * trial * ngs.dx
        mass.Assemble()
        mass = mass.mat

        mass.AsVector().data += dt * stiffness.AsVector()

        return FemMatrices(stiffness=stiffness,
                           exp_a_delta_t=mass.Inverse(self.active_dofs))


    def set_dofs(self, dof_array, region, value):
        """Set the values of the degrees of freedom in a given region.

        :param dof_array: The array of degrees of freedom to set. Changes are
            made in place.
        :param region: The region in which to set the degrees of freedom.
        :param value: The value to set the degrees of freedom to.
        """
        for el in region.Elements():
            for dof in self.blf.space.GetDofNrs(el):
                dof_array[dof] = value


@dataclass
class FemMatrices():
    """Gathers the assembled bilinear and linear forms.
    """
    exp_a_delta_t: ngs.BaseMatrix
    stiffness: ngs.BaseMatrix
