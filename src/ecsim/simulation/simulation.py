import logging
import os
from datetime import datetime

import astropy.units as u
import ngsolve as ngs
from tqdm import trange

from ecsim.evaluation.recorder import Recorder
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.simulation.geometry.simulation_geometry import SimulationGeometry
from ecsim.units import to_simulation_units
from .simulation_agents import ChemicalSpecies


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Simulation:
    """Build and execute a simulation of a reaction-diffusion system.
    """
    def __init__(
            self,
            name: str,
            *,
            result_root: str,
    ):
        """Initialize a new simulation.

        :param name: The name of the simulation (used for naming the result directory).
        :param result_root: The directory under which simulation results will be
            stored.
        """
        self.simulation_geometry = None
        self.species: list[ChemicalSpecies] = []

        # Set up result directory and logging
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.result_directory = os.path.join(result_root, f"{name}_{time_stamp}")
        if not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory)

        file_handler = logging.FileHandler(os.path.join(self.result_directory, "simulation.log"))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Set up empty containers for simulation data
        self._compartment_fes: dict[Compartment, ngs.FESpace] = {}
        self ._rd_fes: ngs.FESpace = None
        self._stiffness: dict[ChemicalSpecies, ngs.BilinearForm] = {}
        self._time_stepping_matrix: dict[ChemicalSpecies, ngs.BaseMatrix] = {}
        self._concentrations: dict[ChemicalSpecies, ngs.GridFunction] = {}
        self._dt = None

        self._recorders: list[Recorder] = []
        self._time_step = None


    def add_geometry(
            self,
            mesh: ngs.Mesh,
    ) -> SimulationGeometry:
        """Add a mesh to the simulation geometry.

        :param mesh: The mesh representing the geometry of the simulation.
        :returns: The :class:`SimulationGeometry` obtained from the mesh.
        :raises ValueError: If the geometry has already been set.
        """
        if self.simulation_geometry is not None:
            raise ValueError("Geometry has already been set.")
        self.simulation_geometry = SimulationGeometry(mesh)
        return self.simulation_geometry


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

        return species


    def add_recorder(
            self,
            recorder: Recorder,
    ) -> None:
        """Add a recorder to the simulation to record data during the simulation.

        :param recorder: An instance of a subclass of :class:`Recorder`.
        """
        if not isinstance(recorder, Recorder):
            raise TypeError("Recorder must be an instance of a subclass of Recorder")
        self._recorders.append(recorder)


    def simulate_until(
            self,
            *,
            end_time: u.Quantity,
            time_step: u.Quantity,
            start_time: u.Quantity = 0 * u.s,
    ) -> None:
        """Run the simulation until a given end time.

        :param end_time: The end time of the simulation.
        :param time_step: The time step to use for the simulation.
        :param start_time: The start time of the simulation.
        :raises ValueError: If the end time is not greater than the start time.
        """
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time.")

        n_steps = int(to_simulation_units(end_time - start_time, 'time') / self._dt)
        self.simulate_for(n_steps=n_steps, time_step=time_step, start_time=start_time)


    def simulate_for(
            self,
            *,
            n_steps: int,
            time_step: u.Quantity,
            start_time: u.Quantity = 0 * u.s,
    ) -> None:
        """Run the simulation for a given number of time steps.

        :param n_steps: The number of time steps to run the simulation for.
        :param time_step: The time step to use for the simulation.
        :param start_time: The start time of the simulation.
        :raises ValueError: If the number of steps is less than 1 or time step is not positive.
        """
        if n_steps < 1:
            raise ValueError("Number of steps must be at least 1.")
        if time_step <= 0 * u.s:
            raise ValueError(f"Time step must be positive, not {time_step}.")
        logger.info("Running simulation for %d steps of size %s.", n_steps, time_step)

        self._dt = to_simulation_units(time_step, 'time')
        self._time_step = time_step

        self._setup()

        name_to_concentration = {s.name: self._concentrations[s] for s in self.species}
        for recorder in self._recorders:
            recorder.setup(
                mesh=self.simulation_geometry.mesh,
                compartments=self.simulation_geometry.compartments.values(),
                directory=self.result_directory,
                concentrations=name_to_concentration,
                start_time=start_time
            )

        # TODO: replace this dummy linear form with a proper one
        f = ngs.LinearForm(self._rd_fes)
        f.Assemble()

        t = start_time
        for _ in trange(n_steps):
            residual = {}

            # Solve the potential equation
            for species in self.species:
                a = self._stiffness[species]
                c = self._concentrations[species]
                residual[species] = self._dt * (f.vec - a.mat * c.vec)

            for species, c in self._concentrations.items():
                c.vec.data += self._time_stepping_matrix[species] * residual[species]

            t += self._time_step
            for recorder in self._recorders:
                recorder.record(current_time=t)

        for recorder in self._recorders:
            recorder.finalize(end_time=t)


    def _setup(self) -> None:
        """Set up the simulation by initializing the finite element matrices.
        
        :param time_step: The time step to use for the simulation.
        """
        # Set up the finite element spaces
        logger.info("Setting up finite element spaces...")
        mesh = self.simulation_geometry.mesh
        compartments = self.simulation_geometry.compartments.values()
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


        for species in self.species:
            concentration, stiffness, time_stepping_matrix = self._setup_lhs(species)
            self._concentrations[species] = concentration
            self._stiffness[species] = stiffness
            self._time_stepping_matrix[species] = time_stepping_matrix


    def _setup_lhs(self, species):
        """Set up the left-hand side of the finite element equations for a given species.
        """
        compartments = self.simulation_geometry.compartments.values()
        mass = ngs.BilinearForm(self._rd_fes, check_unused=False)
        stiffness = ngs.BilinearForm(self._rd_fes, check_unused=False)
        active_dofs = ngs.BitArray(self._rd_fes.ndof)
        test_and_trial = [(test, trial) for test, trial in zip(*self._rd_fes.TnT())]
        concentration = ngs.GridFunction(self._rd_fes)

        for i, compartment in enumerate(compartments):
            # Initialize data structures for the species
            logger.debug("Initializing concentrations for species %s.", species)
            coefficients = compartment.coefficients
            test, trial = test_and_trial[i]

            # Initialize the concentrations in the compartment
            if species in coefficients.initial_conditions:
                c = concentration.components[i]
                c.Set(coefficients.initial_conditions[species])

            # Assemble the stiffness matrix (diffusion terms)
            if species in coefficients.diffusion and \
                    (diffusivity := coefficients.diffusion[species]) is not None:
                stiffness += diffusivity * ngs.grad(trial) * ngs.grad(test) * ngs.dx

            # Set up the time-stepping matrix (inverted perturbed mass matrix)
            mass += test * trial * ngs.dx

        # Assemble the mass and stiffness matrices and invert the time-stepping matrix
        mass.Assemble()
        stiffness.Assemble()
        mass.mat.AsVector().data += self._dt * stiffness.mat.AsVector()
        time_stepping_matrix = mass.mat.Inverse(active_dofs)

        return concentration, stiffness, time_stepping_matrix


def set_dofs(space, dof_array, region, value):
    """Set the values of the degrees of freedom in a given region.

    :param space: The finite element space to which the degrees of freedom belong.
    :param dof_array: The array of degrees of freedom to set. Changes are
        made in place.
    :param region: The region in which to set the degrees of freedom.
    :param value: The value to set the degrees of freedom to.
    """
    for el in region.Elements():
        for dof in space.GetDofNrs(el):
            dof_array[dof] = value
