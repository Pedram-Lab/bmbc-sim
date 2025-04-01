import logging
import os
from datetime import datetime

import astropy.units as u
import ngsolve as ngs
from tqdm import trange

from ecsim.logging import logger
from ecsim.simulation.recorder import Recorder
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.simulation.geometry.simulation_geometry import SimulationGeometry
from ecsim.units import to_simulation_units
from .simulation_agents import ChemicalSpecies


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
        self._mass_inv: ngs.BaseMatrix = None
        self._source_terms: dict[ChemicalSpecies, ngs.LinearForm] = {}

        self._recorders: list[Recorder] = []
        self._time_step = None


    def setup_geometry(
            self,
            mesh: ngs.Mesh,
    ) -> SimulationGeometry:
        """Add a mesh and set up the simulation geometry.

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


    def run(
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
        if time_step <= 0 * u.s:
            raise ValueError(f"Time step must be positive, not {time_step}.")

        n_steps = int((end_time - start_time) / time_step)
        if n_steps < 1:
            raise ValueError("Number of steps must be at least 1.")
        logger.info("Running simulation for %d steps of size %s.", n_steps, time_step)

        self._dt = to_simulation_units(time_step, 'time')
        self._time_step = time_step

        # TODO: make number of threads configurable
        ngs.SetNumThreads(4)
        task_manager = ngs.TaskManager()
        task_manager.__enter__()

        self._setup()

        name_to_concentration = {s.name: self._concentrations[s] for s in self.species}
        for recorder in self._recorders:
            recorder.setup(
                directory=self.result_directory,
                mesh=self.simulation_geometry.mesh,
                compartments=self.simulation_geometry.compartments.values(),
                concentrations=name_to_concentration,
                start_time=start_time.copy()
            )

        t = start_time.copy()
        for _ in trange(n_steps):
            # Update the concentrations via first-order operator splitting
            # Full step for reaction and transport
            self._update_transport(t)
            for species, c in self._concentrations.items():
                f = self._source_terms[species]
                f.Assemble()
                c.vec.data += self._dt * (self._mass_inv * f.vec)

            # Full step for diffusion
            for species, c in self._concentrations.items():
                a = self._stiffness[species]
                m_star = self._time_stepping_matrix[species]
                residual = -self._dt * (a.mat * c.vec)
                c.vec.data += m_star * residual

            t += self._time_step
            for recorder in self._recorders:
                recorder.record(current_time=t)

        for recorder in self._recorders:
            recorder.finalize(end_time=t)

        task_manager.__exit__(None, None, None)


    def _update_transport(self, t: u.Quantity) -> None:
        """Update the transport mechanisms based on the current time.

        :param t: The current time in the simulation.
        """
        # Update all transport mechanisms
        for membrane in self.simulation_geometry.membranes.values():
            for transport in membrane.get_transport().values():
                transport.update_flux(t)


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
            concentration, mass_inv, stiffness, time_stepping_matrix = self._setup_lhs(species)
            self._concentrations[species] = concentration
            self._stiffness[species] = stiffness
            self._time_stepping_matrix[species] = time_stepping_matrix
            self._mass_inv = mass_inv

        self._source_terms = self._setup_rhs()


    def _setup_lhs(self, species):
        """Set up the left-hand side of the finite element equations for a given species.
        """
        compartments = self.simulation_geometry.compartments.values()
        mass = ngs.BilinearForm(self._rd_fes, check_unused=False)
        stiffness = ngs.BilinearForm(self._rd_fes, check_unused=False)
        active_dofs = ngs.BitArray(self._rd_fes.ndof)
        active_dofs[:] = True
        test_and_trial = list(zip(*self._rd_fes.TnT()))
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

        # Assemble the mass and stiffness matrices
        mass.Assemble()
        stiffness.Assemble()

        # Invert the mass matrix and the matrix for the implicit mid-point rule
        m_star = mass.mat.CreateMatrix()
        m_star.AsVector().data = mass.mat.AsVector() + self._dt / 2 * stiffness.mat.AsVector()
        time_stepping_matrix = m_star.Inverse(active_dofs)
        m_inv = mass.mat.Inverse()

        return concentration, m_inv, stiffness, time_stepping_matrix


    def _setup_rhs(self):
        """Set up the right-hand side of the finite element equations for all species.
        """
        test_functions = self._rd_fes.TestFunction()
        source_terms = {s: ngs.LinearForm(self._rd_fes) for s in self.species}
        compartments = list(self.simulation_geometry.compartments.values())
        compartment_to_index = {compartment: i for i, compartment in enumerate(compartments)}

        # Handle reaction terms
        for i, compartment in enumerate(compartments):
            coefficients = compartment.coefficients
            test = test_functions[i]

            for (reactants, products), (k_f, k_r) in coefficients.reactions.items():
                # TODO: find a better default value
                all_reactants = ngs.CoefficientFunction(1.0)
                for reactant in reactants:
                    all_reactants *= self._concentrations[reactant].components[i]
                all_reactants = all_reactants.Compile()
                for reactant in reactants:
                    source_terms[reactant] += -k_f * all_reactants * test * ngs.dx
                for product in products:
                    source_terms[product] += k_f * all_reactants * test * ngs.dx

                all_products = ngs.CoefficientFunction(1.0)
                for product in products:
                    all_products *= self._concentrations[product].components[i]
                all_products = all_products.Compile()
                for reactant in reactants:
                    source_terms[reactant] += k_r * all_products * test * ngs.dx
                for product in products:
                    source_terms[product] += -k_r * all_products * test * ngs.dx

        # Handle transport terms
        for membrane in self.simulation_geometry.membranes.values():
            for (species, source, target), transport in membrane.get_transport().items():
                concentration = self._concentrations[species]

                def get_index_and_concentration(compartment):
                    if compartment is None:
                        return None, None
                    idx = compartment_to_index[compartment]
                    return idx, concentration.components[idx]

                src_idx, src_concentration = get_index_and_concentration(source)
                trg_idx, trg_concentration = get_index_and_concentration(target)

                # Calculate the flux density through the membrane
                area = to_simulation_units(membrane.area, 'area')
                flux_density = transport.flux(src_concentration, trg_concentration) / area

                ds = ngs.ds(membrane.name)
                if src_idx is not None:
                    source_terms[species] += -flux_density * test_functions[src_idx] * ds
                if trg_idx is not None:
                    source_terms[species] += flux_density * test_functions[trg_idx] * ds

        return source_terms


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


def find_latest_results(name: str, results_root: str) -> str:
    """Find the latest results folder with a given name in a directory."

    :param name: The name of the simulation of interest.
    :param results_root: The directory in which to search for results folders.
    :returns: The full path of the latest results folder for the given simulation.
    """
    result_folders = [
        d for d in os.listdir(results_root)
        if d.startswith(name) and os.path.isdir(os.path.join(results_root, d))
    ]
    if not result_folders:
        raise RuntimeError(f"No folders with name {name} found in {results_root}.")

    latest = max(result_folders, key=lambda d: os.path.getctime(os.path.join(results_root, d)))
    return os.path.join(results_root, latest)
