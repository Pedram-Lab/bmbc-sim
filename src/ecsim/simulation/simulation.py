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
from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.simulation.fem_details import FemLhs, FemRhs


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
        self._concentrations: dict[ChemicalSpecies, ngs.GridFunction] = {}
        self._lhs: dict[ChemicalSpecies, FemLhs] = {}
        self._rhs: dict[ChemicalSpecies, FemRhs] = {}

        self._recorders: list[Recorder] = []


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
            n_threads: int = 4,
    ) -> None:
        """Run the simulation until a given end time.

        :param end_time: The end time of the simulation.
        :param time_step: The time step to use for the simulation.
        :param start_time: The start time of the simulation.
        :param n_threads: The number of threads to use for the simulation.
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

        dt = to_simulation_units(time_step, 'time')

        # Main simulation loop (with parallelization)
        ngs.SetNumThreads(n_threads)
        with ngs.TaskManager():
            self._setup(dt)

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
            residual = {}
            for _ in trange(n_steps):
                # Update the concentrations via IMEX approach:
                # Reaction + some transport (explicit)
                self._update_transport(t)
                for species, c in self._concentrations.items():
                    lhs = self._lhs[species].assemble()
                    rhs = self._rhs[species].assemble()
                    residual[species] = dt * (rhs.vec - lhs.stiffness * c.vec)

                # Diffusion + some transport (implicit)
                for species, c in self._concentrations.items():
                    lhs = self._lhs[species]
                    c.vec.data += lhs.time_stepping * residual[species]

                t += time_step
                for recorder in self._recorders:
                    recorder.record(current_time=t)

            for recorder in self._recorders:
                recorder.finalize(end_time=t)


    def _update_transport(self, t: u.Quantity) -> None:
        """Update the transport mechanisms based on the current time.

        :param t: The current time in the simulation.
        """
        # Update all transport mechanisms
        for membrane in self.simulation_geometry.membranes.values():
            for _, _, _, transport in membrane.get_transport():
                transport.update_flux(t)


    def _setup(self, dt) -> None:
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

        # Set up the solution vectors
        for species in self.species:
            logger.debug("Initializing concentrations for species %s.", species)
            self._concentrations[species] = ngs.GridFunction(self._rd_fes)

            # Initialize the concentrations in the compartment
            for i, compartment in enumerate(compartments):
                coefficients = compartment.coefficients
                if species in coefficients.initial_conditions:
                    c = self._concentrations[species].components[i]
                    c.Set(coefficients.initial_conditions[species])

        logger.debug("Setting up finite element matrices...")
        self._lhs = FemLhs.for_all_species(
            self.species,
            self._rd_fes,
            self.simulation_geometry,
            self._concentrations,
            dt
        )
        logger.debug("Setting up finite element right-hand sides...")
        self._rhs = FemRhs.for_all_species(
            self.species,
            self._rd_fes,
            self.simulation_geometry,
            self._concentrations,
        )


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
