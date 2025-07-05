import logging
import os
from datetime import datetime

import astropy.units as u
import ngsolve as ngs
from tqdm import trange

from ecsim.logging import logger
from ecsim.simulation.result_io import Recorder
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.simulation.geometry.simulation_geometry import SimulationGeometry
from ecsim.units import to_simulation_units
from ecsim.simulation.simulation_agents import ChemicalSpecies
from ecsim.simulation.fem_details import DiffusionSolver, ReactionSolver, PnpSolver


class Simulation:
    """Build and execute a simulation of a reaction-diffusion system.
    """
    def __init__(
            self,
            name: str,
            mesh: ngs.Mesh,
            *,
            result_root: str,
            electrostatics: bool = False,
    ):
        """Initialize a new simulation.

        :param name: The name of the simulation (used for naming the result directory).
        :param mesh: The mesh representing the geometry of the simulation.
        :param result_root: The directory under which simulation results will be
            stored.
        :param electrostatics: Whether to include electrostatics in the simulation. If
            yes, compartments must have a permeability.
        """
        self.simulation_geometry = SimulationGeometry(mesh)

        self.species: list[ChemicalSpecies] = []
        self.electrostatics = electrostatics

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
        self ._el_fes: ngs.FESpace = None
        self._concentrations: dict[ChemicalSpecies, ngs.GridFunction] = {}
        self._pnp: PnpSolver = None
        self._diffusion: dict[ChemicalSpecies, DiffusionSolver] = {}
        self._reaction: dict[ChemicalSpecies, ReactionSolver] = {}


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


    def run(
            self,
            *,
            end_time: u.Quantity,
            time_step: u.Quantity,
            start_time: u.Quantity = 0 * u.s,
            record_interval: u.Quantity | None = None,
            n_threads: int = 4,
            chemical_substeps: int = 1,
    ) -> None:
        """Run the simulation until a given end time.

        :param end_time: The end time of the simulation.
        :param time_step: The time step to use for the simulation.
        :param start_time: The start time of the simulation.
        :param record_interval: The interval at which to record data. If None, a record
            is taken every 10 time steps.
        :param n_threads: The number of threads to use for the simulation.
        :param chemical_substeps: The number of substeps to use for the chemical
            reactions within each time step.
        :raises ValueError: If the end time is not greater than the start time.
        """
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time.")
        if time_step <= 0 * u.s:
            raise ValueError(f"Time step must be positive, not {time_step}.")
        record_interval = record_interval if record_interval is not None else 10 * time_step
        if record_interval < 0 * u.s:
            raise ValueError(f"Record interval must be non-negative, not {record_interval}.")

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
            recorder = Recorder(record_interval)
            recorder.setup(
                directory=self.result_directory,
                mesh=self.simulation_geometry.mesh,
                compartments=self.simulation_geometry.compartments.values(),
                concentrations=name_to_concentration,
                potential=self._pnp.potential if self.electrostatics else None,
                start_time=start_time.copy()
            )

            t = start_time.copy()
            residual = {}
            for _ in trange(n_steps):
                # Update the concentrations via a first-order splitting approach:
                # 1. Update the electrostatic potential (if applicable)
                if self.electrostatics:
                    self._pnp.update()

                # 2. Independently update the concentrations via reaction kinetics (explicit)
                for _ in range(chemical_substeps):
                    reaction = {s: self._reaction[s].assemble() for s in self._concentrations}
                    tau = dt / chemical_substeps
                    for species, c in self._concentrations.items():
                        m_inv = self._reaction[species].lumped_mass_inv
                        c.vec.FV().NumPy()[:] += tau * (m_inv * reaction[species].vec.FV().NumPy())

                # 3. Diffuse and transport the concentrations (implicit)
                self._update_transport(t)
                for species, c in self._concentrations.items():
                    diffusion = self._diffusion[species].assemble()
                    source = self._diffusion[species]._source_term.vec
                    residual[species] = dt * (source - diffusion.stiffness * c.vec)

                for species, c in self._concentrations.items():
                    diffusion = self._diffusion[species]
                    c.vec.data += diffusion.time_stepping * residual[species]

                t += time_step
                recorder.record(current_time=t)

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
        self._rd_fes = ngs.FESpace([self._compartment_fes[c] for c in compartments])
        logger.info("Total number of degrees of freedom for reaction-diffusion: %d.",
                    self._rd_fes.ndof)

        if self.electrostatics:
            self._el_fes = ngs.FESpace([self._compartment_fes[c] for c in compartments]
                                        + [ngs.FESpace("number", mesh) for _ in compartments])
            logger.info("Total number of degrees of freedom for electrostatics: %d.",
                        self._el_fes.ndof)

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

        if self.electrostatics:
            logger.debug("Setting up electrostatic finite element matrices...")
            self._pnp = PnpSolver.for_all_species(
                self.species,
                self._el_fes,
                self.simulation_geometry,
                self._concentrations,
            )

        logger.debug("Setting up finite element matrices...")
        self._diffusion = DiffusionSolver.for_all_species(
            self.species,
            self._rd_fes,
            self.simulation_geometry,
            self._concentrations,
            self._pnp,
            dt
        )
        logger.debug("Setting up finite element right-hand sides...")
        self._reaction = ReactionSolver.for_all_species(
            self.species,
            self._rd_fes,
            self.simulation_geometry,
            self._concentrations,
        )
