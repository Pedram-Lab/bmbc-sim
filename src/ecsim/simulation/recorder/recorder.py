import os

import ngsolve as ngs
import astropy.units as u

from ecsim.logging import logger
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.units import to_simulation_units


class Recorder:
    """Base class for recorders that record certain quantities during simulation.
    """
    def __init__(self, recording_interval: u.Quantity):
        """Initialize the recorder with a specified recording interval.
        """
        self.recording_interval = recording_interval
        self._last_recorded_time = float("-inf") * u.s
        self._vtk_output = None


    def setup(
            self,
            directory: str,
            mesh: ngs.Mesh,
            compartments: list[Compartment],
            concentrations: dict[str, ngs.GridFunction],
            potential: ngs.GridFunction | None,
            start_time: u.Quantity
    ) -> None:
        """Set up the recorder with the necessary parameters.
        
        :param directory: Directory where the recorded data will be saved.
        :param mesh: NGSolve mesh object that represents the geometry.
        :param compartments: List of compartments in the simulation geometry.
        :param concentrations: Dictionary mapping species names to their
            respective NGSolve GridFunctions representing concentrations.
        :param potential: NGSolve GridFunction representing the electric
            potential (if applicable).
        :param start_time: The initial time of the simulation.
        """
        # Record the initial state
        # GridFunctions in multi-component spaces cannot automatically be converted
        # to values on the mesh, so we need to set up MaterialCFs manually by a mapping
        #   mesh material -> concentration (i.e., the component of the compartment that
        #                                   containst the material)
        coeff = {}
        for species, concentration in concentrations.items():
            coeff[species] = mesh.MaterialCF({
                region: concentration.components[i]
                for i, compartment in enumerate(compartments)
                for region in compartment.get_region_names(full_names=True)
            })

        if potential is not None:
            coeff['potential'] = mesh.MaterialCF({
                region: potential.components[i]
                for i, compartment in enumerate(compartments)
                for region in compartment.get_region_names(full_names=True)
            })

        # Create a VTK writer for a subfolder in the specified directory
        file_template = os.path.join(directory, "snapshot")
        os.makedirs(os.path.dirname(file_template), exist_ok=True)
        logger.info("Writing VTK output to %s*.vtu", file_template)

        self._vtk_output = ngs.VTKOutput(
            mesh,
            filename=file_template,
            coefs=list(coeff.values()),
            names=list(coeff.keys()),
            floatsize='single'
        )

        self.record(start_time)


    def record(self, current_time: u.Quantity):
        """Record the specified quantities during simulation if the current time
        is at the appropriate interval.
        
        :param current_time: The current time of the simulation.
        """
        # Check if last recording is sufficiently long ago
        time_since_last_record = current_time - self._last_recorded_time
        if (time_since_last_record / self.recording_interval) >= 1 - 1e-6:
            logger.debug("Recording VTK output at time %s", current_time)
            self._vtk_output.Do(to_simulation_units(current_time, 'time'))
            self._last_recorded_time = current_time.copy()


    def finalize(self, end_time: u.Quantity):
        """Finalize the recorder, e.g., close files or clean up resources.

        :param end_time: The end time of the simulation.
        """
        time_since_last_record = end_time - self._last_recorded_time
        if time_since_last_record > self.recording_interval / 2:
            # If we haven't recorded near the end time, do one last recording
            self.record(end_time)
