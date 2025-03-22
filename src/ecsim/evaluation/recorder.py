import abc

import ngsolve as ngs
import astropy.units as u

from ecsim.simulation.geometry.compartment import Compartment


class Recorder(abc.ABC):
    """Base class for recorders that record certain quantities during simulation.
    """
    def __init__(self, recording_interval: u.Quantity):
        """Initialize the recorder with a specified recording interval.
        """
        self.recording_interval = recording_interval
        self._last_recorded_time = float("-inf") * u.s


    def setup(
            self,
            directory: str,
            mesh: ngs.Mesh,
            n_steps: int,
            compartments: list[Compartment],
            concentrations: dict[str, ngs.GridFunction],
            start_time: u.Quantity
    ) -> None:
        """Set up the recorder with the necessary parameters.
        
        :param directory: Directory where the recorded data will be saved.
        :param mesh: NGSolve mesh object that represents the geometry.
        :param n_steps: Number of simulation steps.
        :param compartments: List of compartments in the simulation geometry.
        :param concentrations: Dictionary mapping species names to their
            respective NGSolve GridFunctions representing concentrations.
        """
        # Record the initial state
        self._setup(directory, mesh, n_steps, compartments, concentrations)
        self.record(start_time)


    @abc.abstractmethod
    def _setup(
            self,
            directory: str,
            mesh: ngs.Mesh,
            n_steps: int,
            compartments: list[Compartment],
            concentrations: dict[str, ngs.GridFunction]
    ) -> None:
        """Internal method to set up the recorder with the necessary parameters.

        :param directory: Directory where the recorded data will be saved.
        :param mesh: NGSolve mesh object that represents the geometry.
        :param n_steps: Number of simulation steps.
        :param compartments: List of compartments in the simulation geometry.
        :param concentrations: Dictionary mapping species names to their
            respective NGSolve GridFunctions representing concentrations.
        """


    def record(self, current_time: u.Quantity):
        """Record the specified quantities during simulation if the current time
        is at the appropriate interval.
        
        :param current_time: The current time of the simulation.
        """
        # Check if last recording is sufficiently long ago
        time_since_last_record = current_time - self._last_recorded_time
        if time_since_last_record >= self.recording_interval:
            self._record(current_time)
            self._last_recorded_time = current_time.copy()


    @abc.abstractmethod
    def _record(self, current_time: u.Quantity):
        """Internal method to perform the actual recording.
        
        :param current_time: The current time of the simulation.
        """


    def finalize(self, end_time: u.Quantity):
        """Finalize the recorder, e.g., close files or clean up resources.

        :param end_time: The end time of the simulation.
        """
        time_since_last_record = end_time - self._last_recorded_time
        if time_since_last_record > self.recording_interval / 2:
            # If we haven't recorded near the end time, do one last recording
            self._record(end_time)
        self._finalize()


    @abc.abstractmethod
    def _finalize(self) -> None:
        """Internal method to finalize the recorder, e.g., close files or clean up resources.
        """
