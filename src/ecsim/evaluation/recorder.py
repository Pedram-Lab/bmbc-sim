import abc

import astropy.units as u


class Recorder(abc.ABC):
    """Base class for recorders that record certain quantities during simulation.
    """
    def __init__(self, recording_interval: u.Quantity):
        """Initialize the recorder with a specified recording interval.
        """
        self.recording_interval = recording_interval
        self._last_recorded_time = None

    def setup(self, start_time: u.Quantity, *args, **kwargs):
        """Set up the recorder with the necessary parameters.
        
        :param start_time: The start time of the simulation.
        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments for setup.
        """
        self._record(self, start_time, *args, **kwargs)

    def record(self, current_time: u.Quantity, *args, **kwargs):
        """Record the specified quantities during simulation if the current time
        is at the appropriate interval.
        
        :param current_time: The current time of the simulation.
        :param args: Positional arguments for recording.
        :param kwargs: Keyword arguments for recording.
        """
        time_since_last_record = current_time - self._last_recorded_time
        if time_since_last_record >= self.recording_interval:
            self._record(current_time, *args, **kwargs)
            self._last_recorded_time = current_time


    @abc.abstractmethod
    def _record(self, current_time: u.Quantity, *args, **kwargs):
        """Internal method to perform the actual recording.
        
        :param current_time: The current time of the simulation.
        :param args: Positional arguments for recording.
        :param kwargs: Keyword arguments for recording.
        """

    @abc.abstractmethod
    def finalize(self, end_time: u.Quantity, *args, **kwargs):
        """Finalize the recorder, e.g., close files or clean up resources.

        :param end_time: The end time of the simulation.
        :param args: Positional arguments for finalization.
        :param kwargs: Keyword arguments for finalization.
        """
        self._record(end_time, *args, **kwargs)
