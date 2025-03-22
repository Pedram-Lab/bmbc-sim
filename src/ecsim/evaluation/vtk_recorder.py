import astropy.units as u
import ngsolve as ngs

from ecsim.evaluation.recorder import Recorder
from ecsim.units import to_simulation_units


class Snapshot(Recorder):
    """Record the full mesh with the specified quantities during simulation.
    This 
    """
    def __init__(self, recording_interval: u.Quantity):
        """Initialize the FullMesh recorder.
        """
        super().__init__(recording_interval)
        self._vtk_output = None


    def setup(
            self,
            mesh: ngs.Mesh,
            directory: str,
            concentrations: dict[str, ngs.GridFunction],
            start_time: u.Quantity,
    ) -> None:
        """Set up the recorder with the necessary parameters.
        
        :param mesh: NGSolve mesh object that represents the geometry.
        :param directory: Directory where the recorded data will be saved.
        :param concentrations: Dictionary mapping species names to their
            respective NGSolve GridFunctions representing concentrations.
        """
        self._vtk_output = ngs.VTKOutput(
            mesh,
            filename=directory + "/snapshot",
            coefs=list(concentrations.values()),
            names=list(concentrations.keys()),
            floatsize='single'
        )
        super().setup(start_time)


    def _record(
            self,
            current_time: u.Quantity
    ) -> None:
        """Record the specified quantities during simulation.
        
        :param args: Positional arguments for recording.
        :param kwargs: Keyword arguments for recording.
        """
        self._vtk_output.Do(to_simulation_units(current_time, 'time'))
