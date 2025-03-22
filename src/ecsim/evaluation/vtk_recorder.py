import astropy.units as u
import ngsolve as ngs

from ecsim.evaluation.recorder import Recorder
from ecsim.simulation.geometry.compartment import Compartment
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


    def _setup(
            self,
            mesh: ngs.Mesh,
            compartments: list[Compartment],
            directory: str,
            concentrations: dict[str, ngs.GridFunction],
    ) -> None:
        """Set up the recorder with the necessary parameters.
        
        :param mesh: NGSolve mesh object that represents the geometry.
        :param directory: Directory where the recorded data will be saved.
        :param concentrations: Dictionary mapping species names to their
            respective NGSolve GridFunctions representing concentrations.
        """
        # GridFunctions in multi-component spaces cannot automatically be converted
        # to values on the mesh, so we need to set up MaterialCFs manually by a mapping
        #   mesh material -> concentration (i.e., the component of the compartment that
        #                                   containst the material)
        coeff = {}
        for species, concentration in concentrations.items():
            species_coeff = {}
            for i, compartment in enumerate(compartments):
                for region in compartment.get_region_names(full_names=True):
                    species_coeff[region] = concentration.components[i]
            coeff[species] = mesh.MaterialCF(species_coeff)

        self._vtk_output = ngs.VTKOutput(
            mesh,
            filename=directory + "/snapshot",
            coefs=list(coeff.values()),
            names=list(coeff.keys()),
            floatsize='single'
        )


    def _record(
            self,
            current_time: u.Quantity
    ) -> None:
        """Record the specified quantities during simulation.
        
        :param args: Positional arguments for recording.
        :param kwargs: Keyword arguments for recording.
        """
        self._vtk_output.Do(to_simulation_units(current_time, 'time'))


    def _finalize(self):
        pass
