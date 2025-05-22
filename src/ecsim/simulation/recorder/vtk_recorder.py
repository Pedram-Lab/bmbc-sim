import os
import astropy.units as u
import ngsolve as ngs

from ecsim.logging import logger
from ecsim.simulation.recorder import Recorder
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.units import to_simulation_units


class FullSnapshot(Recorder):
    """Record the full mesh with all concentrations during simulation.
    This saves a snapshot of the entire mesh at specified intervals in
    VTK format, allowing for visualization and analysis in pyvista or
    paraview. Only a single :class:`FullSnapshot` recorder instance
    is allowed per simulation
    """
    def __init__(self, recording_interval: u.Quantity):
        super().__init__(recording_interval)
        self._vtk_output = None


    def _setup(
            self,
            directory: str,
            mesh: ngs.Mesh,
            compartments: list[Compartment],
            concentrations: dict[str, ngs.GridFunction],
            potential: ngs.GridFunction | None
    ) -> None:
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
        file_template = os.path.join(directory, "snapshots", "snapshot")
        os.makedirs(os.path.dirname(file_template), exist_ok=True)
        logger.info("Writing VTK output to %s*.vtu", file_template)

        self._vtk_output = ngs.VTKOutput(
            mesh,
            filename=file_template,
            coefs=list(coeff.values()),
            names=list(coeff.keys()),
            floatsize='single'
        )


    def _record(
            self,
            current_time: u.Quantity
    ) -> None:
        logger.debug("Recording VTK output at time %s", current_time)
        self._vtk_output.Do(to_simulation_units(current_time, 'time'))


    def _finalize(self):
        pass
