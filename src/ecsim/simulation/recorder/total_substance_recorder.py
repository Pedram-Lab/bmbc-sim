import os

import astropy.units as u
import ngsolve as ngs
import numpy as np
import xarray as xr

from ecsim.logging import logger
from ecsim.simulation.recorder import Recorder
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.units import to_simulation_units


class CompartmentSubstance(Recorder):
    """Record the total substance of all species in the simulation per
    compartment. The output is saved as xarray datasets. Only one
    :class:`CompartmentSubstance` recorder is allowed per simulation.
    """
    def __init__(self, recording_interval: u.Quantity):
        super().__init__(recording_interval)
        self._xarray = None
        self._data_list = []
        self._time_list = []
        self._volumes = None
        self._mesh = None
        self._coords = None
        self._concentration_cf = None
        self._directory = None


    def _setup(
            self,
            directory: str,
            mesh: ngs.Mesh,
            compartments: list[Compartment],
            concentrations: dict[str, ngs.GridFunction],
            potential: ngs.GridFunction | None
    ) -> None:
        #TODO: Add potential to the output
        self._mesh = mesh
        self._directory = directory
        self._volumes = [to_simulation_units(comp.volume, 'volume') for comp in compartments]

        # Store the labels for later use
        species = list(concentrations.keys())
        comp_names = [comp.name for comp in compartments]
        self._coords = {
            "compartment": comp_names,
            "species": species
        }

        # Create a vector-valued CoefficientFunction to hold the concentrations
        # for all compartments and species
        cs = []
        for i, _ in enumerate(compartments):
            for spec in species:
                cs.append(concentrations[spec].components[i])
        self._concentration_cf = ngs.CoefficientFunction(tuple(cs))


    def _record(
            self,
            current_time: u.Quantity
    ) -> None:
        # Compute all substances at the same time
        logger.debug("Recording compartment substance at time %s", current_time)
        substance = ngs.Integrate(self._concentration_cf, self._mesh, ngs.VOL, order=1)

        # Store data of the current time step
        shape = (len(self._coords["compartment"]), len(self._coords["species"]))
        self._data_list.append(np.array(substance).reshape(shape))
        self._time_list.append(to_simulation_units(current_time, 'time'))


    def _finalize(self):
        # Create an xarray dataset from the recorded data and save it to disk
        logger.info("storing compartment substance data to %s", self._directory)
        data = np.stack(self._data_list, axis=0)
        ds = xr.Dataset(
            {"substance": (("time", "compartment", "species"), data)},
            coords={
                "time": self._time_list,
                "compartment": self._coords["compartment"],
                "species": self._coords["species"],
            },
            attrs = {
                "time_unit": "ms",
                "substance_unit": "amol",
                "volume_unit": "um^3",
                "compartment_volume": self._volumes,
            }
        )
        output_path = os.path.join(self._directory, "substance_data.zarr")
        ds.to_zarr(output_path, mode="w", zarr_format=2)
