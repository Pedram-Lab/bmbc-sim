import os
import astropy.units as u
import ngsolve as ngs
import numpy as np
import xarray as xr

from ecsim.logging import logger
from ecsim.simulation.recorder import Recorder
from ecsim.simulation.geometry.compartment import Compartment
from ecsim.units import to_simulation_units


class PointValues(Recorder):
    """Record the total substance of all species in the simulation per
    compartment. The output is saved as xarray datasets. Multiple
    :class:`PointValues` can be used in a single simulation.
    """
    def __init__(self, recording_interval: u.Quantity, points: np.ndarray | list[list[float]]):
        """Initialize the recorder with a specified recording interval and
        points.The points are given as (N, 3) array of coordinates in the mesh.

        :param recording_interval: The interval at which to record data.
        :param points: An (N, 3) array of coordinates (in micrometer) in the
            mesh where the values will be recorded.
        """
        super().__init__(recording_interval)
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=float)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an (N, 3) array of coordinates.")
        self._points = points

        self._xarray = None
        self._data_list = []
        self._time_list = []
        self._species_names = None
        self._concentration_cf = {}
        self._mips = None
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
        del compartments  # Unused
        self._directory = directory

        # Store the species labels for later use
        self._species_names = list(concentrations.keys())

        # Compute mapped integration points for later evaluation
        self._mips = mesh(self._points[:, 0], self._points[:, 1], self._points[:, 2])

        # Sum all components (i.e., the GridFunctions in the compartments) to have
        # a single CoefficientFunction to point-evaluate for each species
        for spec in self._species_names:
            self._concentration_cf[spec] = sum(concentrations[spec].components)


    def _record(
            self,
            current_time: u.Quantity
    ) -> None:
        # Compute all substances at the same time
        logger.debug("Recording point evaluations at time %s", current_time)
        point_values = np.vstack([
            cf(self._mips) for cf in self._concentration_cf.values()
        ])

        # Store data of the current time step
        shape = (len(self._species_names), len(self._mips))
        self._data_list.append(point_values.reshape(shape))
        self._time_list.append(to_simulation_units(current_time, 'time'))


    def _finalize(self):
        # Create an xarray dataset from the recorded data and save it to disk
        logger.info("storing compartment substance data to %s", self._directory)
        data = np.stack(self._data_list, axis=0)
        ds = xr.Dataset(
            {"concentration": (("time", "species", "point"), data)},
            coords={
                "time": self._time_list,
                "species": self._species_names,
                "point": np.arange(len(self._mips))
            },
            attrs={
                "time_unit": "ms",
                "concentration_unit": "mmol/L",
                "length_unit": "um",
                "point_coordinates": self._points
            }
        )

        # Write it into a unique zarr file in the specified directory
        suffix = 0
        while True:
            output_path = os.path.join(self._directory, f"point_data_{suffix}.zarr")
            if not os.path.exists(output_path):
                break
            suffix += 1
        ds.to_zarr(output_path, mode="w", zarr_format=2)
