import os
from typing import Sequence
import xml.etree.ElementTree as ET

import pyvista as pv
import numpy as np
import xarray as xr
import h5py

from bmbcsim.units import BASE_UNITS


class ResultLoader:
    """A class to load results from a specified directory."""

    def __init__(self, results_path: str):
        """Initialize the ResultLoader with the given directory for results."""
        self.path = results_path
        self.results_root = os.path.abspath(self.path)

        # Auto-detect format
        h5_path = os.path.join(self.results_root, "snapshot.h5")
        if os.path.exists(h5_path):
            self._format = "hdf5"
            self._h5_path = h5_path
            self._load_hdf5_metadata()
        else:
            self._format = "pvd"
            self._load_pvd_metadata()

        # Initialize variables for caching
        self.cell_to_region, self.regions = self._get_cell_to_region()

    def _load_pvd_metadata(self):
        """Load metadata from PVD/VTU format."""
        snapshot_xml = os.path.join(self.results_root, "snapshot.pvd")
        tree = ET.parse(snapshot_xml)
        root = tree.getroot()

        self.snapshots = [
            (float(ds.get("timestep")) * BASE_UNITS["time"], ds.get("file"))
            for ds in root.findall(".//DataSet")
        ]
        self.snapshots.sort(key=lambda x: x[0])

    def _load_hdf5_metadata(self):
        """Load metadata from HDF5 format."""
        with h5py.File(self._h5_path, "r") as h5:
            times = np.array(h5["data/time"])
            self.snapshots = [
                (float(t) * BASE_UNITS["time"], None)
                for t in times
            ]
            self._compartment_names = list(h5["compartments"].keys())
            self._field_names = [
                k for k in h5["data"].keys() if k != "time"
            ]

    @classmethod
    def find(
        cls,
        *,
        simulation_name: str,
        results_root: str,
        time_stamp: str | None = None
    ) -> "ResultLoader":
        """Find the latest results folder with a given name in a directory.

        :param simulation_name: The name of the simulation of interest.
        :param results_root: The directory in which to search for results folders.
        :param time_stamp: Optional timestamp to filter results folders. If not
            provided, the latest folder is returned.
        :returns: The result loader instance for the found folder.
        """
        # Search for folders that contain results for the given simulation name
        result_folders = [
            d
            for d in os.listdir(results_root)
            if d.startswith(simulation_name) and os.path.isdir(os.path.join(results_root, d))
        ]
        if not result_folders:
            raise RuntimeError(
                f"No folders with name {simulation_name} found in {results_root}."
            )

        # Search for a specific timestamp or return the latest folder
        if time_stamp:
            result_folders = [f for f in result_folders if time_stamp in f]
            if not result_folders:
                raise RuntimeError(
                    f"No folders with name {simulation_name} and "
                    f"timestamp {time_stamp} found in {results_root}."
                )
            if len(result_folders) > 1:
                raise RuntimeError(
                    f"Multiple folders with name {simulation_name} and "
                    f"timestamp {time_stamp} found in {results_root}."
                )
            result_folder = result_folders[0]
        else:
            result_folder = max(
                result_folders,
                key=lambda d: os.path.getctime(os.path.join(results_root, d)),
            )

        return cls(os.path.join(results_root, result_folder))

    def __len__(self) -> int:
        """Return the number of snapshots available."""
        return len(self.snapshots)

    # Compute region sizes (volumes)
    def load_regions(self) -> pv.UnstructuredGrid:
        """Load the geometry of the simulation containing all regions."""
        if self._format == "hdf5":
            return self._load_regions_hdf5()
        return pv.read(os.path.join(self.results_root, "compartments.vtu"))

    def _load_regions_hdf5(self) -> pv.UnstructuredGrid:
        """Load regions from HDF5."""
        with h5py.File(self._h5_path, "r") as h5:
            grid = self._build_grid(h5)
            for name in self._compartment_names:
                grid.point_data[name] = np.array(h5[f"compartments/{name}"])
        return grid

    def _build_grid(self, h5) -> pv.UnstructuredGrid:
        """Build a PyVista UnstructuredGrid from HDF5 mesh data."""
        points = np.array(h5["mesh/points"])
        connectivity = np.array(h5["mesh/connectivity"])
        n_cells = len(connectivity)

        # Build VTK-style cell array: [4, v0, v1, v2, v3, ...]
        cells = np.empty((n_cells, 5), dtype=np.int64)
        cells[:, 0] = 4
        cells[:, 1:] = connectivity
        celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)

        return pv.UnstructuredGrid(cells, celltypes, points)

    def compute_region_sizes(self) -> dict[str, float]:
        """Compute the sizes (volumes) of each region."""
        compartments = self.load_regions()

        # Integrate volumes over regions
        volumes = -compartments.compute_cell_sizes(
            length=False, area=False, vertex_count=False
        )["Volume"]
        region_sizes = np.bincount(
            self.cell_to_region, weights=volumes
        )

        return {n: s for n, s in zip(self.regions, region_sizes)}

    def load_snapshot(self, step: int) -> pv.UnstructuredGrid:
        """Load a snapshot file for a given step.

        :param step: The step number of the snapshot to load.
        :returns: The simulation results at the specified step.
        """
        if step < 0:
            step += len(self)
        if step >= len(self):
            raise IndexError(f"Step {step} is out of range for available snapshots.")

        if self._format == "hdf5":
            return self._load_snapshot_hdf5(step)

        path = os.path.join(self.results_root, self.snapshots[step][1])
        return pv.read(path)

    def _load_snapshot_hdf5(self, step: int) -> pv.UnstructuredGrid:
        """Load a snapshot from HDF5."""
        with h5py.File(self._h5_path, "r") as h5:
            grid = self._build_grid(h5)
            for field_name in self._field_names:
                grid.point_data[field_name] = np.array(
                    h5[f"data/{field_name}/step_{step:05d}"]
                )
        return grid

    def load_point_values(
        self, step: int, points: Sequence[float] | Sequence[Sequence[float]]
    ) -> xr.DataArray:
        """Load point values for a given step.

        :param step: The step number of the snapshot to load.
        :param points: One or more points (x, y, z) to sample at.
        :returns: The path to the point values file.
        """
        # Prepare points for consumption by PyVista (wrap negative indexing)
        if step < 0:
            step += len(self)
        data = self.load_snapshot(step)
        if not isinstance(points[0], (list, tuple)):
            points = [points]
        points = np.array(points, dtype=float)

        # Sample concentration fields at given points
        point_cloud = pv.PolyData(points)
        point_cloud = point_cloud.sample(data)

        # Compile point data into an array
        species = _get_species_names(point_cloud)
        values = np.expand_dims(
            np.stack([point_cloud.point_data[s] for s in species], axis=0), axis=0
        )

        # Create data array consisting of values and appropriate metadata
        concentration_unit = (
            BASE_UNITS["amount of substance"] / BASE_UNITS["length"] ** 3
        )
        ds = xr.DataArray(
            values,
            dims=("time", "species", "point"),
            coords={
                "time": [self.snapshots[step][0].value],
                "point": np.arange(values.shape[-1]),
                "species": species,
            },
            attrs={
                "concentration unit": concentration_unit.to_string(),
                "time unit": BASE_UNITS["time"].to_string(),
            },
        )
        return ds

    def load_total_substance(self, step: int) -> xr.DataArray:
        """Load total substance data per compartment for a given step.

        :param step: The step number of the snapshot to load.
        :returns: An xarray Dataset containing total substance data.
        """
        # Prepare data (wrap negative indexing)
        if step < 0:
            step += len(self)
        data = self.load_snapshot(step)
        species = _get_species_names(data)

        # Precompute cell connectivity and volume
        # NGSolve stores the cells in an orientation that makes pyvista compute negative volumes
        cells = data.cell_connectivity.reshape(-1, 4)
        volumes = -data.compute_cell_sizes(
            length=False, area=False, vertex_count=False
        )["Volume"]
        total_substance = np.zeros((1, len(self.regions), len(species)), dtype=float)

        # Integrate element-wise via midpoint rule
        for i, s in enumerate(species):
            values = data.point_data[s]
            cell_substances = np.mean(values[cells], axis=1) * volumes
            total_substance[:, :, i] = np.bincount(
                self.cell_to_region, weights=cell_substances
            )

        # Create data array consisting of values and appropriate metadata
        return xr.DataArray(
            total_substance,
            dims=("time", "region", "species"),
            coords={
                "time": [self.snapshots[step][0].value],
                "region": self.regions,
                "species": species,
            },
            attrs={
                "substance unit": BASE_UNITS["amount of substance"].to_string(),
                "time unit": BASE_UNITS["time"].to_string(),
            },
        )

    def _get_cell_to_region(self) -> tuple[np.ndarray, list[str]]:
        """Return an array that contains the region index for each cell and the
        name of all regions.
        """
        # Load compartments and accumulate region information per cell
        compartments = self.load_regions().point_data_to_cell_data()
        cell_to_region = np.ndarray(compartments.n_cells, dtype=int)
        regions = list(compartments.cell_data.keys())

        # Enumerate regions and assign indices to cells
        for i, region in enumerate(regions):
            mask = compartments.cell_data[region].astype(bool)
            cell_to_region[mask] = i

        return cell_to_region, regions


def _get_species_names(data: pv.UnstructuredGrid) -> list[str]:
    """Extract species names from the point data of a PyVista UnstructuredGrid."""
    return [name for name in data.point_data.keys() if not name.startswith("vtk")]
