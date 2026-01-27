import os
import tempfile
import xml.etree.ElementTree as ET
import glob
import shutil

import numpy as np
import ngsolve as ngs
import h5py
import pyvista as pv
import astropy.units as u

from bmbcsim.logging import logger
from bmbcsim.simulation.geometry.compartment import Compartment
from bmbcsim.units import to_simulation_units


class XdmfRecorder:
    """Records simulation quantities to XDMF/HDF5 format.

    Uses NGSolve's VTKOutput internally to evaluate CoefficientFunctions at
    mesh vertices (handling boundary vertex duplication correctly), then stores
    the results in HDF5/XDMF format for efficient storage and ParaView access.
    """

    def __init__(self, recording_interval: u.Quantity):
        self.recording_interval = recording_interval
        self._last_recorded_time = float("-inf") * u.s
        self._h5_path: str = ""
        self._xdmf_path: str = ""
        self._n_points: int = 0
        self._n_cells: int = 0
        self._field_names: list[str] = []
        self._compartment_names: list[str] = []
        self._step_count: int = 0
        self._times: list[float] = []
        self._vtk_output: ngs.VTKOutput | None = None
        self._vtk_tmp_dir: str = ""

    def setup(
        self,
        directory: str,
        mesh: ngs.Mesh,
        compartments: list[Compartment],
        concentrations: dict[str, ngs.GridFunction],
        potential: ngs.GridFunction | None,
        start_time: u.Quantity,
    ) -> None:
        absolute_path = os.path.abspath(directory)
        os.makedirs(absolute_path, exist_ok=True)

        self._h5_path = os.path.join(absolute_path, "snapshot.h5")
        self._xdmf_path = os.path.join(absolute_path, "snapshot.xdmf")

        # Build compartment indicator CFs
        indicator_cfs = {}
        for compartment in compartments:
            for region in compartment.get_region_names(full_names=True):
                indicator_cfs[region] = mesh.MaterialCF({region: 1.0})

        # Write compartment indicators via VTKOutput (produces correct
        # boundary vertex handling via vertex duplication)
        indicator_path = os.path.join(absolute_path, "compartments")
        ngs.VTKOutput(
            mesh,
            filename=indicator_path,
            coefs=list(indicator_cfs.values()),
            names=list(indicator_cfs.keys()),
            floatsize='single'
        ).Do()

        # Read the VTK grid to get the duplicated-vertex mesh
        vtk_grid = pv.read(indicator_path + ".vtu")
        points = np.array(vtk_grid.points, dtype=np.float32)
        connectivity = vtk_grid.cell_connectivity.reshape(-1, 4).astype(np.int32)

        self._n_points = vtk_grid.n_points
        self._n_cells = vtk_grid.n_cells
        self._compartment_names = list(indicator_cfs.keys())

        logger.info(
            "Writing XDMF/HDF5 output to %s (%d points, %d cells)",
            self._h5_path, self._n_points, self._n_cells,
        )

        # Build species (and optional potential) CFs
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

        self._field_names = list(coeff.keys())

        # Set up VTKOutput for field evaluation (writes to temp directory)
        self._vtk_tmp_dir = tempfile.mkdtemp(prefix="xdmf_recorder_")
        vtk_path = os.path.join(self._vtk_tmp_dir, "snap")
        self._vtk_output = ngs.VTKOutput(
            mesh,
            filename=vtk_path,
            coefs=list(coeff.values()),
            names=list(coeff.keys()),
            floatsize='single'
        )

        # Chunking parameters
        chunk_1d = (self._n_points,)
        chunk_points = (min(self._n_points, 87381), 3)
        chunk_conn = (min(self._n_cells, 65536), 4)

        # Create HDF5 and write static data
        with h5py.File(self._h5_path, "w") as h5:
            mesh_grp = h5.create_group("mesh")
            mesh_grp.create_dataset(
                "points", data=points,
                compression="gzip", compression_opts=1, chunks=chunk_points,
            )
            mesh_grp.create_dataset(
                "connectivity", data=connectivity,
                compression="gzip", compression_opts=1, chunks=chunk_conn,
            )

            # Compartment indicators from VTK grid
            comp_grp = h5.create_group("compartments")
            for name in self._compartment_names:
                data = np.array(vtk_grid.point_data[name], dtype=np.float32)
                comp_grp.create_dataset(
                    name, data=data,
                    compression="gzip", compression_opts=1, chunks=chunk_1d,
                )

            # Prepare data group with subgroups per field
            data_grp = h5.create_group("data")
            for field_name in self._field_names:
                data_grp.create_group(field_name)

        # Clean up the compartments VTU file (data is now in HDF5)
        os.remove(indicator_path + ".vtu")

        # Record initial state
        self.record(start_time)

    def _read_vtk_fields(self) -> dict[str, np.ndarray]:
        """Read field values from the latest VTK snapshot file."""
        # Find the VTU file that VTKOutput just wrote
        pattern = os.path.join(self._vtk_tmp_dir, "snap*.vtu")
        vtu_files = sorted(glob.glob(pattern))
        if not vtu_files:
            raise RuntimeError(f"No VTU files found in {self._vtk_tmp_dir}")
        vtu_path = vtu_files[-1]
        grid = pv.read(vtu_path)
        result = {}
        for name in self._field_names:
            result[name] = np.array(grid.point_data[name], dtype=np.float32)
        # Clean up the VTU file
        os.remove(vtu_path)
        return result

    def record(self, current_time: u.Quantity) -> None:
        time_since_last_record = current_time - self._last_recorded_time
        if (time_since_last_record / self.recording_interval) >= 1 - 1e-6:
            logger.debug("Recording XDMF output at time %s", current_time)

            sim_time = to_simulation_units(current_time, "time")

            # Use VTKOutput to evaluate CFs with correct boundary handling
            self._vtk_output.Do(float(sim_time))
            fields = self._read_vtk_fields()

            chunk_1d = (self._n_points,)
            with h5py.File(self._h5_path, "a") as h5:
                data_grp = h5["data"]
                for field_name in self._field_names:
                    data_grp[field_name].create_dataset(
                        f"step_{self._step_count:05d}",
                        data=fields[field_name],
                        compression="gzip", compression_opts=1,
                        chunks=chunk_1d,
                    )

            self._times.append(float(sim_time))
            self._step_count += 1
            self._write_xdmf()
            self._last_recorded_time = current_time.copy()

    def finalize(self, end_time: u.Quantity) -> None:
        time_since_last_record = end_time - self._last_recorded_time
        if time_since_last_record > self.recording_interval / 2:
            self.record(end_time)

        # Write final time dataset
        with h5py.File(self._h5_path, "a") as h5:
            if "data/time" in h5:
                del h5["data/time"]
            h5["data"].create_dataset(
                "time", data=np.array(self._times, dtype=np.float32)
            )

        # Clean up temp directory
        if os.path.exists(self._vtk_tmp_dir):
            shutil.rmtree(self._vtk_tmp_dir, ignore_errors=True)

    def _write_xdmf(self) -> None:
        """Generate the XDMF XML metadata file."""
        h5_filename = os.path.basename(self._h5_path)

        xdmf = ET.Element("Xdmf", Version="3.0")
        domain = ET.SubElement(xdmf, "Domain")
        temporal = ET.SubElement(
            domain, "Grid",
            Name="TimeSeries",
            GridType="Collection",
            CollectionType="Temporal",
        )

        for i, time in enumerate(self._times):
            grid = ET.SubElement(
                temporal, "Grid",
                Name=f"mesh_{i:05d}",
                GridType="Uniform",
            )
            ET.SubElement(grid, "Time", Value=str(time))

            # Topology
            topology = ET.SubElement(
                grid, "Topology",
                TopologyType="Tetrahedron",
                NumberOfElements=str(self._n_cells),
            )
            topo_data = ET.SubElement(
                topology, "DataItem",
                Dimensions=f"{self._n_cells} 4",
                NumberType="Int", Precision="4", Format="HDF",
            )
            topo_data.text = f"{h5_filename}:/mesh/connectivity"

            # Geometry
            geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
            geo_data = ET.SubElement(
                geometry, "DataItem",
                Dimensions=f"{self._n_points} 3",
                NumberType="Float", Precision="4", Format="HDF",
            )
            geo_data.text = f"{h5_filename}:/mesh/points"

            # Compartment masks
            for comp_name in self._compartment_names:
                attribute = ET.SubElement(
                    grid, "Attribute",
                    Name=comp_name, AttributeType="Scalar", Center="Node",
                )
                attr_data = ET.SubElement(
                    attribute, "DataItem",
                    Dimensions=f"{self._n_points}",
                    NumberType="Float", Precision="4", Format="HDF",
                )
                attr_data.text = f"{h5_filename}:/compartments/{comp_name}"

            # Time-varying fields
            for field_name in self._field_names:
                attribute = ET.SubElement(
                    grid, "Attribute",
                    Name=field_name, AttributeType="Scalar", Center="Node",
                )
                attr_data = ET.SubElement(
                    attribute, "DataItem",
                    Dimensions=f"{self._n_points}",
                    NumberType="Float", Precision="4", Format="HDF",
                )
                attr_data.text = f"{h5_filename}:/data/{field_name}/step_{i:05d}"

        tree = ET.ElementTree(xdmf)
        ET.indent(tree, space="  ")

        tmp_path = self._xdmf_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            tree.write(f, encoding="unicode", xml_declaration=False)
        os.replace(tmp_path, self._xdmf_path)
