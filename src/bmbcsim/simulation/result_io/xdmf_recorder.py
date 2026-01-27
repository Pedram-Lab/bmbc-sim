import os
import xml.etree.ElementTree as ET

import numpy as np
import ngsolve as ngs
import h5py
import astropy.units as u

from bmbcsim.logging import logger
from bmbcsim.simulation.geometry.compartment import Compartment
from bmbcsim.units import to_simulation_units


class XdmfRecorder:
    """Records simulation quantities to XDMF/HDF5 format.

    Extracts DOF values directly from GridFunction components and maps them
    to mesh vertices. Boundary vertices shared between compartments are
    duplicated (one copy per compartment).
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
        self._grid_functions: dict[str, ngs.GridFunction] = {}
        self._n_components: int = 0

    def setup(
        self,
        directory: str,
        mesh: ngs.Mesh,
        compartments: list[Compartment],
        concentrations: dict[str, ngs.GridFunction],
        potential: ngs.GridFunction | None,
        start_time: u.Quantity,
    ) -> None:
        """Prepare XDMF/HDF5 output files and write static mesh data."""
        absolute_path = os.path.abspath(directory)
        os.makedirs(absolute_path, exist_ok=True)

        self._h5_path = os.path.join(absolute_path, "snapshot.h5")
        self._xdmf_path = os.path.join(absolute_path, "snapshot.xdmf")

        self._n_components = len(compartments)

        # Get compressed H1 spaces from any GridFunction's space components
        any_gf = next(iter(concentrations.values()))
        compressed_spaces = [any_gf.space.components[i] for i in range(self._n_components)]

        # Build DOF-to-vertex and vertex-to-DOF maps
        dof_to_vertex_maps = []
        vertex_to_dof_maps = []
        for fes in compressed_spaces:
            dof_to_vertex = np.full(fes.ndof, -1, dtype=np.int32)
            vertex_to_dof = np.full(mesh.nv, -1, dtype=np.int32)
            for v in range(mesh.nv):
                dofs = fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v))
                for d in dofs:
                    if d >= 0:
                        dof_to_vertex[d] = v
                        vertex_to_dof[v] = d
            dof_to_vertex_maps.append(dof_to_vertex)
            vertex_to_dof_maps.append(vertex_to_dof)

        # Build global point array with boundary duplication
        mesh_coords = np.array(mesh.ngmesh.Coordinates(), dtype=np.float32)
        points = np.concatenate([mesh_coords[dtv] for dtv in dof_to_vertex_maps])

        # Compute offsets for each component in the global point array
        offsets = np.zeros(self._n_components + 1, dtype=np.int32)
        for i, fes in enumerate(compressed_spaces):
            offsets[i + 1] = offsets[i] + fes.ndof

        total_dofs = int(offsets[-1])
        self._n_points = total_dofs

        # Build vertex_to_global lookup: vertex_to_global[comp, vertex] → global index
        vertex_to_global = np.full((self._n_components, mesh.nv), -1, dtype=np.int32)
        for i, vtd in enumerate(vertex_to_dof_maps):
            valid = vtd >= 0
            vertex_to_global[i, valid] = offsets[i] + vtd[valid]

        # Build remapped connectivity — map mesh material names to component index
        mat_to_comp = {}
        for i, compartment in enumerate(compartments):
            for region in compartment.get_region_names(full_names=True):
                mat_to_comp[region] = i

        mesh_conn = np.array(
            [[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)],
            dtype=np.int32,
        )
        el_comp = np.array(
            [mat_to_comp[el.mat] for el in mesh.Elements(ngs.VOL)],
            dtype=np.int32,
        )
        connectivity = vertex_to_global[el_comp[:, None], mesh_conn].astype(np.int32)

        self._n_cells = len(connectivity)

        # Build one indicator per compartment (covers all regions of that compartment)
        self._compartment_names = [c.name for c in compartments]
        indicators = {}
        for i, compartment in enumerate(compartments):
            ind = np.zeros(total_dofs, dtype=np.float32)
            ind[offsets[i]:offsets[i + 1]] = 1.0
            indicators[compartment.name] = ind

        # Store GridFunction references for direct DOF extraction in record()
        for species, gf in concentrations.items():
            self._grid_functions[species] = gf
        if potential is not None:
            self._grid_functions['potential'] = potential

        self._field_names = list(self._grid_functions.keys())

        logger.info(
            "Writing XDMF/HDF5 output to %s (%d points, %d cells)",
            self._h5_path, self._n_points, self._n_cells,
        )

        # Chunking parameters about 1 MB per chunk
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

            # Compartment indicators
            comp_grp = h5.create_group("compartments")
            for name in self._compartment_names:
                comp_grp.create_dataset(
                    name, data=indicators[name],
                    compression="gzip", compression_opts=1, chunks=chunk_1d,
                )

            # Prepare data group with subgroups per field
            data_grp = h5.create_group("data")
            for field_name in self._field_names:
                data_grp.create_group(field_name)

        # Record initial state
        self.record(start_time)

    def record(self, current_time: u.Quantity) -> None:
        """Record simulation state at the given time."""
        time_since_last_record = current_time - self._last_recorded_time
        if (time_since_last_record / self.recording_interval) >= 1 - 1e-6:
            logger.debug("Recording XDMF output at time %s", current_time)

            sim_time = to_simulation_units(current_time, "time")

            # Extract field values directly from GridFunction DOFs
            fields = {}
            for field_name, gf in self._grid_functions.items():
                fields[field_name] = np.concatenate(
                    [gf.components[i].vec.FV().NumPy() for i in range(self._n_components)]
                ).astype(np.float32)

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
        """Finalize recording, ensuring final time point is recorded."""
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
