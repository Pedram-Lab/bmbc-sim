"""
Prototype script to convert VTU/PVD time series to XDMF/HDF5 format.

XDMF (eXtensible Data Model and Format) stores:
- The mesh (points, connectivity) once in an HDF5 file
- Time-varying field data appended to the same HDF5 file
- An XML metadata file (.xdmf) describing the structure

This is more efficient than VTU/PVD because:
1. The mesh geometry is not duplicated for each timestep
2. HDF5 is optimized for appending data
3. The format is well-supported by ParaView and other visualization tools
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pyvista as pv


def parse_pvd(pvd_path: Path) -> list[tuple[float, Path]]:
    """Parse a PVD file and return list of (timestep, vtu_path) tuples."""
    tree = ET.parse(pvd_path)
    root = tree.getroot()

    collection = root.find("Collection")
    if collection is None:
        raise ValueError("PVD file does not contain a Collection element")

    result = []
    for dataset in collection.findall("DataSet"):
        timestep = float(dataset.get("timestep", 0))
        filename = dataset.get("file")
        vtu_path = pvd_path.parent / filename
        result.append((timestep, vtu_path))

    return result


def get_cell_type_name(pyvista_cell_type: int) -> str:
    """Convert PyVista/VTK cell type to XDMF topology type."""
    # VTK cell type constants
    VTK_TETRA = 10
    VTK_HEXAHEDRON = 12
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    VTK_LINE = 3

    mapping = {
        VTK_TETRA: "Tetrahedron",
        VTK_HEXAHEDRON: "Hexahedron",
        VTK_TRIANGLE: "Triangle",
        VTK_QUAD: "Quadrilateral",
        VTK_LINE: "Polyline",
    }

    if pyvista_cell_type not in mapping:
        raise ValueError(f"Unsupported cell type: {pyvista_cell_type}")

    return mapping[pyvista_cell_type]


def get_nodes_per_cell(topology_type: str) -> int:
    """Return the number of nodes per cell for a given topology type."""
    mapping = {
        "Tetrahedron": 4,
        "Hexahedron": 8,
        "Triangle": 3,
        "Quadrilateral": 4,
        "Polyline": 2,
    }
    return mapping[topology_type]


def convert_pvd_to_xdmf(pvd_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Convert a PVD time series to XDMF/HDF5 format.

    Parameters
    ----------
    pvd_path : Path
        Path to the .pvd file
    output_dir : Path, optional
        Output directory. Defaults to same directory as pvd_path.

    Returns
    -------
    tuple[Path, Path]
        Paths to the created .xdmf and .h5 files
    """
    pvd_path = Path(pvd_path)
    if output_dir is None:
        output_dir = pvd_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = pvd_path.stem
    h5_path = output_dir / f"{base_name}.h5"
    xdmf_path = output_dir / f"{base_name}.xdmf"

    # Parse PVD to get timesteps and VTU files
    timesteps = parse_pvd(pvd_path)
    print(f"Found {len(timesteps)} timesteps in PVD file")

    # Read first VTU to get mesh structure
    first_time, first_vtu = timesteps[0]
    print(f"Reading mesh from {first_vtu.name}...")
    mesh = pv.read(first_vtu)

    points = np.array(mesh.points, dtype=np.float64)
    n_points = len(points)
    n_cells = mesh.n_cells

    # Get cell type (assume uniform mesh)
    cell_types = mesh.celltypes
    unique_types = np.unique(cell_types)
    if len(unique_types) != 1:
        raise ValueError(f"Mixed cell types not supported: {unique_types}")

    cell_type = int(unique_types[0])
    topology_type = get_cell_type_name(cell_type)
    nodes_per_cell = get_nodes_per_cell(topology_type)

    print(f"Mesh: {n_points} points, {n_cells} {topology_type} cells")

    # Extract connectivity (reshape from flat to (n_cells, nodes_per_cell))
    # PyVista stores cells as [n, p0, p1, ..., n, p0, p1, ...]
    cells = mesh.cells.reshape(-1, nodes_per_cell + 1)[:, 1:]
    connectivity = np.array(cells, dtype=np.int64)

    # Get field names from point data
    field_names = list(mesh.point_data.keys())
    print(f"Fields: {field_names}")

    # Check for compartments.vtu file
    compartments_path = pvd_path.parent / "compartments.vtu"
    compartment_names: list[str] = []
    compartment_data: dict[str, np.ndarray] = {}

    if compartments_path.exists():
        print(f"Reading compartments from {compartments_path.name}...")
        compartments_mesh = pv.read(compartments_path)
        compartment_names = list(compartments_mesh.point_data.keys())
        print(f"Compartments: {len(compartment_names)} masks found")

        for name in compartment_names:
            compartment_data[name] = np.array(
                compartments_mesh.point_data[name], dtype=np.float64
            )

    # Create HDF5 file and write mesh (once)
    print(f"Writing HDF5 file: {h5_path}")
    with h5py.File(h5_path, "w") as h5:
        # Store mesh geometry
        mesh_grp = h5.create_group("mesh")
        mesh_grp.create_dataset("points", data=points, compression="gzip")
        mesh_grp.create_dataset("connectivity", data=connectivity, compression="gzip")

        # Store compartment masks (static data)
        if compartment_names:
            comp_grp = h5.create_group("compartments")
            for name, data in compartment_data.items():
                comp_grp.create_dataset(name, data=data, compression="gzip")

        # Create groups for time-varying data
        data_grp = h5.create_group("data")
        time_dset = data_grp.create_dataset(
            "time",
            shape=(len(timesteps),),
            dtype=np.float64
        )

        # Create datasets for each field (all timesteps)
        field_datasets = {}
        for field_name in field_names:
            field_datasets[field_name] = data_grp.create_dataset(
                field_name,
                shape=(len(timesteps), n_points),
                dtype=np.float64,
                compression="gzip",
                chunks=(1, n_points),  # Chunk by timestep for efficient appending
            )

        # Process each timestep
        for i, (time, vtu_path) in enumerate(timesteps):
            print(f"  Processing timestep {i+1}/{len(timesteps)}: t={time}")
            time_dset[i] = time

            # Read VTU (reuse first mesh if same file)
            if i == 0:
                step_mesh = mesh
            else:
                step_mesh = pv.read(vtu_path)

            # Extract and store field data
            for field_name in field_names:
                field_data = np.array(step_mesh.point_data[field_name], dtype=np.float64)
                field_datasets[field_name][i, :] = field_data

    # Create XDMF file
    print(f"Writing XDMF file: {xdmf_path}")
    write_xdmf(
        xdmf_path=xdmf_path,
        h5_filename=h5_path.name,
        n_points=n_points,
        n_cells=n_cells,
        topology_type=topology_type,
        nodes_per_cell=nodes_per_cell,
        timesteps=[t for t, _ in timesteps],
        field_names=field_names,
        compartment_names=compartment_names,
    )

    print("Conversion complete!")
    return xdmf_path, h5_path


def write_xdmf(
    xdmf_path: Path,
    h5_filename: str,
    n_points: int,
    n_cells: int,
    topology_type: str,
    nodes_per_cell: int,
    timesteps: list[float],
    field_names: list[str],
    compartment_names: list[str] | None = None,
) -> None:
    """Write the XDMF metadata file."""
    if compartment_names is None:
        compartment_names = []

    # Create XDMF structure
    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")

    # Create temporal collection
    temporal = ET.SubElement(
        domain, "Grid",
        Name="TimeSeries",
        GridType="Collection",
        CollectionType="Temporal",
    )

    # Add each timestep as a grid
    for i, time in enumerate(timesteps):
        grid = ET.SubElement(
            temporal, "Grid",
            Name=f"mesh_{i:05d}",
            GridType="Uniform",
        )

        # Time value
        ET.SubElement(grid, "Time", Value=str(time))

        # Topology (connectivity) - reference same data for all timesteps
        topology = ET.SubElement(
            grid, "Topology",
            TopologyType=topology_type,
            NumberOfElements=str(n_cells),
        )
        topo_data = ET.SubElement(
            topology, "DataItem",
            Dimensions=f"{n_cells} {nodes_per_cell}",
            NumberType="Int",
            Precision="8",
            Format="HDF",
        )
        topo_data.text = f"{h5_filename}:/mesh/connectivity"

        # Geometry (points) - reference same data for all timesteps
        geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        geo_data = ET.SubElement(
            geometry, "DataItem",
            Dimensions=f"{n_points} 3",
            NumberType="Float",
            Precision="8",
            Format="HDF",
        )
        geo_data.text = f"{h5_filename}:/mesh/points"

        # Compartment masks (static data) - same for all timesteps
        for comp_name in compartment_names:
            attribute = ET.SubElement(
                grid, "Attribute",
                Name=comp_name,
                AttributeType="Scalar",
                Center="Node",
            )
            attr_data = ET.SubElement(
                attribute, "DataItem",
                Dimensions=f"{n_points}",
                NumberType="Float",
                Precision="8",
                Format="HDF",
            )
            attr_data.text = f"{h5_filename}:/compartments/{comp_name}"

        # Attributes (field data) - different for each timestep
        for field_name in field_names:
            attribute = ET.SubElement(
                grid, "Attribute",
                Name=field_name,
                AttributeType="Scalar",
                Center="Node",
            )
            attr_data = ET.SubElement(
                attribute, "DataItem",
                ItemType="HyperSlab",
                Dimensions=f"{n_points}",
                Type="HyperSlab",
            )
            # HyperSlab: start, stride, count
            slab_params = ET.SubElement(
                attr_data, "DataItem",
                Dimensions="3 2",
                Format="XML",
            )
            slab_params.text = f"{i} 0  1 1  1 {n_points}"

            slab_data = ET.SubElement(
                attr_data, "DataItem",
                Dimensions=f"{len(timesteps)} {n_points}",
                NumberType="Float",
                Precision="8",
                Format="HDF",
            )
            slab_data.text = f"{h5_filename}:/data/{field_name}"

    # Write with proper formatting
    tree = ET.ElementTree(xdmf)
    ET.indent(tree, space="  ")

    with open(xdmf_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)


def main():
    """Convert the tissue_kinetics results to XDMF format."""
    import sys

    # Find the results directory
    results_dir = Path(__file__).parent.parent.parent / "results"

    # Allow specifying a specific directory via command line
    if len(sys.argv) > 1:
        pvd_path = Path(sys.argv[1])
        if pvd_path.is_dir():
            pvd_path = pvd_path / "snapshot.pvd"
    else:
        pvd_files = list(results_dir.glob("tissue_kinetics_*/snapshot.pvd"))
        if not pvd_files:
            print("No tissue_kinetics PVD files found in results directory")
            return
        # Use the most recent one
        pvd_path = sorted(pvd_files)[-1]

    print(f"Converting: {pvd_path}")

    xdmf_path, h5_path = convert_pvd_to_xdmf(pvd_path)

    # Report file sizes
    original_size = sum(f.stat().st_size for f in pvd_path.parent.glob("*.vtu"))
    original_size += pvd_path.stat().st_size

    new_size = h5_path.stat().st_size + xdmf_path.stat().st_size

    print(f"\nFile size comparison:")
    print(f"  Original (VTU/PVD): {original_size / 1e6:.1f} MB")
    print(f"  New (XDMF/HDF5):    {new_size / 1e6:.1f} MB")
    print(f"  Reduction:          {(1 - new_size/original_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
