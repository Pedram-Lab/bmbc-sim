import os
import xml.etree.ElementTree as ET

import numpy as np
import ngsolve as ngs
import h5py

from bmbcsim.logging import logger
from bmbcsim.simulation.geometry.compartment import Compartment
from bmbcsim.simulation.geometry.membrane import Membrane
from bmbcsim.simulation.simulation_agents import ChemicalSpecies
from bmbcsim.simulation.result_io.xdmf_recorder import extract_dof_values


def write_coefficient_fields(
    directory: str,
    h5_filename: str,
    mesh: ngs.Mesh,
    compartments: dict[str, Compartment],
    membranes: dict[str, Membrane],
    rd_fes: ngs.FESpace,
    membrane_fes: dict[Membrane, ngs.FESpace],
    species: list[ChemicalSpecies],
) -> None:
    """Write out coefficient fields.

    Appends volumetric and surface coefficient data to snapshot.h5 and writes
    separate XDMF metadata files for volume and surface coefficients.

    :param directory: Result directory containing snapshot.h5.
    :param h5_filename: Name of the HDF5 file (typically "snapshot.h5").
    :param mesh: The NGSolve mesh.
    :param compartments: Dict of compartment name -> Compartment.
    :param membranes: Dict of membrane name -> Membrane.
    :param rd_fes: The compound reaction-diffusion FE space.
    :param membrane_fes: Dict of Membrane -> boundary FE space.
    :param species: List of chemical species in the simulation.
    """
    # Initialize paths and collect coefficient data
    h5_path = os.path.join(directory, h5_filename)
    vol_xdmf_path = os.path.join(directory, "volume_coefficients.xdmf")
    surf_xdmf_path = os.path.join(directory, "surface_coefficients.xdmf")

    compartment_list = list(compartments.values())

    volume_fields = _collect_volume_coefficients(
        compartment_list, rd_fes, species
    )
    surface_data = _collect_surface_coefficients(mesh, membranes, membrane_fes)

    if not volume_fields and not surface_data:
        logger.info("No coefficient fields to write.")
        return

    # Append coefficient data to HDF5
    with h5py.File(h5_path, "a") as h5:
        if volume_fields:
            coeff_grp = h5.create_group("coefficients")
            for name, data in volume_fields.items():
                coeff_grp.create_dataset(
                    name, data=data,
                    compression="gzip", compression_opts=1,
                )

        for membrane_name, (points, connectivity, fields) in surface_data.items():
            surf_mesh_grp = h5.require_group("surface_mesh")
            mem_grp = surf_mesh_grp.create_group(membrane_name)
            mem_grp.create_dataset(
                "points", data=points,
                compression="gzip", compression_opts=1,
            )
            mem_grp.create_dataset(
                "connectivity", data=connectivity,
                compression="gzip", compression_opts=1,
            )

            surf_coeff_grp = h5.require_group("surface_coefficients")
            mem_coeff_grp = surf_coeff_grp.create_group(membrane_name)
            for name, data in fields.items():
                mem_coeff_grp.create_dataset(
                    name, data=data,
                    compression="gzip", compression_opts=1,
                )

    # Write separate XDMF files for volume and surface
    if volume_fields:
        with h5py.File(h5_path, "r") as h5:
            n_points = h5["mesh/points"].shape[0]
            n_cells = h5["mesh/connectivity"].shape[0]
        _write_volume_xdmf(vol_xdmf_path, h5_filename, n_points, n_cells, volume_fields)
        logger.info("Wrote %d volume coefficient field(s) to %s", len(volume_fields), vol_xdmf_path)

    if surface_data:
        _write_surface_xdmf(surf_xdmf_path, h5_filename, surface_data)
        n_surface = sum(len(sd[2]) for sd in surface_data.values())
        logger.info("Wrote %d surface coefficient field(s) to %s", n_surface, surf_xdmf_path)


def _collect_volume_coefficients(
    compartments: list[Compartment],
    vol_fes: ngs.FESpace,
    species: list[ChemicalSpecies],
) -> dict[str, np.ndarray]:
    """Collect all volumetric coefficient fields from compartment SimulationDetails.

    After _setup(), coefficient values are NGSolve CoefficientFunctions.
    """
    fields: dict[str, np.ndarray] = {}

    # Collect initial conditions and diffusivities for each species
    for sp in species:
        data = _project_volume_field(
            compartments, vol_fes,
            lambda c, s=sp: c.coefficients.initial_conditions.get(s),
        )
        if data is not None:
            fields[f"initial_condition_{sp.name}"] = data

    for sp in species:
        data = _project_volume_field(
            compartments, vol_fes,
            lambda c, s=sp: c.coefficients.diffusion.get(s),
        )
        if data is not None:
            fields[f"diffusivity_{sp.name}"] = data

    # Collect all unique reaction keys across compartments
    all_reaction_keys = set()
    for compartment in compartments:
        all_reaction_keys.update(compartment.coefficients.reactions.keys())

    for reaction_key in all_reaction_keys:
        reactants, products = reaction_key
        reactant_str = "+".join(s.name for s in reactants)
        product_str = "+".join(s.name for s in products)

        kf_data = _project_volume_field(
            compartments, vol_fes,
            lambda c, k=reaction_key: (
                c.coefficients.reactions[k][0] if k in c.coefficients.reactions else None
            ),
        )
        if kf_data is not None:
            fields[f"kf_{reactant_str}_to_{product_str}"] = kf_data

        kr_data = _project_volume_field(
            compartments, vol_fes,
            lambda c, k=reaction_key: (
                c.coefficients.reactions[k][1] if k in c.coefficients.reactions else None
            ),
        )
        if kr_data is not None:
            fields[f"kr_{reactant_str}_to_{product_str}"] = kr_data

    # Collect permittivity if defined in any compartment
    data = _project_volume_field(
        compartments, vol_fes,
        lambda c: c.coefficients.permittivity,
    )
    if data is not None:
        fields["permittivity"] = data

    return fields


def _project_volume_field(
    compartments: list[Compartment],
    fes: ngs.FESpace,
    get_cf,
) -> np.ndarray | None:
    """Project a coefficient function from each compartment onto the
    reaction-diffusion FE space to have the same layout as species data in
    snapshot.h5.
    """
    gf = ngs.GridFunction(fes)
    gf.vec[:] = 0.0

    any_set = False
    for i, compartment in enumerate(compartments):
        cf = get_cf(compartment)
        if cf is not None:
            gf.components[i].Set(cf)
            any_set = True

    if not any_set:
        return None

    return extract_dof_values(gf)


def _collect_surface_coefficients(
    mesh: ngs.Mesh,
    membranes: dict[str, Membrane],
    membrane_fes: dict[Membrane, ngs.FESpace],
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """Collect surface coefficients from all membranes.

    Returns dict mapping membrane_name -> (points, connectivity, {field_name: values}).
    """
    result = {}

    for membrane_name, membrane in membranes.items():
        # Extract coefficients
        fes = membrane_fes.get(membrane)
        if fes is None or fes.ndof == 0:
            continue

        transport_list = membrane.get_transport()
        if not transport_list:
            continue

        has_coefficients = any(
            len(transport._finalized_coefficients) > 0
            for _, _, _, transport in transport_list
        )
        if not has_coefficients:
            continue

        # If there are coefficients, build the surface mesh for this membrane
        # and interpolate coefficients onto it
        boundary_region = mesh.Boundaries(membrane_name)
        points, connectivity = _build_surface_mesh(boundary_region, fes)

        fields = {}
        for species, _, _, transport in transport_list:
            for attr_name in transport._finalized_coefficients:
                cf = transport._spatial_coefficients.get(attr_name)
                if cf is None:
                    continue

                transport_type = type(transport).__name__
                field_name = f"{species.name}_{transport_type}_{attr_name}"
                gf = ngs.GridFunction(fes)
                gf.Set(cf, definedon=boundary_region)
                fields[field_name] = gf.vec.FV().NumPy().copy().astype(np.float32)

        if fields:
            result[membrane_name] = (points, connectivity, fields)

    return result


def _build_surface_mesh(
    boundary_region: ngs.Region,
    membrane_fes: ngs.FESpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Build surface mesh (points, triangles) for a membrane boundary."""
    # Extract coordinates of all vertices in the boundary region and build a mapping to global DOFs
    all_coords = np.array(boundary_region.mesh.ngmesh.Coordinates(), dtype=np.float32)
    dof_coords = np.zeros((membrane_fes.ndof, 3), dtype=np.float32)
    seen: set[int] = set()
    triangles: list[list[int]] = []

    for el in boundary_region.Elements():
        dofs = membrane_fes.GetDofNrs(el)
        verts = list(el.vertices)
        tri = []
        for dof, vert in zip(dofs, verts):
            if dof >= 0:
                if dof not in seen:
                    dof_coords[dof] = all_coords[vert.nr]
                    seen.add(dof)
                tri.append(dof)
        if len(tri) == 3:
            triangles.append(tri)

    if triangles:
        connectivity = np.array(triangles, dtype=np.int32)
    else:
        connectivity = np.empty((0, 3), dtype=np.int32)

    return dof_coords, connectivity


def _write_volume_xdmf(
    xdmf_path: str,
    h5_filename: str,
    n_points: int,
    n_cells: int,
    volume_fields: dict[str, np.ndarray],
) -> None:
    """Write coefficients.xdmf for volumetric coefficient fields."""
    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")

    grid = ET.SubElement(
        domain, "Grid", Name="VolumeCoefficients", GridType="Uniform",
    )

    topology = ET.SubElement(
        grid, "Topology",
        TopologyType="Tetrahedron",
        NumberOfElements=str(n_cells),
    )
    topo_data = ET.SubElement(
        topology, "DataItem",
        Dimensions=f"{n_cells} 4",
        NumberType="Int", Precision="4", Format="HDF",
    )
    topo_data.text = f"{h5_filename}:/mesh/connectivity"

    geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    geo_data = ET.SubElement(
        geometry, "DataItem",
        Dimensions=f"{n_points} 3",
        NumberType="Float", Precision="4", Format="HDF",
    )
    geo_data.text = f"{h5_filename}:/mesh/points"

    for name in sorted(volume_fields.keys()):
        attribute = ET.SubElement(
            grid, "Attribute",
            Name=name, AttributeType="Scalar", Center="Node",
        )
        attr_data = ET.SubElement(
            attribute, "DataItem",
            Dimensions=str(n_points),
            NumberType="Float", Precision="4", Format="HDF",
        )
        attr_data.text = f"{h5_filename}:/coefficients/{name}"

    _write_xdmf_file(xdmf_path, xdmf)


def _write_surface_xdmf(
    xdmf_path: str,
    h5_filename: str,
    surface_data: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
) -> None:
    """Write surface_coefficients.xdmf for membrane coefficient fields."""
    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")

    for membrane_name, (points, connectivity, fields) in surface_data.items():
        n_surf_points = points.shape[0]
        n_surf_cells = connectivity.shape[0]

        grid = ET.SubElement(
            domain, "Grid",
            Name=f"Surface_{membrane_name}",
            GridType="Uniform",
        )

        topology = ET.SubElement(
            grid, "Topology",
            TopologyType="Triangle",
            NumberOfElements=str(n_surf_cells),
        )
        topo_data = ET.SubElement(
            topology, "DataItem",
            Dimensions=f"{n_surf_cells} 3",
            NumberType="Int", Precision="4", Format="HDF",
        )
        topo_data.text = f"{h5_filename}:/surface_mesh/{membrane_name}/connectivity"

        geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
        geo_data = ET.SubElement(
            geometry, "DataItem",
            Dimensions=f"{n_surf_points} 3",
            NumberType="Float", Precision="4", Format="HDF",
        )
        geo_data.text = f"{h5_filename}:/surface_mesh/{membrane_name}/points"

        for name in sorted(fields.keys()):
            attribute = ET.SubElement(
                grid, "Attribute",
                Name=name, AttributeType="Scalar", Center="Node",
            )
            attr_data = ET.SubElement(
                attribute, "DataItem",
                Dimensions=str(n_surf_points),
                NumberType="Float", Precision="4", Format="HDF",
            )
            attr_data.text = (
                f"{h5_filename}:/surface_coefficients/{membrane_name}/{name}"
            )

    _write_xdmf_file(xdmf_path, xdmf)


def _write_xdmf_file(path: str, xdmf: ET.Element) -> None:
    """Write an XDMF XML element to disk atomically."""
    tree = ET.ElementTree(xdmf)
    ET.indent(tree, space="  ")

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)
    os.replace(tmp_path, path)
