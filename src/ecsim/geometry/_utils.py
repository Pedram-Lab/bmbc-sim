"""
Utility functions for working with Netgen geometries and meshes.
"""
from typing import Dict

import astropy.units as u
from netgen import occ
import netgen.meshing as ngs
from ngsolve import Mesh

from ecsim.units import to_simulation_units

def convert_to_volume_mesh(
        surface_mesh: ngs.Mesh,
        bnd_to_fd: Dict[str, ngs.FaceDescriptor]
):
    """
    Converts a surface mesh to a volume mesh. The surface mesh is optimized in
    the process and may change.
    :param surface_mesh: The surface mesh to convert.
    :param bnd_to_fd: A mapping from boundary names to face descriptors that
        describe how the boundaries map to volume compartments and boundary
    conditions.
    :return: A new volume mesh.
    """
    new_mesh = ngs.Mesh()

    # Copy nodes
    old_to_new = {}
    for e in surface_mesh.Elements2D():
        for v in e.vertices:
            if v not in old_to_new:
                old_to_new[v] = new_mesh.Add(surface_mesh[v])

    # Arrange face descriptors for the new mesh
    boundaries = Mesh(surface_mesh).GetBoundaries()
    bnd_to_fd_index = {bnd: new_mesh.Add(fd) for bnd, fd in bnd_to_fd.items()}
    face_descriptor_indices = [bnd_to_fd_index[bnd] for bnd in boundaries]

    # Copy elements
    for e in surface_mesh.Elements2D():
        fd = face_descriptor_indices[e.index - 1]
        new_mesh.Add(ngs.Element2D(fd, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh()
    return new_mesh

def create_mesh(
        geo: occ.TopoDS_Shape,
        mesh_size: u.Quantity,
):
    """
    Generate a mesh from a geometry.
    :param geo: The geometry to mesh.
    :param mesh_size: The maximum mesh size.
    :return: The mesh.
    """
    return occ.OCCGeometry(geo).GenerateMesh(maxh=to_simulation_units(mesh_size, 'length'))
