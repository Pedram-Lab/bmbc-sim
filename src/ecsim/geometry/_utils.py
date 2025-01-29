"""
Utility functions for working with Netgen geometries and meshes.
"""
from typing import Dict
from netgen.meshing import Mesh as NetgenMesh
from ngsolve import Mesh
from netgen.meshing import Element2D, FaceDescriptor

def convert_to_volume_mesh(
        surface_mesh: NetgenMesh,
        bnd_to_fd: Dict[str, FaceDescriptor]
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
    new_mesh = NetgenMesh()

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
        new_mesh.Add(Element2D(fd, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh()
    return new_mesh
