import numpy as np
import pyvista as pv
from netgen.meshing import Mesh, MeshPoint, Point3d, Element2D, Element3D, FaceDescriptor


def pyvista_surface_to_netgen(
        mesh: pv.PolyData,
        *,
        flip_elements: bool = False
) -> Mesh:
    """Convert a pyvista surface mesh to a netgen mesh.

    :param mesh: The pyvista surface mesh to convert (can only include triangles).
    :param flip_elements: If True, flip the orientation of the elements.
    :return: The netgen mesh.
    """
    if not isinstance(mesh, pv.PolyData):
        raise ValueError("The mesh must be a pyvista PolyData object.")
    if not mesh.is_all_triangles:
        raise ValueError("The mesh must only contain triangles.")

    # Create an empty mesh with a default face descriptor
    ng_mesh = Mesh()
    fd = ng_mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))

    # Add vertices
    for point in mesh.points:
        ng_mesh.Add(MeshPoint(Point3d(*point)))

    # Add elements (vertices lookup is 1-based in netgen)
    cells = mesh.regular_faces[:, ::-1] if flip_elements else mesh.regular_faces
    for cell in cells:
        ng_mesh.Add(Element2D(fd, vertices=cell + 1))

    return ng_mesh


def pyvista_volume_to_netgen(
        mesh: pv.UnstructuredGrid
) -> Mesh:
    """Convert a pyvista volume mesh to a netgen mesh.

    :param mesh: The pyvista volume mesh to convert (can only include tetrahedra).
    :return: The netgen mesh.
    """
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise ValueError("The mesh must be a pyvista UnstructuredGrid object.")
    if any(mesh.celltypes != pv.CellType.TETRA):
        raise ValueError("The mesh must only contain tetrahedra.")

    # Create an empty mesh with a default face descriptor
    ng_mesh = Mesh()
    mat = ng_mesh.AddRegion("inside", 3)

    # Add vertices
    for point in mesh.points:
        ng_mesh.Add(MeshPoint(Point3d(*point)))

    # Add elements (vertices lookup is 1-based in netgen)
    for cell in mesh.cell_connectivity.reshape(-1, 4):
        ng_mesh.Add(Element3D(mat, vertices=cell + 1))

    # Add surface elements (vertices lookup is 1-based in netgen)
    # Note that pyvista surface elements appear to be flipped compared to
    # netgen surface elements. Thus, we flip them again below.
    surface = mesh.extract_surface()
    fd = ng_mesh.Add(FaceDescriptor(surfnr=1, domin=mat, domout=0))

    original_vertices = surface['vtkOriginalPointIds']
    for cell in surface.regular_faces[:, ::-1]:
        vertices = original_vertices[cell] + 1
        ng_mesh.Add(Element2D(fd, vertices=vertices))

    return ng_mesh


def netgen_to_pyvista(
        mesh: Mesh
) -> pv.UnstructuredGrid:
    """Convert a netgen volume mesh to a pyvista mesh.
    
    :param mesh: The netgen mesh to convert.
    :return: The pyvista mesh.
    """
    points = mesh.Coordinates()

    # Extract the cells from the netgen mesh (wich are stored in an internal
    # structure of 1-based point IDs)
    elements = [[point_id.nr for point_id in el.vertices]
                for el in mesh.Elements3D()]
    cells = np.full((len(elements), 5), 4)
    cells[:, 1:] = np.array(elements) - 1
    celltypes = [pv.CellType.TETRA] * len(cells)
    return pv.UnstructuredGrid(cells, celltypes, points)
