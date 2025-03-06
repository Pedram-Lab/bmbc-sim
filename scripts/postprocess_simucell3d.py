import numpy as np
import pyvista as pv


mesh = pv.read("result_3590.vtk")

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, scalars='face_cell_id')
# plotter.show()

cell_ids = np.unique(mesh['face_cell_id'])


def _extract_local_data(mesh, cell_id):
    """Extract nodes and triangles of a cell with given id."""
    # Find the faces that belong to the cell
    face_in_cell = mesh['face_cell_id'] == cell_id
    faces = mesh.cell_connectivity.reshape(-1, 3)

    # Renumber the nodes of the cell and renumber connectivity accordingly
    cell_faces = faces[face_in_cell]
    cell_node_indices, local_connectivity = np.unique(cell_faces, return_inverse=True)
    local_connectivity = local_connectivity.reshape(-1, 3)
    cell_nodes = mesh.points[cell_node_indices]

    return cell_nodes, local_connectivity


def extract_single_cell(mesh, cell_id):
    """Extract a single cell with given id from a mesh."""
    cell_nodes, local_connectivity = _extract_local_data(mesh, cell_id)
    return pv.UnstructuredGrid({pv.CellType.TRIANGLE: local_connectivity}, cell_nodes)


def compute_volume(mesh, cell_id):
    """Compute the volume of a cell with given id."""
    cell_nodes, local_connectivity = _extract_local_data(mesh, cell_id)

    # Compute the center of mass of the cell
    center = np.mean(cell_nodes, axis=0)

    # Compute the volume using the divergence theorem
    # http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
    c0 = cell_nodes[local_connectivity[:, 0]] - center
    c1 = cell_nodes[local_connectivity[:, 1]] - center
    c2 = cell_nodes[local_connectivity[:, 2]] - center
    return np.sum(np.cross(c0, c1) * c2) / 6


def shrink_cells(mesh, cell_ids, factor):
    """Shrink the given cells by a given factor in all directions."""
    faces = mesh.cell_connectivity.reshape(-1, 3)

    for cell_id in cell_ids:
        # Find the nodes of the cell
        face_in_cell = mesh['face_cell_id'] == cell_id
        cell_faces = faces[face_in_cell]
        cell_node_indices = np.unique(cell_faces)

        # Shrink the nodes around the center of mass
        cell_nodes = mesh.points[cell_node_indices]
        center = np.mean(cell_nodes, axis=0)
        cell_nodes = center + factor * (cell_nodes - center)
        mesh.points[cell_node_indices] = cell_nodes

    return mesh


def get_fully_contained_box(mesh):
    """Get the smallest box that fully contains the mesh."""
    # Get min, max, and center of a box that contains the mesh
    min_coords = np.min(mesh.points, axis=0)
    max_coords = np.max(mesh.points, axis=0)
    center = (min_coords + max_coords) / 2

    # Shrink the box to fit inside the mesh
    s = 1 / np.sqrt(3) * 0.9  # theoretical factor and fudge factor
    min_coords = center + s * (min_coords - center)
    max_coords = center + s * (max_coords - center)

    return np.column_stack([min_coords, max_coords]).flatten()


# We want a certain percentage of the volume to become ECS
ECS_PERCENTAGE = 0.5
INTERACTIVE = False
FACTOR = (1 - ECS_PERCENTAGE) ** (1 / 3)

volume = sum(compute_volume(mesh, cell_id) for cell_id in cell_ids)
print(f"total volume: {volume}")


shrink_cells(mesh, cell_ids, FACTOR)
plotter = pv.Plotter()
if INTERACTIVE:
    plotter.add_mesh_clip_box(mesh, scalars='face_cell_id')
else:
    box = get_fully_contained_box(mesh)
    box = pv.Box(box)
    clipped_mesh = mesh.clip_box(box, invert=False)
    plotter.add_mesh(clipped_mesh, scalars='face_cell_id')
    # plotter.add_mesh(box, color='red', opacity=0.5)
plotter.show()
