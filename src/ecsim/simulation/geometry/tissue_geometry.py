import numpy as np
import pyvista as pv


class TissueGeometry:
    """Class to represent the geometry of a tissue in a simulation.
    """
    def __init__(self, cells: list[pv.PolyData]):
        """Initialize the TissueGeometry with a list of cell surface meshes.

        :param cells: A list of PyVista PolyData objects representing cell surfaces.
        """
        self.cells = cells

    def as_single_mesh(self):
        """Combine all cell meshes into a single mesh. The resulting mesh
        has a 'face_cell_id' array that is constant and unique for each cell.
        """
        return self.cells[0].merge(self.cells[1:])

    def scale(self, factor: float) -> 'TissueGeometry':
        """Scale the tissue geometry by a given factor.

        :param factor: The scaling factor.
        :return: A new TissueGeometry object with the scaled cells.
        """
        new_cells = [cell.copy() for cell in self.cells]
        for cell in new_cells:
            cell.points *= factor
        return TissueGeometry(new_cells)
    
    def shrink_cells(
            self,
            factor: float,
            *,
            jitter: float = 0.0
        ) -> 'TissueGeometry':
        """Shrink the given cells by a given factor in all directions. Cells are
        shrunk around their center of mass. Some random jitter can be added to
        the centers of the cells to make spacing between cells more variable.

        :param factor: The volume factor by which to shrink the cells.
        :param jitter: Optional jitter to add to the center of mass while shrinking.
        :return: A new TissueGeometry object with the shrunk cells.
        """
        factor = factor ** (1 / 3)  # Convert volume factor to length factor
        new_cells = [cell.copy() for cell in self.cells]
        for cell in new_cells:
            if jitter > 0:
                # Create a random weight field to ensure center is within the cell (if it's convex)
                cell['jitter_weight'] = np.random.poisson(jitter, cell.n_points)
                cell.set_active_scalars('jitter_weight')
                center = cell.center_of_mass(scalars_weight=True)
                cell.point_data.remove('jitter_weight')
            else:
                center = cell.center_of_mass()

            cell.points = center + factor * (cell.points - center)

        return TissueGeometry(new_cells)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the tissue geometry.

        :return: A tuple of 1D arrays with the min and max coordinates of the
            bounding box in the order [xmin, ymin, zmin], [xmax, ymax, zmax].
        """
        min_coords = np.min([cell.bounds[::2] for cell in self.cells], axis=0)
        max_coords = np.max([cell.bounds[1::2] for cell in self.cells], axis=0)
        return min_coords, max_coords

    def smooth(self, n_iter: int = 10) -> 'TissueGeometry':
        """Smooth the tissue geometry using a Laplacian filter.

        :param n_iter: The number of smoothing iterations.
        :return: A new TissueGeometry object with the smoothed cells.
        """
        new_cells = [cell.smooth(n_iter) for cell in self.cells]
        for cell, new_cell in zip(self.cells, new_cells):
            new_cell['face_cell_id'] = cell['face_cell_id']
        return TissueGeometry(new_cells)

    def decimate(self, factor: float) -> 'TissueGeometry':
        """Decimate the number of triangles in the tissue geometry by a given factor.

        :param factor: The decimation factor.
        :return: A new TissueGeometry object with the decimated cells.
        """
        new_cells = [cell.decimate(factor) for cell in self.cells]
        for cell, new_cell in zip(self.cells, new_cells):
            cell_id = cell['face_cell_id'][0]
            new_cell['face_cell_id'] = np.full(new_cell.n_cells, cell_id)
        return TissueGeometry(new_cells)

    def keep_cells_within(
            self,
            *,
            min_coords: np.ndarray,
            max_coords: np.ndarray,
            touching: bool = True
        ) -> 'TissueGeometry':
        """Keep only the cells that are contained within a given bounding box.

        :param min_coords: The minimum coordinates of the bounding box.
        :param max_coords: The maximum coordinates of the bounding box.
        :param touching: If True, cells that touch the bounding box are kept, otherwise
            only cells that are fully contained within the bounding box are kept.
        :return: A new TissueGeometry object with the cells that are within the bounding box.
        """
        new_cells = []
        for cell in self.cells:
            larger_than_min = np.all(cell.points >= min_coords, axis=1)
            smaller_than_max = np.all(cell.points <= max_coords, axis=1)
            in_box = np.logical_and(larger_than_min, smaller_than_max)

            if touching:
                within_box = np.any(in_box)
            else:
                within_box = np.all(in_box)

            if within_box:
                new_cells.append(cell.copy())
                
        return TissueGeometry(new_cells)

    @classmethod
    def from_file(cls, file_name: str):
        """Load a tissue geometry from a file. The file must be in vtk format
        and containin triangular surface meshes of one or mor cells. It is
        assumed that the mesh has an array called 'face_cell_id' that is
        constant and unique for each cell. This format is typically generated by
        the SimuCell3D software.

        :param file_name: Path to the vtk file containing the tissue geometry.
        """
        raw_geometry = pv.read(file_name)
        cell_ids = np.unique(raw_geometry['face_cell_id'])
        cells = [extract_single_cell(raw_geometry, i) for i in cell_ids]
        for i, cell in enumerate(cells):
            cell['face_cell_id'] = np.full(cell.n_cells, i, dtype=int)
        return cls(cells)


def extract_single_cell(
        mesh: pv.PolyData,
        cell_id: int
    ) -> pv.PolyData:
    """Extract nodes and triangles of a cell with given id."""
    # Find the faces that belong to the cell
    face_in_cell = mesh['face_cell_id'] == cell_id
    faces = mesh.cell_connectivity.reshape(-1, 3)

    # Renumber the nodes of the cell and renumber connectivity accordingly
    cell_faces = faces[face_in_cell]
    cell_node_indices, local_connectivity = np.unique(cell_faces, return_inverse=True)
    local_connectivity = local_connectivity.reshape(-1, 3)
    cell_nodes = mesh.points[cell_node_indices]

    return pv.PolyData.from_regular_faces(points=cell_nodes, faces=local_connectivity)


if __name__ == "__main__":
    # Example usage
    geometry = TissueGeometry.from_file("/Users/innerbergerm/Projects/janelia/ecm-simulations/scripts/result_3590.vtk")
    geometry = geometry.scale(8000)
    min_coords, max_coords = geometry.bounding_box()
    print(f"Bounding box: {min_coords}, {max_coords}")

    geometry = geometry.shrink_cells(0.7, jitter=0.1)
    geometry = geometry.smooth(100)
    geometry = geometry.decimate(0.7)

    geometry = geometry.keep_cells_within(
        min_coords=min_coords / 2,
        max_coords=max_coords / 2,
        touching=True
    )

    combined_mesh = geometry.as_single_mesh()
    combined_mesh.plot(show_edges=True, cmap="tab20b")
    cell = geometry.cells[0]
