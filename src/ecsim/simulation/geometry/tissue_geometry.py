import numpy as np
import pyvista as pv
import ngsolve as ngs
from netgen import occ


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

    def smooth(self, n_iter: int = 10, **kwargs) -> 'TissueGeometry':
        """Smooth the tissue geometry using a Laplacian filter.

        :param n_iter: The number of smoothing iterations.
        :param kwargs: Additional arguments to pass to the
            :method:`pyvista.PolyData.smooth` filter.
        :return: A new TissueGeometry object with the smoothed cells.
        """
        new_cells = [cell.smooth(n_iter, **kwargs) for cell in self.cells]
        for cell, new_cell in zip(self.cells, new_cells):
            new_cell['face_cell_id'] = cell['face_cell_id']
        return TissueGeometry(new_cells)

    def decimate(self, factor: float = 0.7, **kwargs) -> 'TissueGeometry':
        """Decimate the number of triangles in the tissue geometry by a given factor.

        :param factor: The decimation factor.
        :param kwargs: Additional arguments to pass to the
            :method:`pyvista.PolyData.decimate` filter.
        :return: A new TissueGeometry object with the decimated cells.
        """
        new_cells = [cell.decimate(factor, **kwargs) for cell in self.cells]
        for cell, new_cell in zip(self.cells, new_cells):
            cell_id = cell['face_cell_id'][0]
            new_cell['face_cell_id'] = np.full(new_cell.n_cells, cell_id)
        return TissueGeometry(new_cells)

    def keep_cells_within(
            self,
            *,
            min_coords: np.ndarray,
            max_coords: np.ndarray,
            inside_threshold: float = 0.1
        ) -> 'TissueGeometry':
        """Keep only the cells that are contained within a given bounding box.

        :param min_coords: The minimum coordinates of the bounding box.
        :param max_coords: The maximum coordinates of the bounding box.
        :param inside_threshold: A ratio between 0 and 1. All cells that have at
            least this ratio of their nodes within the bounding box are kept.
        :return: A new TissueGeometry object with the cells that are within the
            bounding box.
        """
        new_cells = []
        for cell in self.cells:
            larger_than_min = np.all(cell.points >= min_coords, axis=1)
            smaller_than_max = np.all(cell.points <= max_coords, axis=1)
            inside_box = np.logical_and(larger_than_min, smaller_than_max)
            ratio_inside_box = np.count_nonzero(inside_box) / cell.n_points

            if ratio_inside_box > inside_threshold:
                new_cells.append(cell.copy())

        return TissueGeometry(new_cells)

    def to_ngs_mesh(
            self,
            mesh_size: float,
            *,
            min_coords: np.ndarray,
            max_coords: np.ndarray,
            cell_names: str | list[str] = "cell",
            cell_bnd_names: str | list[str] = "membrane",
            projection_tol: float = None
    ) -> ngs.Mesh:
        """Convert the tissue geometry to a netgen mesh. The mesh is a box with
        given min and max coordinates, and the cells are clipped to fit within
        the box. The extracellular space is named 'ecs', the cells and their
        boundaries can be given a custom name. The boundaries of the box are
        named 'left', 'right', 'top', 'bottom', 'front', 'back'.
        An optional projection step is applied to points near the box faces to
        reduce the number of triangles in the mesh.

        :param mesh_size: The size of the mesh elements.
        :param min_coords: The minimum coordinates of the bounding box.
        :param max_coords: The maximum coordinates of the bounding box.
        :param cell_names: The name of the cells in the mesh. If a list is given,
            the names are assigned in the order of the cells.
        :param cell_bnd_names: The name of the cell boundaries in the mesh. If a
            list is given, the names are assigned in the order of the cells.
        :param projection_tol: The tolerance for projecting points to the box faces.
            If None, no projection is applied.
        :return: A netgen mesh object of the tissue geometry.
        """
        # Project points to nearest box face if below projection_tol
        if projection_tol is not None:
            for cell in self.cells:
                cell.points = snap_points_to_bounds(cell.points, min_coords, projection_tol)
                cell.points = snap_points_to_bounds(cell.points, max_coords, projection_tol)

        # Sort out the cell and boundary names
        if isinstance(cell_names, str):
            cell_names = [cell_names] * len(self.cells)
        if isinstance(cell_bnd_names, str):
            cell_bnd_names = [cell_bnd_names] * len(self.cells)
        if len(cell_names) != len(self.cells) or len(cell_bnd_names) != len(self.cells):
            raise ValueError("Number of cell and boundary names must match number of cells.")

        # Create an occ surface from the triangular mesh
        cell_geometries = []
        for cell, name, bc_name in zip(self.cells, cell_names, cell_bnd_names):
            cell = polydata_to_occ(cell)
            cell.mat(name)
            cell.bc(bc_name)
            cell_geometries.append(cell)

        # Create the bounding box and clip cells
        min_coords = tuple(float(c) for c in min_coords)
        max_coords = tuple(float(c) for c in max_coords)
        bounding_box = occ.Box(min_coords, max_coords)
        for i, name in enumerate(["left", "right", "top", "bottom", "front", "back"]):
            bounding_box.faces[i].bc(name)

        cell_geometries = [cell * bounding_box for cell in cell_geometries]
        cell_geometries = occ.Glue(cell_geometries)
        geometry = occ.OCCGeometry(occ.Glue([cell_geometries, bounding_box - cell_geometries]))

        return ngs.Mesh(geometry.GenerateMesh(maxh=mesh_size))



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


def polydata_to_occ(surface: pv.PolyData) -> occ.Solid:
    """Create an occ solid from a triangular pyvista surface mesh."""
    polys = []
    for trig in surface.cell:
        vertices = [occ.Vertex(tuple(p)) for p in trig.points[[0, 1, 2, 0]]]
        poly = occ.Face(occ.MakePolygon(vertices))
        poly.col = (1, 0, 0)
        polys.append(poly)
    return occ.Solid(occ.Glue(polys))


def snap_points_to_bounds(
    points: np.ndarray,
    bounds: np.ndarray,
    tol: float
) -> np.ndarray:
    """Snap points to the nearest point on the bounding box."""
    n = points.shape[0]
    bounds = np.tile(bounds, (n, 1))
    dist = points - bounds
    project = np.abs(dist) < tol
    points[project] = bounds[project]
    return points


if __name__ == "__main__":
    from ngsolve.webgui import Draw
    # Example usage
    geometry = TissueGeometry.from_file("/Users/innerbergerm/Projects/janelia/ecm-simulations/scripts/result_3590.vtk")
    geometry = geometry.scale(8000)
    min_coords, max_coords = geometry.bounding_box()
    min_coords, max_coords = min_coords / 5, max_coords / 5
    print(f"Bounding box: {min_coords}, {max_coords}")

    geometry = geometry.shrink_cells(0.7, jitter=0.1)
    geometry = geometry.smooth(100)
    geometry = geometry.decimate(0.7)

    geometry = geometry.keep_cells_within(
        min_coords=min_coords,
        max_coords=max_coords,
        inside_threshold=0.1
    )

    combined_mesh = geometry.as_single_mesh()
    plotter = pv.Plotter()
    box = pv.Box((min_coords[0], max_coords[0],
                  min_coords[1], max_coords[1],
                  min_coords[2], max_coords[2]))
    plotter.add_mesh(combined_mesh, scalars='face_cell_id', show_edges=True)
    plotter.add_mesh(box, color='gray', opacity=0.5)
    plotter.show()

    working_cells = [c for i, c in enumerate(geometry.cells) if not i in [1, 4, 31]]

    # geometry.cells[1].plot(show_edges=True)
    # geometry = TissueGeometry(geometry.cells[1:2])
    # mesh = geometry.to_ngs_mesh(
    #     mesh_size=0.1,
    #     min_coords=min_coords,
    #     max_coords=max_coords,
    #     projection_tol=0.004,
    # )
    # print(f"Created mesh with {mesh.nv} vertices and {mesh.ne} elements")
    # Draw(mesh)
