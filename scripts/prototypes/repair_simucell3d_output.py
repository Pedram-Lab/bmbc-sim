"""
Postprocess and repair SimuCell3D output meshes for volume meshing.
This includes smoothing, decimation, and fixing mesh issues like holes,
degeneracies, and intersections.
"""
import pyvista as pv
from pymeshfix import MeshFix

from bmbcsim import TissueGeometry


def repair_mesh(mesh):
    """Repair a PyVista mesh using pymeshfix (holes, degeneracies, intersections)."""
    mesh = MeshFix(mesh)
    mesh.repair(verbose=False)
    mesh = mesh.mesh
    return mesh.triangulate().clean()


# Load tissue geometry from file
all_cells = TissueGeometry.from_file("scripts/prototypes/result_3590.vtk")
print(f"Loaded geometry with {len(all_cells.cells)} cells.")
# faulty_cells = [47, 58, 139, 149, 162, 214, 295, 296, 298, 309, 441, 462, 556, 584, 680, 897]

for i, cell in enumerate(all_cells.cells):
    # Scale it to microns and extract a single cell for testing
    print(f"Processing cell {i}/{len(all_cells.cells)}")
    single_cell = TissueGeometry(cells=cell)
    single_cell = single_cell.scale(1000000)
    min_coords, max_coords = single_cell.bounding_box()
    print(f"Bounding box: {min_coords}, {max_coords}")

    # Post-process the mesh to make the surfaces look nicer
    single_cell = single_cell.smooth(100)
    single_cell = single_cell.decimate(0.7)

    single_cell = TissueGeometry(cells=repair_mesh(single_cell.cells[0]))
    single_cell_mesh = single_cell.to_ngs_mesh(
        mesh_size=10, min_coords=min_coords, max_coords=max_coords
    )


# single_cell = TissueGeometry(cells=all_cells.cells[47])
# single_cell = single_cell.scale(1000000)
# min_coords, max_coords = single_cell.bounding_box()
# print(f"Bounding box: {min_coords}, {max_coords}")

# # Post-process the mesh to make the surfaces look nicer
# single_cell = single_cell.smooth(100)
# single_cell = single_cell.decimate(0.7)

# box_bounds = (min_coords[0], max_coords[0], min_coords[1], max_coords[1], min_coords[2], max_coords[2])
# combined_mesh = single_cell.as_single_mesh()
# combined_mesh = repair_mesh(combined_mesh)
# plotter = pv.Plotter()
# box = pv.Box(box_bounds)
# plotter.add_mesh(combined_mesh, show_edges=True, cmap='tab20b')
# plotter.add_mesh(box, color='gray', opacity=0.5)
# plotter.show()

# cleaned_cell = TissueGeometry(cells=combined_mesh)
# cleaned_mesh = cleaned_cell.to_ngs_mesh(mesh_size=10, min_coords=min_coords, max_coords=max_coords)
# Draw(cleaned_mesh)


# Manually remove some cells that didn't successfully mesh
# all_cells = geometry.cells
# faulty_cells = {47, 58, 139, 149, 162, 214, 295, 296, 298, 309, 441, 462, 556, 584, 680, 897}
# working_cells = [c for i, c in enumerate(all_cells) if not i in faulty_cells]
# geometry = TissueGeometry(working_cells)
