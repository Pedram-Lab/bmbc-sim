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

# Visualize the original mesh
combined_mesh = all_cells.as_single_mesh()
plotter = pv.Plotter()
plotter.add_mesh(combined_mesh, scalars="face_cell_id", show_edges=True, cmap='tab20b')
plotter.show()

repaired_cells = []
for i, cell in enumerate(all_cells.cells[:5]):
    print(f"Processing cell {i}/{len(all_cells.cells)}")

    # Scale the mesh to microns and extract a single cell
    single_cell = TissueGeometry(cells=cell)
    single_cell = single_cell.scale(1000000)

    # Extract bounding box for meshing
    min_coords, max_coords = single_cell.bounding_box()
    print(f"Bounding box: {min_coords}, {max_coords}")

    # Post-process the mesh to make the surfaces look nicer
    single_cell = single_cell.smooth(100)
    single_cell = single_cell.decimate(0.7)
    repaired_cell = repair_mesh(single_cell.cells[0])
    repaired_cells.append(repaired_cell)

    # Create a volume mesh to check for mesh issues
    single_cell = TissueGeometry(cells=repaired_cell)
    single_cell_mesh = single_cell.to_ngs_mesh(
        mesh_size=10, min_coords=min_coords, max_coords=max_coords
    )

# Visualize the repaired meshes
repaired_cells = TissueGeometry(cells=repaired_cells)
print(f"Repaired geometry with {len(repaired_cells.cells)} cells.")

combined_mesh = repaired_cells.as_single_mesh()
plotter = pv.Plotter()
plotter.add_mesh(combined_mesh, show_edges=True, cmap='tab20b')
plotter.show()
