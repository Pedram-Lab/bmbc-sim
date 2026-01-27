"""
Prototype: Explore whether NGSolve FE spaces duplicate DOFs at compartment
boundaries, and how to extract DOF-to-vertex mappings.

Creates a simple two-compartment mesh (two unit cubes glued at x=1) and
inspects H1 spaces defined on each compartment separately vs. the full mesh.
Then builds a DOF-based mesh with duplicated boundary vertices and verifies
it produces correct results compared to VTKOutput.
"""

import ngsolve as ngs
from netgen import occ
import numpy as np
import pyvista as pv


# --- Build a two-compartment mesh ---
left = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1)).mat("left").bc("outer")
right = occ.Box(occ.Pnt(1, 0, 0), occ.Pnt(2, 1, 1)).mat("right").bc("outer")
left.faces[1].bc("membrane")

geo = occ.OCCGeometry(occ.Glue([left, right]))
ngmesh = geo.GenerateMesh(maxh=0.5)
mesh = ngs.Mesh(ngmesh)

materials = tuple(mesh.GetMaterials())
print(f"Mesh: {mesh.nv} vertices, {mesh.ne} volume elements")
print(f"Materials: {materials}")
print()

# --- Build compressed H1 spaces per compartment ---
compartment_fes = []
for mat in materials:
    fes = ngs.Compress(ngs.H1(mesh, order=1, definedon=mat))
    compartment_fes.append(fes)
    print(f"Compressed H1 ('{mat}'): ndof = {fes.ndof}")

total_dofs = sum(fes.ndof for fes in compartment_fes)
print(f"Total DOFs: {total_dofs}  (mesh.nv = {mesh.nv}, "
      f"duplication = {total_dofs - mesh.nv} boundary vertices)")
print()

# --- Build per-component dof-to-vertex and vertex-to-dof maps ---
dof_to_vertex_maps = []  # dof_to_vertex[dof] → mesh vertex index
vertex_to_dof_maps = []  # vertex_to_dof[vertex] → dof or -1

for fes in compartment_fes:
    dof_to_vertex = np.full(fes.ndof, -1, dtype=np.int32)
    vertex_to_dof = np.full(mesh.nv, -1, dtype=np.int32)
    for v in range(mesh.nv):
        dofs = fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, v))
        for d in dofs:
            if d >= 0:
                dof_to_vertex[d] = v
                vertex_to_dof[v] = d
    dof_to_vertex_maps.append(dof_to_vertex)
    vertex_to_dof_maps.append(vertex_to_dof)

print("=== DOF-to-vertex maps ===")
for i, (mat, dtv) in enumerate(zip(materials, dof_to_vertex_maps)):
    assert np.all(dtv >= 0), f"Component {mat} has unmapped DOFs"
    print(f"  {mat}: all {len(dtv)} DOFs mapped to vertices ✓")
print()

# --- Build global point array with boundary duplication ---
mesh_coords = np.array(mesh.ngmesh.Coordinates(), dtype=np.float32)

offsets = np.zeros(len(compartment_fes) + 1, dtype=np.int32)
for i, fes in enumerate(compartment_fes):
    offsets[i + 1] = offsets[i] + fes.ndof

points = np.concatenate([mesh_coords[dtv] for dtv in dof_to_vertex_maps])
print(f"=== Global point array ===")
print(f"Shape: {points.shape}  (total_dofs={total_dofs})")
print()

# --- Build remapped connectivity ---
# Map material name → component index
mat_to_comp = {mat: i for i, mat in enumerate(materials)}

# Extract mesh connectivity and per-element component index
mesh_conn = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)],
                      dtype=np.int32)
el_comp = np.array([mat_to_comp[el.mat] for el in mesh.Elements(ngs.VOL)], dtype=np.int32)

# Build a combined vertex_to_global_dof[comp, vertex] lookup table
vertex_to_global = np.full((len(compartment_fes), mesh.nv), -1, dtype=np.int32)
for i, vtd in enumerate(vertex_to_dof_maps):
    valid = vtd >= 0
    vertex_to_global[i, valid] = offsets[i] + vtd[valid]

# Vectorized connectivity remap: look up each cell's vertices in its component's map
connectivity = vertex_to_global[el_comp[:, None], mesh_conn]
print(f"=== Connectivity ===")
print(f"Shape: {connectivity.shape}")
assert np.all(connectivity >= 0), "Some vertices not mapped to DOFs"
assert np.all(connectivity < total_dofs), "DOF index out of range"
print("All connectivity indices valid ✓")
print()

# --- Build compartment indicators ---
indicators = {}
for i, mat in enumerate(materials):
    ind = np.zeros(total_dofs, dtype=np.float32)
    ind[offsets[i]:offsets[i + 1]] = 1.0
    indicators[mat] = ind

# --- Create a test GridFunction and set different values per compartment ---
product_fes = ngs.FESpace(compartment_fes)
gf = ngs.GridFunction(product_fes)
test_cf = mesh.MaterialCF({"left": 1.5, "right": 3.0})
for i, mat in enumerate(materials):
    gf.components[i].Set(test_cf, definedon=mesh.Materials(mat))

# Extract field values as trivial concatenation
field_values = np.concatenate(
    [gf.components[i].vec.FV().NumPy() for i in range(len(compartment_fes))]
).astype(np.float32)

print(f"=== Field values ===")
print(f"Shape: {field_values.shape}")
for i, mat in enumerate(materials):
    comp_vals = field_values[offsets[i]:offsets[i + 1]]
    print(f"  {mat}: min={comp_vals.min():.4f}, max={comp_vals.max():.4f}")
print()

# --- Build pyvista grid from DOF-based data ---
cells = np.empty((len(connectivity), 5), dtype=np.int64)
cells[:, 0] = 4
cells[:, 1:] = connectivity
celltypes = np.full(len(connectivity), pv.CellType.TETRA, dtype=np.uint8)
dof_grid = pv.UnstructuredGrid(cells, celltypes, points)
dof_grid.point_data["val"] = field_values
for mat, ind in indicators.items():
    dof_grid.point_data[mat] = ind

# --- Compare with VTKOutput ---
print("=== VTKOutput comparison ===")
ngs.VTKOutput(mesh, filename="/tmp/fes_proto", coefs=[test_cf], names=["val"],
              floatsize="single").Do()
vtk_grid = pv.read("/tmp/fes_proto.vtu")
print(f"VTKOutput: {vtk_grid.n_points} points, {vtk_grid.n_cells} cells")
print(f"DOF-based: {dof_grid.n_points} points, {dof_grid.n_cells} cells")
print(f"Reduction: {vtk_grid.n_points / dof_grid.n_points:.1f}x fewer points")
print()

# --- Sample at interior points and compare ---
print("=== Point sampling comparison ===")
sample_points = np.array([
    [0.3, 0.5, 0.5],  # deep inside left
    [1.7, 0.5, 0.5],  # deep inside right
    [0.9, 0.5, 0.5],  # near boundary, left side
    [1.1, 0.5, 0.5],  # near boundary, right side
])
sample_cloud = pv.PolyData(sample_points)

vtk_sampled = sample_cloud.sample(vtk_grid)
dof_sampled = sample_cloud.sample(dof_grid)

for i, pt in enumerate(sample_points):
    vtk_val = vtk_sampled.point_data["val"][i]
    dof_val = dof_sampled.point_data["val"][i]
    match = "✓" if abs(vtk_val - dof_val) < 1e-4 else "✗"
    print(f"  Point {pt}: VTK={vtk_val:.4f}, DOF={dof_val:.4f}  {match}")
print()

# --- Integration comparison (total substance style) ---
print("=== Integration comparison ===")
for label, grid in [("VTKOutput", vtk_grid), ("DOF-based", dof_grid)]:
    cells_conn = grid.cell_connectivity.reshape(-1, 4)
    volumes = -grid.compute_cell_sizes(
        length=False, area=False, vertex_count=False
    )["Volume"]
    values = grid.point_data["val"]
    cell_means = np.mean(values[cells_conn], axis=1)
    total = np.sum(cell_means * volumes)
    print(f"  {label}: total integral = {total:.6f}")

# Also per-region integration
print()
print("=== Per-region integration ===")
for label, grid in [("VTKOutput", vtk_grid), ("DOF-based", dof_grid)]:
    cells_conn = grid.cell_connectivity.reshape(-1, 4)
    volumes = -grid.compute_cell_sizes(
        length=False, area=False, vertex_count=False
    )["Volume"]
    values = grid.point_data["val"]
    cell_means = np.mean(values[cells_conn], axis=1)

    # Determine cell regions from indicator data
    if "left" in grid.point_data:
        cell_grid = grid.point_data_to_cell_data()
        for mat in materials:
            mask = cell_grid.cell_data[mat].astype(bool)
            region_total = np.sum(cell_means[mask] * volumes[mask])
            print(f"  {label} [{mat}]: {region_total:.6f}")
    else:
        # VTK grid doesn't have indicators, use the value to determine region
        for mat, expected_val in [("left", 1.5), ("right", 3.0)]:
            # Cells where all vertices have the expected value
            cell_vals = values[cells_conn]
            mask = np.all(np.abs(cell_vals - expected_val) < 0.1, axis=1)
            region_total = np.sum(cell_means[mask] * volumes[mask])
            print(f"  {label} [{mat}]: {region_total:.6f}")

print()
print("=== Summary ===")
print(f"DOF-based mesh successfully built with {total_dofs} points "
      f"(vs VTKOutput's {vtk_grid.n_points})")
print("Field values are a trivial concatenation of component DOF vectors.")
