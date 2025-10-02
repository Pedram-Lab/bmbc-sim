# %%
from netgen import occ
from ngsolve.webgui import Draw
import ngsolve as ngs
import pyvista as pv

# %%
def explode(solid, distance):
    """Explode a solid by moving its faces and edges away from the center."""
    center = solid.center
    exploded_faces = sum([f.Move(distance * (f.center - center)) for f in solid.faces])
    exploded_edges = sum([e.Move(distance * (e.center - center)) for e in solid.edges])
    return occ.Compound([solid, exploded_faces, exploded_edges])


# %%
# Disassemble a box
box = occ.Box((0, 0, 0), (1, 1, 1))
Draw(explode(box, 0.3))

# %%
# Reassemble the box into a proper solid
new_box = occ.Solid(occ.Glue(box.faces))
Draw(explode(new_box, 0.3))

# %%
# Load a single cell (triangular surface mesh)
high_res_cell = pv.read('single_cell.stl')
cell = high_res_cell.decimate_boundary(0.8)
trigs = list(cell.cell)
# plotter = pv.Plotter()
# plotter.add_mesh(cell, show_edges=True)
# plotter.add_mesh(high_res_cell, show_edges=True, color='red', opacity=0.5)
# plotter.show()

# %%
# Create an occ surface from the triangular mesh
polys = []
for trig in trigs:
    vertices = [occ.Vertex(tuple(p)) for p in trig.points[[0,1,2,0]]]
    polys.append(occ.Face(occ.MakePolygon(vertices)))
polys = occ.Glue(polys)
Draw(polys)

# %%
# Make it a solid
ngs_cell = occ.Solid(polys)
Draw(ngs_cell)

# %%
# This can be clipped easily
center = ngs_cell.center
half_space = occ.HalfSpace(center, occ.Z)
Draw(ngs_cell * half_space)

# %%
# It can also be meshed
geo = occ.OCCGeometry(ngs_cell * half_space)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=100))
print(f"Created mesh with {mesh.nv} vertices and {mesh.ne} elements")
Draw(mesh)

# %%
# For comparison, this is the mesh of the un-clipped cell
geo = occ.OCCGeometry(ngs_cell)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=100))
print(f"Created mesh with {mesh.nv} vertices and {mesh.ne} elements")
Draw(mesh)

# %%
