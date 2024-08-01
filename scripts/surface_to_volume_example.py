# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This is a short example showcasing how to obtain a volume mesh from a surface mesh. It is easy to manipulate surface meshes with NGSolve's OCC module, but converting them to volume meshes is not straightforward: The surfaces don't know that they are supposed to border any solid material.
#
# This information has to be given to the surfaces by `FaceDescriptor`s (which are part of `Element2D`, so that has to happen after meshing the surfaces. For information about how to convert surface meshes to volume meshes using `FaceDescriptor`s, see [this](https://docu.ngsolve.org/latest/i-tutorials/unit-4.3-manualmesh/manualmeshing.html#Merge-three-dimensional-meshes) and [this](https://docu.ngsolve.org/latest/i-tutorials/unit-4.3-manualmesh/manualmeshing.html#Merge-three-dimensional-meshes) part of NGSolve's documentation.

# %%
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw

# %%
# Dissect a cube into its faces
box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
top = box.faces[5]
bottom = box.faces[4]
front = box.faces[2]
back = box.faces[3]
left = box.faces[0]
right = box.faces[1]

# %%
# Tinker with some properties of the single faces
# Note that for these modifications it's not necessary to dissect the cube,
# this is done only for illustration purposes
for f in [bottom, front, back, left, right]:
    f.bc("fancy")
top.maxh = 0.1

# %%
# Fuse the faces back together
# At this point, the geometry is just a collection of surfaces (no contained solid)
geo = Fuse([top, bottom, front, back, left, right])
Draw(geo)

# %%
# The generated mesh is just a surface mesh
surface_mesh = OCCGeometry(geo).GenerateMesh()
mesh = Mesh(surface_mesh)
Draw(mesh)
print(mesh.GetBoundaries())
print(f"Number of 3D elements: {mesh.ne}")

# %%
# Even worse, it seems that the geometry has lost all it's knowledge about any contained solid
# as the following command doesn't actually produce a volume mesh
surface_mesh.GenerateVolumeMesh()
mesh = Mesh(surface_mesh)
Draw(mesh)
print(f"Number of 3D elements: {mesh.ne}")

# %%
# What's missing is information in the face Descriptors about the solid left and right of the face
# domin and domout, respectively (right is in the direction of the normal vector with the righ-hand rule applied)
# A Domain tag of 0 means that no domain is assigned to either left or right side
# Face descriptor handles are just integers (0-based), but the lookup here is 1-based
for i in range(6):
    print(f"Face descriptor {i + 1}: {surface_mesh.FaceDescriptor(i + 1)}")

# %%
# Unfortunately, face descriptors can't just be added to existing elements, so we need to copy the mesh,
# add suitable face descriptors, and add them to the appropriate elements 
from netgen.meshing import FaceDescriptor, Element2D
from netgen.meshing import Mesh as NetgenMesh

new_mesh = NetgenMesh()
default_descriptor = new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
fancy_descriptor = new_mesh.Add(FaceDescriptor(surfnr=2, domin=1, bc=2))

# %%
# Copy the points and keep a map of old to new index
old_to_new = {}
for e in surface_mesh.Elements2D():
    for v in e.vertices:
        if (v not in old_to_new):
            old_to_new[v] = new_mesh.Add(surface_mesh[v])

# %%
# Add elements and swap out the face descriptor (called "index", here)
old_fancy_descriptors = {2, 3, 4, 5, 6}
for e in surface_mesh.Elements2D():
    if e.index in old_fancy_descriptors:
        new_mesh.Add(Element2D(fancy_descriptor, [old_to_new[v] for v in e.vertices]))
    else:
        new_mesh.Add(Element2D(default_descriptor, [old_to_new[v] for v in e.vertices]))

# %%
# Use names to refer to new boundaries (there is a off-by-one mismatch when compared to the bcs above)
# Note that the "fancy" boundary condition is now collapsed into one boundary instead of five
new_mesh.SetBCName(0, "default")
new_mesh.SetBCName(1, "fancy")

# %%
# Et voilá, generating the volume mesh works!
new_mesh.GenerateVolumeMesh()
mesh = Mesh(new_mesh)
Draw(mesh)
print(mesh.GetBoundaries())
print(f"Number of 3D elements: {mesh.ne}")

# %%
