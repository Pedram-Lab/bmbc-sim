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
# This example shows how to use the knowledge obtained in [`surface_to_volume.py`](/scripts/surface_to_volume.py) to make a prototypical channel geometry, which consists of:
# * A cytosolic part on the bottom,
# * A small extracellular space on the top,
# * An interface separating those two and containing a region that models an ion channel.
#
# The interesting part is that the channel region is not a volume but only a region on the surface that represents the interface.

# %%
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw

# %%
# Define general geometrical objects that we'll use to stitch the domain together
cytosol = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
ecs = Box(Pnt(0, 0, 1), Pnt(1, 1, 1.2))
left, right, front, back, bottom, top = (0, 1, 2, 3, 4, 5)

# %%
# Assign boundary conditions
for f in [front, back, left, right]:
    cytosol.faces[f].bc("cyt_bnd")
    ecs.faces[f].bc("ecs_bnd")
cytosol.faces[bottom].bc("cyt_bnd")
ecs.faces[top].bc("ecs_top")

# Cut a hole into the ecs-cytosol interface
channel = Face(Wire(Circle(Pnt(0.5, 0.5, 1), Z, 0.1)))
channel.maxh = 0.03
channel.bc("channel")
channel.col = (1, 0, 0)
membrane = (cytosol.faces[top] - channel).bc("membrane")
interface = Glue([membrane, channel])  # if fused, channel vanishes

# Only take parts that make up the actual geometry
geo = Fuse([interface, ecs.faces[top], cytosol.faces[bottom]]
           + [cytosol.faces[f] for f in [front, back, left, right]]
           + [ecs.faces[f] for f in [front, back, left, right]])
Draw(geo, clipping={"function": True,  "pnt": (0.5, 0.5, 0.5), "vec": (0, 1, 0)})

# %%
# Generate a mesh on the surface (no volume mesh so far)
surface_mesh = OCCGeometry(geo).GenerateMesh()
mesh = Mesh(surface_mesh)
Draw(mesh, clipping={"function": True,  "pnt": (0.5, 0.5, 0.5), "vec": (0, 1, 0)})

# %%
# Automatically get the boundary indices that are used to assign face descriptors later
cyt_bnd = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "cyt_bnd"]
ecs_bnd = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "ecs_bnd"]
ecs_top = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "ecs_top"]
channel = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "channel"]
membrane = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "membrane"]

# %%
# The sub-meshes can be visualized easily
# Note: in this process, the meshes are copied, so using GetSubMesh to add face descriptors doesn't work
membrane_mesh = surface_mesh.GetSubMesh(faces="membrane|channel") 
Draw(membrane_mesh)

# %%
# Generate new face descriptors - note that there are two domains now!
from netgen.meshing import FaceDescriptor, Element2D
from netgen.meshing import Mesh as NetgenMesh

new_mesh = NetgenMesh()
fd_channel = new_mesh.Add(FaceDescriptor(surfnr=1, domin=2, domout=1, bc=1))
fd_membrane = new_mesh.Add(FaceDescriptor(surfnr=2, domin=2, domout=1, bc=2))
fd_ecs_top = new_mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3))
fd_ecs_bnd = new_mesh.Add(FaceDescriptor(surfnr=4, domin=1, domout=0, bc=4))
fd_cyt_bnd = new_mesh.Add(FaceDescriptor(surfnr=5, domin=2, domout=0, bc=5))


# %%
# Copy all nodes to a new mesh
old_to_new = {}
for e in surface_mesh.Elements2D():
    for v in e.vertices:
        if (v not in old_to_new):
            old_to_new[v] = new_mesh.Add(surface_mesh[v])

# Copy all elements (with the appropriate face descriptors)
for e in surface_mesh.Elements2D():
    if e.index in ecs_top:
        new_mesh.Add(Element2D(fd_ecs_top, [old_to_new[v] for v in e.vertices]))
    elif e.index in channel:
        new_mesh.Add(Element2D(fd_channel, [old_to_new[v] for v in e.vertices]))
    elif e.index in membrane:
        new_mesh.Add(Element2D(fd_membrane, [old_to_new[v] for v in e.vertices]))
    elif e.index in cyt_bnd:
        new_mesh.Add(Element2D(fd_cyt_bnd, [old_to_new[v] for v in e.vertices]))
    elif e.index in ecs_bnd:
        new_mesh.Add(Element2D(fd_ecs_bnd, [old_to_new[v] for v in e.vertices]))
    else:
        raise ValueError(f"Can't cope with value {e.index}")

# Draw(new_mesh, clipping={"function": True,  "pnt": (0.5, 0.5, 0.5), "vec": (0, 1, 0)})

# %%
# Generate volume mesh from surface
new_mesh.GenerateVolumeMesh()
mesh = Mesh(new_mesh)
print(f"Number of 3D elements: {mesh.ne}")

# %%
# Assign names to boundaries
new_mesh.SetBCName(0, "channel")
new_mesh.SetBCName(1, "membrane")
new_mesh.SetBCName(2, "ecs_top")
new_mesh.SetBCName(3, "boundary")
new_mesh.SetBCName(4, "boundary")

# Click on any face to see the boundary condition
Draw(mesh, clipping={"function": True,  "pnt": (0.5, 0.5, 0.5), "vec": (0, 1, 0)})
print(mesh.GetBoundaries())

# %%
# Assign names to regions
new_mesh.SetMaterial(1, "ecs")
new_mesh.SetMaterial(2, "cytosol")

print(mesh.GetMaterials())

# %%
