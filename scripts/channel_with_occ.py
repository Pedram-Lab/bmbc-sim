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

# %%
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw

# %%
# Define the sides of a cube
box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
top = box.faces[5]
bottom = box.faces[4]
front = box.faces[2]
back = box.faces[3]
left = box.faces[0]
right = box.faces[1]

# %%
for f in [bottom, front, back, left, right]:
    f.bc("reflective")

channel = Face(Wire(Circle(Pnt(0.5, 0.5, 1), Z, 0.1)))
channel.maxh = 0.01
membrane = (top - channel).bc("membrane")
channel.bc("channel")
geo = Fuse([Glue([membrane, channel]), bottom, front, back, left, right])
Draw(geo)

# %%
ngmesh = OCCGeometry(geo).GenerateMesh()
ngmesh.GenerateVolumeMesh()
mesh = Mesh(ngmesh)
Draw(mesh)
print(mesh.GetBoundaries())
print(f"Number of 3D elements: {mesh.ne}")


# %%
