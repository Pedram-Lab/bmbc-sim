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
from ngsolve.webgui import Draw
from netgen.csg import *

#Create a 3x3x3 unit cube
cube = OrthoBrick(Pnt(0, 0, 0), Pnt(3, 3, 3))

# Create a cylinder for the hole with depth 0.2 units on the bottom face
hole = Cylinder(Pnt(1.5, 1.5, 0), Pnt(1.5, 1.5, 0.2), 0.5) * Plane(Pnt(1.5, 1.5, 2.9), Vec(0, 0, -1))

# Creating the geometry for ECS (Extracellular Space)
#ecs = OrthoBrick(Pnt(0, 0, -0.1), Pnt(3, 3, 0)).bc("ecs")

# Create the geometry and subtract the cylinder from the cube
geo = CSGeometry()
geo.Add(cube - hole)
#geo.Add((cube - hole) + ecs)

# Generate the mesh
mesh = geo.GenerateMesh(maxh=0.25)

# Convert Netgen mesh to NGSolve
ngmesh = Mesh(mesh)

# View the mesh
Draw(ngmesh)


# %%
