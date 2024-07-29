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

# Create a 3x3x3 unit cube with label 'cyt'
cytosol = OrthoBrick(Pnt(0, 0, 0), Pnt(3, 3, 3)).bc("cyt")

# Create a cylinder for the hole with depth 0.2 units on the bottom face
hole_cyt = Cylinder(Pnt(1.5, 1.5, 0), Pnt(1.5, 1.5, 0.2), 0.5) * Plane(Pnt(1.5, 1.5, 2.9), Vec(0, 0, -1))

# Create the ECS outer volume with label 'ecs'
ecs = OrthoBrick(Pnt(0, 0, 3), Pnt(3, 3, 3.5)).bc("ecs")

# Create a cylinder for the hole with depth 0.2 units in the ECS
hole_ecs = Cylinder(Pnt(1.5, 1.5, 3), Pnt(1.5, 1.5, 3.1), 0.5) * Plane(Pnt(1.5, 1.5, 3.1), Vec(0, 0, 1))

# Create the geometry and subtract the cylinders from the corresponding volumes
geo = CSGeometry()
geo.Add((cytosol - hole_cyt).mat("cyt"))
geo.Add((ecs - hole_ecs).mat("ecs"))

# Generate the mesh
mesh = geo.GenerateMesh(maxh=0.25)

# Convert Netgen mesh to NGSolve
ngmesh = Mesh(mesh)

# View the mesh
Draw(ngmesh)

#Define materials with values
material_dict = {
    "cyt": 0.1,
    "ecs": 15
}

# Create a coefficient function that assigns different values ​​depending on the material
cf = CoefficientFunction([material_dict[mat] for mat in ngmesh.GetMaterials()])

#Visualize the coefficient function
Draw(cf, ngmesh, "Material Coefficient")


# %%
print("Boundaries:", ngmesh.GetBoundaries())

# %%
