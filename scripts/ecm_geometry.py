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
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw


def MakeGeometry():
    geometry = CSGeometry()
    
    cytosol = OrthoBrick(Pnt(0,0,0),Pnt(3,3,3)).bc("cytosol")
    ecs = OrthoBrick(Pnt(0,0,3),Pnt(3,3,3.1)).bc("ecs")
    
    #channel = Cylinder ( Pnt(0.5, 0.5, 0), Pnt(0.5, 0.5, 1), 0.2)
    
    #geometry.Add(ecs)
    #geometry.Add(cytosol)
    #geometry.Add(channel)
    #geometry.Add(ecs+cytosol+channel)
    geometry.Add ((ecs-cytosol)+cytosol)
    return geometry

ngmesh = MakeGeometry().GenerateMesh(maxh=0.5)
mesh = Mesh(ngmesh)
Draw(mesh)

# %%
print (mesh.GetMaterials(), mesh.GetBoundaries())

# %%
print("Bnd = ", mesh.GetBoundaries())

# %%
