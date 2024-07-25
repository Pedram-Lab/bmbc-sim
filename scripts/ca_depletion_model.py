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


# %%
def MakeGeometry(side_length, cytosol_height, ecs_height):

    geometry = CSGeometry()
    
    cytosol_left  = Plane (Pnt(0,0,0), Vec(-1,0,0) ).bc("cytosol_left")
    cytosol_right = Plane (Pnt(side_length,side_length,cytosol_height), Vec( 1,0,0) )
    cytosol_front = Plane (Pnt(0,0,0), Vec(0,-1,0) )
    cytosol_back  = Plane (Pnt(side_length,side_length,cytosol_height), Vec(0, 1,0) )
    cytosol_bot   = Plane (Pnt(0,0,0), Vec(0,0,-1) )
    cytosol_top   = Plane (Pnt(side_length,side_length,cytosol_height), Vec(0,0, 1) )
    cytosol = cytosol_left * cytosol_right * cytosol_front * cytosol_back * cytosol_bot * cytosol_top

    ecs_left  = Plane (Pnt(0,0,cytosol_height), Vec(-1,0,0) )
    ecs_right = Plane (Pnt(side_length,side_length,ecs_height), Vec( 1,0,0) )
    ecs_front = Plane (Pnt(0,0,cytosol_height), Vec(0,-1,0) )
    ecs_back  = Plane (Pnt(side_length,side_length,ecs_height), Vec(0, 1,0) )
    ecs_bot   = Plane (Pnt(0,0,cytosol_height), Vec(0,0,-1) ).bc("ecs_bot")
    ecs_top   = Plane (Pnt(side_length,side_length,ecs_height), Vec(0,0, 1) )
    ecs = ecs_left * ecs_right * ecs_front * ecs_back * ecs_bot * ecs_top

    geometry.NameEdge

    geometry.Add(cytosol + ecs)
    return geometry


# %%
ngmesh = MakeGeometry(side_length=3, cytosol_height=3, ecs_height=3.1).GenerateMesh(maxh=0.25)

# %%
mesh = Mesh(ngmesh)

# %%
Draw (mesh)

# %%
print("Bnd = ", mesh.GetBoundaries())

# %%
mesh.ngmesh.SetBCName(0, "other_name")
print("Bnd = ", mesh.GetBoundaries())

# %%
