# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
  
    channel = Cylinder ( Pnt(0.5, 0.5, 0), Pnt(0.5, 0.5, 1), 0.2)
    
    geometry.Add(channel)

    return geometry

ngmesh = MakeGeometry().GenerateMesh(maxh=0.05)
mesh = Mesh(ngmesh)
Draw(mesh)

# %%
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

#cube = OrthoBrick( Pnt(0,0,0), Pnt(3,3,3) )
cube_cutting = OrthoBrick( Pnt(1.4,1.4,2.8), Pnt(1.6,1.6,3) )
hole = Cylinder ( Pnt(1.5, 1.5, 0), Pnt(1.5, 1.5, 1), 0.1)

geo = CSGeometry()
geo.Add (cube-hole)
mesh = geo.GenerateMesh(maxh=0.1)
mesh.Save("cube_hole.vol")
Draw(mesh)

# %%
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

cube = OrthoBrick( Pnt(0,0,0), Pnt(3,3,3) )
hole = Cylinder ( Pnt(1.5, 1.5, 0), Pnt(1.5, 1.5, 1), 0.1)

geo = CSGeometry()
geo.Add (cube-hole)
mesh = geo.GenerateMesh(maxh=0.1)
#mesh.Save("cube_hole.vol")
Draw(mesh)

# %%
