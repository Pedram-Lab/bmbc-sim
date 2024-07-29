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
from netgen.occ import *
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.meshing import IdentificationType

c1 = Circle((0,0),1).Face()
c2 = Circle((0,0),1.1).Face()

trf = gp_Trsf.Scale((0,0,0), 1.1)

# 3 is close surface identification
c1.edges.Identify(c2.edges, "cs", 3, trf)

c2 -= c1

shape = Glue([c1, c2])

geo = OCCGeometry(shape)
mesh = Mesh(geo.GenerateMesh(maxh=0.1))
Draw(mesh)

# %%
