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

# Crear un cubo de 3x3x3 unidades
cube = OrthoBrick(Pnt(0, 0, 0), Pnt(3, 3, 3)).bc("cube")

# Crear un cilindro para el agujero con profundidad 0.2 unidades en la cara inferior
hole = Cylinder(Pnt(1.5, 1.5, 0), Pnt(1.5, 1.5, 0.2), 0.5).bc("hole") * Plane(Pnt(1.5, 1.5, 2.9), Vec(0, 0, -1))

# Crear la geometría y restar el cilindro del cubo
geo = CSGeometry()
geo.Add(cube - hole)

# Generar la malla
mesh = geo.GenerateMesh(maxh=0.25)

# Convertir la malla de Netgen a NGSolve
ngmesh = Mesh(mesh)

# Visualizar la malla
Draw(ngmesh)

# Obtener las fronteras
boundaries = ngmesh.GetBoundaries()

# Imprimir información de las fronteras
print("Number of boundaries:", len(boundaries))
for bc in boundaries:
    print("Boundary:", bc)


# %%
