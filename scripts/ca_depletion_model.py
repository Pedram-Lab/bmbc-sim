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

from ecsim.geometry import create_ca_depletion_mesh

# %%
mesh = create_ca_depletion_mesh(side_length=3, cytosol_height=3, ecs_height=0.1, mesh_size=0.25)

# %%
Draw(mesh)

# %%
fes = H1(mesh, order=2, dirichlet="ecs_top")
u = fes.TrialFunction()
v = fes.TestFunction()

f = LinearForm(fes)
f += 0 * v * dx
f += 1 * v.Trace() * ds(definedon="channel")

a = BilinearForm(fes)
a += grad(u) * grad(v) * dx

a.Assemble()
f.Assemble()

# %%
concentration = GridFunction(fes)
concentration.Set(15, definedon=mesh.Boundaries("ecs_top"))
res = f.vec.CreateVector()
res.data = f.vec - a.mat * concentration.vec
concentration.vec.data += a.mat.Inverse(fes.FreeDofs()) * res
Draw(concentration)

# %%
