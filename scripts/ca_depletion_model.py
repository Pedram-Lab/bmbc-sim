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
# This script takes the `geometry from scripts/channel_geometry_with_occ.py` and adds reaction-diffusion of chemical species on top:
# * Ca can diffuse from the ECS to the cytosol through the channel.

# %%
from ngsolve import *
from ngsolve.webgui import Draw

from ecsim.geometry import create_ca_depletion_mesh

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(side_length=3, cytosol_height=3, ecs_height=0.1, mesh_size=0.25, channel_radius=0.5)
print(mesh.GetBoundaries())

# %%
clipping = {"function": True,  "pnt": (0, 0, 1.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -60}]}}
Draw(mesh, clipping=clipping, settings=settings)

# %%
# Define and assemble the FE-problem
# We set the cytosol boundary to zero for visualization purposes
ecs_fes = H1(mesh, order=2, definedon=mesh.Materials("ecs"), dirichlet="ecs_top")
cytosol_fes = H1(mesh, order=2, definedon=mesh.Materials("cytosol"), dirichlet="boundary")
fes = FESpace([ecs_fes, cytosol_fes])
u_ecs, u_cyt = fes.TrialFunction()
v_ecs, v_cyt = fes.TestFunction()

f = LinearForm(fes)

a = BilinearForm(fes)
a += grad(u_ecs) * grad(v_ecs) * dx("ecs")              # diffusion in ecs
a += grad(u_cyt) * grad(v_cyt) * dx("cytosol")          # diffusion in cytosol
a += (u_ecs - u_cyt) * (v_ecs - v_cyt) * ds("channel")  # interface flux

a.Assemble()
f.Assemble()

# %%
# Set concentration at top to 15 and solve the system
concentration = GridFunction(fes)
concentration.components[0].Set(15, definedon=mesh.Boundaries("ecs_top"))
res = f.vec.CreateVector()
res.data = f.vec - a.mat * concentration.vec
concentration.vec.data += a.mat.Inverse(fes.FreeDofs()) * res

# %%
# Visualize (the colormap is quite extreme for dramatic effect)
visualization = mesh.MaterialCF({"ecs": concentration.components[0], "cytosol": concentration.components[1]})
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 256, "autoscale": False, "max": 3}}
Draw(visualization, mesh, clipping=clipping, settings=settings)

# %%
