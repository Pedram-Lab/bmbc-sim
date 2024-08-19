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

# %% [markdown]
# # Post-processing and Visualization
# This script shows how to use the `LineEvaluator` class to plot calcium traces in both the cytosol and the extracellular space.

# %%
from ngsolve import *
from ngsolve.webgui import Draw
from tqdm.notebook import trange

from ecsim.geometry import create_ca_depletion_mesh, LineEvaluator
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import csv

# %%
# Set to True to write out the results as CSV files
write_as_csv = False

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(side_length=3, cytosol_height=3, ecs_height=0.1, mesh_size=0.25, channel_radius=0.5)

# %%
# Define and assemble the FE-problem
ecs_fes = H1(mesh, order=2, definedon=mesh.Materials("ecs"), dirichlet="ecs_top")
cytosol_fes = H1(mesh, order=2, definedon=mesh.Materials("cytosol"))
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
# Time stepping - set up system matrix
m = BilinearForm(fes)
m += u_ecs * v_ecs * dx("ecs")
m += u_cyt * v_cyt * dx("cytosol")
m.Assemble()

dt = 0.001
mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
mstar_inv = mstar.Inverse(freedofs=fes.FreeDofs())


# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(u, t_end, n_samples):
    n_steps = int(ceil(t_end / dt))
    sample_int = int(ceil(n_steps / n_samples))
    u_t = GridFunction(u.space, multidim=0)
    u_t.AddMultiDimComponent(u.vec)
    
    for i in trange(n_steps):
        res = dt * (f.vec - a.mat * u.vec)
        u.vec.data += mstar_inv * res
        if i % sample_int == 0:
            u_t.AddMultiDimComponent(u.vec)
    return u_t


# Time stepping - set initial conditions and do time stepping
with TaskManager():
    concentration = GridFunction(fes)
    concentration.components[0].Set(15)
    c_t = time_stepping(concentration, t_end=1, n_samples=100)

# %% [markdown]
# # Cytosolic calcium dynamics along a line segment bewteen two points


# %%
# Define the constant values for y and z
y_cyt = 1.5  # Constant value for y
z_cyt = 2.8  # Constant value for z

# Define the range and number of points for x
x_start_cyt = 0.0  # Start of the x range
x_end_cyt = 1.5    # End of the x range
n_points_cyt = 50  # Number of points in the x range

# Create the line evaluator using the LineEvaluator class
line_evaluator_cyt = LineEvaluator(
    mesh, 
    (x_start_cyt, y_cyt, z_cyt),  # Start point (x, y, z)
    (x_end_cyt, y_cyt, z_cyt),    # End point (x, y, z)
    n_points_cyt  # Number of points to evaluate
)

# Evaluate the concentration in the cytosol
concentrations_cyt = line_evaluator_cyt.evaluate(concentration.components[1])

# Get the x-coordinates for the plot
x_coords = line_evaluator_cyt.raw_points[:, 0]  # Extract the x-coordinates

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_coords, concentrations_cyt, marker='o', linestyle='-', color='blue')
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$ (nM)")
plt.grid(True)
plt.show()


# %%
if write_as_csv:
    x_coords = line_evaluator_cyt.raw_points[:, 0]
    with open('concentration_cyt_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Distance from channel cluster (um)', 'Cytosolic calcium (nM)'])  # Column heads 
        for x_coord, val in zip(x_coords, concentrations_cyt):
            writer.writerow([x_coord, val])


# %% [markdown]
# # Extracellular calcium dynamic along a line segment bewteen two points

# %%
# Define the constant values for y and z
y_ecs = 1.5  # Constant value for y
z_ecs = 3.005  # Constant value for z

# Define the range and number of points for x
x_start_ecs = 0.0  # Start of the x range
x_end_ecs = 1.5    # End of the x range
n_points_ecs = 50  # Number of points in the x range

# Create the line evaluator using the LineEvaluator class
line_evaluator_ecs = LineEvaluator(
    mesh, 
    (x_start_ecs, y_ecs, z_ecs),  # Start point (x, y, z)
    (x_end_ecs, y_ecs, z_ecs),    # End point (x, y, z)
    n_points_ecs  # Number of points to evaluate
)

# Evaluate the concentration in the extracellular space (ECS)
concentrations_ecs = line_evaluator_ecs.evaluate(concentration.components[0])

# Get the x-coordinates for the plot
x_coords_ecs = line_evaluator_ecs.raw_points[:, 0]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_coords_ecs, concentrations_ecs, marker='o', linestyle='-', color='red')
plt.ylim([14.5, 15.1])
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ (mM)")
plt.grid(True)
plt.show()


# %%
if write_as_csv:
    with open('concentration_ecs_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Distance from channel cluster (um)', 'Extracellular calcium (mM)'])  # Column heads 
        for x_coord, val in zip(x_coords_ecs, concentrations_ecs):
            writer.writerow([x_coord, val])

# %%
