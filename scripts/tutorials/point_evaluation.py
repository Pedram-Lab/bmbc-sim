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
import csv

import matplotlib.pyplot as plt
from astropy import units as au
from ngsolve import *
from tqdm.notebook import trange

from ecsim.geometry import create_ca_depletion_mesh, LineEvaluator
from ecsim.simulation import Simulation

# %%
# Set to True to write out the results as CSV files
write_as_csv = False

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(
    side_length_x=3 * au.um,
    side_length_y=3 * au.um,
    cytosol_height=3 * au.um,
    ecs_height=0.1 * au.um,
    mesh_size=0.25 * au.um,
    channel_radius=0.5 * au.um
)

# %%
# Define and assemble the FE-problem
simulation = Simulation(mesh, time_step=1 * au.ms, t_end=1 * au.s)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"ecs": 1 * au.um**2 / au.s, "cytosol": 1 * au.um**2 / au.s},
    clamp={"ecs_top": 15 * au.mole / au.liter}
)
simulation.add_channel_flux(
    left="ecs",
    right="cytosol",
    boundary="channel",
    rate=1 * au.millimole / au.s
)

# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(sim: Simulation, n_samples: int):
    n_steps = sim.n_time_steps
    sample_int = int(ceil(n_steps / n_samples))
    u_t = GridFunction(sim._fes_rd, multidim=0)
    u_t.AddMultiDimComponent(sim.concentrations["calcium"].vec)
    
    for i in trange(n_steps):
        sim.time_step()
        if i % sample_int == 0:
            u_t.AddMultiDimComponent(sim.concentrations["calcium"].vec)
    return u_t


# %%
# Time stepping - set initial conditions and do time stepping
with TaskManager():
    simulation.setup_problem()
    simulation.init_concentrations(
        calcium={"ecs": 15 * au.mole / au.liter, "cytosol": 0.1 * au.millimole / au.liter},
    )
    c_t = time_stepping(simulation, n_samples=100)

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
concentrations_cyt = line_evaluator_cyt.evaluate(simulation.concentrations["calcium"].components[1])

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
concentrations_ecs = line_evaluator_ecs.evaluate(simulation.concentrations["calcium"].components[0])

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
