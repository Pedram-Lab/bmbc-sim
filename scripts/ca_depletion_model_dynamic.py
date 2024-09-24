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
# This script takes the geometry from `scripts/channel_geometry_with_occ.py` and adds reaction-diffusion of chemical species on top (units are handled by astropy):
#
#
# \begin{equation}
#     \mathrm{Ca}^{2+}_{\mathrm{cyt}} + \mathrm{EGTA} \rightleftharpoons  \mathrm{EGTA}_{\mathrm{bound}}
# \end{equation}
#
#
# \begin{aligned}
#     \frac{\partial [\text{Ca}^{2+}]_{\text{cyt}}}{\partial t} &= D_{\text{Ca}_{\text{cyt}}} \Delta [\text{Ca}^{2+}]_{\text{cyt}} - k_{\text{EGTA}_+} [\text{Ca}^{2+}]_{\text{cyt}} [\text{EGTA}] + k_{\text{EGTA}_-} [\text{EGTA-Ca}], \\
#     \frac{\partial [\text{EGTA}]}{\partial t} &= D_{\text{EGTA}_{\text{free}}} \Delta [\text{EGTA}] - k_{\text{EGTA}_+} [\text{Ca}^{2+}]_{\text{cyt}} [\text{EGTA}] + k_{\text{EGTA}_-} [\text{EGTA-Ca}], \\
#     \frac{\partial [\text{EGTA-Ca}]}{\partial t} &= D_{\text{EGTA}_{\text{bound}}} \Delta [\text{Ca}^{2+}]_{\text{cyt}} + k_{\text{EGTA}_+} [\text{Ca}^{2+}]_{\text{cyt}} [\text{EGTA}] - k_{\text{EGTA}_-} [\text{EGTA-Ca}].
# \end{aligned}
#
#
#
# *Table: Parameter values*
#
# | Parameter               | Meaning                                   | Value                               | 
# |-------------------------|-------------------------------------------|-------------------------------------|
# | D$^{ext}_{Ca}$      | Diffusion of extracellular Ca$^{2+}$          | 600 μm²/s                           |
# | D$^{int}_{Ca}$      | Diffusion of intracellular Ca$^{2+}$          | 220 μm²/s                           |
# | D$^{free}_{EGTA}$   | Diffusion of free EGTA                        | 113 μm²/s                           | 
# | D$^{bound}_{EGTA}$  | Diffusion of EGTA bound to Ca$^{2+}$          | 113 μm²/s                           |
# | D$^{free}_{BAPTA}$  | Diffusion of free BAPTA                       | 95 μm²/s                            |
# | D$^{bound}_{BAPTA}$ | Diffusion of BAPTA bound to Ca$^{2+}$         | 95 μm²/s                            | 
# | k$^{+}_{EGTA}$      | Forward rate constant for EGTA                | 2.7 μM⁻¹·s⁻¹                        | 
# | k$^{-}_{EGTA}$      | Reverse rate constant for EGTA                | 0.5 s⁻¹                             | 
# | k$^{+}_{BAPTA}$     | Forward rate constant for BAPTA               | 450 μM⁻¹·s⁻¹                        |
# | k$^{-}_{BAPTA}$     | Reverse rate constant for BAPTA               | 80 s⁻¹                              |
#
# *Table: Initial conditions*
#
# | Specie        | Value                          |
# |---------------|--------------------------------|
# | $\text{Ca}^{2+}_{\text{ext}}$ | 15 mM          |
# | $\text{Ca}^{2+}_{\text{cyt}}$ | 0.0001 mM      |
# | EGTA          | 4.5 mM / 40 mM                 |
# | BAPTA         | 1 mM                           |
#
#
# Dimensions:
#
# ECS= 3 μm x 3 μm x 0.1 μm
#
# cytosol = 3 μm x 3 μm x 3 μm
#
# channel$_{radius}$ = 5 nm
#
#
# * Ca can diffuse from the ECS to the cytosol through the channel.
# * The dynamics of the system are resolved in time.

# %%
from math import ceil
import csv

import matplotlib.pyplot as plt
from ngsolve import GridFunction, TaskManager, SetNumThreads
from ngsolve.webgui import Draw
from tqdm.notebook import trange
from astropy import units as u

from ecsim.geometry import create_ca_depletion_mesh, LineEvaluator
from ecsim.simulation import Simulation

# %%
# Create meshed geometry
mesh = create_ca_depletion_mesh(
    side_length=3 * u.um,
    cytosol_height=3 * u.um,
    ecs_height=0.1 * u.um,
    mesh_size=0.25 * u.um,
    channel_radius=5 * u.nm
)

# %%
# Set up a simulation on the mesh with BAPTA as a buffer
simulation = Simulation(mesh, time_step=1 * u.us, t_end=20 * u.ms)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"ecs": 600 * u.um**2 / u.s, "cytosol": 220 * u.um**2 / u.s},
    clamp={"ecs_top": 15 * u.mmol / u.L}
)
free_buffer = simulation.add_species(
    "free_buffer",
    #diffusivity={"cytosol": 95 * u.um**2 / u.s} #BAPTA
    diffusivity={"cytosol": 113 * u.um**2 / u.s} #EGTA
)
bound_buffer = simulation.add_species(
    "bound_buffer",
    #diffusivity={"cytosol": 95 * u.um**2 / u.s} #BAPTA
    diffusivity={"cytosol": 113 * u.um**2 / u.s} #EGTA
)
simulation.add_reaction(
    reactants=(calcium, free_buffer),
    products=bound_buffer,
    kf={"cytosol": 450 / (u.s * u.umol / u.L)}, #BAPTA
    kr={"cytosol": 80 / u.s} #BAPTA
    #kf={"cytosol": 2.7 / (u.s * u.umol / u.L)}, #EGTA
    #kr={"cytosol": 0.5 / u.s} #EGTA
)
simulation.add_channel_flux(
    left="ecs",
    right="cytosol",
    boundary="channel",
    rate = 1e16 * u.um / u.ms
    # old values:
    # rate= 7.04e+03 * u.mmol / (u.L * u.s)  # VOLUME 1
    # rate= 1.86e+09 * u.mmol / (u.L * u.s) #VOLUME 2
)

# %%
# Internally set up all finite element infrastructure
simulation.setup_problem()


# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(simulation, n_samples):
    sample_int = int(ceil(simulation.n_time_steps / n_samples))
    u_ca_t = GridFunction(simulation._fes, multidim=0)
    u_buf_t = GridFunction(simulation._fes, multidim=0)
    u_com_t = GridFunction(simulation._fes, multidim=0)
    u_ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
    u_buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
    u_com_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    
    for i in trange(simulation.n_time_steps):
        simulation.time_step()
        if i % sample_int == 0:
            u_ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
            u_buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
            u_com_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    return u_ca_t, u_buf_t, u_com_t


# %%
# Time stepping - set initial conditions and do time stepping
SetNumThreads(12)
with TaskManager():
    simulation.init_concentrations(
        calcium={"ecs": 15 * u.mmol / u.L, "cytosol": 0.1 * u.umol / u.L},
        free_buffer={"cytosol": 1 * u.mmol / u.L}, # BAPTA
        #free_buffer={"cytosol": 4.5 * u.mmol / u.L}, # low EGTA
        #free_buffer={"cytosol": 40 * u.mmol / u.L}, # high EGTA
        bound_buffer={"cytosol": 0 * u.mmol / u.L}
    )
    ca_t, buffer_t, complex_t = time_stepping(simulation, n_samples=100)

# %%
# Visualize (because of the product structure of the FESpace, the usual
# visualization of time-dependent functions via multidim is not possible)
visualization = mesh.MaterialCF({"ecs": ca_t.components[0], "cytosol": ca_t.components[1]})
clipping = {"function": True,  "pnt": (0, 0, 1.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
Draw(ca_t.components[1], clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)

# %%
# Create a line evaluator that evaluates a line away from the channel in the cytosol
line_evaluator_cyt = LineEvaluator(
    mesh,
    (0.0, 0.0, 2.995),  # Start point (x, y, z)
    (0.6, 0.0, 2.995),  # End point (x, y, z)
    60  # Number of points to evaluate
)

# Evaluate the concentration in the cytosol
step = 100
ca_cyt = line_evaluator_cyt.evaluate(ca_t.components[1].MDComponent(step))
buffer_cyt = line_evaluator_cyt.evaluate(buffer_t.components[1].MDComponent(step))
complex_cyt = line_evaluator_cyt.evaluate(complex_t.components[1].MDComponent(step))

# Get the x-coordinates for the plot
x_coords = line_evaluator_cyt.raw_points[:, 0]  # Extract the x-coordinates

# Plot the results
plt.plot(x_coords, ca_cyt, marker='o', linestyle='-', color='red', label='$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$')
plt.plot(x_coords, buffer_cyt, marker='x', linestyle='-', color='blue', label='$[\mathrm{BAPTA}]_{\mathrm{cyt}}$')
plt.plot(x_coords, complex_cyt, marker='s', linestyle='-', color='magenta', label='$[\mathrm{CaBAPTA}]_{\mathrm{cyt}}$')
plt.title(r"Concentration vs distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"Concentration (mM)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Save the values in a CSV file
# with open('calcium_concentrations_cyt_egta_high_vol2.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['x_coords', 'concentrations_cyt'])
#     for x, conc in zip(x_coords, concentrations_cyt):
#         csvwriter.writerow([x, conc])

# %%
# Create a line evaluator that evaluates a line in the extracellular space (ECS)
line_evaluator_ecs = LineEvaluator(
    mesh,
    (0.0, 0.0, 3.005),  # Start point (x, y, z)
    (0.6, 0.0, 3.005),  # End point (x, y, z)
    60  # Number of points to evaluate
)

# Evaluate the concentration in the extracellular space (ECS)
concentrations_ecs = line_evaluator_ecs.evaluate(simulation.concentrations["calcium"].components[0])

# Get the x-coordinates for the plot
x_coords_ecs = line_evaluator_ecs.raw_points[:, 0]

# Plot the results
plt.plot(x_coords_ecs, concentrations_ecs, marker='o', linestyle='-', color='red')
plt.ylim([14.5, 15.1])
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ (mM)")
plt.grid(True)
plt.show()

# %%
