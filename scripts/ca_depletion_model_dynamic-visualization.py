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
# This script:
# * Takes the `geometry from scripts/channel_geometry_with_occ.py`
# * Adds reaction-diffusion of chemical species on top (units are handled by astropy)
# * Diffuses calcium from the ECS to the cytosol through the channel
# * Resolves the dynamics of the system in time
# * Visualizes the solution on the tetahedral mesh  
# * Displays a plot of the data along a line segment within two points in the geometry

# %%
from ngsolve import *
from ngsolve.webgui import Draw
from tqdm import trange

from ecsim.geometry import create_ca_depletion_mesh
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import csv

# %%
#Parameters and units handle by astropy
ca_ext = 15 * u.millimole
ca_cyt = 0.0001 * u.millimole
egta_1 = 4.5 * u.millimole
egta_2 = 40 * u.millimole
bapta = 1 * u.millimole
diff_ca_ext = 600 * u.um**2 / u.s
diff_ca_cyt = 220 * u.um**2 / u.s
diff_free_egta = 113 * u.um**2 / u.s
diff_bound_egta = 113 * u.um**2 / u.s
diff_free_bapta = 95 * u.um**2 / u.s
diff_bound_bapta = 113 * u.um**2 / u.s
k_f_egta = 2.7 * u.micromole / u.s
k_r_egta = 0.5 / u.s
k_f_bapta = 450 * u.micromole / u.s
k_r_bapta = 80 / u.s
diameter_ch = 10 * u.nm
density_channel = 10000 / u.um**2
i_max = 0.1 * u.picoampere

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


# %%
# Time stepping - set initial conditions and do time stepping
concentration = GridFunction(fes)
concentration.components[0].Set(15)
c_t = time_stepping(concentration, t_end=1, n_samples=100)

# %%
# Visualize (because of the product structure of the FESpace, the usual
# visualization of time-dependent functions via multidim is not possible)
visualization = mesh.MaterialCF({"ecs": c_t.components[0], "cytosol": c_t.components[1]})
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
Draw(c_t.components[1], settings=settings, interpolate_multidim=True, animate=True)

# %% [markdown]
# # Cytosolic calcium dynamics along a line segment bewteen two points

# %%
# This code displays a plot of the data along a line segment between two points (x_start and x_end) on the x-axis, 
# while keeping the y and z coordinates constant throughout the simulation.

# Define constant values for y and z
y_constant = 1.5  #  Constant value for y
z_constant = 2.8  #  Constant value for z

# Define the range and number of points for x
x_start = 0.0     # Inicio del rango de x
x_end = 1.5       # End of the x range
n_points = 50     # Número de puntos en el rango de x

# Generate the x coordinates
x_coord = np.linspace(x_start, x_end, n_points)
print(x_coord)

# Create the full coordinates by combining x with constant y and z
coordinates = np.array([(x, y_constant, z_constant) for x in x_coord])

# print the coordinates
print(coordinates)

# %%
# Define arrays to store concentration values
concentrations_cyt = []

# Calculate the concentration at the cytosol at each point defined previously in coordinates
for point in coordinates:
    mip = mesh(*point)
    concentrations_cyt.append(concentration.components[1](mip))
    print(concentrations_cyt)



# %%
# Plot concentration vs x coordinate
plt.figure(figsize=(10, 6))
plt.plot(x_coord, concentrations_cyt, marker='o', linestyle='-', color='blue')
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$ (nM)")
plt.grid(True)

#Save the plot
plt.savefig('ca_cyt_vs_distance.png')
plt.show()

# %%
with open('concentration_cyt_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Distance from channel cluster (um)', 'calcium_cyt'])  # Column heads 
    for x_coord, concentration_cyt in zip(x_coord, concentrations_cyt):
        writer.writerow([x_coord, concentration_cyt])

# %% [markdown]
# # Extracellular calcium dynamic along a line segment bewteen two points

# %%
# This code displays a plot of the data along a line segment between two points (x_start and x_end) on the x-axis, 
# while keeping the y and z coordinates constant throughout the simulation.

# Define constant values for y and z
y_constant_ecs = 1.5  #  Constant value for y
z_constant_ecs = 3.005  #  Constant value for z

# Define the range and number of points for x
x_start_ecs = 0.0     # Inicio del rango de x
x_end_ecs = 1.5       # End of the x range
n_points_ecs = 50     # Número de puntos en el rango de x

# Generate the x coordinates
x_coordinates_ecs = np.linspace(x_start_ecs, x_end_ecs, n_points_ecs)

# Create the full coordinates by combining x with constant y and z
coordinates_ecs = np.array([(x, y_constant_ecs, z_constant_ecs) for x in x_coordinates_ecs])

# print the coordinates
print(coordinates_ecs)

# %%
# Calculate the concentration at the extracellular at the coordinates defined previously
for point in coordinates_ecs:
    mip_ecs = mesh(*point)
    print(f"Concentration at {point}: {concentration.components[0](mip_ecs)}")

# %%
# Define arrays to store concentration values at the extracellular space
concentrations_ext = []

# Calculate the concentration at the extracellular at the coordinates defined previously
for point in coordinates_ecs:
    mip_ecs = mesh(*point)
    #print(f"Concentration at {point}: {concentration.components[0](mip_ecs)}")
    concentrations_ext.append(concentration.components[0](mip_ecs))
    print(concentrations_ext)


# %%

# Plot concentration vs x coordinate
plt.figure(figsize=(10, 6))
plt.plot(x_coordinates_ecs, concentrations_ext, marker='o', linestyle='-', color='red')
plt.ylim([14.5, 15.1])
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$) ")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ (mM)")
plt.grid(True)
plt.show()

# %%
with open('calcium_ecs_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Distance from channel cluster (um)', 'calcium_ecs'])  # Columns head
    for x_coord_ecs, concentration_ecs in zip(x_coordinates_ecs, concentrations_ext):
        writer.writerow([x_coord_ecs, concentration_ecs])
