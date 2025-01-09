# %% [markdown]
# # Simulate diffusion through a membrane
# We have two 1um x 1um x 1um boxes separated by a semi-permeable membrane. Initially, the left box contains 2mM calcium
# and the right box contains 0mM calcium. We then simulate the diffusion of calcium through the membrane.
# The diffusion coefficient is set high compared to the flux through the membrane to make the simulation comparable to
# a simple dynamical system involving two compartments with exchange in between the two.

# %%
from math import ceil

import astropy.units as u
import numpy as np
from netgen.occ import Box, Glue, Pnt, OCCGeometry
from ngsolve import Mesh, TaskManager, GridFunction, Integrate
from ngsolve.webgui import Draw
from tqdm.notebook import trange
import matplotlib.pyplot as plt

from ecsim.simulation import Simulation
from ecsim.units import LENGTH, convert

# %%
s = convert(1 * u.um, LENGTH) / 2
mesh_size = convert(0.1 * u.um, LENGTH)
left = Box(Pnt(-2 * s, -s, -s), Pnt(0, s, s)).mat("left").bc("reflective")
right = Box(Pnt(0, -s, -s), Pnt(2 * s, s, s)).mat("right").bc("reflective")

geo = Glue([left, right])
geo.faces[1].bc("interface")
mesh = Mesh(OCCGeometry(geo).GenerateMesh(maxh=mesh_size))

# %%
# Set up a simulation
time_step = 10 * u.us
t_end = 20.0 * u.ms
n_samples = 100
simulation = Simulation(mesh, time_step=time_step, t_end=t_end)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"left": 10.0 * u.um**2 / u.ms, "right": 10.0 * u.um**2 / u.ms},
)
simulation.add_channel_flux(
    left="left",
    right="right",
    boundary="interface",
    rate= 0.1 * u.um / u.ms
)


# %%
# Time stepping - define a function that computes all time steps
def time_stepping(simulation, n_samples):
    sample_int = int(ceil(simulation.n_time_steps / n_samples))
    ca_t = GridFunction(simulation._fes, multidim=0)
    ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
    
    for i in trange(simulation.n_time_steps):
        simulation.time_step()
        if i % sample_int == 0:
            ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
    return ca_t


# %%
# Time stepping - set initial conditions and do time stepping
simulation.setup_problem()
with TaskManager():
    simulation.init_concentrations(calcium={"left": 2 * u.mmol / u.L, "right": 0 * u.mmol / u.L})
    ca_t = time_stepping(simulation, n_samples=n_samples)

# %%
# Visualize
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
# Draw(ca_t.components[0], settings=settings, interpolate_multidim=True, animate=True)

# %%
# Evaluate the concentration in both compartments
concentration_left = np.zeros(n_samples)
concentration_right = np.zeros(n_samples)

time = np.arange(n_samples) * (t_end / n_samples).to(u.ms).value
for i in range(n_samples):
    concentration_left[i] = Integrate(ca_t.components[0].MDComponent(i), mesh, definedon=mesh.Materials("left"))
    concentration_right[i] = Integrate(ca_t.components[1].MDComponent(i), mesh, definedon=mesh.Materials("right"))

# Plot the results
plt.plot(time, concentration_left, marker='o', linestyle='-', color='blue', label="Left")
plt.plot(time, concentration_right, marker='x', linestyle='-', color='red', label="Right")
plt.title(r"$[\mathrm{Ca}^{2+}]$ amount over time")
plt.xlabel(r"Time ($\mathrm{ms}$)")
plt.ylabel(r"Amount of $\mathrm{Ca}^{2+}$ (amol)")
plt.grid(True)
plt.show()

# %%

# %%
