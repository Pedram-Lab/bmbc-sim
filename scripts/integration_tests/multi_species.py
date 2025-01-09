# %% [markdown]
# # Simulate a simple diffusion model including multiple species 
# We start with 0mM calcium in a 1um x 1um x 1um box, which is clamped to 1mM on one side. We then simulate the
# diffusion of calcium in the box with a buffer (BAPTA) being present, but not reacting with the calcium.

# %%
from math import ceil

import astropy.units as u
from netgen.csg import OrthoBrick, CSGeometry
from ngsolve import Mesh, TaskManager, GridFunction
from ngsolve.webgui import Draw
from tqdm.notebook import trange
import matplotlib.pyplot as plt

from ecsim.simulation import Simulation
from ecsim.units import LENGTH, convert
from ecsim.geometry import LineEvaluator

# %%
s = convert(1 * u.um, LENGTH) / 2
mesh_size = convert(0.1 * u.um, LENGTH)
cube = OrthoBrick((-s, -s, -s), (s, s, s)).mat("solution").bc("reflective")

geo = CSGeometry()
geo.Add(cube)
mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
mesh.ngmesh.SetBCName(1, "clamped")
Draw(mesh)

# %%
# Set up a simulation on the mesh with BAPTA as a buffer
simulation = Simulation(mesh, time_step=10 * u.us, t_end=1 * u.ms)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"solution": 0.6 * u.um**2 / u.ms},
    clamp={"clamped": 1 * u.mmol / u.L}
)
free_buffer = simulation.add_species(
    "free_buffer",
    diffusivity={"solution": 95 * u.um**2 / u.s} #BAPTA
)
bound_buffer = simulation.add_species(
    "bound_buffer",
    diffusivity={"solution": 95 * u.um**2 / u.s} #BAPTA
)


# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(simulation, n_samples):
    sample_int = int(ceil(simulation.n_time_steps / n_samples))
    ca_t = GridFunction(simulation._fes, multidim=0)
    buf_t = GridFunction(simulation._fes, multidim=0)
    comp_t = GridFunction(simulation._fes, multidim=0)
    ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
    buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
    comp_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    
    for i in trange(simulation.n_time_steps):
        simulation.time_step()
        if i % sample_int == 0:
            ca_t.AddMultiDimComponent(simulation.concentrations["calcium"].vec)
            buf_t.AddMultiDimComponent(simulation.concentrations["free_buffer"].vec)
            comp_t.AddMultiDimComponent(simulation.concentrations["bound_buffer"].vec)
    return ca_t, buf_t, comp_t


# %%
# Time stepping - set initial conditions and do time stepping
simulation.setup_problem()
with TaskManager():
    simulation.init_concentrations(
        calcium={"solution": 0 * u.mmol / u.L},
        free_buffer={"solution": 1 * u.mmol / u.L},
        bound_buffer={"solution": 0 * u.mmol / u.L})
    ca_t = time_stepping(simulation, n_samples=100)

# %%
# Visualize
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
# Draw(ca_t.components[0], settings=settings, interpolate_multidim=True, animate=True)

# %%
# Create a line evaluator that evaluates a line from the left to right right side in the middle of the cube
line_evaluator = LineEvaluator(
    mesh,
    (-0.5, 0.0, 0.0),  # Start point (x, y, z)
    (0.5, 0.0, 0.0),  # End point (x, y, z)
    50  # Number of points to evaluate
)

# Evaluate the concentration in the cytosol
ca = line_evaluator.evaluate(simulation.concentrations["calcium"].components[0])
buf = line_evaluator.evaluate(simulation.concentrations["free_buffer"].components[0])
comp = line_evaluator.evaluate(simulation.concentrations["bound_buffer"].components[0])

# Get the x-coordinates for the plot
x = line_evaluator.raw_points[:, 0]  # Extract the x-coordinates

# Plot the results
plt.plot(x, ca, marker='o', linestyle='-', color='blue', label="$[\mathrm{Ca}^{2+}]$")
plt.plot(x, buf, marker='x', linestyle='-', color='red', label="$[\mathrm{BAPTA}]$")
plt.plot(x, comp, marker='.', linestyle='-', color='orange', label="$[\mathrm{CaBAPTA}]$")
plt.title(r"Concentrations through the middle line at the end of simulation")
plt.xlabel(r"x-position ($\mathrm{\mu m}$)")
plt.ylabel(r"concentration (nM)")
plt.legend()
plt.grid(True)
plt.show()

# %%
