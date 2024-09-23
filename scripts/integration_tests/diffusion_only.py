# %% [markdown]
# # Simulate a simple diffusion-only model
# We start with 0mM calcium in a 1um x 1um x 1um box, which is clamped to 1mM on one side. We then simulate the
# diffusion of calcium in the box.

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
# Set up a simulation
simulation = Simulation(mesh, time_step=10 * u.us, t_end=1.0 * u.ms)
calcium = simulation.add_species(
    "calcium",
    diffusivity={"solution": 0.6 * u.um**2 / u.ms},
    clamp={"clamped": 1 * u.mmol / u.L}
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
    simulation.init_concentrations(calcium={"solution": 0 * u.mmol / u.L})
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
c = line_evaluator.evaluate(simulation.concentrations["calcium"].components[0])

# Get the x-coordinates for the plot
x = line_evaluator.raw_points[:, 0]  # Extract the x-coordinates

# Plot the results
plt.plot(x, c, marker='o', linestyle='-', color='blue')
plt.title(r"$[\mathrm{Ca}^{2+}]$ vs position")
plt.xlabel(r"x-position ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]$ (nM)")
plt.grid(True)
plt.show()

# %%
