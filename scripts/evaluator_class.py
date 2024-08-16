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
# This Python code proposes the class LineEvaluator to plot the calcium traces to both the cytosol and the extracellular space. 

# %%
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


# Time stepping - set initial conditions and do time stepping
concentration = GridFunction(fes)
concentration.components[0].Set(15)
c_t = time_stepping(concentration, t_end=1, n_samples=100)

# Visualize (because of the product structure of the FESpace, the usual
# visualization of time-dependent functions via multidim is not possible)
visualization = mesh.MaterialCF({"ecs": c_t.components[0], "cytosol": c_t.components[1]})
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
Draw(c_t.components[1], settings=settings, interpolate_multidim=True, animate=True)


# %%
#Definiton of the class LineEvaluator

# %%
class LineEvaluator:
    def __init__(self, mesh, start, end, n):
        """
        Initializes the LineEvaluator with the mesh, start, and end points of the line, and the number of points to evaluate.

        Parameters:
        - mesh: The NGSolve mesh object.
        - start: The starting point of the line segment (x, y, z).
        - end: The ending point of the line segment (x, y, z).
        - n: Number of evaluation points along the line segment.
        """
        # Generate the coordinates for the line segment
        x_coords = np.linspace(start[0], end[0], n)
        y_coords = np.linspace(start[1], end[1], n)
        z_coords = np.linspace(start[2], end[2], n)
        
        # Store the raw points for later use
        self._raw_points = np.column_stack((x_coords, y_coords, z_coords))
        
        # Generate the corresponding mesh evaluation points
        self._eval_points = [mesh(x, y, z) for x, y, z in zip(x_coords, y_coords, z_coords)]

    def evaluate(self, coefficient_function):
        """
        Evaluates the given coefficient function at the points defined by the evaluator.

        Parameters:
        - coefficient_function: The NGSolve coefficient function to evaluate (e.g., concentration.components[0]).

        Returns:
        - A numpy array of evaluated values.
        """
        return np.array([coefficient_function(point) for point in self._eval_points])

    def get_points(self):
        """
        Returns the raw points (x, y, z coordinates) used for evaluation.

        Returns:
        - A numpy array of shape (n, 3) representing the points along the line segment.
        """
        return self._raw_points




# %%
#Evaluation of the class for calcium at the cytosol

# %%
# Define the constant values for y and z
y_constant_cyt = 1.5  # Constant value for y
z_constant_cyt = 2.8  # Constant value for z

# Define the range and number of points for x
x_start_cyt = 0.0  # Start of the x range
x_end_cyt = 1.5    # End of the x range
n_points_cyt = 50  # Number of points in the x range

# Create the line evaluator using the LineEvaluator class
line_evaluator_cyt = LineEvaluator(
    mesh, 
    (x_start_cyt, y_constant_cyt, z_constant_cyt),  # Start point (x, y, z)
    (x_end_cyt, y_constant_cyt, z_constant_cyt),    # End point (x, y, z)
    n_points_cyt  # Number of points to evaluate
)

concentrations_cyt = line_evaluator_cyt.evaluate(concentration.components[1])
x_coords = line_evaluator_cyt.get_points()[:, 0]  # Extract the x-coordinates

plt.figure(figsize=(10, 6))
plt.plot(x_coords, concentrations_cyt, marker='o', linestyle='-', color='blue')
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{cyt}}$ (nM)")
plt.grid(True)
plt.show()


# %%
#Evaluation of the class for calcium at the ecs

# %%
# Define the constant values for y and z
y_constant_ecs = 1.5  # Constant value for y
z_constant_ecs = 3.005  # Constant value for z

# Define the range and number of points for x
x_start_ecs = 0.0  # Start of the x range
x_end_ecs = 1.5    # End of the x range
n_points_ecs = 50  # Number of points in the x range

# Create the line evaluator using the LineEvaluator class
line_evaluator_ecs = LineEvaluator(
    mesh, 
    (x_start_ecs, y_constant_ecs, z_constant_ecs),  # Start point (x, y, z)
    (x_end_ecs, y_constant_ecs, z_constant_ecs),    # End point (x, y, z)
    n_points_ecs  # Number of points to evaluate
)

# Evaluate the concentration in the extracellular space (ECS)
concentrations_ext = line_evaluator_ecs.evaluate(concentration.components[0])

# Get the x-coordinates for the plot
x_coords_ecs = line_evaluator_ecs.get_points()[:, 0]  # Extract the x-coordinates

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x_coords_ecs, concentrations_ext, marker='o', linestyle='-', color='red')
plt.ylim([14.5, 15.1])
plt.title(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ vs Distance from the channel")
plt.xlabel(r"Distance from the channel ($\mathrm{\mu m}$)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_{\mathrm{ecs}}$ (mM)")
plt.grid(True)
plt.show()


# %%
