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
# This script takes the `geometry from scripts/channel_geometry_with_occ.py` and adds reaction-diffusion of chemical species on top and adds the library Astropy to handle units.

# %%
from ngsolve import *
from ngsolve.webgui import Draw
from tqdm import trange

from ecsim.geometry import create_ca_depletion_mesh
from astropy import units as u
import matplotlib.pyplot as plt

# %%
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
concentration.components[0].Set(15, definedon=mesh.Boundaries("ecs_top"))
c_t = time_stepping(concentration, t_end=1, n_samples=100)

# %%
# Visualize (because of the product structure of the FESpace, the usual visualization of time-dependent functions via multidim
visualization = mesh.MaterialCF({"ecs": c_t.components[0], "cytosol": c_t.components[1]})
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
Draw(c_t.components[1], clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)

# %%
