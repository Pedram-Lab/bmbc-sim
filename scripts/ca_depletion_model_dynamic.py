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
# * Ca can diffuse from the ECS to the cytosol through the channel.
#
# The dynamics of the system are resolved in time.

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

# %%
# Define and assemble the FE-problem
ecs_fes = H1(mesh, order=2, definedon=mesh.Materials("ecs"), dirichlet="ecs_top")
cytosol_fes = H1(mesh, order=2, definedon=mesh.Materials("cytosol"))
fes = FESpace([ecs_fes, cytosol_fes])
u_ecs, u_cyt = fes.TrialFunction()
v_ecs, v_cyt = fes.TestFunction()
u1_cyt, v1_cyt = cytosol_fes.TnT()

calcium = GridFunction(fes)
buffer = GridFunction(cytosol_fes)
complex = GridFunction(cytosol_fes)

a_ca = BilinearForm(fes)
a_ca += grad(u_ecs) * grad(v_ecs) * dx("ecs")              # diffusion in ecs
a_ca += grad(u_cyt) * grad(v_cyt) * dx("cytosol")          # diffusion in cytosol
a_ca += (u_ecs - u_cyt) * (v_ecs - v_cyt) * ds("channel")  # interface flux

a_buffer = BilinearForm(cytosol_fes)
a_buffer += grad(u1_cyt) * grad(v1_cyt) * dx

f_ca = LinearForm(fes)
f_ca += -buffer * calcium.components[1] * v_cyt * dx("cytosol")
f_ca += complex * v_cyt * dx("cytosol")

f_buf = LinearForm(cytosol_fes)
f_buf += -buffer * calcium.components[1] * v1_cyt * dx
f_buf += complex * v1_cyt * dx

f_com = LinearForm(cytosol_fes)
f_com += buffer * calcium.components[1] * v1_cyt * dx
f_com += -complex * v1_cyt * dx

a_ca.Assemble()
a_buffer.Assemble()

# %%
# Time stepping - set up system matrix
m_ca = BilinearForm(fes)
m_ca += u_ecs * v_ecs * dx("ecs")
m_ca += u_cyt * v_cyt * dx("cytosol")
m_ca.Assemble()

dt = 0.001
mstar_ca = m_ca.mat.CreateMatrix()
mstar_ca.AsVector().data = m_ca.mat.AsVector() + dt * a_ca.mat.AsVector()
mstar_ca_inv = mstar_ca.Inverse(freedofs=fes.FreeDofs())

m_buffer = BilinearForm(cytosol_fes)
m_buffer += u1_cyt * v1_cyt * dx
m_buffer.Assemble()

mstar_buffer = m_buffer.mat.CreateMatrix()
mstar_buffer.AsVector().data = m_buffer.mat.AsVector() + dt * a_buffer.mat.AsVector()
mstar_buffer_inv = mstar_buffer.Inverse(freedofs=cytosol_fes.FreeDofs())

# %%
# Time stepping - define a function that pre-computes all timesteps
def time_stepping(u_ca, u_buf, t_end, n_samples):
    n_steps = int(ceil(t_end / dt))
    sample_int = int(ceil(n_steps / n_samples))
    u_com = GridFunction(u_buf.space)
    u_com.vec.data = 0 * u_buf.vec
    u_ca_t = GridFunction(u_ca.space, multidim=0)
    u_buf_t = GridFunction(u_buf.space, multidim=0)
    u_com_t = GridFunction(u_com.space, multidim=0)
    u_ca_t.AddMultiDimComponent(u_ca.vec)
    u_buf_t.AddMultiDimComponent(u_buf.vec)
    u_com_t.AddMultiDimComponent(u_com.vec)
    
    for i in trange(n_steps):
        f_ca.Assemble()
        f_buf.Assemble()
        f_com.Assemble()
        res_ca = dt * (f_ca.vec - a_ca.mat * u_ca.vec)
        res_buf = dt * (f_buf.vec - a_buffer.mat * u_buf.vec)
        res_com = dt * (f_com.vec - a_buffer.mat * u_com.vec)
        u_ca.vec.data += mstar_ca_inv * res_ca
        u_buf.vec.data += mstar_buffer_inv * res_buf
        u_com.vec.data += mstar_buffer_inv * res_com
        if i % sample_int == 0:
            u_ca_t.AddMultiDimComponent(u_ca.vec)
            u_buf_t.AddMultiDimComponent(u_buf.vec)
            u_com_t.AddMultiDimComponent(u_com.vec)
    return u_ca_t, u_buf_t, u_com_t


# %%
# Time stepping - set initial conditions and do time stepping
with TaskManager():
    calcium.components[0].Set(15)
    buffer.Set(1)
    ca_t, buffer_t, complex_t = time_stepping(calcium, buffer, t_end=1, n_samples=100)

# %%
# Visualize (because of the product structure of the FESpace, the usual
# visualization of time-dependent functions via multidim is not possible)
visualization = mesh.MaterialCF({"ecs": ca_t.components[0], "cytosol": ca_t.components[1]})
clipping = {"function": True,  "pnt": (0, 0, 1.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -90}]}, "Colormap": {"ncolors": 32, "autoscale": False, "max": 15}}
# Draw(ca_t.components[1], clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)
# Draw(buffer_t, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)
Draw(complex_t, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True)

# %%
