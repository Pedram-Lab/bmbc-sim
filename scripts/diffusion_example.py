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
# # Diffusion equation
# This script solves a simple diffusion equation in an essentially 1D geometry to show how to choose units within a finite element simulation. Parameter values are taken from (Horgmo Jæger, Tveito; _Differential Equations for Studies in Computational Electrophysiology_ 2023).
#
# If we take the initial concentration $u(x,y,z,0) = \sin(\pi x)$ and set the concentration to 0 on the left and right boundaries, the time evolution is given by $u(x,y,z,t) = \exp(-D\pi^2t)\sin(\pi x)$.

# %%
from math import ceil, pi

from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as au
from astropy import constants as const
from tqdm.notebook import trange

# %%
# Fix units; this means time in ms, length in um, and concentration in mM
time = au.ms
length = au.um
substance = au.amol

diffusion_unit = length**2 / time
concentration_unit = substance / length**3

# %%
# Parameters (of calcium in extracellular space)
# Caution is needed when converting units: we want the geometry in um and concentrations in mM (= mmol / L = pmol / um^3)
diffusivity = (0.71e6 * au.nm ** 2 / au.ms).to(diffusion_unit).value
ca_max = (1 * au.mmol / au.L).to(concentration_unit).value
tau = (1 * au.us).to(time).value

# %%
# Define geometry (all numbers are in "length" units)
ecs = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1)).mat("ecs")
evaluation_point = (0.5, 0.5, 0.5)
geo = CSGeometry()
geo.Add(ecs)
mesh = Mesh(geo.GenerateMesh(maxh=0.1))
mesh.ngmesh.SetBCName(1, "reservoir")
mesh.ngmesh.SetBCName(3, "reservoir")

# %%
# Define FE space
ecs_fes = H1(mesh, order=2, dirichlet="reservoir")
concentration = GridFunction(ecs_fes)
concentration.Set(ca_max * sin(pi * x))
Draw(concentration)

# %%
# Define diffusion problem
u_ecs, v_ecs = ecs_fes.TnT()
a_ecs = BilinearForm(ecs_fes)
a_ecs += diffusivity * grad(u_ecs) * grad(v_ecs) * dx
m_ecs = BilinearForm(ecs_fes)
m_ecs += u_ecs * v_ecs * dx

a_ecs.Assemble()
m_ecs.Assemble()
m_ecs.mat.AsVector().data += tau * a_ecs.mat.AsVector()
mstar_inv = m_ecs.mat.Inverse(ecs_fes.FreeDofs())


# %%
# Time stepping
def time_stepping(ca, t_end, dt, n_samples):
    n_steps = int(ceil(t_end / dt))
    sample_int = int(ceil(n_steps / n_samples))
    ca_t = GridFunction(ecs_fes, multidim=0)
    ca_t.AddMultiDimComponent(ca.vec)

    for i in trange(n_steps):
        # Solve the diffusion equation
        res = -dt * (a_ecs.mat * ca.vec)
        ca.vec.data += mstar_inv * res
        if i % sample_int == 0:
            ca_t.AddMultiDimComponent(ca.vec)
    return ca_t

# %%
# Evaluation of solutions
def evaluate_solution(sol, t_end, dt, n_samples):
    n_steps = int(ceil(t_end / dt))
    sample_int = int(ceil(n_steps / n_samples))
    mip = mesh(*evaluation_point)

    time, value, mass = [], [], []
    k = 0
    for i in range(n_steps):
        if i % sample_int == 0:
            time.append(i * tau)
            value.append(sol.MDComponent(k)(mip))
            mass.append(Integrate(sol.MDComponent(k), mesh))
            k += 1
    return np.array(time), value, mass

# %%
# Compute time evolution
n_samples = 100
t_end = (1 * au.ms).to(time).value
SetNumThreads(12)
with TaskManager():
    concentration.Set(ca_max * sin(pi * x))
    ca_t = time_stepping(concentration, t_end=t_end, dt=tau, n_samples=n_samples)
    
t, ca_value, ca_mass = evaluate_solution(ca_t, t_end, tau, n_samples)

# %%
# Visualize whole solution if desired
clipping = {"function": True,  "pnt": (0, 0.5, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
# Draw(ca_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=0.0, max=1.0)

# %%
plt.scatter(t, ca_value, facecolors='none', edgecolors='royalblue', label="simulation")
plt.plot(t, ca_max * np.exp(-diffusivity * pi**2 * t), color="darkorange", label="analytic")
plt.title("Evaluation at concentration peak")
plt.xlabel("Time [ms]")
plt.ylabel("Ca concentration [mM]")
plt.legend()
plt.show()

# %%
plt.scatter(t, ca_mass, facecolors='none', edgecolors='royalblue', label="simulation")
integral = 2 * ca_max / pi
plt.plot(t, integral * np.exp(-diffusivity * pi**2 * t), color="darkorange", label="analytic")
plt.title("Total amount of ca-ions")
plt.xlabel("Time [ms]")
plt.ylabel("Ca amount [amol]")
plt.legend()
plt.show()

# %%
