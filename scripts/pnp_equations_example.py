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
# # Poisson-Nernst-Planck equations
# This script solves the Poisson-Nernst-Planck equations for a simple geometry to show how to couple diffusion and
# electrostatics.

# %%
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as au
from astropy import constants as const
from tqdm.notebook import trange

# %%
# Parameters (of calcium in extracellular space)
# Caution is needed when converting units: we want the geometry in um and concentrations in mM (= zmol/um^3)
diffusivity = 0.71 * au.um ** 2 / au.s
relative_permittivity = 80.0
permittivity = relative_permittivity * const.eps0.to(au.F / au.um)
F = (96485.3365 * au.C / au.mol).to(au.C / au.zmol)
valence = 2
beta = valence * const.e.si / (const.k_B * 310 * au.K)
ca_ecs = 2 * au.mmol / au.m**3
source_point = (-0.4, 0.0, 0.1)
tau = 100 * au.us

# %%
# Define geometry
ecs = OrthoBrick(Pnt(-0.5, -0.5, 0), Pnt(0.5, 0.5, 1)).mat("ecs").bc("side")
geo = CSGeometry()
geo.Add(ecs)
mesh = Mesh(geo.GenerateMesh(maxh=0.1))

# %%
# Define FE spaces
ecs_fes = H1(mesh, order=2)
constraint_fes = FESpace("number", mesh)
potential_fes = FESpace([ecs_fes, constraint_fes])
concentration = GridFunction(ecs_fes)
potential = GridFunction(potential_fes)

# %%
# Define diffusion problem
u_ecs, v_ecs = ecs_fes.TnT()
a_ecs = BilinearForm(ecs_fes)
a_ecs += diffusivity.value * grad(u_ecs) * grad(v_ecs) * dx
m_ecs = BilinearForm(ecs_fes)
m_ecs += u_ecs * v_ecs * dx

f_ecs = LinearForm(ecs_fes)
f_ecs += (ca_ecs.to(au.mmol / au.m**3).value * v_ecs) (*source_point)  # point source of calcium
f_ecs += -diffusivity.value * beta.value * concentration * InnerProduct(grad(potential.components[0]), grad(v_ecs)) * dx

a_ecs.Assemble()
m_ecs.Assemble()
m_ecs.mat.AsVector().data += tau.to(au.s).value * a_ecs.mat.AsVector()
mstar_inv = m_ecs.mat.Inverse(ecs_fes.FreeDofs())

# %%
# Define problem for electrostatic potential
(u_ecs, p), (v_ecs, q) = potential_fes.TnT()
a_pot = BilinearForm(potential_fes)
a_pot += permittivity.value * grad(u_ecs) * grad(v_ecs) * dx
a_pot += p * v_ecs * dx
a_pot += q * u_ecs * dx

f_pot = LinearForm(potential_fes)
f_pot += F.value * valence * concentration * v_ecs * dx

a_pot.Assemble()
a_pot_inv = a_pot.mat.Inverse(potential_fes.FreeDofs())

# %%
# Time stepping
def time_stepping(ca, pot, t_end, tau, n_samples, use_pot):
    dt = tau.to(au.s).value
    n_steps = int(ceil(t_end.to(au.s).value / dt))
    sample_int = int(ceil(n_steps / n_samples))
    ca_t = GridFunction(ecs_fes, multidim=0)
    pot_t = GridFunction(ecs_fes, multidim=0)
    ca_t.AddMultiDimComponent(ca.vec)
    pot_t.AddMultiDimComponent(pot.components[0].vec)

    for i in trange(n_steps):
        # Solve the potential equation
        if use_pot:
            f_pot.Assemble()
            pot.vec.data = a_pot_inv * f_pot.vec
        # Solve the diffusion equation
        f_ecs.Assemble()
        res = dt * (f_ecs.vec - a_ecs.mat * ca.vec)
        ca.vec.data += mstar_inv * res
        if i % sample_int == 0:
            ca_t.AddMultiDimComponent(ca.vec)
            pot_t.AddMultiDimComponent(pot.components[0].vec)
    return ca_t, pot_t

# %%
# Evaluation of solutions
def evalutate_solution(sol, t_end, tau, n_samples):
    dt = tau.to(au.s).value
    n_steps = int(ceil(t_end.to(au.s).value / dt))
    sample_int = int(ceil(n_steps / n_samples))
    mip_near = mesh(*source_point)
    mip_far = mesh(0.4, 0.0, 0.9)

    time, near, far = [], [], []
    k = 0
    for i in range(n_steps):
        if i % sample_int == 0:
            time.append(i * tau.value)
            near.append(sol.MDComponent(k)(mip_near))
            far.append(sol.MDComponent(k)(mip_far))
            k += 1
    return time, near, far

# %%
# Compute time evolution with the potential
n_samples = 100
t_end = 0.1 * au.s
with TaskManager():
    concentration.Set(0)
    potential.components[0].Set(0)
    ca_t, potential_t = time_stepping(concentration, potential, t_end=t_end, tau=tau, n_samples=n_samples, use_pot=True)
    
time, ca_full_near, ca_full_far = evalutate_solution(ca_t, t_end, tau, n_samples)

# %%
# Visualize whole solution if desired
clipping = {"function": True,  "pnt": (0, 0, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
# Draw(ca_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=0.0, max=2.0)
# Draw(potential_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=-0.01, max=0.01)

# %%
# Compute time evolution without the potential
with TaskManager():
    concentration.Set(0)
    potential.components[0].Set(0)
    ca_t, potential_t = time_stepping(concentration, potential, t_end=t_end, tau=tau, n_samples=n_samples, use_pot=False)

time, ca_only_near, ca_only_far = evalutate_solution(ca_t, t_end, tau, n_samples)

# %%
# Visualize whole solution if desired
clipping = {"function": True,  "pnt": (0, 0, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
# Draw(ca_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=0.0, max=2.0)
# Draw(potential_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=-0.01, max=0.01)

# %%
plt.plot(time, ca_full_near, label="with potential")
plt.plot(time, ca_only_near, label="without potential")
plt.title("Evaluation at ca-source")
plt.xlabel("Time [us]")
plt.ylabel("Ca concentration [mM]")
plt.legend()
plt.show()

# %%
plt.plot(time, ca_full_far, label="with potential")
plt.plot(time, ca_only_far, label="without potential")
plt.title("Evaluation far away from ca-source")
plt.xlabel("Time [us]")
plt.ylabel("Ca concentration [mM]")
plt.legend()
plt.show()

# %%
