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
# # Benchmarks for electro-diffusion
# This script aims to provide a simple benchmark for electro-diffusion in a 3D geometry by decoupling electrostatics and
# diffusion.

# %%
from math import ceil

import numpy as np
from netgen.occ import *
from ngsolve import *
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
import astropy.units as au
from astropy import constants as const
from tqdm.notebook import trange

from ecsim.units import *

# %%
# Parameters (of calcium in extracellular space)
# Caution is needed when converting units: we want the geometry in um and concentrations in mM (= mmol / L = amol/um^3)
diffusivity = convert(1 * au.um ** 2 / au.ms, DIFFUSIVITY)
relative_permittivity = 80.0
permittivity = convert(relative_permittivity * const.eps0, PERMITTIVITY)

F = convert(96485.3365 * au.C / au.mol, CHARGE / SUBSTANCE)
valence = 2
beta = convert(valence * const.e.si / (const.k_B * 310 * au.K), CHARGE / ENERGY)

ca_ecs = convert(2 * au.mol / au.L, CONCENTRATION)
tau = convert(1 * au.us, TIME)
t_end = convert(1 * au.ms, TIME)

# %%
# Define geometry
ecs = Box(Pnt(-1, -0.1, -0.1), Pnt(1, 0.1, 0.1)).mat("ecs").bc("side")
ecs.faces[0].bc("clamped")

geo = OCCGeometry(ecs)
mesh = Mesh(geo.GenerateMesh(maxh=0.02))
Draw(mesh)


# %%
# Define FE spaces
concentration_fes = H1(mesh, order=1, dirichlet="clamped")
constraint_fes = FESpace("number", mesh)
potential_fes = FESpace([concentration_fes, constraint_fes])
concentration = GridFunction(concentration_fes)
potential = GridFunction(potential_fes)

# %%
# Define diffusion problem
u_ecs, v_ecs = concentration_fes.TnT()
a_ecs = BilinearForm(concentration_fes)
a_ecs += diffusivity * grad(u_ecs) * grad(v_ecs) * dx
m_ecs = BilinearForm(concentration_fes)
m_ecs += u_ecs * v_ecs * dx

f_ecs = LinearForm(concentration_fes)
f_ecs += -diffusivity * beta * concentration * InnerProduct(grad(potential.components[0]), grad(v_ecs)) * dx

a_ecs.Assemble()
m_ecs.Assemble()
m_ecs.mat.AsVector().data += tau * a_ecs.mat.AsVector()
mstar_inv = m_ecs.mat.Inverse(concentration_fes.FreeDofs())

# %%
# Define problem for electrostatic potential
(u_ecs, p), (v_ecs, q) = potential_fes.TnT()
a_pot = BilinearForm(potential_fes)
a_pot += permittivity * grad(u_ecs) * grad(v_ecs) * dx
a_pot += p * v_ecs * dx
a_pot += q * u_ecs * dx

f_pot = LinearForm(potential_fes)
f_pot += F * valence * concentration * v_ecs * dx

a_pot.Assemble()
a_pot_inv = a_pot.mat.Inverse(potential_fes.FreeDofs())

# %%
# Time stepping
def time_stepping(ca, pot, t_end, tau, use_pot):
    n_steps = int(ceil(t_end / tau))
    pot_start = GridFunction(concentration_fes)
    ca_end = GridFunction(concentration_fes)

    ca.Set(0)
    ca.Set(1, BND)
    pot.components[0].Set(0)
    pot.components[1].Set(0)

    for i in trange(n_steps):
        # Solve the potential equation
        if use_pot:
            f_pot.Assemble()
            pot.vec.data = a_pot_inv * f_pot.vec
        # Solve the diffusion equation
        f_ecs.Assemble()
        res = tau * (f_ecs.vec - a_ecs.mat * ca.vec)
        ca.vec.data += mstar_inv * res
        if i == 0:
            # Save the potential of the initial configuration
            pot_start.vec.data = pot.components[0].vec

        # Save the concentration of the final configuration
        ca_end.vec.data = ca.vec

    return ca_end, pot_start

# %%
# Evaluation of solutions
def evaluate_solution(sol, n_samples):
    xs = np.linspace(-1, 1, n_samples)

    values = []
    for x in xs:
        mip = mesh(x, 0.0, 0.0)
        values.append(sol(mip))

    return np.array(values), xs

# %%
# Compute time evolution with the potential
SetNumThreads(8)
with TaskManager():
    concentration.Set(0)
    potential.components[0].Set(0)
    ca_end, potential_start = time_stepping(concentration, potential, t_end=t_end, tau=tau, use_pot=True)

n_samples = 100
ca_end_with, xs = evaluate_solution(ca_end, n_samples)
potential_start_with, _ = evaluate_solution(potential_start, n_samples)

# %%
# Visualize whole solution if desired
clipping = {"function": True,  "pnt": (0, 0, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
# Draw(ca_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=0.0, max=0.02)
# Draw(potential_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=-0.01, max=0.01)

# %%
# Compute time evolution without the potential
SetNumThreads(8)
with TaskManager():
    concentration.Set(0)
    potential.components[0].Set(0)
    ca_end, potential_start = time_stepping(concentration, potential, t_end=t_end, tau=tau, use_pot=False)

ca_end_without, _ = evaluate_solution(ca_end, n_samples)
potential_start_without, _ = evaluate_solution(potential_start, n_samples)

# %%
# Visualize whole solution if desired
clipping = {"function": True,  "pnt": (0, 0, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
# Draw(ca_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=0.0, max=0.02)
# Draw(potential_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=-0.01, max=0.01)

# %%
plt.plot(xs, ca_end_with, label="with potential")
plt.plot(xs, ca_end_without, label="without potential")
plt.title("Evaluation at t_end")
plt.xlabel("x [um]")
plt.ylabel("Ca concentration [mM]")
plt.legend()
plt.show()

# %%
plt.plot(xs, potential_start_with, label="with potential")
plt.plot(xs, potential_start_without, label="without potential")
plt.title("Initial potential")
plt.xlabel("x [um]")
plt.ylabel("Potential [V]")
plt.legend()
plt.show()
