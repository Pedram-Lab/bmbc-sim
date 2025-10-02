# %% [markdown]
# # Benchmarks for electro-diffusion
# This script aims to provide a simple benchmark for electro-diffusion in a 3D
# geometry by decoupling electrostatics and diffusion.

# %%
from math import ceil, pi

import numpy as np
from scipy.special import erfc
from netgen import occ
import ngsolve as ngs
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
import astropy.units as au
from astropy import constants as const
from tqdm.notebook import trange

from bmbcsim.units import to_simulation_units, uM

# %%
# Parameters (of calcium in extracellular space)
diffusivity = to_simulation_units(1 * au.um ** 2 / au.ms, 'diffusivity')
REL_PERMITTIVITY = 80.0
permittivity = to_simulation_units(REL_PERMITTIVITY * const.eps0, 'permittivity')

F = to_simulation_units(96485.3365 * au.C / au.mol)
VALENCE = 2
beta = to_simulation_units(VALENCE * const.e.si / (const.k_B * 310 * au.K))

ca_ecs = to_simulation_units(1 * uM, 'molar concentration')
tau = to_simulation_units(1 * au.us, 'time')
t_end = to_simulation_units(1 * au.ms, 'time')

# %%
# Define geometry
s = 0.05
ecs_left = occ.Box((-1, -s, -s), (0, s, s)).mat("left").bc("side")
ecs_right = occ.Box((0, -s, -s), (1, s, s)).mat("right").bc("side")

geo = occ.OCCGeometry(occ.Glue([ecs_left, ecs_right]))
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.02))
Draw(mesh)


# %%
# Define FE spaces
concentration_fes = ngs.H1(mesh, order=1)
constraint_fes = ngs.FESpace("number", mesh)
potential_fes = ngs.FESpace([concentration_fes, constraint_fes])
concentration = ngs.GridFunction(concentration_fes)
potential = ngs.GridFunction(potential_fes)

# %%
# Define diffusion problem
u_ecs, v_ecs = concentration_fes.TnT()
a_ecs = ngs.BilinearForm(concentration_fes)
a_ecs += diffusivity * ngs.grad(u_ecs) * ngs.grad(v_ecs) * ngs.dx
m_ecs = ngs.BilinearForm(concentration_fes)
m_ecs += u_ecs * v_ecs * ngs.dx

f_ecs = ngs.LinearForm(concentration_fes)
f_ecs += -diffusivity * beta * concentration * \
    ngs.InnerProduct(ngs.grad(potential.components[0]), ngs.grad(v_ecs)) * ngs.dx

a_ecs.Assemble()
m_ecs.Assemble()
m_ecs.mat.AsVector().data += tau * a_ecs.mat.AsVector()
mstar_inv = m_ecs.mat.Inverse(concentration_fes.FreeDofs())

# %%
# Define problem for electrostatic potential
(u_ecs, p), (v_ecs, q) = potential_fes.TnT()
a_pot = ngs.BilinearForm(potential_fes)
a_pot += permittivity * ngs.grad(u_ecs) * ngs.grad(v_ecs) * ngs.dx
a_pot += p * v_ecs * ngs.dx
a_pot += q * u_ecs * ngs.dx

f_pot = ngs.LinearForm(potential_fes)
f_pot += F * VALENCE * concentration * v_ecs * ngs.dx

a_pot.Assemble()
a_pot_inv = a_pot.mat.Inverse(potential_fes.FreeDofs())

# %%
# Time stepping
def time_stepping(ca, pot, t_end, tau, use_pot):
    n_steps = int(ceil(t_end / tau))
    pot_start = ngs.GridFunction(concentration_fes)
    ca_end = ngs.GridFunction(concentration_fes)

    initial_ca = concentration_fes.mesh.MaterialCF({"left": ca_ecs, "right": 0.0})
    ca.Set(initial_ca)
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
concentration.Set(0)
potential.components[0].Set(0)
ca_end, potential_start = time_stepping(concentration, potential, t_end=t_end, tau=tau, use_pot=True)

n_samples = 51
ca_end_with, xs = evaluate_solution(ca_end, n_samples)
potential_start_with, _ = evaluate_solution(potential_start, n_samples)

# %%
# Visualize whole solution if desired
# Draw(ca_end, mesh)
# Draw(potential_start, mesh)

# %%
# Compute time evolution without the potential
concentration.Set(0)
potential.components[0].Set(0)
ca_end, potential_start = time_stepping(concentration, potential, t_end=t_end, tau=tau, use_pot=False)

ca_end_without, _ = evaluate_solution(ca_end, n_samples)
potential_start_without, _ = evaluate_solution(potential_start, n_samples)

# %%
# Visualize whole solution if desired
# Draw(ca_end, mesh)
# Draw(potential_start, mesh)

# %%
# We can compute the analytical solution for the corresponding 1D problem to compare with our solution for the diffusion part
p = lambda n: pi * (2 * n + 1) / 2
exact_solution = 1/2 + sum((-1) ** n / p(n) * np.exp(-p(n) ** 2) * np.cos(p(n) * (xs + 1)) for n in range(100))
plt.plot(xs, 1e-3 * exact_solution, label="Theoretical solution")
plt.plot(xs, ca_end_with, 'x', label="With potential")
plt.plot(xs, ca_end_without, '.', label="Without potential")
plt.title("Evaluation after 1ms")
plt.xlabel("x [µm]")
plt.ylabel("Ca concentration [mM]")
plt.legend(loc='lower left')
plt.show()

# %%
# In order to compute the theoretical potential at the beginning, we need to compute the charge density
initial_ca = concentration_fes.mesh.MaterialCF({"left": 1.0, "right": 0.0})
concentration.Set(initial_ca)
charge_density = F * ca_ecs * VALENCE
source_strength = charge_density / permittivity
# Draw(concentration)

# %%
# Again, we can analytically compute the solution to the corresponding 1D Poisson equation for the potential
# To this end, we must take charges and the permeability into account, which we do manually for comparison reasons
exact_solution = source_strength * sum((-1) ** n / p(n) ** 3 * np.cos(p(n) * (xs + 1)) for n in range(100))
plt.plot(xs, exact_solution, label="Theoretical potential")
plt.plot(xs, potential_start_with, 'x', label="With potential")
plt.plot(xs, potential_start_without, '.', label="Without potential")
plt.title("Initial potential")
plt.xlabel("x [µm]")
plt.ylabel("Potential [mV]")
plt.legend()
plt.show()

# %%
