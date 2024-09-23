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
# # Verify mechanical parameters based on the Paszek paper
# This script recreates the setup of Paszek et al. *Integrin Clustering Is Driven by Mechanical Resistance from the Glycocalyx and the Substrate* (2009).
#
# Accoding to this paper, Young's modulus is determined by the following expression:
#
# $Y = \frac{5\sigma}{2 \Delta x}$
#
# where $\sigma$ is the Hookean spring constant, and $\Delta x$ is the LSM lattice node spacing. If $\sigma_g = 0.02 pN/nm$, $\sigma_m = 0.02 pN/nm$, and $\Delta x = 20 nm$. We get the following parameter values for the stress-strain ECM. 
#
# | Parameter                          | Value                          |
# |------------------------------------|--------------------------------|
# | Poisson ratio ($ν$)                | 0.25                           |       
# | Young's modulus for glycocalyx     | 0.0025 pN/nm$^2$               |
# | Young's modulus for PM             | 0.05 pN/nm$^2$                 |

# %%
import matplotlib.pyplot as plt
import numpy as np

from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

# %%
# Define units and parameters for the simulation
from ecsim.units import *
PRESSURE = MASS / (LENGTH * TIME ** 2)
FORCE = MASS * LENGTH / TIME ** 2

s = convert(1.4 * u.um, LENGTH) / 2
ecs_height = convert(43 * u.nm, LENGTH)
membrane_height = convert(40 * u.nm, LENGTH)
mesh_size = convert(20 * u.nm, LENGTH)
cutout_size = convert(300 * u.nm, LENGTH)

pulling_force = convert(1 * u.pN, FORCE)
young_ecs = convert(2.5 * u.fN / u.nm ** 2, PRESSURE)
young_membrane = convert(0.05 * u.pN / u.nm ** 2, PRESSURE)
poisson_ratio = 0.25

# %%
# Use the NGSolve parameter class to change parameters after defining everything
mu = 200   # [E] = Pa       (e.g., steel = 200 GPa, cork = 30MPa)
lam = 200  # [nu] = number  (e.g., steel = 0.3, cork = 0.0)
clipping = {"function": False,  "pnt": (0, 0, 0), "vec": (0, 1, 0)}
visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}

# %%
# Define geometry
ecs = OrthoBrick(Pnt(-s, -s, 0), Pnt(s, s, ecs_height)).mat("ecs").bc("side")
membrane = OrthoBrick(Pnt(-s, -s, ecs_height), Pnt(s, s, ecs_height + membrane_height)).mat("membrane").bc("side")

geo = CSGeometry()
geo.Add(ecs)
geo.Add(membrane)

mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
mesh.ngmesh.SetBCName(0, "substrate")
mesh.ngmesh.SetBCName(4, "top")
mesh.ngmesh.SetBCName(9, "interface")
# Draw(mesh)

# %%
def stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * Id(3)

# %%
# Define mechanic problem
fes = VectorH1(mesh, order=2, dirichlet="substrate")
u, v = fes.TnT()

blf = BilinearForm(fes)
blf += InnerProduct(stress(Sym(Grad(u))), Sym(Grad(v))).Compile() * dx
blf.Assemble()
a_inv = blf.mat.Inverse(fes.FreeDofs())


# %%
def compute_solution(points):
    deformation = GridFunction(fes)
    lf = LinearForm(fes)
    for p in points:
        lf += (-pulling_force * v[2])(*p)
    
    with TaskManager():
        lf.Assemble()
        deformation.vec.data = a_inv * lf.vec
    return deformation


# %%
# Visualize elastic deformation of steel and cork under atmospheric pressure
# The deformation will be under about 2%, which is where the use of linear elasticity is usually justified
points = [(0, 0, ecs_height)]
deformation_1 = compute_solution(points)
Draw(deformation_1, settings=visualization_settings, clipping=clipping)

# fig, ax = plt.subplots()
# ax.loglog(f_list, absolute_deformation, 'b-', label="Deformation in negative z-direction [m]")
# ax.loglog(f_list, absolute_volume, 'r-', label="Absolute change in volume [m^3]")
# ax.set_xlabel("Pressure [Pa]")
# ax.legend()
# plt.show()

# %%
