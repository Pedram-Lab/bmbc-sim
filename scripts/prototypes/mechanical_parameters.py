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
# # Mechanical parameters
# This script defines an ECS that's tethered to a fixed substrate at the bottom and a pushed on at the top. Mechanical parameters are changed to show their interpretation.

# %%
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
import numpy as np

# %%
# Use the NGSolve parameter class to change parameters after defining everything
mu_param = Parameter(0)   # [E] = Pa       (e.g., steel = 200 GPa, cork = 30MPa)
lam_param = Parameter(0)  # [nu] = number  (e.g., steel = 0.3, cork = 0.0)
f_param = Parameter(0)    # [f] = Pa       (e.g., 1 atmosphere = 101 kPa)
visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}

# %%
# Define geometry
s = 1 / 2
ecs = OrthoBrick(Pnt(-s, -s, 0), Pnt(s, s, 1.0)).mat("ecs").bc("side")

geo = CSGeometry()
geo.Add(ecs)

mesh = Mesh(geo.GenerateMesh(maxh=0.1))
mesh.ngmesh.SetBCName(0, "substrate")
mesh.ngmesh.SetBCName(4, "top")
fes = VectorH1(mesh, order=2, dirichlet="substrate")

# %%
def stress(strain):
    return 2 * mu_param * strain + lam_param * Trace(strain) * Id(3)

# %%
# Define mechanic problem
u, v = fes.TnT()

blf = BilinearForm(fes)
blf += InnerProduct(stress(Sym(Grad(u))), Sym(Grad(v))).Compile() * dx

force = CoefficientFunction((0, 0, -f_param))
lf = LinearForm(fes)
lf += force * v * ds("top")


# %%
def compute_solution(E, nu, f):
    mu_param.Set(E / (2 * (1 + nu)))
    lam_param.Set(E * nu / ((1 + nu) * (1 - 2 * nu)))
    f_param.Set(f)
    deformation = GridFunction(fes)
    
    with TaskManager():
        blf.Assemble()
        lf.Assemble()
        deformation.vec.data = blf.mat.Inverse(fes.FreeDofs()) * lf.vec
    return deformation


# %%
# Visualize elastic deformation of steel and cork under atmospheric pressure
# The deformation will be under about 2%, which is where the use of linear elasticity is usually justified
deformation_steel = compute_solution(E=200e9, nu=0.3, f=101e3)
Draw(deformation_steel, settings=visualization_settings)
deformation_cork = compute_solution(E=30e6, nu=0.0, f=101e3)
Draw(deformation_cork, settings=visualization_settings)

# %%
# Visualize elastic deformation of cork-like synthetic materials with positive / negative poisson ratio under very high loads
deformation_synthetic2 = compute_solution(E=30e6, nu=0.4, f=101e5)
Draw(deformation_synthetic2, settings=visualization_settings, deformation=True)
deformation_synthetic2 = compute_solution(E=30e6, nu=-0.9, f=101e5)
Draw(deformation_synthetic2, settings=visualization_settings, deformation=True)

# %%
# Compute elastic deformation of steel under 1-1e^6 x atmospheric pressure; plot deformation in z and volume change
f_list = 101e3 * 4 ** np.array([i for i in range(11)])
absolute_deformation, absolute_volume = [], []
mip = mesh(0.5, 0.5, 1.0)

for f in f_list:
    deformation_steel = compute_solution(E=200e9, nu=0.3, f=f)
    absolute_deformation.append(-deformation_steel(mip)[2])
    mesh.SetDeformation(deformation_steel)
    absolute_volume.append(abs(Integrate(CoefficientFunction(1), mesh) - 1.0))
    mesh.UnsetDeformation()
    
fig, ax = plt.subplots()
ax.loglog(f_list, absolute_deformation, 'b-', label="Deformation in negative z-direction [m]")
ax.loglog(f_list, absolute_volume, 'r-', label="Absolute change in volume [m^3]")
ax.set_xlabel("Pressure [Pa]")
ax.legend()
plt.show()

# %%
# Compute elastic deformation of steel under 1000 x atmospheric pressure with different Poisson's ratios
# Volume is preserved for a Poisson ratio of ~0.5, no bulging happens for 0.0, and the deformation is small due to compression for negative values
nu_list = np.linspace(-0.9, 0.49, 25)
absolute_deformation, absolute_volume = [], []
mip = mesh(0.5, 0.5, 1.0)

for nu in nu_list:
    deformation_steel = compute_solution(E=200e9, nu=nu, f=101e3)
    absolute_deformation.append(-deformation_steel(mip)[2])
    mesh.SetDeformation(deformation_steel)
    absolute_volume.append(abs(Integrate(CoefficientFunction(1), mesh) - 1.0))
    mesh.UnsetDeformation()
    
fig, ax_l = plt.subplots()
ax_r = ax_l.twinx()
ax_l.semilogy(nu_list, absolute_deformation, 'b-')
ax_l.set_xlabel("Poisson's ratio")
ax_l.set_ylabel("Deformation in negative z-direction [m]", color='b')
ax_r.semilogy(nu_list, absolute_volume, 'r-')
ax_r.set_ylabel("Absolute change in volume [m^3]", color='r')
plt.show()

# %%
# Compute elastic deformation of incompressible material under different pressures to test volume preservation
# The volume volume preservation deteriorates with increasing load, once again showing that linear elasticity is only valid for small strains
f_list = 101e3 * 4 ** np.array([i for i in range(11)])
absolute_deformation, absolute_volume = [], []
mip = mesh(0.5, 0.5, 1.0)

for f in f_list:
    deformation_incomp = compute_solution(E=200e9, nu=0.49999999, f=f)
    absolute_deformation.append(-deformation_incomp(mip)[2])
    mesh.SetDeformation(deformation_incomp)
    absolute_volume.append(abs(Integrate(CoefficientFunction(1), mesh) - 1.0))
    mesh.UnsetDeformation()
    
fig, ax = plt.subplots()
ax.loglog(f_list, absolute_deformation, 'b-', label="Deformation in negative z-direction [m]")
ax.loglog(f_list, absolute_volume, 'r-', label="Absolute change in volume [m^3]")
ax.set_xlabel("Pressure [Pa]")
ax.legend()
plt.show()

# %%
