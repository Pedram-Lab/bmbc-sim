# %% [markdown]
# # Mass conservation upon deformation
# If a computational domain is deformed, ionic mass in the domain should be conserved. This necessitates that the
# concentrations of the species are updated in a way that conserves mass but also maintains locality. This tutorial
# demonstrates how to achieve this using NGSolve by simulating 1mM Ca in a progressively deforming cube of 1um side
# length.

# %%
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt
import numpy as np

from bmbcsim.units import *

# %%
# Define units and parameters for the simulation (mechanical parameters are for typical extracellular matrix)
s = convert(1 * u.um, LENGTH) / 2
mesh_size = convert(100 * u.nm, LENGTH)

pressure = convert(1 * u.fN / u.nm ** 2, PRESSURE)
youngs_modulus = convert(2.5 * u.fN / u.nm ** 2, PRESSURE)
poisson_ratio = 0.25

# %%
# Define geometry: a cube of 1um side length fixed at the bottom
cytosol = OrthoBrick(Pnt(-s, -s, -s), Pnt(s, s, s)).mat("cytosol").bc("side")

geo = CSGeometry()
geo.Add(cytosol)

mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
mesh.ngmesh.SetBCName(0, "bottom")
mesh.ngmesh.SetBCName(4, "top")
# Draw(mesh)

# %%
# Set Lamé constants based on Young's modulus and Poisson ratio
mu = youngs_modulus / (2 * (1 + poisson_ratio))
lam = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}, "deformation": 1.0}

# %%
# A linear stress-strain relationship
def stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * Id(3)

# %%
# Define mechanic problem
mechanic_fes = VectorH1(mesh, order=2, dirichlet="bottom")
concentration_fes = H1(mesh, order=2)
u, v = mechanic_fes.TnT()

blf = BilinearForm(mechanic_fes)
blf += InnerProduct(stress(Sym(Grad(u))), Sym(Grad(v))).Compile() * dx
blf.Assemble()
a_inv = blf.mat.Inverse(mechanic_fes.FreeDofs())

lf = LinearForm(mechanic_fes)
lf += - ((x - 0.5) ** 2 + (y - 0.5) ** 2) * v[2] * ds("top")
lf.Assemble()

# %%
# Compute the solution given some pressure
def compute_solution(pressure):
    deformation = GridFunction(mechanic_fes)
    with TaskManager():
        deformation.vec.data = pressure * (a_inv * lf.vec)
    return deformation

# %%
# Define a linear form that represents the integral of local shape functions
# i.e., the volume associated with a certain coefficient in a FEM function
u, v = concentration_fes.TnT()
local_integration = LinearForm(concentration_fes)
local_integration += v * dx

# %%
# Compute deformations with progressively higher pressures at the top
concentration = GridFunction(concentration_fes)
concentration.Set(1)
previous_local_volume = local_integration.Assemble().vec.FV().NumPy().copy()
volume = []
total_substance = []
steps = np.linspace(0, 1, 10)

for k in steps:
    mesh.UnsetDeformation()
    deformation = compute_solution(pressure * k)
    mesh.SetDeformation(deformation)

    # The correction factor for the local concentration is the ratio between local volumes
    local_volume = local_integration.Assemble().vec.FV().NumPy().copy()
    concentration.vec.FV().NumPy()[:] *= previous_local_volume / local_volume

    # Measure volume and substance in the computational domain
    volume.append(Integrate(1, mesh))
    total_substance.append(Integrate(concentration, mesh))
    previous_local_volume = local_volume

# %%
# Plot volume, substance, and concentration
# NOTE: in order to conserve local substance, the concentration has to change locally to accommodate for volume changes
avg_concentration = [m / v for m, v in zip(total_substance, volume)]
fig, ax = plt.subplots()
ax.plot(steps, volume, label="volume [µm³]")
ax.plot(steps, total_substance, label="total substance [amol]")
ax.plot(steps, avg_concentration, label="average concentration [mM]")
ax.set_xlabel("pressure [fN/nm²]")
ax.legend()
ax.grid()
plt.show()

# %%
# Visualize deformation for largest pressure...
visualization_settings["deformation"] = 0.0
Draw(deformation, settings=visualization_settings)

# %%
# ... and corresponding concentration
visualization_settings["deformation"] = 1.0
Draw(concentration, settings=visualization_settings)

# %%
