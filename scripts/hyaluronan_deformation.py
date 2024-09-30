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
# # Simulate simultaneous pulling of several hyaluronan chains
# This script simulates the deformation of the extracellular matrix (ECS) and the cell membrane due to the pulling of
# several hyaluronan chains. The parameters are based on the paper by Paszek et al. *Integrin Clustering Is Driven by
# Mechanical Resistance from the Glycocalyx and the Substrate* (2009).
#
# | Parameter                          | Definition                              | Value                      |
# |------------------------------------|-----------------------------------------| ---------------------------|
# | $ν$                                | Poisson ratio                           | 0.25                       |
# | Y$_g$                              | Young's modulus for glycocalyx          | 0.0025 pN/nm$^2$           |
# | Y$_{m}$                            | Young's modulus for PM                  | 0.05 pN/nm$^2$             |
# | l$_{g}$                            | Glycocalyx thickness                    | 45 nm                      |
#
#
# | Compartment                          | Size                         |
# |------------------------------------|--------------------------------|
# | Cell membrane                | 1 μm x 1 μm x 40 nm
# | Glycocalyx                   | 1 μm x 1 μm x 45 nm                               |

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

from ecsim.units import *

# %%
# Define units and parameters for the simulation
s = convert(1 * u.um, LENGTH) / 2
ecs_height = convert(45 * u.nm, LENGTH)
membrane_height = convert(40 * u.nm, LENGTH)
mesh_size = convert(40 * u.nm, LENGTH)

pulling_force = convert(5 * u.pN, FORCE)
youngs_modulus_ecs = convert(2.5 * u.fN / u.nm ** 2, PRESSURE)
youngs_modulus_membrane = convert(50 * u.fN / u.nm ** 2, PRESSURE)
poisson_ratio = 0.25

# %%
# Define geometry (we consider the substrate as fixed and don't model it explicitly)
ecs = OrthoBrick(Pnt(-s, -s, 0), Pnt(s, s, ecs_height)).mat("ecs").bc("side")
membrane = OrthoBrick(Pnt(-s, -s, ecs_height), Pnt(s, s, ecs_height + membrane_height)).mat("membrane").bc("side")

geo = CSGeometry()
geo.Add(ecs)
geo.Add(membrane)

mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
mesh.ngmesh.SetBCName(0, "substrate")
mesh.ngmesh.SetBCName(4, "top")
mesh.ngmesh.SetBCName(9, "interface")
Draw(mesh)

# %%
# Set Lamé constants based on Young's modulus and Poisson ratio
E = mesh.MaterialCF({"ecs": youngs_modulus_ecs, "membrane": youngs_modulus_membrane})
nu = poisson_ratio

mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

clipping = {"function": False,  "pnt": (0, 0, ecs_height + membrane_height / 2), "vec": (0, 0, -1)}

# %%
def stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * Id(3)

# %%
# Define mechanic problem
scalar_fes = H1(mesh, order=1)
fes = VectorH1(mesh, order=1, dirichlet="substrate")
u, v = fes.TnT()

blf = BilinearForm(fes)
blf += InnerProduct(stress(Sym(Grad(u))), Sym(Grad(v))).Compile() * dx
blf.Assemble()
a_inv = blf.mat.Inverse(fes.FreeDofs())

# %%
def compute_membrane_points(mesh, nX, nY):
    # Extract the interface points of the mesh
    mesh_points = mesh.ngmesh.Coordinates()
    interface_indices = np.where(np.isclose(mesh_points[:, 2], ecs_height))[0]
    mesh_points = mesh_points[interface_indices, :]
    tree = KDTree(mesh_points[:, :2])

    # Find points closest to a regular grid on the interior of the membrane
    x = np.linspace(-s, s, nX + 2)[1:-1]
    y = np.linspace(-s, s, nY + 2)[1:-1]
    grid = np.column_stack([v.flatten() for v in np.meshgrid(x, y)])
    distances, indices = tree.query(grid)
    indices = np.unique(indices)
    
    # Compute a function that is 1 at the chosen points and 0 elsewhere
    v = GridFunction(scalar_fes)
    v.Set(0)
    v.vec.FV().NumPy()[interface_indices[indices]] = 1

    return mesh_points[indices, :], v, indices.shape[0] / interface_indices.shape[0], np.median(distances)


# %%
def compute_solution(points, force):
    deformation = GridFunction(fes)
    lf = LinearForm(fes)
    for p in points:
        lf += (-force * v[2])(*p)
    
    with TaskManager():
        lf.Assemble()
        deformation.vec.data = a_inv * lf.vec
    return deformation


# %%
def sample_surface(mesh, deformation, n_samples=300):
    x = np.linspace(-s, s, n_samples)
    y = np.linspace(-s, s, n_samples)
    X, Y = np.meshgrid(x, y)
    z = ecs_height

    points = [(x, y, z) for x, y in zip(X.flatten(), Y.flatten())]
    mips = [mesh(*p) for p in points]
    deformation_z = [deformation.components[2](p) for p in mips]

    return X, Y, z + np.array(deformation_z).reshape((n_samples, n_samples))

# %%
# Find grid of points on the membrane surface and visualize weight
_, w, surface_ratio, median_distance = compute_membrane_points(mesh, 3, 3)
print(f"Part of surface nodes covered: {surface_ratio * 100:.2f}%")
print(f"Median distance to regular grid points: {median_distance * 1000:.3g}nm")
Draw(w, clipping=clipping)

# %%
# Compute elastic deformation of membrane and ECS for different densities
points, _, surface_ratio, median_distance = compute_membrane_points(mesh, 3, 3)
print(f"Part of surface covered: {surface_ratio * 100:.2f}%")
print(f"Median distance to regular grid points: {median_distance * 1000:.3g}nm")
deformation_1 = compute_solution(points, pulling_force)
print(f"Maximum deformation: {np.max(np.abs(deformation_1.vec.FV().NumPy())) * 1000:.2g}nm")
# Draw(deformation_1, clipping=clipping)

# %%
points, _, surface_ratio, median_distance = compute_membrane_points(mesh, 10, 10)
print(f"Part of surface covered: {surface_ratio * 100:.2f}%")
print(f"Median distance to regular grid points: {median_distance * 1000:.3g}nm")
deformation_2 = compute_solution(points, pulling_force)
print(f"Maximum deformation: {np.max(np.abs(deformation_2.vec.FV().NumPy())) * 1000:.2g}nm")
# Draw(deformation_2, clipping=clipping)

# %%
points, _, surface_ratio, median_distance = compute_membrane_points(mesh, 20, 20)
print(f"Part of surface covered: {surface_ratio * 100:.2f}%")
print(f"Median distance to regular grid points: {median_distance * 1000:.3g}nm")
deformation_3 = compute_solution(points, pulling_force)
print(f"Maximum deformation: {np.max(np.abs(deformation_3.vec.FV().NumPy())) * 1000:.2g}nm")
# Draw(deformation_3, clipping=clipping)

# %%
def visualize_deformation_2d(mesh, deformation):
    X, Y, Z = sample_surface(mesh, deformation)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, (ecs_height - Z) * 1000, cmap="gnuplot", shading='gouraud', vmin=0, vmax=30)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    cbar = fig.colorbar(c, ax=ax, label='deformation [nm]')
    plt.show()

# %%
# Visualize elastic deformation of membrane and ECS after some post-processing
visualize_deformation_2d(mesh, deformation_1)
visualize_deformation_2d(mesh, deformation_2)
visualize_deformation_2d(mesh, deformation_3)

# %%
