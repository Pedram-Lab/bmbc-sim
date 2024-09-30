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
# According to this paper, Young's modulus is determined by the following expression:
#
# $Y = \frac{5\sigma}{2 \Delta x}$
#
# where $\sigma$ is the Hookean spring constant, and $\Delta x$ is the LSM lattice node spacing.
#
# | Parameter                          | Definition                              | Value                      |
# |------------------------------------|-----------------------------------------| ---------------------------|
# | $\sigma_g$                         | Glycocalyx spring constant              | 0.01 pN/nm                 |
# | $\sigma_m$                         | Membrane spring constant                | 0.4 pN/nm                  |
# | $\sigma_b$                         | Bond spring constant                    | 2 pN/nm                    |
# | $\sigma_s$                         | Substrate spring constant               | 1000 pN/nm                 |
# | $\Delta x$                         | Lattice node spacing                    | 20 nm                      |
# | $ν$                                | Poisson ratio                           | 0.25                       |
# | Y$_g$                              | Young's modulus for glycocalyx          | 0.0025 pN/nm$^2$           |
# | Y$_{m}$                            | Young's modulus for PM                  | 0.05 pN/nm$^2$             |
# | Y$_{b}$                            | Young's modulus for bond                | 0.25 pN/nm$^2$             |
# | Y$_{s}$                            | Young's modulus for substrate           | 125 pN/nm$^2$              |
# | l$_{g}$                            | Glycocalyx thickness                    | 45 nm                      |
# | l$_{b}$                            | Equilibrium bond length                 | 18 nm                      |
# | F                                  | Bond force                              | 4 pN                       |
# | l$_{d}$                            | Equilibrium separation distance         | 13.5 nm                    |
#
#
# | Compartment                          | Size                         |
# |------------------------------------|--------------------------------|
# | Cell membrane                | 1.4 μm x 1.4 μm x 40 nm   (Nodes = 70x70x3)           |
# | ECM substrate                | 1.4 μm x 1.4 μm x 400 nm    (Nodes = 70x70x21)        |
# | Glycocalyx                   | 1.4 μm x 1.4 μm x 45 nm                               |
# | Bond formation geometry      | 240 nm x 240 nm x height of the compartment                               |
#
# According to Ashurst and Hoover, to compare the elastic energy according to the linear-finite-element theory with the energy from the Hooke's law springs, the Lamé constants $\lambda$ and $\eta$ need to be expressed in terms of the spring constant:
#
# $\lambda$ = $\eta = \frac{1}{4}\sqrt{3}*k$,
#
# where $k$ is the spring constant, with units of pN/nm. Using this formula, the Lamé constants can be computed using above values.  
#
# Note: Normally, in the continuum elasticity theory, $\lambda$ describes the material's resistance to uniform compression, and it has units of pressure because it's derived from stress-strain relationships (where stress = force/area). However, in the model described by Ashurt and Hoover, $\lambda$ is connected directly to a spring constant. Springs don't directly involve an area, so the spring constant $k$ has units of force per length (N/m), and the Lamé constant is expressed with the same units in the simplified spring model.
#

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

# %%
# Define units and parameters for the simulation
from ecsim.units import *
FORCE = MASS * LENGTH / TIME ** 2
PRESSURE = FORCE / LENGTH ** 2

s = convert(1.4 * u.um, LENGTH) / 2
ecs_height = convert(45 * u.nm, LENGTH)
membrane_height = convert(40 * u.nm, LENGTH)
mesh_size = convert(10 * u.nm, LENGTH)
cutout_size = convert(300 * u.nm, LENGTH)

pulling_force = convert(10 * u.pN, FORCE)
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
# Draw(mesh)

# %%
# Set Lamé constants based on Young's modulus and Poisson ratio
E = mesh.MaterialCF({"ecs": youngs_modulus_ecs, "membrane": youngs_modulus_membrane})
nu = poisson_ratio

mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

clipping = {"function": False,  "pnt": (0, 0, 0), "vec": (0, 1, 0)}
visualization_settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}, "deformation": 3.0}


# %%
def stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * Id(3)

# %%
# Define mechanic problem
fes = VectorH1(mesh, order=1, dirichlet="substrate")
u, v = fes.TnT()

blf = BilinearForm(fes)
blf += InnerProduct(stress(Sym(Grad(u))), Sym(Grad(v))).Compile() * dx
blf.Assemble()
a_inv = blf.mat.Inverse(fes.FreeDofs())


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
def sample_cutout(mesh, deformation, n_samples=300):
    x = np.linspace(-cutout_size / 2, cutout_size / 2, n_samples)
    y = np.linspace(-cutout_size / 2, cutout_size / 2, n_samples)
    X, Y = np.meshgrid(x, y)
    z = ecs_height

    points = [(x, y, z) for x, y in zip(X.flatten(), Y.flatten())]
    mips = [mesh(*p) for p in points]
    deformation_z = [deformation.components[2](p) for p in mips]

    return X, Y, z + np.array(deformation_z).reshape((n_samples, n_samples))

# %%
# Compute elastic deformation of membrane and ECS for one ...
points = [(0, 0, ecs_height)]
deformation_1 = compute_solution(points, pulling_force)
# Draw(deformation_1, settings=visualization_settings, clipping=clipping)

# %%
# ... two ...
points = [(-0.04, 0, ecs_height), (0.04, 0, ecs_height)]
deformation_2 = compute_solution(points, 0.9 * pulling_force)
# Draw(deformation_2, settings=visualization_settings, clipping=clipping)

# %%
# ... and three points
points = [(-0.03, -0.02, ecs_height), (0.03, -0.02, ecs_height), (0, 0.03, ecs_height)]
deformation_3 = compute_solution(points, 0.7 * pulling_force)
# Draw(deformation_3, settings=visualization_settings, clipping=clipping)

# %%
def visualize_deformation_2d(mesh, deformation):
    X, Y, Z = sample_cutout(mesh, deformation)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, (ecs_height - Z) * 1000, cmap="gnuplot", shading='gouraud', vmin=0, vmax=18)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    cbar = fig.colorbar(c, ax=ax, label='deformation [nm]')
    plt.show()

# %%
def visualize_deformation_3d(mesh, deformation):
    X, Y, Z = sample_cutout(mesh, deformation)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    c = ax.plot_surface(X, Y, (ecs_height - Z) * 1000, cmap="gnuplot", vmin=0, vmax=18, antialiased=False)
    ax.set_xlim3d(-0.05, 0.05)
    ax.set_ylim3d(-0.05, 0.05)
    ax.view_init(160, 70, 0)
    ax.set_facecolor('black')
    ax.set_box_aspect((10, 10, 5))
    ax.set_axis_off()
    cbar = fig.colorbar(c, ax=ax, label='deformation [nm]')
    plt.show()

# %%
# Visualize elastic deformation of membrane and ECS after some post-processing
visualize_deformation_2d(mesh, deformation_1)
visualize_deformation_2d(mesh, deformation_2)
visualize_deformation_2d(mesh, deformation_3)

# %%
# Visualize elastic deformation of membrane and ECS after some post-processing
visualize_deformation_3d(mesh, deformation_1)
visualize_deformation_3d(mesh, deformation_2)
visualize_deformation_3d(mesh, deformation_3)

# %%
