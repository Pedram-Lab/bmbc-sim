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
# # Computing the tortuosity of a porous medium
# # This script demonstrates how to compute the tortuosity of a porous medium
# using a simple diffusion model. The tortuosity is a measure of the effective
# path length of a substance in a porous medium compared to the straight-line
# distance. It is defined as the ratio of the effective diffusion coefficient to
# the diffusion coefficient in free space. The effective diffusion coefficient
# can be computed by solving the diffusion equation in the porous medium and
# comparing the concentration profile to the one in free space. The tortuosity
# is then given by the square root of the ratio of the effective diffusion
# coefficient to the diffusion coefficient in free space.

# %%
from math import pi
from typing import Tuple

import numpy as np
from netgen.occ import Box, Sphere, Pnt, OCCGeometry
from ngsolve import (Mesh, H1, GridFunction, BilinearForm, grad, dx, Integrate,
    SetNumThreads, TaskManager)
from ngsolve.webgui import Draw
from astropy.units import um, ms, us

import ecsim.units

# %%
# Parameters (some generic solution in extracellular space)
DIFFUSIVITY = ecsim.units.convert(1 * um ** 2 / ms, ecsim.units.DIFFUSIVITY)
TAU = ecsim.units.convert(1 * us, ecsim.units.TIME)

# %%
# Define geometry
def create_porous_geometry(
        n_spheres_per_dim: int,
        volume_fraction: float,
        mesh_size: float
) -> Mesh:
    """
    Function to create a porous geometry with an array of spheres in a cubic box.
    :param n_spheres_per_dim: Number of spheres per dimension, i.e., the total
        number of spheres is n_spheres_per_dim^3
    :param volume_fraction: Volume fraction of the spheres; can't be larger than
        ratio of sphere volume to box volume (=pi/6 ~ 0.52)
    :param mesh_size: Mesh size
    :return: Netgen geometry
    """

    if volume_fraction > pi / 6:
        raise ValueError("Volume fraction of spheres can't be larger than pi/6")

    # Create a cubic box
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1)).bc("reflective")
    box.faces.col = (0, 1, 0)
    box.faces[0].bc("left")
    box.faces[1].bc("right")

    # Create spheres
    if n_spheres_per_dim > 0:
        sphere_midpoints_per_dim = np.linspace(0, 1, 2 * n_spheres_per_dim + 1)[1::2]
        max_radius = 1 / (2 * n_spheres_per_dim)
        radius = (volume_fraction / (pi / 6)) ** (1 / 3)  * max_radius

        for x in sphere_midpoints_per_dim:
            for y in sphere_midpoints_per_dim:
                for z in sphere_midpoints_per_dim:
                    sphere = Sphere(Pnt(x, y, z), radius)
                    sphere.faces.col = (1, 0, 0)
                    sphere.bc("reflective")
                    box = box - sphere

    return Mesh(OCCGeometry(box).GenerateMesh(maxh=mesh_size))

# %%
ecs = create_porous_geometry(n_spheres_per_dim=5, volume_fraction=0.1, mesh_size=0.1)
clipping_settings = {"pnt": (0.5, 0.5, 0.99), "vec": (0, 0, -1)}
visualization_settings = {"Light": {"ambient": 1.0}}
Draw(ecs, clipping=clipping_settings, settings=visualization_settings)

# %%
def compute_diffusion_time(
        mesh: Mesh,
        diffusivity: float,
        tau: float
) -> Tuple[float, GridFunction]:
    """
    Function to compute the time needed for a substance to diffuse through a
    porous medium. The time is given by the time it takes for the substance to
    reach a concentration of 50% of the concentration at the left boundary at
    the right boundary.
    :param mesh: Netgen mesh
    :param diffusivity: Diffusion coefficient
    :param tau: Time step
    :return: Time needed for diffusion and final concentration
    """
    # Define FE space
    fes = H1(mesh, order=1, dirichlet="left")
    concentration = GridFunction(fes)

    # Define diffusion problem
    v_test, v_trial = fes.TnT()
    a = BilinearForm(fes)
    a += diffusivity * grad(v_test) * grad(v_trial) * dx
    m = BilinearForm(fes)
    m += v_test * v_trial * dx

    a.Assemble()
    m.Assemble()
    m.mat.AsVector().data += tau * a.mat.AsVector()
    mstar_inv = m.mat.Inverse(fes.FreeDofs())

    # Time stepping
    concentration.Set(0)
    concentration.Set(1, definedon=mesh.Boundaries("left"))
    left_concentration = Integrate(concentration, mesh, definedon=mesh.Boundaries("left"))
    right_concentration = 0
    t = 0

    SetNumThreads(8)
    with TaskManager():
        while right_concentration < 0.5 * left_concentration:
            # Solve the diffusion equation
            res = -tau * (a.mat * concentration.vec)
            concentration.vec.data += mstar_inv * res
            right_concentration = Integrate(concentration, mesh, definedon=mesh.Boundaries("right"))
            t += tau

    return t, concentration

# %%
# Compute diffusion time in free space
mesh = create_porous_geometry(n_spheres_per_dim=0, volume_fraction=0.0, mesh_size=0.1)
t_unhindered, concentration = compute_diffusion_time(mesh, diffusivity=DIFFUSIVITY, tau=TAU)
print(f"Time needed for diffusion: {t_unhindered:.2f} ms")
Draw(concentration, clipping=clipping_settings, settings=visualization_settings)

# %%
# Compute diffusion time in porous medium
# Since the diffusion time is inversely proportional to the diffusion
# coefficient, we can use it to compute the tortuosity (which is sqrt(D_eff / D))
mesh = create_porous_geometry(n_spheres_per_dim=5, volume_fraction=0.49, mesh_size=0.1)
t_hindered, concentration = compute_diffusion_time(mesh, diffusivity=DIFFUSIVITY, tau=TAU)
print(f"Time needed for diffusion: {t_hindered:.2f} ms")
print(f"Tortuosity: {np.sqrt(t_hindered / t_unhindered):.2f}")
Draw(concentration, clipping=clipping_settings, settings=visualization_settings)

# %%
