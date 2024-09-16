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
# # Scaling a diffusion problem
# This script solves the diffusion equation on a 1m^3 cube with a point source of calcium ions during 1s. The diffusion
# constant and all other parameters are converted to different systems of base units to assess the scaling behavior of
# the method with respect to physical units.
#
# The result shows that no matter the SI-base units that are used for time, length, and substance, the system shows the same behavior, verifying that the system is invariant under scaling.

# %%
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as au
from tqdm.notebook import trange
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

# %%
# Fix base units; all other units are derived from these (change prefixes to see the effect of scaling, e.g., au.m to au.dm)
time = au.s
length = au.m
substance = au.mol

# %%
# Parameters (do not change)
D = (1 * au.m ** 2 / au.s).to(length**2 / time).value
ca_source = (1 * au.mol / au.s).to(substance / time).value
t_end = (1 * au.s).to(time).value
side_length = (1 * au.m).to(length).value
mesh_size = (0.05 * au.m).to(length).value
n_steps = 1000
sample_interval = 10
dt = (1 * au.s / n_steps).to(time).value

# %%
# Define geometry
s = side_length
c = 0.1 * s
d = c / 2
cube = OrthoBrick(Pnt(0, 0, 0), Pnt(s, s, s))
source = OrthoBrick(Pnt(c - d, c - d, c - d), Pnt(c + d, c + d, c + d))
source_point = (c, c, c)
far_point = (s - c, s - c, s - c)
geo = CSGeometry()
geo.Add((cube - source).mat("volume"))
geo.Add(source.mat("source"))
mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))

# %%
# Define FE space
fes = H1(mesh, order=1)
concentration = GridFunction(fes)

# %%
# normalize the source inflow wrt volume of the compartment
source_volume = Integrate(1, mesh, definedon=mesh.Materials("source"))
source = mesh.MaterialCF({"source": ca_source / source_volume, "volume": 0})

# %%
# Define diffusion problem
u, v = fes.TnT()
a = BilinearForm(fes)
a += D * grad(u) * grad(v) * dx
m = BilinearForm(fes)
m += u * v * dx

f = LinearForm(fes)
f += source * v * dx  # approximated point source of calcium
f.Assemble()

a.Assemble()
m.Assemble()
m.mat.AsVector().data += dt * a.mat.AsVector()
mstar_inv = m.mat.Inverse(fes.FreeDofs())

# %%
# Time stepping
def time_stepping(ca):
    ca_t = GridFunction(fes, multidim=0)
    ca_t.AddMultiDimComponent(ca.vec)
    mip_source = mesh(*source_point)
    mip_far = mesh(*far_point)

    t = [0]
    near = [ca(mip_source)]
    far = [ca(mip_far)]
    mass = [Integrate(ca, mesh)]
    for i in trange(n_steps):
        # Solve the diffusion equation
        res = dt * (f.vec - a.mat * ca.vec)
        ca.vec.data += mstar_inv * res
        if (i + 1) % sample_interval == 0:
            ca_t.AddMultiDimComponent(ca.vec)
            t.append((i + 1) * dt)
            near.append(ca(mip_source))
            far.append(ca(mip_far))
            mass.append(Integrate(ca, mesh))
    return np.array(t), near, far, mass, ca_t

# %%
# Compute time evolution with the potential
SetNumThreads(12)
with TaskManager():
    concentration.Set(0)
    t, ca_near, ca_far, ca_mass, ca_t = time_stepping(concentration)
t = (t * time).to(au.s)

# %%
# Visualize whole solution if desired
clipping = {"function": True,  "pnt": (5, 5, 5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
# Draw(ca_t, mesh, clipping=clipping, settings=settings, interpolate_multidim=True, animate=True, autoscale=False, min=0.0, max=1.0)

# %%
plt.plot(t, (ca_near * substance / length**3).to(au.mol / au.m**3), label="At ca-source [mol / m^3]")
plt.plot(t, (ca_far * substance / length**3).to(au.mol / au.m**3), label="Far from ca-source [mol / m^3]")
plt.plot(t, (ca_mass * substance).to(au.mol), label="Total amount [mol]")
plt.xlabel("Time [s]")
plt.legend()
plt.show()

# %%
