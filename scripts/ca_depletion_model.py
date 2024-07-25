# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.csg import *

from ecsim.geometry import create_axis_aligned_plane, create_axis_aligned_cylinder


# %%
def create_geometry(*, side_length, cytosol_height, ecs_height, mesh_size):
    geometry = CSGeometry()

    left = create_axis_aligned_plane(0, -side_length / 2, -1)
    right = create_axis_aligned_plane(0, side_length / 2, 1)
    front = create_axis_aligned_plane(1, -side_length / 2, -1)
    back = create_axis_aligned_plane(1, side_length / 2, 1)

    cytosol_cutout = create_axis_aligned_cylinder(2, 0, 0, 0.1) \
             * create_axis_aligned_plane(2, cytosol_height, 1, "channel") \
             * create_axis_aligned_plane(2, cytosol_height - ecs_height / 2, -1)
    cytosol_cutout.maxh(mesh_size / 2)

    cytosol_bot = create_axis_aligned_plane(2, 0, -1)
    cytosol_top = create_axis_aligned_plane(2, cytosol_height, 1, "membrane")
    cytosol = left * right * front * back * cytosol_bot * cytosol_top
    cytosol.maxh(mesh_size)

    ecs_cutout = create_axis_aligned_cylinder(2, 0, 0, 0.1) \
             * create_axis_aligned_plane(2, cytosol_height + ecs_height / 2, 1) \
             * create_axis_aligned_plane(2, cytosol_height, -1, "channel")
    ecs_cutout.maxh(mesh_size / 2)
    
    ecs_bot = create_axis_aligned_plane(2, cytosol_height, -1, "membrane")
    ecs_top = create_axis_aligned_plane(2, cytosol_height + ecs_height, 1, "ecs_top")
    ecs = left * right * front * back * ecs_bot * ecs_top
    ecs.maxh(mesh_size)

    geometry.Add(cytosol - cytosol_cutout)
    geometry.Add(cytosol * cytosol_cutout)
    geometry.Add(ecs - ecs_cutout)
    geometry.Add(ecs * ecs_cutout)
    return geometry


# %%
ngmesh = create_geometry(side_length=3, cytosol_height=3, ecs_height=0.1, mesh_size=0.25).GenerateMesh()

# %%
mesh = Mesh(ngmesh)

# %%
Draw(mesh)

# %%
fes = H1(mesh, order=2, dirichlet="ecs_top")
u = fes.TrialFunction()
v = fes.TestFunction()

f = LinearForm(fes)
f += 0 * v * dx
f += 1 * v.Trace() * ds(definedon="membrane")

a = BilinearForm(fes)
a += grad(u) * grad(v) * dx

a.Assemble()
f.Assemble()

# %%
concentration = GridFunction(fes)
concentration.Set(15, definedon=mesh.Boundaries("ecs_top"))
res = f.vec.CreateVector()
res.data = f.vec - a.mat * concentration.vec
concentration.vec.data += a.mat.Inverse(fes.FreeDofs()) * res
Draw(concentration)

# %%
