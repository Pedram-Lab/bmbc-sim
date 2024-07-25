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


# %%
def create_axis_aligned_plane(axis, offset, direction, boundary_condition=None):
    point = Pnt(*[0 if i != axis else offset for i in range(3)])
    normal = Vec(*[0 if i != axis else direction for i in range(3)])
    plane = Plane(point, normal)
    plane.bc(boundary_condition if boundary_condition is not None else "default")
    return plane


def create_geometry(*, side_length, cytosol_height, ecs_height):
    geometry = CSGeometry()

    left = create_axis_aligned_plane(0, -side_length / 2, -1)
    right = create_axis_aligned_plane(0, side_length / 2, 1)
    front = create_axis_aligned_plane(1, -side_length / 2, -1)
    back = create_axis_aligned_plane(1, side_length / 2, 1)

    cytosol_bot = create_axis_aligned_plane(2, 0, -1)
    cytosol_top = create_axis_aligned_plane(2, cytosol_height, 1)
    cytosol = left * right * front * back * cytosol_bot * cytosol_top

    ecs_bot = create_axis_aligned_plane(2, cytosol_height, -1)
    ecs_top = create_axis_aligned_plane(2, cytosol_height + ecs_height, 1, "fixed")
    ecs = left * right * front * back * ecs_bot * ecs_top

    geometry.Add(cytosol + ecs)
    return geometry


# %%
ngmesh = create_geometry(side_length=3, cytosol_height=3, ecs_height=0.1).GenerateMesh(maxh=0.25)

# %%
mesh = Mesh(ngmesh)

# %%
Draw(mesh)

# %%
print("Bnd = ", mesh.GetBoundaries())
