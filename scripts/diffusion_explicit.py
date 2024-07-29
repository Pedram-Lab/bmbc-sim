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

# Create the geometry and mesh
ngmesh = create_geometry(side_length=3, cytosol_height=3, ecs_height=0.1, mesh_size=0.25).GenerateMesh()
mesh = Mesh(ngmesh)

Draw(mesh)

fes = H1(mesh, order=2, dirichlet="ecs_top")
u, v = fes.TnT()

# Initial conditions and parameters
time = 0.0
dt = 0.00001

# Bilinear and Linear forms
a = BilinearForm(fes, symmetric=True)
a += grad(u) * grad(v) * dx

m = BilinearForm(fes, symmetric=True)
m += u * v * dx

a.Assemble() #Stiffness Matrix
m.Assemble() #Mass Matrix 
minv = m.mat.Inverse(freedofs=fes.FreeDofs())

# Initialize the solution
gfu = GridFunction(fes)
gfu.Set(0)  # Set initial condition

# Define the source term f
f = LinearForm(fes)
f += 1 * v.Trace() * ds(definedon="membrane")
f.Assemble()

# Time-stepping
scene = Draw(gfu, mesh, "u")

def TimeSteppingExplicit(t0=0, tend=30, nsamples=10):
    cnt = 0
    time = t0
    sample_int = int(floor(tend / dt / nsamples) + 1)
    gfut = GridFunction(gfu.space, multidim=0)
    gfut.AddMultiDimComponent(gfu.vec)
    while time < tend - 0.5 * dt:
        res = dt * (f.vec - a.mat * gfu.vec)
        gfu.vec.data += minv * res
        print("\r", time, end="")
        scene.Redraw()
        if cnt % sample_int == 0:
            gfut.AddMultiDimComponent(gfu.vec)
        cnt += 1
        time = cnt * dt
    return gfut

gfut = TimeSteppingExplicit()

Draw(gfut, mesh, interpolate_multidim=True, animate=True)


# %%

# %%

# %%
