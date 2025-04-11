# %%
import time

from netgen import occ
import ngsolve as ngs
from ngsolve.webgui import Draw

# %%
# Parameters
D = 2
n_steps = 1000
dt = 1 / n_steps
n_threads = 1

# %%
# Geometry
box = occ.Box((0, 0, 0), (1, 1, 1))
for k in range(2, 6):
    box.faces[k].bc("reflective")
box.faces[0].bc("left")
box.faces[1].bc("right")
box.mat("ecs")
mesh = ngs.Mesh(occ.OCCGeometry(box).GenerateMesh(maxh=0.1))
print(f"Created mesh with {mesh.nv} vertices and {mesh.ne} elements")
Draw(mesh)

# %%
# FEM system (diffusion with influx on the left boundary, efflux on the right)
fes = ngs.H1(mesh, order=1)
u, v = fes.TnT()
influx = ngs.Parameter(1)
efflux = ngs.Parameter(0)

a = ngs.BilinearForm(fes)
a += D * ngs.grad(u) * ngs.grad(v) * ngs.dx
a += u * v * ngs.ds("left")

m = ngs.BilinearForm(fes)
m += u * v * ngs.dx

u = ngs.GridFunction(fes)
us = ngs.GridFunction(fes, multidim=0)

f = ngs.LinearForm(fes)
f += influx * v * ngs.ds("left")
f += efflux * v * ngs.ds("right")


# %%
start = time.time()
ngs.SetNumThreads(n_threads)
with ngs.TaskManager():
    a.Assemble()
    m.Assemble()

    m.mat.AsVector().data += dt * a.mat.AsVector().data
    mstar_inv = m.mat.Inverse()
    setup_end = time.time()
    print("Setup time:", setup_end - start)

    t = 0
    k = 0
    us.AddMultiDimComponent(u.vec)
    for i in range(n_steps):
        f.Assemble()
        res = dt * (f.vec - a.mat * u.vec)
        u.vec.data += mstar_inv * res
        t += dt
        if t > 0.5:
            influx.Set(0)
            efflux.Set(-1)
        k += 1
        if k == 10:
            us.AddMultiDimComponent(u.vec)
            k = 0

end = time.time()
print("Total time:", end - start)

# %%
# Plot
settings = {'Multidim': {'animate': True, 'speed': 10}}
Draw(us, min=0, max=1, settings=settings)

# %%
