# %%
import time

from netgen import occ
import ngsolve as ngs
from ngsolve.webgui import Draw

# %%
# Parameters
D = 2
N_STEPS = 1000
dt = 1 / N_STEPS
N_THREADS = 16

# %%
# Geometry
box1 = occ.Box((0, 0, 0), (1, 1, 1))
box2 = occ.Box((1, 0, 0), (2, 1, 1))
for k in range(2, 6):
    box1.faces[k].bc("reflective")
    box2.faces[k].bc("reflective")
box1.faces[0].bc("left")
box1.faces[1].bc("interface")
box2.faces[0].bc("interface")
box2.faces[1].bc("right")
box1.mat("cell:left")
box2.mat("cell:right")
geo = occ.Glue([box1, box2])
mesh = ngs.Mesh(occ.OCCGeometry(geo).GenerateMesh(maxh=0.05))
print(f"Created mesh with {mesh.nv} vertices and {mesh.ne} elements")
Draw(mesh)

# %%
# FEM system (diffusion with influx on the left boundary, transmission through the interface)
fesl = ngs.Compress(ngs.H1(mesh, order=1, definedon="cell:left"))
fesr = ngs.Compress(ngs.H1(mesh, order=1, definedon="cell:right"))
fes = fesl * fesr
(ul, ur), (vl, vr) = fes.TnT()
transmission = ngs.Parameter(1)

a = ngs.BilinearForm(fes)
a += D * ngs.grad(ul) * ngs.grad(vl) * ngs.dx("cell:left")
a += D * ngs.grad(ur) * ngs.grad(vr) * ngs.dx("cell:right")

m = ngs.BilinearForm(fes)
m += ul * vl * ngs.dx("cell:left")
m += ur * vr * ngs.dx("cell:right")

u = ngs.GridFunction(fes)
u.components[0].Set(1)
us = ngs.GridFunction(fes, multidim=0)

f = ngs.LinearForm(fes)
cf = transmission * (u.components[0] - u.components[1]) * (vr - vl)
f += cf.Compile(maxderiv=0) * ngs.ds("interface")

# %%
start = time.time()
ngs.SetNumThreads(N_THREADS)
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
    for i in range(N_STEPS):
        f.Assemble()
        res = dt * (f.vec - a.mat * u.vec)
        u.vec.data += mstar_inv * res
        t += dt
        if t > 0.5:
            transmission.Set(0)
        k += 1
        if k == 10:
            us.AddMultiDimComponent(u.vec)
            k = 0

end = time.time()
print("Total time:", end - start)

# %%
# Plot
settings = {'Multidim': {'animate': True, 'speed': 10}}
Draw(us.components[0], mesh, min=0, max=1, settings=settings)

# %%
