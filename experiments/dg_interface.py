from time import time

import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt

left = occ.Box((0, 0, 0), (1, 1, 1)).mat("left")
left.faces[0].bc("influx")
left.faces[1].bc("interface")
right = occ.Box((1, 0, 0), (2, 1, 1)).mat("right")
right.faces[2].bc("outflux")
geo = occ.OCCGeometry(occ.Glue([left, right]))
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))
print(f"Mesh has {mesh.nv} vertices and {mesh.ne} elements.")
Draw(mesh)


ORDER = 1
DT = 0.001
fes = ngs.L2(mesh, order=ORDER, dgjumps=True)
u, v = fes.TnT()

jump_u = u - u.Other()
jump_v = v - v.Other()
n = ngs.specialcf.normal(3)
h = ngs.specialcf.mesh_size
mean_dudn = 0.5 * n * (ngs.grad(u) + ngs.grad(u.Other()))
mean_dvdn = 0.5 * n * (ngs.grad(v) + ngs.grad(v.Other()))


def create_diffusion(mat, coeff):
    diff = coeff * ngs.grad(u) * ngs.grad(v) * ngs.dx(mat)
    reg = 2 * coeff * (ORDER + 1) * (ORDER + 3) / (3 * h)
    reg = reg * jump_u * jump_v * ngs.dx(mat, skeleton=True)
    bnd = coeff * (mean_dudn * jump_v + mean_dvdn * jump_u) * ngs.dx(mat, skeleton=True)
    return (diff + reg - bnd).Compile()

coeff_left = 0.01
coeff_right = 0.1
# This doesn't work...
coeff = mesh.MaterialCF({"left": coeff_left, "right": coeff_right})
reg = (coeff_left + coeff_right) * (ORDER + 1) * (ORDER + 3) / (3 * h)
mean_dudn = 0.5 * n * (coeff * ngs.grad(u) + coeff.Other() * ngs.grad(u.Other()))
mean_dvdn = 0.5 * n * (coeff * ngs.grad(v) + coeff.Other() * ngs.grad(v.Other()))
diffusion = create_diffusion("left", coeff_left) + create_diffusion("right", coeff_right) \
    + reg * jump_u * jump_v * ngs.ds("interface", skeleton=True) \
    + 0.5 * (mean_dudn * jump_v + mean_dvdn * jump_u) * ngs.ds("interface", skeleton=True)

a = ngs.BilinearForm(diffusion).Assemble()
m = ngs.BilinearForm(u * v * ngs.dx).Assemble()

gfu = ngs.GridFunction(fes, name="uDG")
gfu.Set(10 * ngs.x + ngs.cos(10 * ngs.y) + ngs.sin(10 * ngs.z))
Draw(gfu)
u_t = ngs.GridFunction(fes, multidim=0)
u_t.AddMultiDimComponent(gfu.vec)

n = 0
res = gfu.vec.CreateVector()
ngs.SetNumThreads(8)
integrals = []
with ngs.TaskManager():
    start = time()
    m_star = m.mat.CreateMatrix()
    m_star.AsVector().data = m.mat.AsVector() + DT * a.mat.AsVector()
    pre = m_star.CreateSmoother(fes.FreeDofs(), GS=True)
    m_star_inv = ngs.GMRESSolver(m_star, pre=pre)

    while n * DT < 1:
        n += 1
        a.Apply(gfu.vec, res)
        gfu.vec.data += -DT * (m_star_inv * res)
        if n % 10 == 0:
            u_t.AddMultiDimComponent(gfu.vec)
            integrals.append(ngs.Integrate(gfu, mesh))
            print("n =", n, "integral =", integrals[-1])
    print("Time taken:", time() - start)

Draw(u_t, mesh, "uDG")


integrals = np.array(integrals)
dev = np.abs(integrals - integrals[0])
plt.plot(np.arange(len(integrals)) * DT, dev)
plt.ylim(0, dev.max() * 1.1)
plt.show()
