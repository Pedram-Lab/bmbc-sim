from time import time

import ngsolve as ngs
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05))

order=1
dt = 0.001
fes = ngs.L2(mesh, order=order, dgjumps=True)
u, v = fes.TnT()

jump_u = u - u.Other()
jump_v = v - v.Other()
n = ngs.specialcf.normal(2)
h = ngs.specialcf.mesh_size
mean_dudn = 0.5 * n * (ngs.grad(u) + ngs.grad(u.Other()))
mean_dvdn = 0.5 * n * (ngs.grad(v) + ngs.grad(v.Other()))

alpha = 4
reg = alpha * order**2 / h if order > 0 else alpha / h
diffusion = ngs.grad(u) * ngs.grad(v) * ngs.dx \
    + reg * jump_u * jump_v * ngs.dx(skeleton=True) \
    - (mean_dudn * jump_v + mean_dvdn * jump_u) * ngs.dx(skeleton=True) \
    # The following lines vanish because of the Neumann boundary condition
    # + reg * u * v * ngs.ds(skeleton=True) \
    # - (n * ngs.grad(u) * v + n * ngs.grad(v) * u) * ngs.ds(skeleton=True)

b = ngs.CoefficientFunction((20, 5))
uup = ngs.IfPos(b * n, u.Other(), u)
convection = -b * u * ngs.grad(v) * ngs.dx \
    + b * n * uup * jump_v * ngs.dx(skeleton=True) \
    # The following line vanishes because of the Neumann boundary condition
    # + b * n * ngs.IfPos(b * n, 0, u) * v * ngs.ds(skeleton=True)

a = ngs.BilinearForm(0.1 * (diffusion + convection)).Assemble()
m = ngs.BilinearForm(u * v * ngs.dx).Assemble()

gfu = ngs.GridFunction(fes, name="uDG")
gfu.Set(ngs.sin(10 * ngs.x) + ngs.cos(10 * ngs.y))
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
    m_star.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
    pre = m_star.CreateSmoother(fes.FreeDofs(), GS=True)
    m_star_inv = ngs.GMRESSolver(m_star, pre=pre)

    while n * dt < 1:
        n += 1
        a.Apply(gfu.vec, res)
        gfu.vec.data += -dt * (m_star_inv * res)
        if n % 10 == 0:
            u_t.AddMultiDimComponent(gfu.vec)
            integrals.append(ngs.Integrate(gfu, mesh))
    print("Time taken:", time() - start)

Draw(u_t, mesh, "uDG")


integrals = np.array(integrals)
dev = np.abs(integrals - integrals[0])
plt.plot(np.arange(len(integrals)) * dt, dev)
plt.ylim(0, dev.max() * 1.1)
plt.show()