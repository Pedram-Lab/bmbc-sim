from time import time

import ngsolve as ngs
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.02))

order=1
dt = 0.001
fes = ngs.H1(mesh, order=order)
u, v = fes.TnT()

diffusion = ngs.grad(u) * ngs.grad(v) * ngs.dx

b = ngs.CoefficientFunction((20, 5))
convection = -b * u * ngs.grad(v) * ngs.dx

a = ngs.BilinearForm(0.1 * (diffusion + convection)).Assemble()
m = ngs.BilinearForm(u * v * ngs.dx).Assemble()

gfu = ngs.GridFunction(fes)
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
    # m_star_inv = ngs.krylovspace.GMResSolver(m_star, pre=pre)
    m_star_inv = ngs.GMRESSolver(m_star, pre=pre)
    # m_star_inv = m_star.Inverse(fes.FreeDofs(), inverse="umfpack")

    while n * dt < 1:
        n += 1
        a.Apply(gfu.vec, res)
        # gfu.vec.data += -dt * ngs.krylovspace.GMRes(m_star, res, pre=pre, printrates=False)
        gfu.vec.data += -dt * (m_star_inv * res)
        if n % 10 == 0:
            u_t.AddMultiDimComponent(gfu.vec)
            integrals.append(ngs.Integrate(gfu, mesh))
    print("Time taken:", time() - start)

Draw(u_t, mesh)


integrals = np.array(integrals)
dev = np.abs(integrals - integrals[0])
plt.plot(np.arange(len(integrals)) * dt, dev)
plt.ylim(0, dev.max() * 1.1)
plt.show()