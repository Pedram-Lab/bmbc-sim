# %% [markdown]
# # Hydrogel swelling
# This script defines a block of hydrogel that is weakly anchored in a
# surrounding medium and swells under the influence of a chemical concentration.

# %%
from netgen import occ
import ngsolve as ngs
from ngsolve.webgui import Draw

# %%
# Use the NGSolve parameter class to change parameters after defining everything
E, nu = 210, 0.2
MU  = E / (2 * (1 + nu))
LAM = E * nu / ((1 + nu) * (1 - 2 * nu))

# %%
# Define geometry
S = 1 / 2
ecs = occ.Box((-S, -S, 0), (S, S, 1.0)).mat("ecs").bc("side")
ecs.faces[2].bc("substrate")

geo = occ.OCCGeometry(ecs)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))
fes = ngs.VectorH1(mesh, order=1, dirichlet="substrate")
u  = fes.TrialFunction()

# %%
I = ngs.Id(mesh.dim)
F = I + ngs.Grad(u)
C = F.trans * F
E = 0.5 * (C - I)
PRESSURE = -1e0  # should depend on the concentration

def neo_hooke(C):
    det_c = ngs.Det(C)
    return MU * (0.5 * ngs.Trace(C - I) + MU / LAM * det_c ** (-LAM / (2 * MU)) - 1 - PRESSURE * det_c)

# %%
# Define mechanic problem
a = ngs.BilinearForm(fes, symmetric=False)
a += ngs.Variation(neo_hooke(C).Compile() * ngs.dx)

u = ngs.GridFunction(fes)
u.vec[:] = 0

res = u.vec.CreateVector()
w = u.vec.CreateVector()

for it in range(5):
    print("Newton iteration", it)
    print("energy = ", a.Energy(u.vec))
    a.Apply(u.vec, res)
    a.AssembleLinearization(u.vec)
    inv = a.mat.Inverse(fes.FreeDofs())
    w.data = inv * res
    print("err^2 = ", ngs.InnerProduct(w, res))
    u.vec.data -= w

Draw(u, mesh, "displacement")

# %%
