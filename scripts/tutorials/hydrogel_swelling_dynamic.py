# %% [markdown]
# # Hydrogel swelling
# This script defines a block of hydrogel that is weakly anchored in a
# surrounding medium and swells/shrinks due to some ion concentration.

# %%
from netgen import occ
import ngsolve as ngs
from ngsolve.webgui import Draw

# %%
# Use parameters from the NGSolve non-linear elasticity tutorial
E, nu = 210, 0.2
MU  = E / (2 * (1 + nu))
LAM = E * nu / ((1 + nu) * (1 - 2 * nu))
ALPHA = 0.1  # diffusion coefficient
TAU = 0.05  # time step size

# %%
# Define geometry
S = 1 / 2
ecs = occ.Box((-S, -S, 0), (S, S, 1.0)).mat("ecs").bc("side")
ecs.faces[2].bc("substrate")

geo = occ.OCCGeometry(ecs)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))
elasticity_fes = ngs.VectorH1(mesh, order=1, dirichlet="substrate")
diffusion_fes = ngs.H1(mesh, order=1)
u  = elasticity_fes.TrialFunction()
concentration = ngs.GridFunction(diffusion_fes)

# %%
# Define mechanic parameters
I = ngs.Id(mesh.dim)
F = I + ngs.Grad(u)
C = F.trans * F
E = 0.5 * (C - I)

def neo_hooke(c):
    """Neo-Hookean material model with internal pressure."""
    det_c = ngs.Det(c)
    return MU * (
        0.5 * ngs.Trace(c - I)
        + MU / LAM * det_c ** (-LAM / (2 * MU))
        - 1
        + concentration * det_c
    )

# %%
# Define mechanic problem
elasticity_stiffness = ngs.BilinearForm(elasticity_fes, symmetric=False)
elasticity_stiffness += ngs.Variation(neo_hooke(C).Compile() * ngs.dx)

deformation = ngs.GridFunction(elasticity_fes)
deformation.vec[:] = 0

elasticity_res = deformation.vec.CreateVector()
deformation_history = ngs.GridFunction(elasticity_fes, multidim=0)
deformation_history.AddMultiDimComponent(deformation.vec)

# %%
# Define diffusion problem
phi, psi = diffusion_fes.TnT()
diffusion_stiffness = ngs.BilinearForm(diffusion_fes)
diffusion_stiffness += ALPHA * ngs.grad(psi) * ngs.grad(phi) * ngs.dx
diffusion_mass = ngs.BilinearForm(diffusion_fes)
diffusion_mass += psi * phi * ngs.dx

concentration.Set(ngs.exp(-10 * ngs.x * ngs.x))
diffusion_res = concentration.vec.CreateVector()

# %%
# Time stepping
N_STEPS = int(0.5 / TAU)
concentration_history = ngs.GridFunction(diffusion_fes, multidim=0)
concentration_history.AddMultiDimComponent(concentration.vec)

for step in range(N_STEPS):
    # Solve elasticity problem
    for it in range(5):
        elasticity_stiffness.Apply(deformation.vec, elasticity_res)
        elasticity_stiffness.AssembleLinearization(deformation.vec)
        inv = elasticity_stiffness.mat.Inverse(elasticity_fes.FreeDofs())
        deformation.vec.data -= inv * elasticity_res
    deformation_history.AddMultiDimComponent(deformation.vec)

    # Solve diffusion problem
    diffusion_stiffness.Assemble()
    diffusion_mass.Assemble()
    diffusion_mass.mat.AsVector().data += TAU * diffusion_stiffness.mat.AsVector()
    m_star_inv = diffusion_mass.mat.Inverse(diffusion_fes.FreeDofs())

    diffusion_res = -TAU * (diffusion_stiffness.mat * concentration.vec)
    concentration.vec.data += m_star_inv * diffusion_res
    concentration_history.AddMultiDimComponent(concentration.vec)

settings = dict(deformation=1)
animation_settings = dict(interpolate_multidim=True, animate=True)
Draw(deformation_history, mesh, "displacement", settings=settings, **animation_settings)
Draw(concentration_history, mesh, **animation_settings)

# %%
