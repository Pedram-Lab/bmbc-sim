# %% [markdown]
# # Hydrogel swelling
# This script defines a block of hydrogel that is weakly anchored in a
# surrounding medium and swells/shrinks due to some ion concentration.

# %%
from netgen import occ
import ngsolve as ngs
from ngsolve.webgui import Draw
from ngsolve import VTKOutput

# %%
# Use parameters from the NGSolve non-linear elasticity tutorial
E, nu = 210, 0.2
MU  = E / (2 * (1 + nu))
LAM = E * nu / ((1 + nu) * (1 - 2 * nu))
ALPHA = 0.1  # diffusion coefficient
TAU = 0.01  # time step size

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

def neo_hooke(f):
    """Neo-Hookean material model with internal pressure."""
    det_f = ngs.Det(f)
    return MU * (
        0.5 * ngs.Trace(f.trans * f - I)
        + MU / LAM * det_f ** (-LAM / MU)
        - 1
        + concentration * det_f
    )

# %%
# Define mechanic problem
elasticity_stiffness = ngs.BilinearForm(elasticity_fes, symmetric=False)
elasticity_stiffness += ngs.Variation(neo_hooke(F).Compile() * ngs.dx)

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

patch_mass = ngs.LinearForm(diffusion_fes)
patch_mass += psi * ngs.dx
patch_mass.Assemble()
prev_mass = patch_mass.vec.CreateVector()
curr_mass = patch_mass.vec.CreateVector()
prev_mass.data = patch_mass.vec

concentration.Set(ngs.exp(-10 * ngs.x * ngs.x))
diffusion_res = concentration.vec.CreateVector()

# %%
# Time stepping
N_STEPS = int(0.5 / TAU)
concentration_history = ngs.GridFunction(diffusion_fes, multidim=0)
concentration_history.AddMultiDimComponent(concentration.vec)

vtk_out = VTKOutput(ma=mesh, coefs=[concentration], names=["concentration"], filename="results/hydrogel/hydrogel")
vtk_out.Do()

for step in range(N_STEPS):
    # Solve elasticity problem
    for it in range(5):
        elasticity_stiffness.Apply(deformation.vec, elasticity_res)
        elasticity_stiffness.AssembleLinearization(deformation.vec)
        inv = elasticity_stiffness.mat.Inverse(elasticity_fes.FreeDofs())
        deformation.vec.data -= inv * elasticity_res
    deformation_history.AddMultiDimComponent(deformation.vec)
    mesh.SetDeformation(deformation)

    # Adjust previous concentration to new volume
    patch_mass.Assemble()
    curr_mass.data = patch_mass.vec
    concentration.vec.FV().NumPy()[:] *=  prev_mass.FV().NumPy() / curr_mass.FV().NumPy()
    prev_mass.data = curr_mass

    # Solve diffusion problem
    diffusion_stiffness.Assemble()
    diffusion_mass.Assemble()
    diffusion_mass.mat.AsVector().data += TAU * diffusion_stiffness.mat.AsVector()
    m_star_inv = diffusion_mass.mat.Inverse(diffusion_fes.FreeDofs())

    diffusion_res = -TAU * (diffusion_stiffness.mat * concentration.vec)
    concentration.vec.data += m_star_inv * diffusion_res
    concentration_history.AddMultiDimComponent(concentration.vec)
    vtk_out.Do()


mesh.UnsetDeformation()
animation_settings = dict(interpolate_multidim=True, animate=True)
el_settings = dict(min=-0.2, max=0.1, autoscale=False)
diff_settings = dict(min=0, max=1.5, autoscale=False)
Draw(deformation_history, mesh, settings=dict(deformation=1), **animation_settings, **el_settings)
Draw(concentration_history, mesh, settings=diff_settings, **animation_settings, **diff_settings)

# %%
