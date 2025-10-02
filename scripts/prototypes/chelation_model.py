# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ca chelation model
# This script defines an ECS that's tethered to a fixed substrate at the bottom and a membrane at the top. The
# ECS is filled with diffusing Ca ions that bind to a chelator. The chelator's density is higher in the middle of the
# ECS.

# %%
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

# %%
# Parameters
ecs_side = 3.0
ecs_height = 1.0
cutout_side = 1.0
membrane_height = 0.2

# %%
# Define geometry
s = ecs_side / 2
left = OrthoBrick(Pnt(-s, -s, 0), Pnt(0, s, ecs_height)).mat("ecs_left").bc("side")
right = OrthoBrick(Pnt(0, -s, 0), Pnt(s, s, ecs_height)).mat("ecs_right").bc("side")
membrane = OrthoBrick(Pnt(-s, -s, ecs_height), Pnt(s, s, ecs_height + membrane_height)).mat("membrane").bc("side")

geo = CSGeometry()
geo.Add(membrane)
geo.Add(left)
geo.Add(right)

mesh = Mesh(geo.GenerateMesh(maxh=0.2))
# Usually, interface names can be set with `bcmod = [(other, "name")]`, but in this case too many interfaces are renamed
# Therefore, it's easier to rename them by hand
mesh.ngmesh.SetBCName(5, "substrate")
mesh.ngmesh.SetBCName(9, "substrate")
mesh.ngmesh.SetBCName(14, "membrane_top")
mesh.ngmesh.SetBCName(0, "ecs_membrane_interface")
mesh.ngmesh.SetBCName(11, "ecs_membrane_interface")
mesh.ngmesh.SetBCName(8, "ecs_interface")

# %%
clipping = {"function": True,  "pnt": (0, 0, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
Draw(mesh, clipping=clipping, settings=settings)

# %%
# Define FE spaces
ecs_fes = Compress(H1(mesh, order=2, dirichlet="substrate", definedon=mesh.Materials("ecs_left|ecs_right")))
elastic_fes = VectorH1(mesh, order=2, dirichlet="substrate")
fes = FESpace([ecs_fes, elastic_fes])

# %%
# Define and solve diffusion problem
u_ecs, v_ecs = ecs_fes.TnT()

f_ecs = LinearForm(ecs_fes)
f_ecs += v_ecs * dx

D = mesh.MaterialCF({"ecs_left": 1, "ecs_right": 5})
a_ecs = BilinearForm(ecs_fes)
a_ecs += D * grad(u_ecs) * grad(v_ecs) * dx

a_ecs.Assemble()
f_ecs.Assemble()

concentration = GridFunction(ecs_fes)
concentration.vec.data = a_ecs.mat.Inverse(ecs_fes.FreeDofs()) * f_ecs.vec

# %%
# Visualize (diffusion only)
visualization = mesh.MaterialCF({"ecs_left": concentration, "ecs_right": concentration, "membrane": 0})
Draw(visualization, mesh, clipping=clipping, settings=settings)

# %%
# Define and solve linear elasticity problem
u_el, v_el = elastic_fes.TnT()

E = mesh.MaterialCF({"ecs_left": 50, "ecs_right": 1000, "membrane": 100})
nu = 0.49
mu  = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))


def stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * Id(3)


with TaskManager():
    a_el = BilinearForm(elastic_fes)
    a_el += InnerProduct(stress(Sym(Grad(u_el))), Sym(Grad(v_el))).Compile() * dx
    a_el.Assemble()

    force = CoefficientFunction((0, 0, -10))
    f_el = LinearForm(elastic_fes)
    f_el += force * v_el * ds("membrane_top")
    f_el.Assemble()
    
deformation = GridFunction(elastic_fes)
deformation.vec.data = a_el.mat.Inverse(elastic_fes.FreeDofs()) * f_el.vec

# %%
# Visualize (elastic deformation only)
Draw(deformation, settings=settings)

# %%
