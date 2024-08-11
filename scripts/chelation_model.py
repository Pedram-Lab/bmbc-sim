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
# This script defines a periodic ECS that's tethered to a fixed substrate at the bottom and a membrane at the top. The
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
s = ecs_side
c = cutout_side / 2
left = Plane(Pnt(-s / 2, 0, 0), Vec(-1, 0, 0)).bc("side")
right = Plane(Pnt(s / 2, 0, 0), Vec(1, 0, 0)).bc("side")
front = Plane(Pnt(0, s / 2, 0), Vec(0, 1, 0)).bc("side")
back = Plane(Pnt(0, -s / 2, 0), Vec(0, -1, 0)).bc("side")
ecs = OrthoBrick(Pnt(-s, -s, 0), Pnt(s, s, ecs_height))
cutout = OrthoBrick(Pnt(-c, -c, 0), Pnt(c, c, ecs_height)).mat("ecs_high_density")
ecs = (ecs - cutout).mat("ecs")
membrane = OrthoBrick(Pnt(-s, -s, ecs_height), Pnt(s, s, ecs_height + membrane_height)).mat("membrane")

geo = CSGeometry()
geo.Add(ecs * left * right * front * back)
geo.Add(cutout, bcmod=[(ecs, "ecs_interface")])
geo.Add(membrane * left * right * front * back, bcmod=[(ecs, "ecs_membrane_interface"), (cutout, "ecs_membrane_interface")])
geo.PeriodicSurfaces(left, right)
geo.PeriodicSurfaces(front, back)

mesh = Mesh(geo.GenerateMesh(maxh=0.2))
mesh.ngmesh.SetBCName(0, "substrate")
mesh.ngmesh.SetBCName(13, "substrate")
mesh.ngmesh.SetBCName(2, "membrane_top")

# %%
clipping = {"function": True,  "pnt": (0, 0, 0.5), "vec": (0, 1, 0)}
settings = {"camera": {"transformations": [{"type": "rotateX", "angle": -80}]}}
Draw(mesh, clipping=clipping, settings=settings)

# %%
# Define FE spaces
ecs_fes = Compress(Periodic(H1(mesh, order=2, dirichlet="substrate", definedon=mesh.Materials("ecs|ecs_high_density"))))
elastic_fes = Compress(Periodic(VectorH1(mesh, order=2, dirichlet="substrate")))
fes = FESpace([ecs_fes, elastic_fes])

# %%
# Define and solve diffusion problem
u_ecs, v_ecs = ecs_fes.TnT()

f_ecs = LinearForm(ecs_fes)
f_ecs += v_ecs * dx

a_ecs = BilinearForm(ecs_fes)
a_ecs += grad(u_ecs) * grad(v_ecs) * dx

a_ecs.Assemble()
f_ecs.Assemble()

concentration = GridFunction(ecs_fes)
concentration.vec.data = a_ecs.mat.Inverse(ecs_fes.FreeDofs()) * f_ecs.vec

# %%
# Visualize (diffusion only)
visualization = mesh.MaterialCF({"ecs": concentration, "ecs_high_density": concentration, "membrane": 0})
Draw(visualization, mesh, clipping=clipping, settings=settings)

# %%
# Define and solve linear elasticity problem
u_el, v_el = elastic_fes.TnT()

E = mesh.MaterialCF({"ecs": 50, "ecs_high_density": 1000, "membrane": 100})
nu = 0.2
mu  = E / (2 * (1 + nu))
lam = E * (nu * ((1 + nu) * (1 - 2 * nu)))


def stress(strain):
    return 2 * mu * strain + lam * Trace(strain) * Id(3)


with TaskManager():
    a_el = BilinearForm(elastic_fes)
    a_el += InnerProduct(stress(Sym(Grad(u_el))), Sym(Grad(v_el))).Compile() * dx
    a_el.Assemble()

    force = CoefficientFunction((10, 0, 0))
    f_el = LinearForm(elastic_fes)
    f_el += force * v_el * ds("membrane_top")
    f_el.Assemble()
    
deformation = GridFunction(elastic_fes)
deformation.vec.data = a_el.mat.Inverse(elastic_fes.FreeDofs()) * f_el.vec

# %%
# Visualize (elastic deformation only)
Draw(deformation, deformation=deformation, settings=settings, clipping=clipping)

# %%
