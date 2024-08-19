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
# This is an example of how NGSolve uses `BitArray`s to handle degrees of freedom (dofs) and how one can manipulate them to compute stuff on arbitrary subdomains.

# %%
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *

# %%
# Set up geometry with different boundaries and materials (all of which are "regions")
outer = Rectangle(2, 2).Face()
outer.edges.name="outer"
outer.edges.Max(X).name = "r"
outer.edges.Min(X).name = "l"
outer.edges.Min(Y).name = "b"
outer.edges.Max(Y).name = "t"

inner = MoveTo(1, 0.9).Rectangle(0.3, 0.5).Face()
inner.edges.name="interface"
outer = outer - inner

inner.faces.name="inner"
inner.faces.col = (1, 0, 0)
outer.faces.name="outer"

geo = Glue([inner, outer])
mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=0.4))
Draw(mesh)

# %%
# The dofs are internally set as used (1) or unused (0)
# For spaces on subdomains, one can compress the space to get rid of unused dofs
# Compound spaces (with FESpace) consist of dofs from both space (note how the dofs numbers add up,
# i.e., the dofs on the interface are present twice in the compound space)
uncompressed = H1(mesh, order=1, definedon="inner")
fes1 = Compress(H1(mesh, order=1, definedon="inner"))
fes2 = Compress(H1(mesh, order=2, definedon="outer"))
fes = FESpace([fes1, fes2])
print(f"{uncompressed.ndof=}, {fes1.ndof=}, {fes2.ndof=}, {fes.ndof=}")
print("uncompressed dofs (inner):")
print(uncompressed.FreeDofs())
print("compressed dofs (inner):")
print(fes1.FreeDofs())
print("compressed dofs (outer):")
print(fes2.FreeDofs())
print("compound fes dofs (inner + outer):")
print(fes.FreeDofs())

# %%
# Dofs can be used or not by using BitArrays
# The mesh knows about regions (in the VOLume or the BouNDary) and elements in those regions
# The FESpace knows which dofs belong to those elements
dofs = BitArray(fes.ndof)
dofs[:] = False

for i, el in enumerate(mesh.Region(VOL, "outer").Elements()):
    for d in fes.GetDofNrs(el):
        dofs[d] = True

for i, el in enumerate(mesh.Region(BND, "r|b").Elements()):
    if i % 2 == 0:
        for d in fes.GetDofNrs(el):
            dofs[d] = False

print(dofs)

# %%
# By using a BitArray to mask dofs, one can use (Bi)LinearForms defined on the
# whole mesh to compute solutions on only parts of the mesh
gfu = GridFunction(fes)

ui, uo = fes.TrialFunction()
vi, vo = fes.TestFunction()
a = BilinearForm(fes)
a += grad(uo) * grad(vo) * dx("outer")

f = LinearForm(fes)
f += vo * dx("outer")

a.Assemble()
f.Assemble()

# %%
# The BilinearForm a has only zeros for the dofs of the "inner" region
# By supplying a BitArray, we tell it not to invert on these dofs, so the inversion is not a problem
gfu.vec.data = a.mat.Inverse(dofs) * f.vec
Draw(gfu.components[1], settings={"deformation": 0.5})

# %%
