from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw

inner = Rectangle(1,1).Face()
inner.edges.name="interface"
inner.edges.Max(X).name="interface_right"

outer = MoveTo(-1,-1).Rectangle(3,3).Face()
outer.edges.name="dir"
outer = outer-inner

inner.faces.name="inner"
outer.faces.name="outer"
geo = Glue ([inner,outer])
mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=0.2))

Draw (mesh)

print (mesh.GetMaterials(), mesh.GetBoundaries())