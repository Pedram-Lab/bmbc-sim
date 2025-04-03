"""This script recreates the geometry of [Tour, Tsien; 2007] using float-based microns."""

import ngsolve as ngs
from netgen import occ
from netgen.occ import Pnt, Box, Cylinder, Dir
from ngsolve.webgui import Draw

# Dimensiones en micras (μm)
side = 3.0              # μm
ecs_height = 0.1        # μm
cytosol_height = 3.0    # μm
channel_radius = 0.005   # μm = 50 nm

# Crear cubo ECS (encima del citosol)
ecs_box = Box(
    Pnt(0, 0, cytosol_height),
    Pnt(side, side, cytosol_height + ecs_height)
).mat("ecs")

# Crear cubo citosol
cytosol_box = Box(
    Pnt(0, 0, 0),
    Pnt(side, side, cytosol_height)
).mat("cytosol")

# Crear cilindro canal en el centro, conectando ECS con citosol
channel_cylinder = Cylinder(
    Pnt(side / 2, side / 2, cytosol_height),  # base point
    Dir(0, 0, 1),                             # direction vector
    channel_radius,
    ecs_height
).mat("channel")

# Combinar geometrías
domain = ecs_box + cytosol_box + channel_cylinder

# Etiquetar fronteras opcionales
domain.faces.Max().bc("top")                # Parte superior del ECS
domain.faces.Min().bc("bottom")             # Fondo del citosol
channel_cylinder.faces.curve.bc("channel")   # Superficie lateral del canal

# # Crear geometría OCC y generar malla
# geo = occ.OCCGeometry(domain)
# mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.3))  # maxh también en μm

# # Visualizar la malla
# Draw(mesh)


# # Combinar geometrías
# domain = ecs_box + cytosol_box + channel_cylinder

# # Etiquetar fronteras opcionales
# ecs_box.faces.MaxZ().bc("top")                # Parte superior del ECS
# cytosol_box.faces.MinZ().bc("bottom")         # Fondo del citosol
# channel_cylinder.faces.curve.bc("channel")    # Superficie lateral del canal

# # Crear geometría OCC y generar malla
# geo = occ.OCCGeometry(domain)
# mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.3))  # maxh también en μm

# # Visualizar la malla
# Draw(mesh)





# """This script recreates the geometry of [Tour, Tsien; 2007]."""

# import ngsolve as ngs
# from netgen import occ
# from netgen.occ import Pnt, Box, Cylinder
# from ngsolve.webgui import Draw
# import astropy.units as u

# # Conversión de unidades
# um = 1e-6

# # Dimensiones
# side = 3 * um
# ecs_height = 0.1 * um
# cytosol_height = 3 * um
# channel_radius = 0.05 * um  # ◉ cambiado de 5 nm a 50 nm para evitar errores de geometría

# # Crear cubo ECS (encima del citosol)
# ecs_box = Box(
#     Pnt(0, 0, cytosol_height),
#     Pnt(side, side, cytosol_height + ecs_height)
# ).mat("ecs")

# # Crear cubo citosol
# cytosol_box = Box(
#     Pnt(0, 0, 0),
#     Pnt(side, side, cytosol_height)
# ).mat("cytosol")

# # Crear cilindro canal en el centro, conectando ECS con citosol
# channel_cylinder = Cylinder(
#     Pnt(side / 2, side / 2, cytosol_height),
#     Pnt(side / 2, side / 2, cytosol_height + ecs_height),
#     r=channel_radius
# ).mat("channel")

# # Combinar geometrías
# domain = ecs_box + cytosol_box + channel_cylinder

# # Etiquetar fronteras opcionales
# ecs_box.faces.MaxZ().bc("top")                # Parte superior del ECS
# cytosol_box.faces.MinZ().bc("bottom")         # Fondo del citosol
# channel_cylinder.faces.curve.bc("channel")    # Superficie lateral del canal

# # Crear geometría OCC y generar malla
# geo = occ.OCCGeometry(domain)
# mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.3 * um))

# # Visualizar la malla
# Draw(mesh)


# """This script recreates the geometry of [Tour, Tsien; 2007]."""
# import ngsolve as ngs
# from netgen import occ
# from netgen.occ import Pnt, Box, Cylinder
# from ngsolve.webgui import Draw
# import astropy.units as u

# # Conversión de unidades
# um = 1e-6
# nm = 1e-9

# # Dimensiones
# side = 3 * um
# ecs_height = 0.1 * um
# cytosol_height = 3 * um
# channel_radius = 5 * nm

# # Crear cubo ECS (encima del citosol)
# ecs_box = Box(
#     Pnt(0, 0, cytosol_height),
#     Pnt(side, side, cytosol_height + ecs_height)
# ).mat("ecs")

# # Crear cubo citosol
# cytosol_box = Box(
#     Pnt(0, 0, 0),
#     Pnt(side, side, cytosol_height)
# ).mat("cytosol")

# # Crear cilindro canal en el centro, conectando ECS con citosol
# channel_cylinder = Cylinder(
#     Pnt(side / 2, side / 2, cytosol_height),
#     Pnt(side / 2, side / 2, cytosol_height + ecs_height),
#     r=channel_radius
# ).mat("channel")

# # Combinar geometrías
# domain = ecs_box + cytosol_box + channel_cylinder

# # Etiquetar fronteras opcionales
# ecs_box.faces.MaxZ().bc("top")                # Parte superior del ECS
# cytosol_box.faces.MinZ().bc("bottom")         # Fondo del citosol
# channel_cylinder.faces.curve.bc("channel")    # Superficie lateral del canal

# # Crear geometría OCC y generar malla
# geo = occ.OCCGeometry(domain)
# mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.3 * um))

# # Visualizar la malla
# Draw(mesh)


# """This script recreates the simulation of [Tour, Tsien; 2007]."""
# import ngsolve as ngs
# from netgen import occ
# from netgen.occ import Pnt, Box, Cylinder
# import astropy.units as u

# import ecsim
# from ecsim.simulation import recorder, transport

# # Create a geometry 


# # Conversión de unidades
# um = 1e-6
# nm = 1e-9

# # Sizes
# side = 3 * um
# ecs_height = 0.1 * um
# cytosol_height = 3 * um
# channel_radius = 5 * nm
# channel_height = ecs_height

# # Create ECS cube
# ecs_box = Box(Pnt(0, 0, cytosol_height), Pnt(side, side, cytosol_height + ecs_height)).mat("ecs")

# # Create cytosol cube
# cytosol_box = Box(Pnt(0, 0, 0), Pnt(side, side, cytosol_height)).mat("cytosol")

# # Crate cilinder/channel
# channel_cylinder = Cylinder(
#     Pnt(side/2, side/2, cytosol_height),
#     Pnt(side/2, side/2, cytosol_height + ecs_height),
#     r=channel_radius
# ).mat("channel")

# # Combine solids
# domain = ecs_box + cytosol_box + channel_cylinder

# # Label borders
# domain.faces.MinZ().bc("bottom")
# domain.faces.MaxZ().bc("top")
# channel_cylinder.faces.curve.bc("channel")

# # Crear geometría OCC
# geo = OCCGeometry(domain)

# # Generar malla
# mesh = Mesh(geo.GenerateMesh(maxh=0.3 * um))