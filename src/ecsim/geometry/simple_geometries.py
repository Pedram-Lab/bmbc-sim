
"""
Geometries consisting of simple geometric shapes for testing purposes.
"""
from netgen.occ import Box, Pnt, Z, Face, Wire, Circle, Glue, Fuse, OCCGeometry
from netgen.meshing import FaceDescriptor
from ngsolve import Mesh
from astropy.units import Quantity

from ecsim.units import LENGTH, convert
from ecsim.geometry._utils import convert_to_volume_mesh



def create_ca_depletion_mesh(
        *,
        side_length_x: Quantity,
        side_length_y: Quantity,
        cytosol_height: Quantity,
        ecs_height: Quantity,
        channel_radius: Quantity,
        mesh_size: Quantity
):
    """
    Creates a cuboid geometry with two compartments: ECS on top and cytosol on
    the bottom. The compartments are separated by a membrane with a circular
    channel in it.
    :param side_length_x: The side length of the cuboid in the x-direction.
    :param side_length_y: The side length of the cuboid in the y-direction.
    :param cytosol_height: The height of the cytosol compartment.
    :param ecs_height: The height of the ECS compartment.
    :param channel_radius: The radius of the channel in the membrane.
    :param mesh_size: The maximum mesh size.
    """
    sx = convert(side_length_x, LENGTH) / 2
    sy = convert(side_length_y, LENGTH) / 2
    cytosol_height = convert(cytosol_height, LENGTH)
    ecs_height = convert(ecs_height, LENGTH)
    channel_radius = convert(channel_radius, LENGTH)
    mesh_size = convert(mesh_size, LENGTH)

    cytosol = Box(Pnt(-sx, -sy, 0), Pnt(sx, sy, cytosol_height))
    ecs = Box(Pnt(-sx, -sy, cytosol_height), Pnt(sx, sy, cytosol_height + ecs_height))
    left, right, front, back, bottom, top = (0, 1, 2, 3, 4, 5)

    # Assign boundary conditions
    for f in [front, back, left, right]:
        cytosol.faces[f].bc("cyt_bnd")
        ecs.faces[f].bc("ecs_bnd")
    cytosol.faces[bottom].bc("cyt_bnd")
    ecs.faces[top].bc("ecs_top")

    # Cut a hole into the ecs-cytosol interface
    channel = Face(Wire(Circle(Pnt(0, 0, cytosol_height), Z, channel_radius)))
    channel.maxh = mesh_size / 2
    channel.bc("channel")
    membrane = (cytosol.faces[top] - channel).bc("membrane")
    interface = Glue([membrane, channel])  # if fused, channel vanishes

    # Only take parts that make up the actual geometry
    geo = Fuse([interface, ecs.faces[top], cytosol.faces[bottom]]
               + [cytosol.faces[f] for f in [front, back, left, right]]
               + [ecs.faces[f] for f in [front, back, left, right]])

    # Generate a mesh on the surface and convert it to a volume mesh
    surface_mesh = OCCGeometry(geo).GenerateMesh(maxh=mesh_size)
    bnd_to_fd = {
        "channel": FaceDescriptor(surfnr=1, domin=2, domout=1, bc=1),
        "membrane": FaceDescriptor(surfnr=2, domin=2, domout=1, bc=2),
        "ecs_top": FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3),
        "ecs_bnd": FaceDescriptor(surfnr=4, domin=1, domout=0, bc=4),
        "cyt_bnd": FaceDescriptor(surfnr=5, domin=2, domout=0, bc=5),
    }
    mesh = convert_to_volume_mesh(surface_mesh, bnd_to_fd)

    # Assign names to boundaries
    mesh.SetBCName(0, "channel")
    mesh.SetBCName(1, "membrane")
    mesh.SetBCName(2, "ecs_top")
    mesh.SetBCName(3, "boundary")
    mesh.SetBCName(4, "boundary")

    # Assign names to regions
    mesh.SetMaterial(1, "ecs")
    mesh.SetMaterial(2, "cytosol")

    return Mesh(mesh)
