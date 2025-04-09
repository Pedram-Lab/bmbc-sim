
"""
Geometries consisting of simple geometric shapes for testing purposes.
"""
from netgen import occ
from ngsolve import Mesh
from astropy.units import Quantity

from ecsim.units import to_simulation_units


def create_ca_depletion_mesh(
        *,
        side_length_x: Quantity,
        side_length_y: Quantity,
        cytosol_height: Quantity,
        ecs_height: Quantity,
        channel_radius: Quantity,
        mesh_size: Quantity,
        channel_mesh_size: Quantity = None
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
    :param channel_mesh_size: The maximum mesh size for the channel. Defaults to
        mesh_size.
    """
    sx = to_simulation_units(side_length_x, 'length') / 2
    sy = to_simulation_units(side_length_y, 'length') / 2
    cytosol_height = to_simulation_units(cytosol_height, 'length')
    ecs_height = to_simulation_units(ecs_height, 'length')
    channel_radius = to_simulation_units(channel_radius, 'length')
    mesh_size = to_simulation_units(mesh_size, 'length')

    # Model ECS
    ecs = occ.Box(occ.Pnt(-sx, -sy, cytosol_height), occ.Pnt(sx, sy, cytosol_height + ecs_height))
    cytosol = occ.Box(occ.Pnt(-sx, -sy, 0), occ.Pnt(sx, sy, cytosol_height))
    left, right, front, back, bottom, top = (0, 1, 2, 3, 4, 5)

    # Assign boundary conditions and materials to ECS
    for f in [front, back, left, right]:
        ecs.faces[f].bc("ecs_bnd")
    ecs.faces[top].bc("ecs_top")
    ecs.mat("ecs")

    # Assign boundary conditions to cytosol
    for f in [front, back, left, right]:
        cytosol.faces[f].bc("cyt_bnd")
    cytosol.faces[bottom].bc("cyt_bnd")

    # Cut a hole into the ecs-cytosol interface
    channel = occ.Face(occ.Wire(occ.Circle(occ.Pnt(0, 0, cytosol_height), occ.Z, channel_radius)))
    channel.maxh = mesh_size if channel_mesh_size is None \
        else to_simulation_units(channel_mesh_size, 'length')
    membrane = cytosol.faces[top] - channel
    membrane.bc("membrane")
    channel.bc("channel")

    # Reassemble the cytosol, make it a proper solid, and glue everything together
    cytosol = occ.Solid(occ.Glue([membrane, channel]  # if fused, channel vanishes \
                                 + [cytosol.faces[d] for d in [front, back, left, right, bottom]]))
    cytosol.mat("cytosol")
    geo = occ.OCCGeometry(occ.Glue([ecs, cytosol]))

    return Mesh(geo.GenerateMesh(maxh=mesh_size))
