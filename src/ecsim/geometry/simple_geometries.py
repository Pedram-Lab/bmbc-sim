
"""
Geometries consisting of simple geometric shapes for testing purposes.
"""
from netgen import occ
from netgen.meshing import FaceDescriptor
from ngsolve import Mesh
from astropy.units import Quantity

from ecsim.units import to_simulation_units
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
    sx = to_simulation_units(side_length_x, 'length') / 2
    sy = to_simulation_units(side_length_y, 'length') / 2
    cytosol_height = to_simulation_units(cytosol_height, 'length')
    ecs_height = to_simulation_units(ecs_height, 'length')
    channel_radius = to_simulation_units(channel_radius, 'length')
    mesh_size = to_simulation_units(mesh_size, 'length')

    cytosol = occ.Box(occ.Pnt(-sx, -sy, 0), occ.Pnt(sx, sy, cytosol_height))
    ecs = occ.Box(occ.Pnt(-sx, -sy, cytosol_height), occ.Pnt(sx, sy, cytosol_height + ecs_height))
    left, right, front, back, bottom, top = (0, 1, 2, 3, 4, 5)

    # Assign boundary conditions
    for f in [front, back, left, right]:
        cytosol.faces[f].bc("cyt_bnd")
        ecs.faces[f].bc("ecs_bnd")
    cytosol.faces[bottom].bc("cyt_bnd")
    ecs.faces[top].bc("ecs_top")

    # Cut a hole into the ecs-cytosol interface
    channel = occ.Face(occ.Wire(occ.Circle(occ.Pnt(0, 0, cytosol_height), occ.Z, channel_radius)))
    channel.maxh = mesh_size / 2
    channel.bc("channel")
    membrane = (cytosol.faces[top] - channel).bc("membrane")
    interface = occ.Glue([membrane, channel])  # if fused, channel vanishes

    # Only take parts that make up the actual geometry
    geo = occ.Fuse([interface, ecs.faces[top], cytosol.faces[bottom]]
                   + [cytosol.faces[f] for f in [front, back, left, right]]
                   + [ecs.faces[f] for f in [front, back, left, right]])

    # Generate a mesh on the surface and convert it to a volume mesh
    surface_mesh = occ.OCCGeometry(geo).GenerateMesh(maxh=mesh_size)
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


def create_dish_geometry(
        *,
        dish_height: Quantity,
        sidelength: Quantity,
        mesh_size: Quantity,
        substrate_height: Quantity = None
) -> Mesh:
    """Create a simple columnar geometry with given sidelength and height
    represeting a slice in the middle of a dish. Optionally, a substrate can be
    added as a region at the bottom of the dish.

    :param dish_height: Height of the dish.
    :param sidelength: Length of the sides of the slice.
    :param mesh_size: Size of the mesh.
    :param substrate_height: Height of the substrate (can be None).
    :return: Mesh of the geometry.
    """
    h = to_simulation_units(dish_height, 'length')
    s = to_simulation_units(sidelength, 'length')

    dish = occ.Box((-s, -s, 0), (s, s, h))
    dish.mat("dish:free")
    for i in [0, 1, 2, 3]:
        dish.faces[i].bc("side")
    dish.faces[4].bc("bottom")
    dish.faces[5].bc("top")

    if substrate_height is not None:
        sh = to_simulation_units(substrate_height, 'length')
        substrate = occ.Box((-2*s, -2*s, -sh), (2*s, 2*s, sh))
        substrate.faces[5].bc("interface")
        substrate = dish * substrate
        substrate.mat("dish:substrate")
        substrate.col = (1, 0, 0)
        geo = occ.Glue([substrate, dish - substrate])
    else:
        geo = dish

    geo = occ.OCCGeometry(geo)
    mesh_size = to_simulation_units(mesh_size, 'length')
    return Mesh(geo.GenerateMesh(maxh=mesh_size))
