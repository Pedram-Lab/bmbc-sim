
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


def create_dish_geometry(
        *,
        dish_height: Quantity,
        slice_width: Quantity,
        slice_depth: Quantity,
        mesh_size: Quantity,
        substrate_height: Quantity = None
) -> Mesh:
    """Create a simple columnar geometry with given sidelength and height
    represeting a slice in the middle of a dish. Optionally, a substrate can be
    added as a region at the bottom of the dish.

    :param dish_height: Height of the dish.
    :param slice_width: Width of the slice.
    :param slice_depth: Depth of the slice (this controls how fast species
        can reach the boundary and thus be removed from the domain).
    :param mesh_size: Size of the mesh.
    :param substrate_height: Height of the substrate (can be None).
    :return: Mesh of the geometry.
    """
    h = to_simulation_units(dish_height, 'length')
    sx = to_simulation_units(slice_width, 'length')
    sy = to_simulation_units(slice_depth, 'length')

    dish = occ.Box((-sx, -sy, 0), (sx, sy, h))
    dish.mat("dish:free")
    for i in [0, 1, 4, 5]:
        dish.faces[i].bc("reflective")
    dish.faces[2].bc("side")
    dish.faces[3].bc("side")

    if substrate_height is not None:
        sh = to_simulation_units(substrate_height, 'length')
        substrate = occ.Box((-2*sx, -2*sy, -sh), (2*sx, 2*sy, sh))
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


def create_sensor_geometry(
        *,
        side_length: Quantity | tuple[Quantity, Quantity, Quantity],
        compartment_ratio: float,
        sphere_position_x: Quantity,
        sphere_radius: Quantity,
        mesh_size: Quantity
) -> Mesh:
    """
    Creates a cube split into two compartments along the x-axis, each containing
    a centered sphere. The length ratio between compartments is controlled by
    compartment_ratio.

    :param total_length: Side length of the cube along all axes or side lengths
        per axis.
    :param compartment_ratio: Ratio of length for the first compartment (0-1).
    :param sphere_position_x: Position of the sphere along the x-axis, between
        0 and the length in the x-axis.
    :param sphere_radius: Radius of sphere inside compartment.
    :param mesh_size: Maximum mesh size for meshing.
    :return: Mesh object for the geometry.
    """
    if isinstance(side_length, Quantity):
        side_length = 3 * (side_length,)

    # Convert units
    L = to_simulation_units(side_length[0], 'length')
    W = to_simulation_units(side_length[1], 'length')
    H = to_simulation_units(side_length[2], 'length')
    PX = to_simulation_units(sphere_position_x, 'length')
    R = to_simulation_units(sphere_radius, 'length')
    maxh = to_simulation_units(mesh_size, 'length')

    # Validate ratio
    if not 0 < compartment_ratio < 1:
        raise ValueError("Compartment_ratio must be between 0 and 1 (exclusive).")

    L1 = L * compartment_ratio
    L2 = L * (1 - compartment_ratio)

    # Check that the sphere is completely within one compartment
    if W / 2 <= R or H / 2 <= R \
            or PX <= R or PX >= L - R or abs(PX - L1) <= R:
        raise ValueError("Sphere is not completely contained within one compartment.")

    # Regions
    box1 = occ.Box(occ.Pnt(0, -W / 2, -H / 2), occ.Pnt(L1, W / 2, H / 2))
    box2 = occ.Box(occ.Pnt(L1, -W / 2, -H / 2), occ.Pnt(L1 + L2, W / 2, H / 2))
    box1.mat("cube:left")
    box2.mat("cube:right")
    regions = [box1, box2]

    # Spheres at specified position
    if R > 0:
        sphere = occ.Sphere((PX, 0, 0), R)
        sphere.mat("cube:sphere")
        regions.append(sphere)

    # Glue everything into one geometry
    geo = occ.Glue(regions)
    occ_geo = occ.OCCGeometry(geo)
    return Mesh(occ_geo.GenerateMesh(maxh=maxh))
