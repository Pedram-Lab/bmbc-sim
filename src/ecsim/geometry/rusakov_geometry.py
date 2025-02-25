"""
Create the simulation geometry from [Rusakov 2001].
"""
from math import pi

from netgen.occ import Box, Cone, Dir, Glue, HalfSpace, Pnt, Sphere, gp_Ax2, Z, OCCGeometry
import numpy as np
import astropy.units as u

from ecsim.units import ANGLE, LENGTH, convert

def create_rusakov_geometry(
    total_size: u.Quantity,
    synapse_radius: u.Quantity,
    cleft_size: u.Quantity,
    glia_distance: u.Quantity,
    glia_width: u.Quantity,
    glial_coverage_angle: u.Quantity,
) -> OCCGeometry:
    """
    Create the geometry from [Rusakov 2001]. It consists of two semi-spheric
    synaptic terminals separated by a synaptic cleft. The terminals are
    surrounded in some distance by a glial cell with an adjustable coverage
    angle (measured from the top). The whole geometry is contained in an
    enclosing box of porous neuropil.

    :param total_size: The side-length of the enclosing cube.
    :param synapse_radius: The radius of the synaptic terminals.
    :param cleft_size: The size of the synaptic cleft (distance between the terminals).
    :param glia_distance: The distance of the glial cell from the synaptic terminals.
    :param glia_width: The width of the ensheathing glial cell.
    :param glial_coverage_angle: The angle of the glial coverage from the top.
    :return: The assembled geometry.
    """
    # Convert all quantities to the same set of base units
    ts = convert(total_size, LENGTH)
    cs = convert(cleft_size, LENGTH)
    sr = convert(synapse_radius, LENGTH)
    gd = convert(glia_distance, LENGTH)
    gw = convert(glia_width, LENGTH)

    gca = convert(glial_coverage_angle, ANGLE)

    # Create the containing box
    box = Box(Pnt(-ts/2, -ts/2, -ts/2), Pnt(ts/2, ts/2, ts/2))
    box.faces.col = (0.5, 0.5, 0.5)
    box.bc("ecs_boundary")
    box.mat("ecs")

    # Create the synaptic terminals separated by the synaptic cleft
    pre_synapse_cutout = HalfSpace(Pnt(0, 0, -cs/2), Dir(0, 0, -1)).bc("presynaptic_membrane")
    pre_synapse = Sphere(Pnt(0, 0, 0), sr).bc("terminal_membrane") - pre_synapse_cutout
    pre_synapse.faces.col = (1, 0, 0)
    pre_synapse.mat("presynapse")

    post_synapse_cutout = HalfSpace(Pnt(0, 0, cs/2), Dir(0, 0, 1)).bc("postsynaptic_membrane")
    post_synapse = Sphere(Pnt(0, 0, 0), sr).bc("terminal_membrane") - post_synapse_cutout
    post_synapse.faces.col = (0, 0, 1)
    post_synapse.mat("postsynapse")

    synapse_ecs = Sphere(Pnt(0, 0, 0), sr + gd).bc("synapse_boundary")
    synapse_ecs.faces.col = (0.5, 0, 0.5)
    synapse_ecs.mat("synapse_ecs")

    # Create the glial cell with given coverage
    box = box - synapse_ecs
    glia = Sphere(Pnt(0, 0, 0), sr + gd + gw) - synapse_ecs
    glia.faces.col = (0, 1, 0)
    if np.isclose(gca, pi) or gca > pi:
        # No cutout
        pass
    elif np.isclose(gca, 0) or gca < 0:
        # Cut out the whole glial cell
        glia = glia - box
    elif np.isclose(gca, pi / 2):
        # Cut exactly half of the shperical shell
        glia = glia - HalfSpace(Pnt(0, 0, 0), Dir(0, 0, 1))
    elif gca < pi / 2:
        # Intersect with a cone of the given angle in the upper half plane
        h = (sr + gd + gw) * 1.1
        r_base = h * np.tan(gca)
        glial_cutout = Cone(gp_Ax2(Pnt(0, 0, h), -Z), r_base, 0.0, h, 2 * pi)
        glia = glia * glial_cutout
    elif gca > pi / 2:
        # Subtract a cone of the given angle in the lower half plane
        h = (sr + gd + gw) * 1.1
        r_base = h * np.tan(pi - gca)
        glial_cutout = Cone(gp_Ax2(Pnt(0, 0, -h), Z), r_base, 0.0, h, 2 * pi)
        glia = glia - glial_cutout

    glia.bc("glial_membrane")
    glia.mat("glia")

    return Glue([box, pre_synapse, post_synapse, synapse_ecs, glia])
