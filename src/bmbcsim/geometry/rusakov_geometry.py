"""
Create the simulation geometry from [Rusakov 2001].
"""
from math import pi

from netgen import occ
from ngsolve import Mesh
import numpy as np
import astropy.units as u

from bmbcsim.units import to_simulation_units

def create_rusakov_geometry(
    total_size: u.Quantity,
    synapse_radius: u.Quantity,
    cleft_size: u.Quantity,
    glia_distance: u.Quantity,
    glia_width: u.Quantity,
    glial_coverage_angle: u.Quantity,
    mesh_size: u.Quantity,
) -> Mesh:
    """
    Create the geometry from [Rusakov 2001]. It consists of two semi-spheric
    synaptic terminals separated by a synaptic cleft. The terminals are
    surrounded in some distance by a glial cell with an adjustable coverage
    angle (measured from the top). The whole geometry is contained in an
    enclosing box of porous neuropil. The compartments / membranes involved are:
    - presynapse: the interior of the presynaptic terminal
    - postsynapse: the interior of the postsynaptic terminal
    - glia: the interior of the glial covering
    - synapse_ecs: the space between the synaptic terminals, and the surrounding
        glial cell and neuropil
    - neuropil: the porous region surrounding the synapse
    - glial membrane: the membrane surrounding the glial cell
    - presynaptic_membrane: the membrane of the presynaptic terminal facing the
        postsynaptic terminal
    - postsynaptic_membrane: the membrane of the postsynaptic terminal facing
        the presynaptic terminal
    - terminal_membrane: the membranes of both terminals that don't face
        each other
    - synapse_boundary: the boundary of the synapse connecting to the neuropil
    - neuropil_boundary: the outer boundary of the porous neuropil

    :param total_size: The side-length of the enclosing cube.
    :param synapse_radius: The radius of the synaptic terminals.
    :param cleft_size: The size of the synaptic cleft (distance between the terminals).
    :param glia_distance: The distance of the glial cell from the synaptic terminals.
    :param glia_width: The width of the ensheathing glial cell.
    :param glial_coverage_angle: The angle of the glial coverage from the top.
    :param mesh_size: The maximum size of the mesh elements.
    :return: The mesh of the geometry.
    """
    # Convert all quantities to the same set of base units
    ts = to_simulation_units(total_size, 'length')
    cs = to_simulation_units(cleft_size, 'length')
    sr = to_simulation_units(synapse_radius, 'length')
    gd = to_simulation_units(glia_distance, 'length')
    gw = to_simulation_units(glia_width, 'length')

    gca = to_simulation_units(glial_coverage_angle, 'angle')

    # Create the containing box
    box = occ.Box(occ.Pnt(-ts/2, -ts/2, -ts/2), occ.Pnt(ts/2, ts/2, ts/2))
    box.faces.col = (0.5, 0.5, 0.5)
    box.bc("neuropil_boundary")
    box.mat("neuropil")

    # Create the synaptic terminals separated by the synaptic cleft
    pre_synapse_cutout = occ.HalfSpace(occ.Pnt(0, 0, -cs/2), occ.Dir(0, 0, -1)) \
        .bc("presynaptic_membrane")
    pre_synapse = occ.Sphere(occ.Pnt(0, 0, 0), sr).bc("terminal_membrane") - pre_synapse_cutout
    pre_synapse = pre_synapse.MakeFillet(pre_synapse.edges, sr / 12.)
    pre_synapse.faces.col = (1, 0, 0)
    pre_synapse.faces[1].bc("terminal_membrane")
    pre_synapse.mat("presynapse")

    post_synapse_cutout = occ.HalfSpace(occ.Pnt(0, 0, cs/2), occ.Dir(0, 0, 1)) \
        .bc("postsynaptic_membrane")
    post_synapse = occ.Sphere(occ.Pnt(0, 0, 0), sr).bc("terminal_membrane") - post_synapse_cutout
    post_synapse = post_synapse.MakeFillet(post_synapse.edges, sr / 10)
    post_synapse.faces.col = (0, 0, 1)
    post_synapse.faces[1].bc("terminal_membrane")
    post_synapse.mat("postsynapse")

    synapse_ecs = occ.Sphere(occ.Pnt(0, 0, 0), sr + gd).bc("synapse_boundary")
    synapse_ecs.faces.col = (0.5, 0, 0.5)
    synapse_ecs.mat("synapse_ecs")

    # Create the glial cell with given coverage
    box = box - synapse_ecs
    glia = occ.Sphere(occ.Pnt(0, 0, 0), sr + gd + gw) - synapse_ecs
    glia.faces.col = (0, 1, 0)
    if np.isclose(gca, pi) or gca > pi:
        # No cutout
        pass
    elif np.isclose(gca, 0) or gca < 0:
        # Cut out the whole glial cell
        glia = glia - box
    elif np.isclose(gca, pi / 2):
        # Cut exactly half of the shperical shell
        glia = glia - occ.HalfSpace(occ.Pnt(0, 0, 0), occ.Dir(0, 0, 1))
    elif gca < pi / 2:
        # Intersect with a cone of the given angle in the upper half plane
        h = (sr + gd + gw) * 1.1
        r_base = h * np.tan(gca)
        glial_cutout = occ.Cone(occ.gp_Ax2(occ.Pnt(0, 0, h), -occ.Z), r_base, 0.0, h, 2 * pi)
        glia = glia * glial_cutout
    elif gca > pi / 2:
        # Subtract a cone of the given angle in the lower half plane
        h = (sr + gd + gw) * 1.1
        r_base = h * np.tan(pi - gca)
        glial_cutout = occ.Cone(occ.gp_Ax2(occ.Pnt(0, 0, -h), occ.Z), r_base, 0.0, h, 2 * pi)
        glia = glia - glial_cutout

    glia = glia.MakeFillet(glia.edges, gw / 4)
    glia.bc("glial_membrane")
    glia.mat("glia")

    geo = occ.Glue([box, pre_synapse, post_synapse, synapse_ecs, glia])
    geo = occ.OCCGeometry(geo)
    return Mesh(geo.GenerateMesh(maxh=to_simulation_units(mesh_size, 'length')))
