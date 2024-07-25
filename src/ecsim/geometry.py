from netgen.csg import *


def create_axis_aligned_plane(axis, offset, direction, boundary_condition=None):
    point = Pnt(*[0 if i != axis else offset for i in range(3)])
    normal = Vec(*[0 if i != axis else direction for i in range(3)])
    plane = Plane(point, normal)
    plane.bc(boundary_condition if boundary_condition is not None else "default")
    return plane
