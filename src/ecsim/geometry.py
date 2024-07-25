from netgen.csg import *


def create_axis_aligned_plane(axis, offset, direction, boundary_condition=None):
    point = Pnt(*[0 if i != axis else offset for i in range(3)])
    normal = Vec(*[0 if i != axis else direction for i in range(3)])
    plane = Plane(point, normal)
    plane.bc(boundary_condition if boundary_condition is not None else "default")
    return plane


def create_axis_aligned_cylinder(axis, c1, c2, radius, boundary_condition=None):
    match axis:
        case 0:
            point_1 = Pnt(0, c1, c2)
            point_2 = Pnt(1, c1, c2)
        case 1:
            point_1 = Pnt(c1, 0, c2)
            point_2 = Pnt(c1, 1, c2)
        case 2:
            point_1 = Pnt(c1, c2, 0)
            point_2 = Pnt(c1, c2, 1)
        case _:
            raise ValueError(f"Invalid axis {axis}")
    cylinder = Cylinder(point_1, point_2, radius)
    cylinder.bc(boundary_condition if boundary_condition is not None else "default")
    return cylinder
