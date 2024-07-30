from netgen.occ import *
from ngsolve import Mesh


def create_ca_depletion_mesh(*, side_length, cytosol_height, ecs_height, mesh_size):
    cytosol = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    ecs = Box(Pnt(0, 0, 1), Pnt(1, 1, 1.2))
    left, right, front, back, bottom, top = (0, 1, 2, 3, 4, 5)

    # %%
    # Assign boundary conditions
    for f in [front, back, left, right]:
        cytosol.faces[f].bc("cyt_bnd")
        ecs.faces[f].bc("ecs_bnd")
    cytosol.faces[bottom].bc("cyt_bnd")
    ecs.faces[top].bc("ecs_top")

    # Cut a hole into the ecs-cytosol interface
    channel = Face(Wire(Circle(Pnt(0.5, 0.5, 1), Z, 0.1)))
    channel.maxh = 0.03
    channel.bc("channel")
    channel.col = (1, 0, 0)
    membrane = (cytosol.faces[top] - channel).bc("membrane")
    interface = Glue([membrane, channel])  # if fused, channel vanishes

    # Only take parts that make up the actual geometry
    geo = Fuse([interface, ecs.faces[top], cytosol.faces[bottom]]
               + [cytosol.faces[f] for f in [front, back, left, right]]
               + [ecs.faces[f] for f in [front, back, left, right]])

    # %%
    # Generate a mesh on the surface (no volume mesh so far)
    surface_mesh = OCCGeometry(geo).GenerateMesh()
    mesh = Mesh(surface_mesh)

    # %%
    # Automatically get the boundary indices that are used to assign face descriptors later
    cyt_bnd = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "cyt_bnd"]
    ecs_bnd = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "ecs_bnd"]
    ecs_top = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "ecs_top"]
    channel = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "channel"]
    membrane = [i + 1 for i, bnd in enumerate(mesh.GetBoundaries()) if bnd == "membrane"]

    # %%
    # The sub-meshes can be visualized easily
    # Note: in this process, the meshes are copied, so using GetSubMesh to add face descriptors doesn't work
    membrane_mesh = surface_mesh.GetSubMesh(faces="membrane|channel")

    # %%
    # Generate new face descriptors - note that there are two domains now!
    from netgen.meshing import FaceDescriptor, Element2D
    from netgen.meshing import Mesh as NetgenMesh

    new_mesh = NetgenMesh()
    fd_channel = new_mesh.Add(FaceDescriptor(surfnr=1, domin=2, domout=1, bc=1))
    fd_membrane = new_mesh.Add(FaceDescriptor(surfnr=2, domin=2, domout=1, bc=2))
    fd_ecs_top = new_mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3))
    fd_ecs_bnd = new_mesh.Add(FaceDescriptor(surfnr=4, domin=1, domout=0, bc=4))
    fd_cyt_bnd = new_mesh.Add(FaceDescriptor(surfnr=5, domin=2, domout=0, bc=5))


    # %%
    # Copy all nodes to a new mesh
    old_to_new = {}
    for e in surface_mesh.Elements2D():
        for v in e.vertices:
            if (v not in old_to_new):
                old_to_new[v] = new_mesh.Add(surface_mesh[v])

    # Copy all elements (with the appropriate face descriptors)
    for e in surface_mesh.Elements2D():
        if e.index in ecs_top:
            new_mesh.Add(Element2D(fd_ecs_top, [old_to_new[v] for v in e.vertices]))
        elif e.index in channel:
            new_mesh.Add(Element2D(fd_channel, [old_to_new[v] for v in e.vertices]))
        elif e.index in membrane:
            new_mesh.Add(Element2D(fd_membrane, [old_to_new[v] for v in e.vertices]))
        elif e.index in cyt_bnd:
            new_mesh.Add(Element2D(fd_cyt_bnd, [old_to_new[v] for v in e.vertices]))
        elif e.index in ecs_bnd:
            new_mesh.Add(Element2D(fd_ecs_bnd, [old_to_new[v] for v in e.vertices]))
        else:
            raise ValueError(f"Can't cope with value {e.index}")


    # %%
    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh()
    mesh = Mesh(new_mesh)

    # %%
    # Assign names to boundaries
    new_mesh.SetBCName(0, "channel")
    new_mesh.SetBCName(1, "membrane")
    new_mesh.SetBCName(2, "ecs_top")
    new_mesh.SetBCName(3, "boundary")
    new_mesh.SetBCName(4, "boundary")

    # Click on any face to see the boundary condition

    # %%
    # Assign names to regions
    new_mesh.SetMaterial(1, "ecs")
    new_mesh.SetMaterial(2, "cytosol")

    return mesh

    # geometry = CSGeometry()
    # left = create_axis_aligned_plane(0, -side_length / 2, -1)
    # right = create_axis_aligned_plane(0, side_length / 2, 1)
    # front = create_axis_aligned_plane(1, -side_length / 2, -1)
    # back = create_axis_aligned_plane(1, side_length / 2, 1)
    #
    # cytosol_cutout = create_axis_aligned_cylinder(2, 0, 0, 0.1) \
    #                  * create_axis_aligned_plane(2, cytosol_height, 1, "channel") \
    #                  * create_axis_aligned_plane(2, cytosol_height - ecs_height / 2, -1)
    # cytosol_cutout.maxh(mesh_size / 2)
    #
    # cytosol_bot = create_axis_aligned_plane(2, 0, -1)
    # cytosol_top = create_axis_aligned_plane(2, cytosol_height, 1, "membrane")
    # cytosol = left * right * front * back * cytosol_bot * cytosol_top
    # cytosol.maxh(mesh_size)
    #
    # ecs_cutout = create_axis_aligned_cylinder(2, 0, 0, 0.1) \
    #              * create_axis_aligned_plane(2, cytosol_height + ecs_height / 2, 1) \
    #              * create_axis_aligned_plane(2, cytosol_height, -1, "channel")
    # ecs_cutout.maxh(mesh_size / 2)
    #
    # ecs_bot = create_axis_aligned_plane(2, cytosol_height, -1, "membrane")
    # ecs_top = create_axis_aligned_plane(2, cytosol_height + ecs_height, 1, "ecs_top")
    # ecs = left * right * front * back * ecs_bot * ecs_top
    # ecs.maxh(mesh_size)
    #
    # geometry.Add(cytosol - cytosol_cutout)
    # geometry.Add(cytosol * cytosol_cutout)
    # geometry.Add(ecs - ecs_cutout)
    # geometry.Add(ecs * ecs_cutout)
    # return geometry
