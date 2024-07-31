from netgen.occ import *
from ngsolve import Mesh
from netgen.meshing import FaceDescriptor, Element2D
from netgen.meshing import Mesh as NetgenMesh


def _convert_to_volume_mesh(surface_mesh, bnd_to_fd):
    new_mesh = NetgenMesh()

    # Copy nodes
    old_to_new = {}
    for e in surface_mesh.Elements2D():
        for v in e.vertices:
            if (v not in old_to_new):
                old_to_new[v] = new_mesh.Add(surface_mesh[v])

    # Arrange face descriptors for the new mesh
    boundaries = Mesh(surface_mesh).GetBoundaries()
    print(boundaries)
    bnd_to_fd_index = {bnd: new_mesh.Add(fd) for bnd, fd in bnd_to_fd.items()}
    face_descriptor_indices = [bnd_to_fd_index[bnd] for bnd in boundaries]

    # Copy elements
    for e in surface_mesh.Elements2D():
        fd = face_descriptor_indices[e.index - 1]
        new_mesh.Add(Element2D(fd, [old_to_new[v] for v in e.vertices]))

    # Generate volume mesh from surface
    new_mesh.GenerateVolumeMesh()
    return new_mesh


def create_ca_depletion_mesh(*, side_length, cytosol_height, ecs_height, mesh_size):
    cytosol = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    ecs = Box(Pnt(0, 0, 1), Pnt(1, 1, 1.2))
    left, right, front, back, bottom, top = (0, 1, 2, 3, 4, 5)

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

    # Generate a mesh on the surface and convert it to a volume mesh
    surface_mesh = OCCGeometry(geo).GenerateMesh()
    bnd_to_fd = {
        "channel": FaceDescriptor(surfnr=1, domin=2, domout=1, bc=1),
        "membrane": FaceDescriptor(surfnr=2, domin=2, domout=1, bc=2),
        "ecs_top": FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3),
        "ecs_bnd": FaceDescriptor(surfnr=4, domin=1, domout=0, bc=4),
        "cyt_bnd": FaceDescriptor(surfnr=5, domin=2, domout=0, bc=5),
    }
    mesh = _convert_to_volume_mesh(surface_mesh, bnd_to_fd)

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
