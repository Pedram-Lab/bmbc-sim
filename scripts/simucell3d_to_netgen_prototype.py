# %%
import pyvista as pv
from netgen import occ
import netgen.libngpy._meshing as ng
import ngsolve as ngs
from ngsolve.webgui import Draw
import numpy as np



cells = pv.read('cells.stl')
cells.points /= 10  # to make it reasonably sized
TOL = 1e-3

cells_clip = cells.clip(normal='z', invert=False).triangulate()
cells_slice = cells.slice(normal='z')
lines = cells_slice.lines.reshape(-1, 3)[:, 1:]

# Remove duplicate points and update connectivity indices in the slice
# Assuming 'cells_slice' is already defined as cells.slice(normal='z')
orig_points = cells_slice.points
unique_points, inverse = np.unique(orig_points, axis=0, return_inverse=True)
cells_slice.points = unique_points

# Update the lines connectivity based on the new unique points
lines = inverse[lines.flatten()].reshape(lines.shape)

# Reorder line segments to form a closed polygonal chain
ordered_indices = list(lines[0])
remaining = lines[1:].copy()  # ensure remaining is a numpy array

while remaining.shape[0] > 0:
    last = ordered_indices[-1]
    # Vectorized check: find rows where either endpoint matches 'last'
    match = (remaining[:, 0] == last) | (remaining[:, 1] == last)
    if not np.any(match):
        break
    # Get the first matching row index
    idx = np.where(match)[0][0]
    # Append the other endpoint
    if remaining[idx, 0] == last:
        ordered_indices.append(remaining[idx, 1])
    else:
        ordered_indices.append(remaining[idx, 0])
    # Remove the used segment
    remaining = np.delete(remaining, idx, axis=0)

if ordered_indices[0] != ordered_indices[-1]:
    ordered_indices.append(ordered_indices[0])
lines = np.array([[ordered_indices[i], ordered_indices[i+1]] for i in range(len(ordered_indices)-1)])
midpoint = cells_slice.center

origin = (0, 0, midpoint[2])
cell_wp = occ.WorkPlane(axes=occ.Axes(origin, n=-occ.Z, h=occ.X))
pnt = unique_points[lines[0, 0]]
cell_wp.MoveTo(pnt[0], pnt[1])
for seg in lines:
    pnt = unique_points[seg[1]]
    cell_wp.LineTo(pnt[0], pnt[1])
cell_face = cell_wp.Face()
cell_face.bc("cell_boundary")
cell_face.col = [1, 0, 0]

bottom_wp = occ.WorkPlane(axes=occ.Axes(origin, n=-occ.Z, h=occ.X))
bottom_wp.MoveTo(midpoint[0], midpoint[1])
bottom = bottom_wp.RectangleC(4, 4).Face()
bottom.bc("ecs_boundary")

origin = (midpoint[0] - 2, -midpoint[1] - 2, midpoint[2])
left_wp = occ.WorkPlane(axes=occ.Axes(origin, n=-occ.X, h=occ.Y))
left_wp.MoveTo(0, -4)
left = left_wp.Rectangle(4, 4).Face()
left.bc("ecs_boundary")

origin = (midpoint[0] + 2, -midpoint[1] - 2, midpoint[2])
right_wp = occ.WorkPlane(axes=occ.Axes(origin, n=occ.X, h=occ.Y))
right_wp.MoveTo(0, 0)
right = right_wp.Rectangle(4, 4).Face()
right.bc("ecs_boundary")

origin = (midpoint[0] - 2, -midpoint[1] - 2, midpoint[2])
front_wp = occ.WorkPlane(axes=occ.Axes(origin, n=-occ.Y, h=occ.X))
front_wp.MoveTo(0, 0)
front = front_wp.Rectangle(4, 4).Face()
front.bc("ecs_boundary")

origin = (midpoint[0] - 2, -midpoint[1] + 2, midpoint[2])
back_wp = occ.WorkPlane(axes=occ.Axes(origin, n=occ.Y, h=occ.X))
back_wp.MoveTo(0, -4)
back = back_wp.Rectangle(4, 4).Face()
back.bc("ecs_boundary")

origin = (midpoint[0] - 2, -midpoint[1] - 2, midpoint[2] + 4)
top_wp = occ.WorkPlane(axes=occ.Axes(origin, n=occ.Z, h=occ.X))
top_wp.MoveTo(0, 0)
top = top_wp.Rectangle(4, 4).Face()
top.bc("ecs_boundary")

geo = occ.OCCGeometry(occ.Glue([bottom - cell_face, cell_face,
                                left, right, front, back, top]))
surface_mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.3))
cell_bnd = [i + 1 for i, bnd in enumerate(surface_mesh.GetBoundaries()) if bnd == "cell_boundary"]
ecs_bnd = [i + 1 for i, bnd in enumerate(surface_mesh.GetBoundaries()) if bnd == "ecs_boundary"]
Draw(surface_mesh)

# %%
volume_mesh = ng.Mesh()
ecs_outside_fd = volume_mesh.Add(ng.FaceDescriptor(surfnr=1, domin=1, domout=0, bc=0))
cell_outside_fd = volume_mesh.Add(ng.FaceDescriptor(surfnr=2, domin=2, domout=0, bc=1))
ecs_cell_fd = volume_mesh.Add(ng.FaceDescriptor(surfnr=3, domin=2, domout=1, bc=2))

dummy_fd = volume_mesh.Add(ng.FaceDescriptor(surfnr=4, domin=1, domout=1, bc=3))

for pnt in surface_mesh.ngmesh.Coordinates():
    volume_mesh.Add(ng.MeshPoint(ng.Point3d(pnt[0], pnt[1], pnt[2])))

indices = []
for el in surface_mesh.ngmesh.Elements2D():
    if el.index in ecs_bnd:
        volume_mesh.Add(ng.Element2D(ecs_outside_fd, el.vertices))
    else:
        # volume_mesh.Add(ng.Element2D(cell_outside_fd, el.vertices))
        volume_mesh.Add(ng.Element2D(ecs_outside_fd, el.vertices))

existing_coords = surface_mesh.ngmesh.Coordinates()
n_existing = existing_coords.shape[0]
pv_to_ng_points = np.zeros(cells_clip.points.shape[0], dtype=int)

for i, pnt in enumerate(cells_clip.points):
    # Find any existing point within tolerance
    # TODO: there is a sign flip here!
    pnt = pnt * np.array([1, -1, 1])
    diff = np.linalg.norm(existing_coords - pnt, axis=1)
    matches = np.where(diff < TOL)[0]
    if matches.size > 0:
        pv_to_ng_points[i] = int(matches[0]) + 1
    else:
        point_id = volume_mesh.Add(ng.MeshPoint(ng.Point3d(pnt[0], pnt[1], pnt[2])))
        pv_to_ng_points[i] = point_id.nr

for el in cells_clip.regular_faces:
    vertices = pv_to_ng_points[[i for i in reversed(el)]]
    if np.count_nonzero(vertices <= n_existing) != 2:
        # Element somewhere on the surface
        # volume_mesh.Add(ng.Element2D(ecs_cell_fd, vertices))
        volume_mesh.Add(ng.Element2D(dummy_fd, vertices))
    else:
        # Element touches the clipping boundary
        # Find the base point and the two points on the boundary
        idx = np.where(vertices > n_existing)[0][0]
        match idx:
            case 0:
                base_idx, idx1, idx2 = vertices
            case 1:
                idx2, base_idx, idx1 = vertices
            case 2:
                idx1, idx2, base_idx = vertices

        pnt1 = np.array(existing_coords[idx1 - 1])
        pnt2 = np.array(existing_coords[idx2 - 1])
        line_vec = pnt2 - pnt1
        line_length2 = np.dot(line_vec, line_vec)

        # Vectorized approach: compute projection factors for all points.
        pts = existing_coords
        t = np.dot(pts - pnt1, line_vec) / line_length2
        proj = pnt1 + np.outer(t, line_vec)
        dists = np.linalg.norm(pts - proj, axis=1)

        # Identify points whose projection lies within the segment and are close to the line.
        mask = (t >= 0) & (t <= 1) & (dists < TOL)
        points_between = np.where(mask)[0] + 1
        t = t[mask]

        # Sort the points based on their projection factor
        sorted_indices = np.argsort(t)
        points_between = points_between[sorted_indices]

        # Add all elements between the base point and the boundary points
        for i in range(len(points_between) - 1):
            vertices = [base_idx, points_between[i], points_between[i + 1]]
            # volume_mesh.Add(ng.Element2D(ecs_cell_fd, vertices))
            volume_mesh.Add(ng.Element2D(dummy_fd, vertices))

# volume_mesh.GenerateVolumeMesh()
# volume_mesh.SetBCName(0, "ecs_boundary")
# volume_mesh.SetBCName(1, "cell_boundary")
# volume_mesh.SetBCName(2, "cell_membrane")
Draw(volume_mesh)

# %%
