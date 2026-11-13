"""Tissue meshes: names, domain numbers, and .vtu file I/O.

A tissue mesh has the regions `ecs` and `cell_k`, and the boundaries `membrane_k` (between the ECS and cell k),
`cell_boundary` (cells on the box) and the walls `left`, `right`, `top`, `bottom`, `front`, `back` (ECS on the box;
x-, x+, y-, y+, z-, z+). `save_tissue_mesh` / `load_tissue_mesh` store it as a .vtu file with one `cell_id` per
tet (-1 = ECS, k = cell k) and rebuild the boundary names on load; this needs no meshing extras.
"""
import numpy as np
import ngsolve as ngs
import pyvista as pv
from netgen.meshing import FaceDescriptor, Mesh

WALLS = ["left", "right", "top", "bottom", "front", "back"]  # x-, x+, y-, y+, z-, z+
OUTSIDE, ECS, CELL0 = 0, 1, 2   # netgen domain numbers: outside the box, the ECS, cell k = CELL0 + k


def box_walls(box):
    """(axis, value, side) of each wall of the box [0, box] in `WALLS` order; side = outward direction (+-1)."""
    return [(k // 2, 0.0 if k % 2 == 0 else float(box[k // 2]), -1.0 if k % 2 == 0 else 1.0) for k in range(6)]


def cell_name(k):
    return f"cell_{k}"


def membrane_name(k):
    return f"membrane_{k}"


def tissue_names(mesh):
    """(cell names, membrane names) of a tissue mesh, in cell order."""
    n = len(mesh.GetMaterials()) - 1
    return [cell_name(k) for k in range(n)], [membrane_name(k) for k in range(n)]


def _face_keys(*faces):
    """Integer key of each triangle (rows of each array): equal for the same three points in any order."""
    F = np.sort(np.concatenate(faces), axis=1)
    key = np.unique(F, axis=0, return_inverse=True)[1].ravel()
    return np.split(key, np.cumsum([len(f) for f in faces])[:-1])


def tissue_mesh_from_labels(nodes, tets, cell_id):
    """NGSolve mesh with the tissue names from labelled tets (cell_id -1 = ECS, k = cell k).

    Interfaces between the ECS and cell k become `membrane_k`; faces on the box (the bounding box of the nodes)
    become `cell_boundary` (cell side) or the wall name (ECS side).
    """
    nodes = np.asarray(nodes, float)
    tets = np.asarray(tets, np.int64).copy()
    treg = np.asarray(cell_id, np.int64) + CELL0     # cell_id -1 = ECS
    lo, hi = nodes.min(0), nodes.max(0)
    X = nodes[tets]
    sv = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
    tets[sv > 0] = tets[sv > 0][:, [1, 0, 2, 3]]     # netgen orientation
    faces = np.r_[tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]
    opp = np.r_[tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]]
    freg = np.tile(treg, 4)
    fk, = _face_keys(faces)
    o = np.lexsort((freg, fk))
    faces, fk, opp, freg = faces[o], fk[o], opp[o], freg[o]
    pair = np.r_[fk[1:] == fk[:-1], False]
    first = np.r_[True, ~pair[:-1]]                  # first face of each key
    single = first & ~pair
    iface = np.nonzero(pair & (freg != np.r_[freg[1:], -1]))[0]   # faces i, i + 1 in different domains
    if np.any((freg[iface] >= CELL0) & (freg[iface + 1] >= CELL0)):
        raise ValueError("cells touch: tissue meshes need ECS between all cells")
    # interfaces: domin = cell (freg[i + 1], the higher domain), domout = ECS; boundary faces: domout = 0
    F = np.r_[faces[iface], faces[single]]
    domin = np.r_[freg[iface + 1], freg[single]]
    domout = np.r_[freg[iface], np.full(single.sum(), OUTSIDE)]
    opp_in = np.r_[opp[iface + 1], opp[single]]
    names = [membrane_name(d - CELL0) for d in domin[:len(iface)]]
    C = nodes[faces[single]].mean(1)
    tol = 1e-6 * np.max(hi - lo)
    on = np.c_[np.abs(C - lo) < tol, np.abs(C - hi) < tol][:, [0, 3, 1, 4, 2, 5]]   # WALLS order
    if not np.all(on.any(1)):
        raise ValueError("boundary face off the box: the mesh is not a box")
    names += ["cell_boundary" if d >= CELL0 else WALLS[w] for d, w in zip(domin[len(iface):], on.argmax(1))]
    Y = nodes[F]
    s = np.einsum("ij,ij->i", np.cross(Y[:, 1] - Y[:, 0], Y[:, 2] - Y[:, 0]), nodes[opp_in] - Y[:, 0])
    F[s > 0] = F[s > 0][:, [1, 0, 2]]                # normal out of domin

    m = Mesh(dim=3)
    m.AddPoints(np.ascontiguousarray(nodes))
    for d in range(ECS, treg.max() + 1):
        m.SetMaterial(d, "ecs" if d == ECS else cell_name(d - CELL0))
        m.AddElements(dim=3, index=d, data=np.ascontiguousarray(tets[treg == d], np.int32), base=0)
    groups = {}
    for i, key in enumerate(zip(domin, domout, names)):
        groups.setdefault(key, []).append(i)
    for gi, ((di, do, name), idx) in enumerate(sorted(groups.items())):
        fd = m.Add(FaceDescriptor(surfnr=gi + 1, domin=int(di), domout=int(do), bc=gi + 1))
        m.SetBCName(fd - 1, name)
        m.AddElements(dim=2, index=fd, data=np.ascontiguousarray(F[idx], np.int32), base=0)
    return ngs.Mesh(m)


def save_tissue_mesh(path, mesh, **meta):
    """Write a tissue mesh as .vtu: tets with a `cell_id` (-1 = ECS, k = cell_k); `meta` goes to the field data."""
    ngm = mesh.ngmesh
    el = ngm.Elements3D().NumPy()
    tets = el["nodes"][:, :4].astype(np.int64) - 1
    if list(mesh.GetMaterials()) != ["ecs"] + tissue_names(mesh)[0]:
        raise ValueError(f"not a tissue mesh: materials {mesh.GetMaterials()}")
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, ngm.Coordinates())
    grid.cell_data["cell_id"] = el["index"].astype(np.int32) - CELL0
    for key, value in meta.items():
        grid.field_data[key] = np.atleast_1d(value)
    grid.save(path)


def load_tissue_mesh(path):
    """Read a .vtu file from `save_tissue_mesh`. Returns (mesh, meta), size-1 field data as scalars."""
    grid = pv.read(path)
    tets = grid.cells_dict[pv.CellType.TETRA]
    mesh = tissue_mesh_from_labels(grid.points, tets, grid.cell_data["cell_id"])
    meta = {k: (v.item() if v.size == 1 else v) for k, v in ((k, grid.field_data[k]) for k in grid.field_data)}
    return mesh, meta
