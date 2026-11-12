import numpy as np
import ngsolve as ngs
import pytest

for module in ("pymeshlab", "triangle", "tetgen"):
    pytest.importorskip(module)

from bmbcsim.geometry.rounded_voronoi import _box_hs, facet_triangles, make_cells
from bmbcsim.geometry.tube_cells import make_tube_cells
from bmbcsim.meshing import tissue_mesher as tm

BOX = np.array([6.0, 6.0, 1.0])


@pytest.fixture(scope="module")
def tissue():
    """Small tissue mesh: 7 rounded Voronoi cells, gap 0.1, in a 6 x 6 x 1 box."""
    cells, ids, _ = make_cells(0.1, 0.5, BOX, seed=0)
    G, T, tags = tm.surface_mesh(cells, BOX, 0.3, workers=2)
    return len(cells), ids, tm.surface_ecs_fraction(G, T, tags, BOX), tm.volume_mesh(G, T, tags, len(cells))


def _volume(mesh, name):
    return ngs.Integrate(1, mesh, definedon=mesh.Materials(name))


def _area(mesh, name):
    return ngs.Integrate(1, mesh, ngs.BND, definedon=mesh.Boundaries(name))


def test_tissue_mesh_names_and_volume(tissue):
    n_cells, _, alpha_surface, mesh = tissue
    assert mesh.GetMaterials() == ("ecs", *(f"cell_{k}" for k in range(n_cells)))
    walls = set(tm.WALLS) | {"cell_boundary"} | {f"membrane_{k}" for k in range(n_cells)}
    assert set(mesh.GetBoundaries()) == walls
    assert sum(_area(mesh, w) for w in tm.WALLS + ["cell_boundary"]) == pytest.approx(2 * (6 * 6 + 2 * 6))
    # TetGen and netgen keep the surface, so the mesh encloses the same volumes as the surface mesh
    assert _volume(mesh, "ecs") / BOX.prod() == pytest.approx(alpha_surface, abs=1e-9)
    assert 0.05 < alpha_surface < 0.15


def test_save_load_round_trip(tissue, tmp_path):
    n_cells, ids, _, mesh = tissue
    path = tmp_path / "tissue.vtu"
    tm.save_tissue_mesh(path, mesh, w=0.1, seed=0, seed_ids=ids)
    loaded, meta = tm.load_tissue_mesh(path)
    assert meta["w"] == 0.1 and meta["seed"] == 0 and list(meta["seed_ids"]) == ids
    assert (loaded.ne, loaded.nv) == (mesh.ne, mesh.nv)
    assert loaded.GetMaterials() == mesh.GetMaterials()
    assert set(loaded.GetBoundaries()) == set(mesh.GetBoundaries())
    for name in mesh.GetMaterials():
        assert _volume(loaded, name) == pytest.approx(_volume(mesh, name), rel=1e-12)
    for name in set(mesh.GetBoundaries()):
        assert _area(loaded, name) == pytest.approx(_area(mesh, name), rel=1e-12)


def _tri_area(tris):
    return np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2


def test_facet_triangles_cube():
    V = np.array(np.meshgrid([0, 1], [0, 2], [0, 3], indexing="ij")).reshape(3, -1).T.astype(float)
    tris, k = facet_triangles(V, _box_hs(np.zeros(3), [1, 2, 3]))
    for j, area in enumerate([6, 6, 3, 3, 2, 2]):
        assert _tri_area(tris[k == j]) == pytest.approx(area)


def test_tube_cells_mesh():
    """Tube cells in a 6 x 6 x 2 box mesh, and their reference faces do not depend on w."""
    box = np.array([6.0, 6.0, 2.0])
    cells, ids, info = make_tube_cells(0.1, 0.25, box, seed=0)
    _, ids2, info2 = make_tube_cells(0.2, 0.25, box, seed=0)
    assert ids == ids2 and len(cells) > 3
    for f, f2 in zip(info["faces"], info2["faces"]):
        assert len(f) and np.array_equal(f, f2)
    G, T, tags = tm.surface_mesh(cells, box, 0.25, workers=8)   # ~1 CPU min per tube cell
    mesh = tm.volume_mesh(G, T, tags, len(cells))
    assert _volume(mesh, "ecs") / box.prod() == pytest.approx(tm.surface_ecs_fraction(G, T, tags, box), abs=1e-9)
