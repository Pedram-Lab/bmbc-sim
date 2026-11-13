import importlib.util
import multiprocessing

import numpy as np
import ngsolve as ngs
import pytest

from bmbcsim.geometry.cells import RoundedPolytope, box_halfspaces, facet_triangles
from bmbcsim.geometry.rounded_voronoi import make_rounded_voronoi_cells
from bmbcsim.geometry.tube_cells import make_tube_cells
from bmbcsim.meshing import tissue_io as tio
from bmbcsim.meshing import tissue_mesher as tm

BOX = np.array([6.0, 6.0, 1.0])
needs_extra = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("pymeshlab", "triangle", "tetgen")),
    reason="needs the geometry extra (pymeshlab, triangle, tetgen)")


@pytest.fixture(scope="module")
def pool():
    with multiprocessing.get_context("spawn").Pool(4) as p:
        yield p


@pytest.fixture(scope="module")
def round_cells():
    return make_rounded_voronoi_cells(0.1, 0.5, BOX, seed=0)


@pytest.fixture(scope="module")
def tissue(round_cells, pool):
    """Small tissue mesh: 7 rounded Voronoi cells, gap 0.1, in a 6 x 6 x 1 box."""
    surf = tm.surface_mesh(round_cells.cells, BOX, 0.3, round_cells.max_offset, pool)
    return surf.ecs_fraction(BOX), tm.volume_mesh(surf)


def _volume(mesh, name):
    return ngs.Integrate(1, mesh, definedon=mesh.Materials(name))


def _area(mesh, name):
    return ngs.Integrate(1, mesh, ngs.BND, definedon=mesh.Boundaries(name))


def _centroid(mesh, name):
    region = mesh.Materials(name)
    return np.array([ngs.Integrate(x, mesh, definedon=region) for x in (ngs.x, ngs.y, ngs.z)]) / _volume(mesh, name)


def test_rounded_polytope_is_exact_sdf():
    """Cube core [-a, a]^3 rounded by r: exact distances in the face, edge and vertex regions and inside."""
    a, r, s = 0.5, 0.3, 0.2
    cell = RoundedPolytope(box_halfspaces(-a * np.ones(3), a * np.ones(3)), np.zeros(3), r)
    X = np.array([[a + s, 0, 0], [a + s, a + s, 0], [a + s, a + s, a + s], [0, 0, 0], [a - 0.1, 0, 0]])
    expected = np.array([s, np.sqrt(2) * s, np.sqrt(3) * s, -a, -0.1]) - r
    assert cell.phi(X) == pytest.approx(expected, abs=1e-12)
    # beyond the band only a lower bound of at least the band, exact below it
    phi = cell.phi(X, band=0.05)
    assert np.all((phi >= 0.05) & (phi <= expected + 1e-12) | (np.abs(phi - expected) < 1e-12))
    near = np.array([[a + r + 0.01, a + 0.01, 0.0]])
    assert cell.phi(near, band=0.05) == pytest.approx(cell.phi(near), abs=1e-12)
    Q = cell.project(np.random.default_rng(0).uniform(-1, 1, (500, 3)) * (a + r + 0.5))
    assert np.abs(cell.phi(Q)).max() < 1e-12


def test_rounded_voronoi_cells_do_not_depend_on_w(round_cells):
    """Same labels and reference faces at another w, and neighbours at least w/2 apart (w unless shifted)."""
    w = 0.3
    cs = make_rounded_voronoi_cells(w, 0.5, BOX, seed=0)
    assert len(cs.cells) == 7 and np.array_equal(cs.labels, round_cells.labels)
    for f, f2 in zip(cs.faces, round_cells.faces):
        assert len(f) and np.array_equal(f, f2)
    rng = np.random.default_rng(0)
    for k, cell in enumerate(cs.cells):
        lo, hi = cell.bounds()
        X = rng.uniform(lo, hi, (2000, 3))
        Q = cell.project(X[cell.phi(X) > -cell.r + 1e-6])
        gap = min(other.phi(Q).min() for j, other in enumerate(cs.cells) if j != k)
        assert gap >= w / 2 - 1e-9


@needs_extra
def test_tissue_mesh_names_and_volume(round_cells, tissue):
    alpha_surface, mesh = tissue
    n_cells = len(round_cells.cells)
    assert mesh.GetMaterials() == ("ecs", *(f"cell_{k}" for k in range(n_cells)))
    walls = set(tio.WALLS) | {"cell_boundary"} | {f"membrane_{k}" for k in range(n_cells)}
    assert set(mesh.GetBoundaries()) == walls
    assert sum(_area(mesh, w) for w in tio.WALLS + ["cell_boundary"]) == pytest.approx(2 * (6 * 6 + 2 * 6))
    # TetGen and netgen keep the surface, so the mesh encloses the same volumes as the surface mesh
    assert _volume(mesh, "ecs") / BOX.prod() == pytest.approx(alpha_surface, abs=1e-9)
    assert alpha_surface == pytest.approx(0.0756, abs=2e-3)
    for k, cell in enumerate(round_cells.cells):      # region cell_k is cells[k]
        assert cell.phi(_centroid(mesh, f"cell_{k}")[None])[0] < 0


@needs_extra
def test_save_load_round_trip(round_cells, tissue, tmp_path):
    _, mesh = tissue
    path = tmp_path / "tissue.vtu"
    faces = np.concatenate(round_cells.faces).reshape(-1, 9)
    tio.save_tissue_mesh(path, mesh, w=0.1, seed=0, cell_labels=round_cells.labels, ref_faces=faces)
    loaded, meta = tio.load_tissue_mesh(path)
    assert meta["w"] == 0.1 and meta["seed"] == 0 and np.array_equal(meta["cell_labels"], round_cells.labels)
    assert np.array_equal(meta["ref_faces"], faces)
    assert (loaded.ne, loaded.nv) == (mesh.ne, mesh.nv)
    assert loaded.GetMaterials() == mesh.GetMaterials()
    assert set(loaded.GetBoundaries()) == set(mesh.GetBoundaries())
    for name in mesh.GetMaterials():
        assert _volume(loaded, name) == pytest.approx(_volume(mesh, name), rel=1e-12)
    for name in set(mesh.GetBoundaries()):
        assert _area(loaded, name) == pytest.approx(_area(mesh, name), rel=1e-12)


@needs_extra
def test_corner_cell_matches_steiner_formula(pool):
    """A rounded cube centred on a box corner: an eighth of its volume and area lie in the box (Steiner)."""
    a, r, B = 0.5, 0.3, np.array([2.0, 2.0, 2.0])
    cell = RoundedPolytope(box_halfspaces(-a * np.ones(3), a * np.ones(3)), np.zeros(3), r)
    volume = (8 * a**3 + 24 * a**2 * r + 6 * np.pi * a * r**2 + 4 / 3 * np.pi * r**3) / 8
    area = (24 * a**2 + 12 * np.pi * a * r + 4 * np.pi * r**2) / 8
    mesh = tm.volume_mesh(tm.surface_mesh([cell], B, 0.1, pool=pool))
    assert _volume(mesh, "cell_0") == pytest.approx(volume, rel=3e-4)
    assert _area(mesh, "membrane_0") == pytest.approx(area, rel=4e-3)


def _tri_area(tris):
    return np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2


def test_facet_triangles_cube():
    V = np.array(np.meshgrid([0, 1], [0, 2], [0, 3], indexing="ij")).reshape(3, -1).T.astype(float)
    tris, k = facet_triangles(V, box_halfspaces(np.zeros(3), [1, 2, 3]))
    for j, area in enumerate([6, 6, 3, 3, 2, 2]):
        assert _tri_area(tris[k == j]) == pytest.approx(area)


@needs_extra
def test_tube_cells_mesh(pool):
    """Tube cells in a 4 x 4 x 1 box mesh, and their reference faces do not depend on w."""
    box = np.array([4.0, 4.0, 1.0])
    cs = make_tube_cells(0.1, 0.25, box, seed=2)
    cs2 = make_tube_cells(0.2, 0.25, box, seed=2, n_r=0, n_shift=0)  # no r_k / shift search: faster
    assert len(cs.cells) == 4 and np.array_equal(cs.labels, cs2.labels)
    for f, f2 in zip(cs.faces, cs2.faces):
        assert len(f) and np.array_equal(f, f2)
    surf = tm.surface_mesh(cs.cells, box, 0.25, cs.max_offset, pool)
    assert not surf.open_domains()
    mesh = tm.volume_mesh(surf)
    assert _volume(mesh, "ecs") / box.prod() == pytest.approx(surf.ecs_fraction(box), abs=1e-9)


def test_merge_outside_edge_point_past_corner():
    """A point on the box edge x+/z- that lies past the y- wall, without a corner neighbour, becomes the corner."""
    B = np.array([14.0, 14.0, 7.0])
    P = np.array([[14.0, -0.015, 0.0], [14.0, 0.3, 0.2], [13.8, 0.2, 0.0]])
    mask = np.array([0b010010, 0b000010, 0b010000])
    P, F, mask = tm._merge_outside(P, np.array([[0, 1, 2]]), mask, B)
    assert np.array_equal(P[0], [14.0, 0.0, 0.0]) and mask[0] == 0b010110
    assert np.array_equal(F, [[0, 1, 2]])
