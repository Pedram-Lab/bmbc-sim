import pytest

import ngsolve as ngs
from netgen import occ

import bmbcsim


@pytest.fixture(scope="function")
def simulation(tmp_path_factory):
    """Create a simple test simulation with three regions that are sorted into two
    compartments."
    """
    left = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    middle = occ.Box(occ.Pnt(1, 0, 0), occ.Pnt(2, 1, 1))
    right = occ.Box(occ.Pnt(2, 0, 0), occ.Pnt(3, 1, 1))

    left.mat("ecm:left").bc("reflective")
    middle.mat("ecm:right").bc("reflective")
    right.mat("cell").bc("reflective")

    geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.5))

    mesh.ngmesh.SetBCName(5, "clamped")
    mesh.ngmesh.SetBCName(1, "left_membrane")
    mesh.ngmesh.SetBCName(6, "right_membrane")

    tmp_path = tmp_path_factory.mktemp("results")
    return bmbcsim.Simulation("test_simulation", mesh, result_root=tmp_path)


def test_added_species_are_present(simulation):
    """Test that the added species are present in the geometry description."""
    species = simulation.add_species("test_species", valence=1)

    assert species in simulation.species


def test_add_species_twice_raises_error(simulation):
    """Test that adding the same species twice raises an error."""
    _ = simulation.add_species("test_species", valence=1)

    with pytest.raises(ValueError):
        simulation.add_species("test_species", valence=1)
