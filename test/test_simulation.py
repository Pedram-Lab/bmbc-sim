import pytest

import ngsolve as ngs
from netgen import occ

import ecsim


@pytest.fixture(scope="function")
def simulation(tmp_path_factory):
    """Create a simple test simulation with three regions that are sorted into two
    compartments."
    """
    left = occ.Box((0, 0, 0), (1, 1, 1)).mat('ecm:left').bc('reflective')
    middle = occ.Box((1, 0, 0), (2, 1, 1)).mat('ecm:right').bc('reflective')
    right = occ.Box((2, 0, 0), (3, 1, 1)).mat('cell').bc('reflective')

    geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.5))

    mesh.ngmesh.SetBCName(5, 'clamped')
    mesh.ngmesh.SetBCName(1, 'left_membrane')
    mesh.ngmesh.SetBCName(6, 'right_membrane')

    tmp_path = tmp_path_factory.mktemp("results")
    simulation = ecsim.Simulation('test_simulation', result_root=tmp_path)
    simulation.setup_geometry(mesh)
    return simulation


def test_added_species_are_present(simulation):
    """Test that the added species are present in the geometry description.
    """
    species = simulation.add_species('test_species', valence=1)

    assert species in simulation.species


def test_add_species_twice_raises_error(simulation):
    """Test that adding the same species twice raises an error.
    """
    _ = simulation.add_species('test_species', valence=1)

    with pytest.raises(ValueError):
        simulation.add_species('test_species', valence=1)
