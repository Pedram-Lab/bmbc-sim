import pytest

import ngsolve as ngs
from netgen import occ

import ecsim


@pytest.fixture(scope="module")
def geometry_description():
    """Create a simple test geometry with three regions that are sorted into two
    compartments."
    """
    left = occ.Box((0, 0, 0), (1, 1, 1)).mat('ecm:left').bc('reflective')
    middle = occ.Box((1, 0, 0), (2, 1, 1)).mat('ecm:right').bc('reflective')
    right = occ.Box((2, 0, 0), (3, 1, 1)).mat('cell').bc('reflective')

    geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=1.0))

    mesh.ngmesh.SetBCName(5, 'clamped')
    mesh.ngmesh.SetBCName(1, 'left_membrane')
    mesh.ngmesh.SetBCName(6, 'right_membrane')

    return ecsim.SimulationGeometry(mesh)


def test_added_species_are_present(geometry_description):
    """Test that the added species are present in the geometry description.
    """
    simulation = ecsim.Simulation(geometry_description)
    species = simulation.add_species(ecsim.ChemicalSpecies('test_species', valence=1))

    assert species in simulation.species


def test_add_species_twice_raises_error(geometry_description):
    """Test that adding the same species twice raises an error.
    """
    simulation = ecsim.Simulation(geometry_description)
    species = simulation.add_species(ecsim.ChemicalSpecies('test_species', valence=1))

    with pytest.raises(ValueError):
        simulation.add_species(species)
