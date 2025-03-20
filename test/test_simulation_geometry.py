import pytest

from netgen import occ
import ngsolve as ngs

import ecsim


@pytest.fixture(scope="module")
def simulation_geometry():
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


def test_geometry_contains_everything(simulation_geometry):
    """Test geometry should contain all regions compartments, and membranes.
    """
    assert set(simulation_geometry.regions) == {'ecm:left', 'ecm:right', 'cell'}
    assert set(simulation_geometry.compartment_names) == {'ecm', 'cell'}
    assert set(simulation_geometry.membrane_names) == {'right_membrane', 'clamped', 'reflective'}


def test_geometry_identifies_full_subregions_correctly(simulation_geometry):
    """Test that the geometry identifies regions that make up compartments
    correctly.
    """
    ecm_regions = simulation_geometry.get_regions('ecm', full_names=True)
    cell_regions = simulation_geometry.get_regions('cell', full_names=True)

    assert set(ecm_regions) == {'ecm:left', 'ecm:right'}
    assert set(cell_regions) == {'cell'}


def test_geometry_identifies_subregions_correctly(simulation_geometry):
    """Test that the geometry identifies regions that make up compartments
    correctly.
    """
    ecm_regions = simulation_geometry.get_regions('ecm')
    cell_regions = simulation_geometry.get_regions('cell')

    assert set(ecm_regions) == {'left', 'right'}
    assert set(cell_regions) == {'cell'}


def test_geometry_identifies_membrane_neighbors_correctly(simulation_geometry):
    """Test that the geometry identifies membrane neighbors correctly.
    """
    assert 'ecm' in simulation_geometry.get_membrane_neighbors('right_membrane')
    assert 'cell' in simulation_geometry.get_membrane_neighbors('right_membrane')
    assert 'ecm' in simulation_geometry.get_membrane_neighbors('clamped')
    assert 'cell' in simulation_geometry.get_membrane_neighbors('reflective')
    assert 'ecm' in simulation_geometry.get_membrane_neighbors('reflective')
