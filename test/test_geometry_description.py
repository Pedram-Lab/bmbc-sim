import pytest

from netgen import occ
import ngsolve as ngs

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

    return ecsim.GeometryDescription(mesh)


def test_geometry_contains_everything(geometry_description):
    """Test geometry should contain all regions compartments, and membranes.
    """
    assert set(geometry_description.regions) == {'ecm:left', 'ecm:right', 'cell'}
    assert set(geometry_description.compartments) == {'ecm', 'cell'}
    assert set(geometry_description.membranes) == {'right_membrane', 'clamped', 'reflective'}


def test_geometry_identifies_full_subregions_correctly(geometry_description):
    """Test that the geometry identifies regions that make up compartments
    correctly.
    """
    ecm_regions = geometry_description.get_regions('ecm', full_names=True)
    cell_regions = geometry_description.get_regions('cell', full_names=True)

    assert set(ecm_regions) == {'ecm:left', 'ecm:right'}
    assert set(cell_regions) == {'cell'}


def test_geometry_identifies_subregions_correctly(geometry_description):
    """Test that the geometry identifies regions that make up compartments
    correctly.
    """
    ecm_regions = geometry_description.get_regions('ecm')
    cell_regions = geometry_description.get_regions('cell')

    assert set(ecm_regions) == {'left', 'right'}
    assert set(cell_regions) == {'cell'}


def test_geometry_identifies_membrane_neighbors_correctly(geometry_description):
    """Test that the geometry identifies membrane neighbors correctly.
    """
    assert 'ecm' in geometry_description.get_membrane_neighbors('right_membrane')
    assert 'cell' in geometry_description.get_membrane_neighbors('right_membrane')
    assert 'ecm' in geometry_description.get_membrane_neighbors('clamped')
    assert 'cell' in geometry_description.get_membrane_neighbors('reflective')
    assert 'ecm' in geometry_description.get_membrane_neighbors('reflective')
