import pytest

from netgen import occ
import ngsolve as ngs

import ecsim


@pytest.fixture(scope="module")
def geometry():
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


def test_geometry_contains_everything(geometry):
    """Test geometry should contain all regions compartments, and membranes.
    """
    assert set(geometry.region_names) == {'ecm:left', 'ecm:right', 'cell'}
    assert set(geometry.compartment_names) == {'ecm', 'cell'}
    assert set(geometry.membrane_names) == {'right_membrane', 'clamped', 'reflective'}


def test_geometry_identifies_full_subregions_correctly(geometry):
    """Test that the geometry identifies regions that make up compartments
    correctly.
    """
    ecm_regions = geometry.compartments['ecm'].get_region_names(full_names=True)
    cell_regions = geometry.compartments['cell'].get_region_names(full_names=True)

    assert set(ecm_regions) == {'ecm:left', 'ecm:right'}
    assert set(cell_regions) == {'cell'}


def test_geometry_identifies_subregions_correctly(geometry):
    """Test that the geometry identifies regions that make up compartments
    correctly.
    """
    ecm_regions = geometry.compartments['ecm'].get_region_names()
    cell_regions = geometry.compartments['cell'].get_region_names()

    assert set(ecm_regions) == {'left', 'right'}
    assert set(cell_regions) == {'cell'}


def test_geometry_identifies_membrane_neighbors_correctly(geometry):
    """Test that the geometry identifies membrane neighbors correctly.
    """
    ecm = geometry.compartments['ecm']
    cell = geometry.compartments['cell']

    right_membrane = geometry.membranes['right_membrane']
    assert cell == right_membrane.neighbor(ecm)
    assert ecm == right_membrane.neighbor(cell)

    clamped = geometry.membranes['clamped']
    assert None is clamped.neighbor(ecm)

    reflective = geometry.membranes['reflective']
    assert None is reflective.neighbor(ecm)
    assert None is reflective.neighbor(cell)
