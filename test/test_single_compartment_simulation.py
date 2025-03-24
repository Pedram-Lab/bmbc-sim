import os
import tempfile
import numpy as np
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u
import xarray as xr

import ecsim
from ecsim.simulation import recorder


def get_point_values(dir):
    """Read the point values from a simulation."""
    zarr_path = os.path.join(dir, "point_data_0.zarr")
    point_data = xr.open_zarr(zarr_path)

    species_list = point_data.coords['species'].values
    time = point_data.coords['time'].values
    point = point_data.coords['point'].values[0]

    values = {}
    for species in species_list:
        ts = point_data.sel(species=species, point=point)
        values[species] = ts.to_array().values.squeeze()

    return values, time


def get_substance_values(dir):
    """Read the substance values from a simulation."""
    zarr_path = os.path.join(dir, "substance_data.zarr")
    substance_data = xr.open_zarr(zarr_path)

    species_list = substance_data.coords['species'].values
    time = substance_data.coords['time'].values
    compartment = substance_data.coords['compartment'].values[0]

    values = {}
    for species in species_list:
        ts = substance_data.sel(species=species, compartment=compartment)
        values[species] = ts.to_array().values.squeeze()

    return values, time


def create_simulation(tmp_path):
    """Create a simple test simulation with three regions that are sorted into two
    compartments."
    """
    box = occ.Box((0, 0, 0), (1, 1, 1)).mat('cell').bc('reflective')

    geo = occ.OCCGeometry(box)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    mesh.ngmesh.SetBCName(5, 'transport')

    simulation = ecsim.Simulation('test_simulation', result_root=tmp_path)
    simulation.add_geometry(mesh)
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=[(0.5, 0.5, 0.5)]))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.ms))

    return simulation


@pytest.fixture(scope="function")
def simulation(tmp_path):
    """Wrap the creation of a simulation in a fixture.
    """
    return create_simulation(tmp_path)


def test_single_species_stays_constant(simulation, visualize=False):
    """Test that a the concentration of a single species stays constant with
    only diffusion.
    """
    species = simulation.add_species('test_species', valence=0)
    cell = simulation.simulation_geometry.compartments['cell']

    cell.initialize_species(species, 1.5 * u.mmol / u.L)
    cell.add_diffusion(species, 100 * u.um**2 / u.s)
    simulation.run(end_time=1 * u.s, time_step=10 * u.ms)

    values, time = get_point_values(simulation.result_directory)
    point_results = values['test_species']
    assert len(point_results) == 101
    assert point_results[0] == pytest.approx(1.5)
    assert point_results[-1] == pytest.approx(1.5)

    values, time = get_substance_values(simulation.result_directory)
    substance_results = values['test_species']
    assert len(substance_results) == 101
    assert substance_results[0] == pytest.approx(1.5)
    assert substance_results[-1] == pytest.approx(1.5)

    if visualize:
        plt.figure()
        plt.semilogy(time / 1000, np.abs(point_results.T - 1.5), label='Midpoint')
        plt.xlabel("Time [s]")
        plt.ylabel("Error in concentration [mM]")
        plt.title(f"Species: {species.name}")
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(time / 1000, np.abs(substance_results.T - 1.5), label='Cell')
        plt.xlabel("Time [s]")
        plt.ylabel("Error in substance [amol]")
        plt.title(f"Species: {species.name}")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        sim = create_simulation(tmpdir)
        test_single_species_stays_constant(sim, visualize=True)
