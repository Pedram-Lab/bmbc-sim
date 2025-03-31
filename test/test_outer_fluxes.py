import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u

import ecsim
from ecsim.simulation import recorder, transport
from conftest import get_point_values, get_substance_values


def create_simulation(tmp_path):
    """Create a simple test simulation with three regions that are sorted into two
    compartments."
    """
    box = occ.Box((0, 0, 0), (1, 1, 1)).mat('cell').bc('reflective')

    geo = occ.OCCGeometry(box)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    mesh.ngmesh.SetBCName(0, 'left')
    mesh.ngmesh.SetBCName(1, 'right')

    simulation = ecsim.Simulation('test_simulation', result_root=tmp_path)
    simulation.setup_geometry(mesh)
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=[(0.5, 0.5, 0.5)]))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.ms))

    return simulation


def test_single_compartment_fluxes(tmp_path, visualize=False):
    """Test that, in a single compartment:
    - linear flux drives a species to a constant value (from above and below)
    - Michaelis-Menten efflux depletes a species
    - constant influx increases a species linearly
    - time dependent influx adds a defined amount of substance
    """
    simulation = create_simulation(tmp_path)
    cell = simulation.simulation_geometry.compartments['cell']
    left_membrane = simulation.simulation_geometry.membranes['left']

    # Species that should stay increase to the outside value
    too_low = simulation.add_species('too-low', valence=0)
    cell.initialize_species(too_low, 0.5 * u.mmol / u.L)
    cell.add_diffusion(too_low, 1 * u.um**2 / u.ms)
    t = transport.Linear(permeability=10 * u.nm / u.ms, outside_concentration=0.7 * u.mmol / u.L)
    left_membrane.add_transport(species=too_low, transport=t, source=cell, target=None)

    # Species that should stay decrease to the outside value
    too_high = simulation.add_species('too-high', valence=0)
    cell.initialize_species(too_high, 0.8 * u.mmol / u.L)
    cell.add_diffusion(too_high, 1 * u.um**2 / u.ms)
    left_membrane.add_transport(species=too_high, transport=t, source=cell, target=None)

    # Michaelis-Menten transport that depletes the species
    deplete = simulation.add_species('deplete', valence=0)
    cell.initialize_species(deplete, 0.4 * u.mmol / u.L)
    cell.add_diffusion(deplete, 1 * u.um**2 / u.ms)
    t = transport.MichaelisMenten(v_max=10 * u.amol / (u.um**2 * u.s), km=1 * u.mmol / u.L)
    left_membrane.add_transport(species=deplete, transport=t, source=cell, target=None)

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms)

    # Test point values
    values, time = get_point_values(simulation.result_directory)
    too_low_results = values['too-low']
    assert too_low_results[0] == pytest.approx(0.5)
    assert too_low_results[-1] == pytest.approx(0.7, rel=1e-4)

    too_high_results = values['too-high']
    assert too_high_results[0] == pytest.approx(0.8)
    assert too_high_results[-1] == pytest.approx(0.7, rel=1e-4)

    deplete_results = values['deplete']
    assert deplete_results[0] == pytest.approx(0.4)
    assert deplete_results[-1] == pytest.approx(0.0, abs=1e-4)

    # Test substance values
    values, time = get_substance_values(simulation.result_directory)
    too_low_results = values['too-low']
    assert too_low_results[0] == pytest.approx(0.5)
    assert too_low_results[-1] == pytest.approx(0.7, rel=1e-4)

    too_high_results = values['too-high']
    assert too_high_results[0] == pytest.approx(0.8)
    assert too_high_results[-1] == pytest.approx(0.7, rel=1e-4)

    deplete_results = values['deplete']
    assert deplete_results[0] == pytest.approx(0.4)
    assert deplete_results[-1] == pytest.approx(0.0, abs=1e-4)

    if visualize:
        species = ['too-low', 'too-high', 'deplete']
        plt.figure()
        for s in species:
            plt.plot(time / 1000, values[s].T, label=s)
        plt.xlabel("Time [s]")
        plt.ylabel("Concentration [mM]")
        plt.title('Value in the center of the cell')
        plt.legend()
        plt.show()

        plt.figure()
        for s in species:
            plt.plot(time / 1000, values[s].T, label=s)
        plt.xlabel("Time [s]")
        plt.ylabel("Substance [amol]")
        plt.title('Value over the whole cell')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_single_compartment_fluxes(tmpdir, visualize=True)
