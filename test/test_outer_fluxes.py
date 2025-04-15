import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u

import ecsim
from ecsim.simulation import recorder, transport
from conftest import get_point_values, get_substance_values


def create_simulation(tmp_path, width):
    """Create a simple test geometry with one compartment and two named membranes.
    """
    box = occ.Box((0, 0, 0), (1, 1, width)).mat('cell').bc('reflective')
    box.faces[0].bc('left')
    box.faces[1].bc('right')

    geo = occ.OCCGeometry(box)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation(f'outside_fluxes_{width}_test', result_root=tmp_path)
    simulation.setup_geometry(mesh)
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=[(0.5, 0.5, 0.5)]))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.ms))

    return simulation


@pytest.mark.parametrize('width', [1, 2])
def test_fluxes_from_to_outside(tmp_path, width, visualize=False):
    """Test that, in a single compartment:
    - linear flux drives a species to a constant value (from above and below)
    - Michaelis-Menten efflux depletes a species
    - constant influx increases a species linearly
    - time dependent influx adds a defined amount of substance
    """
    simulation = create_simulation(tmp_path, width)
    cell = simulation.simulation_geometry.compartments['cell']
    left_membrane = simulation.simulation_geometry.membranes['left']
    right_membrane = simulation.simulation_geometry.membranes['right']

    # Species that starts too low increases to the outside value
    too_low = simulation.add_species('too-low', valence=0)
    cell.initialize_species(too_low, 0.5 * u.mmol / u.L)
    cell.add_diffusion(too_low, 1 * u.um**2 / u.ms)
    permeability = 10 * u.nm / u.ms * left_membrane.area
    t = transport.Passive(permeability=permeability, outside_concentration=0.7 * u.mmol / u.L)
    left_membrane.add_transport(species=too_low, transport=t, source=cell, target=None)

    # Species that starts too high decreases to the outside value
    too_high = simulation.add_species('too-high', valence=0)
    cell.initialize_species(too_high, 0.8 * u.mmol / u.L)
    cell.add_diffusion(too_high, 1 * u.um**2 / u.ms)
    left_membrane.add_transport(species=too_high, transport=t, source=cell, target=None)

    # Michaelis-Menten transport that depletes the species
    deplete = simulation.add_species('deplete', valence=0)
    cell.initialize_species(deplete, 0.4 * u.mmol / u.L)
    cell.add_diffusion(deplete, 1 * u.um**2 / u.ms)
    t = transport.Active(v_max=50 * u.amol / u.s, km=1 * u.mmol / u.L)
    left_membrane.add_transport(species=deplete, transport=t, source=cell, target=None)

    # Constant influx (x5) that increases the species linearly
    constant_influx = simulation.add_species('constant-influx', valence=0)
    cell.initialize_species(constant_influx, 0.1 * u.mmol / u.L)
    cell.add_diffusion(constant_influx, 1 * u.um**2 / u.ms)
    t = transport.GeneralFlux(0.2 * u.amol / (u.s))
    for _ in range(5):
        right_membrane.add_transport(species=constant_influx, transport=t, source=None, target=cell)

    # Time-dependent influx that adds a defined amount of substance
    variable_influx = simulation.add_species('variable-influx', valence=0)
    cell.initialize_species(variable_influx, 0.2 * u.mmol / u.L)
    cell.add_diffusion(variable_influx, 1 * u.um**2 / u.ms)
    t = transport.GeneralFlux(lambda t: t * (1 * u.s - t) * 6 * u.amol / u.s**3)
    right_membrane.add_transport(species=variable_influx, transport=t, source=None, target=cell)

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms)

    # Test point values
    pnt_values, time = get_point_values(simulation.result_directory)
    too_low_results = pnt_values['too-low']
    assert too_low_results[0] == pytest.approx(0.5)
    assert too_low_results[-1] == pytest.approx(0.7, rel=1e-3)

    too_high_results = pnt_values['too-high']
    assert too_high_results[0] == pytest.approx(0.8)
    assert too_high_results[-1] == pytest.approx(0.7, rel=1e-3)

    deplete_results = pnt_values['deplete']
    assert deplete_results[0] == pytest.approx(0.4)
    assert deplete_results[-1] == pytest.approx(0.0, abs=1e-3)

    influx_results = pnt_values['constant-influx']
    assert influx_results[0] == pytest.approx(0.1)
    assert influx_results[-1] == pytest.approx(1 / width + 0.1, rel=1e-3)

    limited_influx_results = pnt_values['variable-influx']
    assert limited_influx_results[0] == pytest.approx(0.2)
    assert limited_influx_results[-1] == pytest.approx(1 / width + 0.2, rel=1e-3)

    # Test substance values
    sbst_values, _ = get_substance_values(simulation.result_directory)
    too_low_results = sbst_values['too-low']
    assert too_low_results[0] == pytest.approx(0.5 * width)
    assert too_low_results[-1] == pytest.approx(0.7 * width, rel=1e-3)

    too_high_results = sbst_values['too-high']
    assert too_high_results[0] == pytest.approx(0.8 * width)
    assert too_high_results[-1] == pytest.approx(0.7 * width, rel=1e-3)

    deplete_results = sbst_values['deplete']
    assert deplete_results[0] == pytest.approx(0.4 * width)
    assert deplete_results[-1] == pytest.approx(0.0, abs=1e-3)

    influx_results = sbst_values['constant-influx']
    assert influx_results[0] == pytest.approx(0.1 * width)
    assert influx_results[-1] == pytest.approx(1 + 0.1 * width, rel=1e-3)

    limited_influx_results = sbst_values['variable-influx']
    assert limited_influx_results[0] == pytest.approx(0.2 * width)
    assert limited_influx_results[-1] == pytest.approx(1 + 0.2 * width, rel=1e-3)

    if visualize:
        species = ['too-low', 'too-high', 'deplete', 'constant-influx', 'variable-influx']
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0})
        fig.suptitle(f"Volume: {cell.volume}")

        for s in species:
            ax1.plot(time / 1000, pnt_values[s].T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_title('Concentration [mM]')
        ax1.grid(True)
        ax1.legend()

        for s in species:
            ax2.plot(time / 1000, sbst_values[s].T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title('Substance [amol]')
        ax2.grid(True)

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_fluxes_from_to_outside(tmpdir, 2, visualize=True)
