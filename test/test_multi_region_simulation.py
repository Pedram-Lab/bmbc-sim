import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u

import ecsim
from ecsim.simulation import recorder, transport
from conftest import get_point_values


def create_simulation(tmp_path):
    """Create a simple test simulation with three regions that are sorted into two
    compartments."
    """
    left = occ.Box((0, 0, 0), (1, 1, 1)).mat('cell:left').bc('reflective')
    right = occ.Box((1, 0, 0), (2, 1, 1)).mat('cell:right').bc('reflective')
    left.faces[0].bc('left')
    right.faces[1].bc('right')

    geo = occ.OCCGeometry(occ.Glue([left, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation('multi_region_test', result_root=tmp_path)
    simulation.setup_geometry(mesh)
    points = [(0.2, 0.5, 0.5), (1.8, 0.5, 0.5)]
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=points))

    return simulation


def test_multi_region_dynamics(tmp_path, visualize=False):
    """Test that, in a two-region compartment:
    - a single species can stay constant
    - different diffusion coefficients lead to different gradients
    - a single species with different initial values relaxes to the same value
    - an immobile species reacts only in one region
    """
    simulation = create_simulation(tmp_path)
    cell = simulation.simulation_geometry.compartments['cell']
    left_membrane = simulation.simulation_geometry.membranes['left']
    right_membrane = simulation.simulation_geometry.membranes['right']

    # Species that should stay constant
    fixed = simulation.add_species('fixed', valence=0)
    cell.initialize_species(fixed, 1.1 * u.mmol / u.L)
    cell.add_diffusion(fixed, 1 * u.um**2 / u.ms)

    # Species with efflux left and influx right should have a piecewise linear profile
    gradient = simulation.add_species('gradient', valence=0)
    cell.initialize_species(gradient, 1 * u.mmol / u.L)
    cell.add_diffusion(gradient, {'left': 10 * u.um**2 / u.s, 'right': 1 * u.um**2 / u.s})
    flux = 1 * u.amol /  u.s
    left_membrane.add_transport(
        species=gradient, source=cell, target=None,
        transport=transport.Channel(flux=flux)
    )
    right_membrane.add_transport(
        species=gradient, source=None, target=cell,
        transport=transport.Channel(flux=flux)
    )

    # Species with different initial values should relax to the same value
    equilibrium = simulation.add_species('equilibrium', valence=0)
    cell.initialize_species(equilibrium, {'left': 1.0 * u.mmol / u.L, 'right': 1.4 * u.mmol / u.L})
    cell.add_diffusion(equilibrium, 3 * u.um**2 / u.s)

    # Immobile species that annihilates a mobile species in one region
    mobile = simulation.add_species('mobile', valence=0)
    immobile = simulation.add_species('immobile', valence=0)
    cell.initialize_species(mobile, 1.3 * u.mmol / u.L)
    cell.initialize_species(immobile, {'left': 0 * u.mmol / u.L, 'right': 0.6 * u.mmol / u.L})
    cell.add_diffusion(mobile, 10 * u.um**2 / u.s)
    cell.add_reaction(
        reactants=[mobile, immobile], products=[],
        k_f=10 / u.s, k_r=0 * u.mmol / (u.L * u.s)
    )

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms)

    # Test point values
    p0, time = get_point_values(simulation.result_directory, point_id=0)
    p1, time = get_point_values(simulation.result_directory, point_id=1)
    assert p0['fixed'][0] == pytest.approx(1.1)
    assert p1['fixed'][0] == pytest.approx(1.1)
    assert p0['fixed'][-1] == pytest.approx(1.1)
    assert p1['fixed'][-1] == pytest.approx(1.1)

    left_final, right_final = p0['gradient'][-1], p1['gradient'][-1]
    assert p0['gradient'][0] == pytest.approx(1)
    assert p1['gradient'][0] == pytest.approx(1)
    assert left_final < 1
    assert right_final > 1
    assert (1 - left_final) < (right_final - 1)

    assert p0['equilibrium'][0] == pytest.approx(1.0)
    assert p1['equilibrium'][0] == pytest.approx(1.4)
    assert p0['equilibrium'][-1] == pytest.approx(1.2, rel=1e-2)
    assert p1['equilibrium'][-1] == pytest.approx(1.2, rel=1e-2)

    assert p0['mobile'][0] == pytest.approx(1.3)
    assert p1['mobile'][0] == pytest.approx(1.3)
    assert p0['immobile'][0] == pytest.approx(0)
    assert p1['immobile'][0] == pytest.approx(0.6)
    assert p0['mobile'][-1] == pytest.approx(1.0, rel=1e-2)
    assert p1['mobile'][-1] == pytest.approx(1.0, rel=1e-2)
    assert p0['immobile'][-1] == pytest.approx(0, abs=1e-2)
    assert p1['immobile'][-1] == pytest.approx(0, abs=1e-2)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ['fixed', 'gradient', 'equilibrium', 'mobile', 'immobile']
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, gridspec_kw={'wspace': 0})

        # Left panel: Value in the left region
        for s in species:
            ax1.plot(time / 1000, p0[s].T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_title("Value in the left region")
        ax1.grid(True)
        ax1.legend()

        # Right panel: Value in the right region
        for s in species:
            ax2.plot(time / 1000, p1[s].T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Value in the right region")
        ax2.grid(True)

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_multi_region_dynamics(tmpdir, visualize=True)
