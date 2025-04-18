import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u

import ecsim
from ecsim.simulation import recorder
from conftest import get_point_values, get_substance_values


def create_simulation(tmp_path):
    """Create a simple test geometry with two compartments that are connected
    by a membrane.
    """
    left = occ.Box((0, 0, 0), (1, 1, 1)).mat('left').bc('reflective')
    middle = occ.Box((1, 0, 0), (2, 1, 1)).mat('comp:middle').bc('reflective')
    right = occ.Box((2, 0, 0), (3, 1, 1)).mat('comp:right').bc('reflective')
    left.faces[1].bc('interface')

    geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation(
        'multi_compartment_test',
        result_root=tmp_path,
        electrostatics=True
    )
    simulation.setup_geometry(mesh)
    points = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5)]
    simulation.add_recorder(recorder.PointValues(10 * u.us, points=points))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.us))

    return simulation


def test_pnp_dynamics(tmp_path, visualize=False):
    """Test that, in a two-compartment geometry:
    - two charged species in equilibrium stay in equilibrium
    - two charged species with spatial variance equilibrate
    """
    simulation = create_simulation(tmp_path)
    left = simulation.simulation_geometry.compartments['left']
    comp = simulation.simulation_geometry.compartments['comp']

    left.add_relative_permittivity(80.0)
    comp.add_relative_permittivity({'middle': 150.0, 'right': 100.0})

    # Introduce two species
    eq1 = simulation.add_species('eq1', valence=2)
    eq2 = simulation.add_species('eq2', valence=-1)

    # Both species are already in equilibrium in the left compartment
    left.initialize_species(eq1, 0.6 * u.mmol / u.L)
    left.initialize_species(eq2, 1.1 * u.mmol / u.L)
    left.add_diffusion(eq1, 1 * u.um**2 / u.s)
    left.add_diffusion(eq2, 1 * u.um**2 / u.s)

    # One species is in equilibrium, the other is not and pulls the first away from equilibrium
    comp.initialize_species(eq1, {'middle': 0.4 * u.mmol / u.L, 'right': 0.6 * u.mmol / u.L})
    comp.initialize_species(eq2, 1.0 * u.mmol / u.L)
    comp.add_diffusion(eq1, 0 * u.um**2 / u.s)
    comp.add_diffusion(eq2, 1 * u.um**2 / u.s)

    # Run the simulation
    simulation.run(end_time=1.0 * u.ms, time_step=1.0 * u.us)

    # Test point values
    pl, time = get_point_values(simulation.result_directory, point_id=0)
    pm, _ = get_point_values(simulation.result_directory, point_id=1)
    pr, _ = get_point_values(simulation.result_directory, point_id=2)
    assert pl['eq1'][0] == pytest.approx(0.6)
    assert pm['eq1'][0] == pytest.approx(0.4)
    assert pr['eq1'][0] == pytest.approx(0.6)
    assert pl['eq2'][0] == pytest.approx(1.1)
    assert pm['eq2'][0] == pytest.approx(1.0)
    assert pr['eq2'][0] == pytest.approx(1.0)

    assert pl['eq1'][-1] == pytest.approx(0.6)
    assert pm['eq1'][-1] == pytest.approx(0.4)
    assert pr['eq1'][-1] == pytest.approx(0.6)
    assert pl['eq2'][-1] == pytest.approx(1.1)
    assert pm['eq2'][-1] < 0.9
    assert pr['eq2'][-1] > 1.1

    # Test total substance values (0 = right compartment, 1 = left compartment)
    sl, _ = get_substance_values(simulation.result_directory, compartment_name='left')
    sr, _ = get_substance_values(simulation.result_directory, compartment_name='comp')
    assert sl['eq1'][0] == pytest.approx(0.6, rel=1e-3)
    assert sr['eq1'][0] == pytest.approx(1.0, rel=1e-3)
    assert sl['eq2'][0] == pytest.approx(1.1, rel=1e-3)
    assert sr['eq2'][0] == pytest.approx(2.0, rel=1e-3)

    assert sl['eq1'][-1] == pytest.approx(0.6, rel=1e-3)
    assert sr['eq1'][-1] == pytest.approx(1.0, rel=1e-3)
    assert sl['eq2'][-1] == pytest.approx(1.1, rel=1e-3)
    assert sr['eq2'][-1] == pytest.approx(2.0, rel=1e-3)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ['eq1', 'eq2']
        _, ((ax1, ax2, ax3), (ax4, ax5, _)) = \
            plt.subplots(2, 3, figsize=(15, 10), sharey=True, gridspec_kw={'wspace': 0})

        # Top row: point values in left, middle, right regions
        for s in species:
            ax1.plot(time / 1000, pl[s].T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Concentration [mM]")
        ax1.set_title("Left region")
        ax1.grid(True)
        ax1.legend()

        for s in species:
            ax2.plot(time / 1000, pm[s].T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Middle region")
        ax2.grid(True)

        for s in species:
            ax3.plot(time / 1000, pr[s].T, label=s)
        ax3.set_xlabel("Time [s]")
        ax3.set_title("Right region")
        ax3.grid(True)

        # Bottom row: total substance in left/middle and right regions
        for s in species:
            ax4.plot(time / 1000, sl[s].T, label=s)
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Total substance [amol]")
        ax4.set_title("Total substance in left and middle regions")
        ax4.grid(True)

        for s in species:
            ax5.plot(time / 1000, sr[s].T, label=s)
        ax5.set_xlabel("Time [s]")
        ax5.set_title("Total substance in right region")
        ax5.grid(True)
        plt.tight_layout()

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_pnp_dynamics(tmpdir, visualize=True)
