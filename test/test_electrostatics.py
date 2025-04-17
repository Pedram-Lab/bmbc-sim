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
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=points))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.ms))

    return simulation


def total_substance(left, right, species, time):
    """Calculate the total substance of a species in two compartments."""
    return left[species][time] + right[species][time]


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

    # It's already in equilibrium in the left compartment
    left.initialize_species(eq1, 0.6 * u.mmol / u.L)
    left.initialize_species(eq2, 1.1 * u.mmol / u.L)
    left.add_diffusion(eq1, 1 * u.um**2 / u.s)
    left.add_diffusion(eq2, 1 * u.um**2 / u.s)

    # It's not in equilibrium in the right compartment and it should equilibrate
    comp.initialize_species(eq1, {'middle': 0.4 * u.mmol / u.L, 'right': 1.0 * u.mmol / u.L})
    comp.initialize_species(eq2, 1.2 * u.mmol / u.L)
    comp.add_diffusion(eq1, 1 * u.um**2 / u.s)
    comp.add_diffusion(eq2, 1 * u.um**2 / u.s)

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms)

    # Test point values
    p0, time = get_point_values(simulation.result_directory, point_id=0)
    p1, _ = get_point_values(simulation.result_directory, point_id=1)
    p2, _ = get_point_values(simulation.result_directory, point_id=2)
    assert p0['eq1'][0] == pytest.approx(0.6)
    assert p1['eq1'][0] == pytest.approx(0.4)
    assert p2['eq1'][0] == pytest.approx(1.0)
    assert p0['eq2'][0] == pytest.approx(1.1)
    assert p1['eq2'][0] == pytest.approx(1.2)
    assert p2['eq2'][0] == pytest.approx(1.2)

    assert p0['eq1'][1] == pytest.approx(0.6)
    assert p1['eq1'][1] == pytest.approx(0.7)
    assert p2['eq1'][1] == pytest.approx(0.7)
    assert p0['eq2'][1] == pytest.approx(1.1)
    assert p1['eq2'][1] == pytest.approx(1.2)
    assert p2['eq2'][1] == pytest.approx(1.2)


    # Test total substance values
    s0, _ = get_substance_values(simulation.result_directory, compartment_id=0)
    s1, _ = get_substance_values(simulation.result_directory, compartment_id=1)
    assert s0['eq1'][0] == pytest.approx(0.6)
    assert s1['eq1'][0] == pytest.approx(0.4)
    assert s0['eq2'][0] == pytest.approx(1.1)
    assert s1['eq2'][0] == pytest.approx(1.2)

    assert s0['eq1'][1] == pytest.approx(0.6)
    assert s1['eq1'][1] == pytest.approx(0.7)
    assert s0['eq2'][1] == pytest.approx(1.1)
    assert s1['eq2'][1] == pytest.approx(1.2)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ['eq1', 'eq2']
        _, ((ax1, ax2, ax3), (ax4, ax5, _)) = \
            plt.subplots(3, 2, figsize=(15, 10), sharey=True, gridspec_kw={'wspace': 0})

        # Top row: point values in left, middle, right regions
        for s in species:
            ax1.plot(time / 1000, p0[s].T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Concentration [mM]")
        ax1.set_title("Left region")
        ax1.grid(True)
        ax1.legend()

        for s in species:
            ax2.plot(time / 1000, p1[s].T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Middle region")
        ax2.grid(True)

        for s in species:
            ax3.plot(time / 1000, p2[s].T, label=s)
        ax3.set_xlabel("Time [s]")
        ax3.set_title("Right region")
        ax3.grid(True)

        # Bottom row: total substance in left/middle and right regions
        for s in species:
            ax4.plot(time / 1000, s0[s].T, label=s)
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Total substance [amol]")
        ax4.set_title("Total substance in left and middle regions")
        ax4.grid(True)

        for s in species:
            ax5.plot(time / 1000, s1[s].T, label=s)
        ax5.set_xlabel("Time [s]")
        ax5.set_title("Total substance in right region")
        ax5.grid(True)
        plt.tight_layout()

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_pnp_dynamics(tmpdir, visualize=True)
