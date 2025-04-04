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
    """Create a simple test geometry with two compartments that are connected
    by a membrane.
    """
    left = occ.Box((0, 0, 0), (1, 1, 1)).mat('left').bc('reflective')
    right = occ.Box((1, 0, 0), (2, 1, 1)).mat('right').bc('reflective')
    left.faces[1].bc('middle')

    geo = occ.OCCGeometry(occ.Glue([left, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation('multi_compartment_test', result_root=tmp_path)
    simulation.setup_geometry(mesh)
    points = [(0.2, 0.5, 0.5), (1.8, 0.5, 0.5)]
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=points))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.ms))

    return simulation


def total_substance(left, right, species, time):
    """Calculate the total substance of a species in two compartments."""
    return left[species][time] + right[species][time]


def test_multi_compartment_dynamics(tmp_path, visualize=False):
    """Test that, in a two-compartment geometry:
    - linear flux equilibrates a species
    - Michaelis-Menten flux works against a gradient
    - channel flux works against a gradient
    - reverse channel flux results are exactly mirrored
    """
    simulation = create_simulation(tmp_path)
    left = simulation.simulation_geometry.compartments['left']
    right = simulation.simulation_geometry.compartments['right']
    membrane = simulation.simulation_geometry.membranes['middle']

    # Species that should equilibrate
    equilibrium = simulation.add_species('equilibrium', valence=0)
    left.initialize_species(equilibrium, 0.6 * u.mmol / u.L)
    right.initialize_species(equilibrium, 1.2 * u.mmol / u.L)
    left.add_diffusion(equilibrium, 10 * u.um**2 / u.s)
    right.add_diffusion(equilibrium, 10 * u.um**2 / u.s)
    permeability = 10 * u.nm / u.ms * membrane.area
    t = transport.Passive(permeability=permeability)
    membrane.add_transport(species=equilibrium, transport=t, source=left, target=right)

    # Species with Michaelis-Menten transport that creates a discontinuity
    michaelis_menten = simulation.add_species('michaelis_menten', valence=0)
    left.initialize_species(michaelis_menten, 0.9 * u.mmol / u.L)
    right.initialize_species(michaelis_menten, 0.9 * u.mmol / u.L)
    left.add_diffusion(michaelis_menten, 10 * u.um**2 / u.s)
    right.add_diffusion(michaelis_menten, 10 * u.um**2 / u.s)
    t = transport.Active(v_max=1 * u.amol / u.s, km=0.5 * u.mmol / u.L)
    membrane.add_transport(species=michaelis_menten, transport=t, source=left, target=right)

    # Species with channel transport that creates a discontinuity
    channel_l2r = simulation.add_species('channel_l2r', valence=0)
    left.initialize_species(channel_l2r, 1.1 * u.mmol / u.L)
    right.initialize_species(channel_l2r, 0 * u.mmol / u.L)
    left.add_diffusion(channel_l2r, 10 * u.um**2 / u.s)
    right.add_diffusion(channel_l2r, 10 * u.um**2 / u.s)
    t = transport.GeneralFlux(1 * u.amol / u.s)
    membrane.add_transport(species=channel_l2r, transport=t, source=left, target=right)

    # Species with reverse channel transport that creates a discontinuity
    channel_r2l = simulation.add_species('channel_r2l', valence=0)
    left.initialize_species(channel_r2l, 0 * u.mmol / u.L)
    right.initialize_species(channel_r2l, 1.3 * u.mmol / u.L)
    left.add_diffusion(channel_r2l, 10 * u.um**2 / u.s)
    right.add_diffusion(channel_r2l, 10 * u.um**2 / u.s)
    membrane.add_transport(species=channel_r2l, transport=t, source=right, target=left)

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms)

    # Test point values
    p0, time = get_point_values(simulation.result_directory, point_id=0)
    p1, _ = get_point_values(simulation.result_directory, point_id=1)
    assert p0['equilibrium'][0] == pytest.approx(0.6)
    assert p1['equilibrium'][0] == pytest.approx(1.2)
    assert p0['equilibrium'][-1] == pytest.approx(0.9, rel=1e-3)
    assert p1['equilibrium'][-1] == pytest.approx(0.9, rel=1e-3)

    assert p0['michaelis_menten'][0] == pytest.approx(0.9)
    assert p1['michaelis_menten'][0] == pytest.approx(0.9)
    assert p0['michaelis_menten'][-1] < 0.8
    assert p1['michaelis_menten'][-1] > 1.0

    assert p0['channel_l2r'][0] == pytest.approx(1.1)
    assert p1['channel_l2r'][0] == pytest.approx(0)
    assert p0['channel_l2r'][-1] < 0.6
    assert p1['channel_l2r'][-1] > 0.5

    assert p0['channel_r2l'][0] == pytest.approx(0)
    assert p1['channel_r2l'][0] == pytest.approx(1.3)
    assert p0['channel_r2l'][-1] > 0.5
    assert p1['channel_r2l'][-1] < 0.8

    # Test total substance values
    s0, _ = get_substance_values(simulation.result_directory, compartment_id=0)
    s1, _ = get_substance_values(simulation.result_directory, compartment_id=1)
    assert total_substance(s0, s1, 'equilibrium', 0) == pytest.approx(1.8, rel=1e-3)
    assert total_substance(s0, s1, 'equilibrium', -1) == pytest.approx(1.8, rel=1e-3)

    assert total_substance(s0, s1, 'michaelis_menten', 0) == pytest.approx(1.8, rel=1e-3)
    assert total_substance(s0, s1, 'michaelis_menten', -1) == pytest.approx(1.8, rel=1e-3)

    assert total_substance(s0, s1, 'channel_l2r', 0) == pytest.approx(1.1, rel=1e-3)
    assert total_substance(s0, s1, 'channel_l2r', -1) == pytest.approx(1.1, rel=1e-3)

    assert total_substance(s0, s1, 'channel_r2l', 0) == pytest.approx(1.3, rel=1e-3)
    assert total_substance(s0, s1, 'channel_r2l', -1) == pytest.approx(1.3, rel=1e-3)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ['equilibrium', 'michaelis_menten', 'channel_l2r', 'channel_r2l']
        _, ((ax1, ax2), (ax3, ax4)) = \
            plt.subplots(2, 2, figsize=(10, 10), sharey=True, gridspec_kw={'wspace': 0})

        # Left panel: Value in the left region
        for s in species:
            ax1.plot(time / 1000, p0[s].T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Concentration [mM]")
        ax1.set_title("Left compartment")
        ax1.grid(True)
        ax1.legend()

        # Right panel: Value in the right region
        for s in species:
            ax2.plot(time / 1000, p1[s].T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Right compartment")
        ax2.grid(True)

        # Bottom left panel: Total substance in the left region
        for s in species:
            ax3.plot(time / 1000, s0[s].T, label=s)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Total substance [amol]")
        ax3.grid(True)

        # Bottom right panel: Total substance in the right region
        for s in species:
            ax4.plot(time / 1000, s1[s].T, label=s)
        ax4.set_xlabel("Time [s]")
        ax4.grid(True)
        plt.tight_layout()
        # plt.subplots_adjust(top=0.88)

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_multi_compartment_dynamics(tmpdir, visualize=True)
