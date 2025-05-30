import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u
import xarray as xr

import ecsim
from ecsim.simulation import transport


def create_simulation(tmp_path):
    """Create a simple test geometry with two compartments that are connected
    by a membrane.
    """
    left = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1)).mat('left').bc('reflective')
    right = occ.Box(occ.Pnt(1, 0, 0), occ.Pnt(2, 1, 1)).mat('right').bc('reflective')
    left.faces[1].bc('middle')

    geo = occ.OCCGeometry(occ.Glue([left, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation('multi_compartment_test', mesh, result_root=tmp_path)

    return simulation


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
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=10 * u.ms)

    # Test point values (new ResultLoader syntax)
    result_loader = ecsim.ResultLoader(simulation.result_directory)
    assert len(result_loader) == 101
    points = [(0.2, 0.5, 0.5), (1.8, 0.5, 0.5)]
    point_values = xr.concat([result_loader.load_point_values(i, points=points) for i in range(len(result_loader))], dim='time')
    p0 = point_values.isel(point=0)
    p1 = point_values.isel(point=1)

    equilibrium_0 = p0.sel(species="equilibrium")
    equilibrium_1 = p1.sel(species="equilibrium")
    michaelis_menten_0 = p0.sel(species="michaelis_menten")
    michaelis_menten_1 = p1.sel(species="michaelis_menten")
    channel_l2r_0 = p0.sel(species="channel_l2r")
    channel_l2r_1 = p1.sel(species="channel_l2r")
    channel_r2l_0 = p0.sel(species="channel_r2l")
    channel_r2l_1 = p1.sel(species="channel_r2l")

    assert equilibrium_0.isel(time=0) == pytest.approx(0.6)
    assert equilibrium_1.isel(time=0) == pytest.approx(1.2)
    assert equilibrium_0.isel(time=-1) == pytest.approx(0.9, rel=1e-3)
    assert equilibrium_1.isel(time=-1) == pytest.approx(0.9, rel=1e-3)

    assert michaelis_menten_0.isel(time=0) == pytest.approx(0.9)
    assert michaelis_menten_1.isel(time=0) == pytest.approx(0.9)
    assert michaelis_menten_0.isel(time=-1) < 0.8
    assert michaelis_menten_1.isel(time=-1) > 1.0

    assert channel_l2r_0.isel(time=0) == pytest.approx(1.1)
    assert channel_l2r_1.isel(time=0) == pytest.approx(0)
    assert channel_l2r_0.isel(time=-1) < 0.6
    assert channel_l2r_1.isel(time=-1) > 0.5

    assert channel_r2l_0.isel(time=0) == pytest.approx(0)
    assert channel_r2l_1.isel(time=0) == pytest.approx(1.3)
    assert channel_r2l_0.isel(time=-1) > 0.5
    assert channel_r2l_1.isel(time=-1) < 0.8

    # Test total substance values (new ResultLoader syntax)
    total_substance = xr.concat(
        [result_loader.load_total_substance(i) for i in range(len(result_loader))],
        dim="time"
    )
    left = total_substance.sel(region="left")
    right = total_substance.sel(region="right")
    def total_substance_sum(species, t):
        return left.sel(species=species).isel(time=t) + right.sel(species=species).isel(time=t)

    assert total_substance_sum('equilibrium', 0) == pytest.approx(1.8, rel=1e-3)
    assert total_substance_sum('equilibrium', -1) == pytest.approx(1.8, rel=1e-3)
    assert total_substance_sum('michaelis_menten', 0) == pytest.approx(1.8, rel=1e-3)
    assert total_substance_sum('michaelis_menten', -1) == pytest.approx(1.8, rel=1e-3)
    assert total_substance_sum('channel_l2r', 0) == pytest.approx(1.1, rel=1e-3)
    assert total_substance_sum('channel_l2r', -1) == pytest.approx(1.1, rel=1e-3)
    assert total_substance_sum('channel_r2l', 0) == pytest.approx(1.3, rel=1e-3)
    assert total_substance_sum('channel_r2l', -1) == pytest.approx(1.3, rel=1e-3)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ['equilibrium', 'michaelis_menten', 'channel_l2r', 'channel_r2l']
        _, ((ax1, ax2), (ax3, ax4)) = \
            plt.subplots(2, 2, figsize=(10, 10), sharey=True, gridspec_kw={'wspace': 0})
        time = point_values.coords['time'].values

        # Left panel: Value in the left region
        for s in species:
            ax1.plot(time / 1000, p0.sel(species=s).T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Concentration [mM]")
        ax1.set_title("Left compartment")
        ax1.grid(True)
        ax1.legend()

        # Right panel: Value in the right region
        for s in species:
            ax2.plot(time / 1000, p1.sel(species=s).T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Right compartment")
        ax2.grid(True)

        # Bottom left panel: Total substance in the left region
        for s in species:
            ax3.plot(time / 1000, left.sel(species=s).T, label=s)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Total substance [amol]")
        ax3.grid(True)

        # Bottom right panel: Total substance in the right region
        for s in species:
            ax4.plot(time / 1000, right.sel(species=s).T, label=s)
        ax4.set_xlabel("Time [s]")
        ax4.grid(True)
        plt.tight_layout()

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_multi_compartment_dynamics(tmpdir, visualize=True)
