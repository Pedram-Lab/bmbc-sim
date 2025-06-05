import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u
import xarray as xr

import ecsim
from ecsim.simulation import transport
from ecsim.units import mM


def create_simulation(tmp_path):
    """Create a simple test geometry with a compartment that is split into two
    regions.
    """
    left = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    right = occ.Box(occ.Pnt(1, 0, 0), occ.Pnt(2, 1, 1))

    left.mat("cell:left").bc("reflective")
    right.mat("cell:right").bc("reflective")
    left.faces[0].bc("left")
    right.faces[1].bc("right")

    geo = occ.OCCGeometry(occ.Glue([left, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation("multi_region_test", mesh, result_root=tmp_path)

    return simulation


def test_multi_region_dynamics(tmp_path, visualize=False):
    """Test that, in a two-region compartment:
    - a single species can stay constant
    - different diffusion coefficients lead to different gradients
    - a single species with different initial values relaxes to the same value
    - an immobile species reacts only in one region
    """
    simulation = create_simulation(tmp_path)
    cell = simulation.simulation_geometry.compartments["cell"]
    left_membrane = simulation.simulation_geometry.membranes["left"]
    right_membrane = simulation.simulation_geometry.membranes["right"]

    # Species that should stay constant
    fixed = simulation.add_species("fixed", valence=0)
    cell.initialize_species(fixed, 1.1 * mM)
    cell.add_diffusion(fixed, 1 * u.um**2 / u.ms)

    # Species with efflux left and influx right should have a piecewise linear profile
    gradient = simulation.add_species("gradient", valence=0)
    cell.initialize_species(gradient, 1 * mM)
    cell.add_diffusion(
        gradient, {"left": 10 * u.um**2 / u.s, "right": 1 * u.um**2 / u.s}
    )
    flux = 1 * u.amol / u.s
    left_membrane.add_transport(
        species=gradient,
        source=cell,
        target=None,
        transport=transport.GeneralFlux(flux=flux),
    )
    right_membrane.add_transport(
        species=gradient,
        source=None,
        target=cell,
        transport=transport.GeneralFlux(flux=flux),
    )

    # Species with different initial values should relax to the same value
    equilibrium = simulation.add_species("equilibrium", valence=0)
    cell.initialize_species(equilibrium, {"left": 1.0 * mM, "right": 1.4 * mM})
    cell.add_diffusion(equilibrium, 3 * u.um**2 / u.s)

    # Immobile species that annihilates a mobile species in one region
    mobile = simulation.add_species("mobile", valence=0)
    immobile = simulation.add_species("immobile", valence=0)
    cell.initialize_species(mobile, 1.3 * mM)
    cell.initialize_species(immobile, {"left": 0 * mM, "right": 0.6 * mM})
    cell.add_diffusion(mobile, 10 * u.um**2 / u.s)
    cell.add_reaction(
        reactants=[mobile, immobile], products=[], k_f=10 / u.s, k_r=0 * mM / u.s
    )

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=10 * u.ms)

    # Test point values (new ResultLoader syntax)
    result_loader = ecsim.ResultLoader(simulation.result_directory)
    assert len(result_loader) == 101
    points = [(0.2, 0.5, 0.5), (1.8, 0.5, 0.5)]
    point_values = xr.concat(
        [
            result_loader.load_point_values(i, points=points)
            for i in range(len(result_loader))
        ],
        dim="time",
    )
    p0 = point_values.isel(point=0)
    p1 = point_values.isel(point=1)

    fixed_0 = p0.sel(species="fixed")
    fixed_1 = p1.sel(species="fixed")
    gradient_0 = p0.sel(species="gradient")
    gradient_1 = p1.sel(species="gradient")
    equilibrium_0 = p0.sel(species="equilibrium")
    equilibrium_1 = p1.sel(species="equilibrium")
    mobile_0 = p0.sel(species="mobile")
    mobile_1 = p1.sel(species="mobile")
    immobile_0 = p0.sel(species="immobile")
    immobile_1 = p1.sel(species="immobile")

    assert fixed_0.isel(time=0) == pytest.approx(1.1)
    assert fixed_1.isel(time=0) == pytest.approx(1.1)
    assert fixed_0.isel(time=-1) == pytest.approx(1.1)
    assert fixed_1.isel(time=-1) == pytest.approx(1.1)

    left_final, right_final = gradient_0.isel(time=-1), gradient_1.isel(time=-1)
    assert gradient_0.isel(time=0) == pytest.approx(1)
    assert gradient_1.isel(time=0) == pytest.approx(1)
    assert left_final < 1
    assert right_final > 1
    assert (1 - left_final) < (right_final - 1)

    assert equilibrium_0.isel(time=0) == pytest.approx(1.0)
    assert equilibrium_1.isel(time=0) == pytest.approx(1.4)
    assert equilibrium_0.isel(time=-1) == pytest.approx(1.2, rel=1e-2)
    assert equilibrium_1.isel(time=-1) == pytest.approx(1.2, rel=1e-2)

    assert mobile_0.isel(time=0) == pytest.approx(1.3)
    assert mobile_1.isel(time=0) == pytest.approx(1.3)
    assert immobile_0.isel(time=0) == pytest.approx(0)
    assert immobile_1.isel(time=0) == pytest.approx(0.6)
    assert mobile_0.isel(time=-1) == pytest.approx(1.0, rel=1e-2)
    assert mobile_1.isel(time=-1) == pytest.approx(1.0, rel=1e-2)
    assert immobile_0.isel(time=-1) == pytest.approx(0, abs=1e-2)
    assert immobile_1.isel(time=-1) == pytest.approx(0, abs=1e-2)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ["fixed", "gradient", "equilibrium", "mobile", "immobile"]
        _, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 5), sharey=True, gridspec_kw={"wspace": 0}
        )
        time = point_values.coords["time"].values

        # Left panel: Value in the left region
        for s in species:
            ax1.plot(time / 1000, p0.sel(species=s).T, label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_title("Value in the left region")
        ax1.grid(True)
        ax1.legend()

        # Right panel: Value in the right region
        for s in species:
            ax2.plot(time / 1000, p1.sel(species=s).T, label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Value in the right region")
        ax2.grid(True)

        plt.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_multi_region_dynamics(tmpdir, visualize=True)
