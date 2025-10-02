import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u
import xarray as xr

import bmbcsim
from bmbcsim.units import mM


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

    simulation = bmbcsim.Simulation("simple_diffusion_test", mesh, result_root=tmp_path)

    return simulation


def test_simple_diffusion(tmp_path, visualize=False):
    """Test that, in a two-region compartment:
    - a single species with inhomogeneous initial conditions can equilibrate
    """
    simulation = create_simulation(tmp_path)
    cell = simulation.simulation_geometry.compartments["cell"]

    # Species with different initial values should relax to the same value
    equilibrium = simulation.add_species("equilibrium", valence=0)
    cell.initialize_species(equilibrium, {"left": 1.0 * mM, "right": 1.4 * mM})
    cell.add_diffusion(equilibrium, 3 * u.um**2 / u.s)

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=10 * u.ms)

    # Test point values (new ResultLoader syntax)
    result_loader = bmbcsim.ResultLoader(simulation.result_directory)
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

    equilibrium_0 = p0.sel(species="equilibrium")
    equilibrium_1 = p1.sel(species="equilibrium")

    assert equilibrium_0.isel(time=0) == pytest.approx(1.0)
    assert equilibrium_1.isel(time=0) == pytest.approx(1.4)
    assert equilibrium_0.isel(time=-1) == pytest.approx(1.2, rel=1e-2)
    assert equilibrium_1.isel(time=-1) == pytest.approx(1.2, rel=1e-2)

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        _, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 5), sharey=True, gridspec_kw={"wspace": 0}
        )
        time = point_values.coords["time"].values

        # Left panel: Value in the left region
        ax1.plot(time / 1000, p0.sel(species="equilibrium").T, label="Equilibrium")
        ax1.set_xlabel("Time [s]")
        ax1.set_title("Value in the left region")
        ax1.grid(True)
        ax1.legend()

        # Right panel: Value in the right region
        ax2.plot(time / 1000, p1.sel(species="equilibrium").T, label="Equilibrium")
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Value in the right region")
        ax2.grid(True)

        plt.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_simple_diffusion(tmpdir, visualize=True)
