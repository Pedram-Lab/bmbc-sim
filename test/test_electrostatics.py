import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u
import xarray as xr

import ecsim
from ecsim.units import mM


def create_simulation(tmp_path):
    """Create a simple test geometry with two compartments that are connected
    by a membrane.
    """
    left = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    middle = occ.Box(occ.Pnt(1, 0, 0), occ.Pnt(2, 1, 1))
    right = occ.Box(occ.Pnt(2, 0, 0), occ.Pnt(3, 1, 1))

    left = left.mat("left").bc("reflective")
    middle = middle.mat("comp:middle").bc("interface")
    right = right.mat("comp:right").bc("reflective")
    left.faces[1].bc("interface")

    geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    return ecsim.Simulation(
        "electrostatics_test",
        mesh,
        result_root=tmp_path,
        electrostatics=True
    )


def test_pnp_dynamics(tmp_path, visualize=False):
    """Test that, in a two-compartment geometry:
    - two charged species in equilibrium stay in equilibrium
    - two charged species with spatial variance equilibrate
    """
    simulation = create_simulation(tmp_path)
    left = simulation.simulation_geometry.compartments["left"]
    comp = simulation.simulation_geometry.compartments["comp"]

    left.add_relative_permittivity(80.0)
    comp.add_relative_permittivity({"middle": 150.0, "right": 100.0})

    # Introduce two species
    eq1 = simulation.add_species("eq1", valence=2)
    eq2 = simulation.add_species("eq2", valence=-1)

    # Both species are already in equilibrium in the left compartment
    left.initialize_species(eq1, 0.6 * mM)
    left.initialize_species(eq2, 1.1 * mM)
    left.add_diffusion(eq1, 1 * u.um**2 / u.s)
    left.add_diffusion(eq2, 1 * u.um**2 / u.s)

    # One species is in equilibrium, the other is not and pulls the first away from equilibrium
    comp.initialize_species(eq1, {"middle": 0.4 * mM, "right": 0.6 * mM})
    comp.initialize_species(eq2, 1.0 * mM)
    comp.add_diffusion(eq1, 0 * u.um**2 / u.s)
    comp.add_diffusion(eq2, 1 * u.um**2 / u.s)

    # Run the simulation
    simulation.run(
        end_time=1.0 * u.ms,
        time_step=1.0 * u.us,
        record_interval=1.0 * u.us
    )

    # Test point values (new ResultLoader syntax)
    result_loader = ecsim.ResultLoader(simulation.result_directory)
    assert len(result_loader) == 1001
    points = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5)]
    point_values = xr.concat(
        [
            result_loader.load_point_values(i, points=points)
            for i in range(len(result_loader))
        ],
        dim="time",
    )
    pl = point_values.isel(point=0)
    pm = point_values.isel(point=1)
    pr = point_values.isel(point=2)

    eq1_left = pl.sel(species="eq1")
    eq1_middle = pm.sel(species="eq1")
    eq1_right = pr.sel(species="eq1")
    eq2_left = pl.sel(species="eq2")
    eq2_middle = pm.sel(species="eq2")
    eq2_right = pr.sel(species="eq2")

    assert eq1_left.isel(time=0) == pytest.approx(0.6)
    assert eq1_middle.isel(time=0) == pytest.approx(0.4)
    assert eq1_right.isel(time=0) == pytest.approx(0.6)
    assert eq2_left.isel(time=0) == pytest.approx(1.1)
    assert eq2_middle.isel(time=0) == pytest.approx(1.0)
    assert eq2_right.isel(time=0) == pytest.approx(1.0)

    assert eq1_left.isel(time=-1) == pytest.approx(0.6)
    assert eq1_middle.isel(time=-1) == pytest.approx(0.4)
    assert eq1_right.isel(time=-1) == pytest.approx(0.6)
    assert eq2_left.isel(time=-1) == pytest.approx(1.1)
    assert eq2_middle.isel(time=-1) < 0.9
    assert eq2_right.isel(time=-1) > 1.1

    # Test total substance values (new ResultLoader syntax)
    total_substance = xr.concat(
        [result_loader.load_total_substance(i) for i in range(len(result_loader))],
        dim="time",
    )
    left = total_substance.sel(region="left")
    middle = total_substance.sel(region="comp:middle")
    right = total_substance.sel(region="comp:right")
    comp = middle + right

    eq1_left_s = left.sel(species="eq1")
    eq1_comp_s = comp.sel(species="eq1")
    eq2_left_s = left.sel(species="eq2")
    eq2_comp_s = comp.sel(species="eq2")

    assert eq1_left_s.isel(time=0) == pytest.approx(0.6, rel=1e-3)
    assert eq1_comp_s.isel(time=0) == pytest.approx(1.0, rel=1e-3)
    assert eq2_left_s.isel(time=0) == pytest.approx(1.1, rel=1e-3)
    assert eq2_comp_s.isel(time=0) == pytest.approx(2.0, rel=1e-3)

    assert eq1_left_s.isel(time=-1) == pytest.approx(0.6, rel=1e-3)
    assert eq1_comp_s.isel(time=-1) == pytest.approx(1.0, rel=1e-3)
    assert eq2_left_s.isel(time=-1) == pytest.approx(1.1, rel=1e-3)
    assert eq2_comp_s.isel(time=-1) == pytest.approx(2.0, rel=1e-3)

    if visualize:
        species = ["eq1", "eq2"]
        time = pl.coords["time"].values / 1000
        _, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(
            2, 3, figsize=(15, 10), sharey=True, gridspec_kw={"wspace": 0}
        )
        for s in species:
            ax1.plot(time, pl.sel(species=s), label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Concentration [mM]")
        ax1.set_title("Left region")
        ax1.grid(True)
        ax1.legend()
        for s in species:
            ax2.plot(time, pm.sel(species=s), label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Middle region")
        ax2.grid(True)
        for s in species:
            ax3.plot(time, pr.sel(species=s), label=s)
        ax3.set_xlabel("Time [s]")
        ax3.set_title("Right region")
        ax3.grid(True)
        for s in species:
            ax4.plot(time, left.sel(species=s), label=s)
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Substance [amol]")
        ax4.set_title("Left region (substance)")
        ax4.grid(True)
        for s in species:
            ax5.plot(time, comp.sel(species=s), label=s)
        ax5.set_xlabel("Time [s]")
        ax5.set_title("Comp region (substance)")
        ax5.grid(True)
        plt.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_pnp_dynamics(tmpdir, visualize=True)
