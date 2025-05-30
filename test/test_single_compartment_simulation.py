import tempfile
import pytest

import ngsolve as ngs
from netgen import occ
from matplotlib import pyplot as plt
import astropy.units as u
import xarray as xr

import ecsim


def create_simulation(tmp_path):
    """Create a simple test geometry with a single compartment.
    """
    box = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1)).mat('cell').bc('reflective')

    geo = occ.OCCGeometry(box)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    simulation = ecsim.Simulation('single_compartment_test', mesh, result_root=tmp_path)

    return simulation


def test_single_compartment_dynamics(tmp_path, visualize=False):
    """Test that, in a single compartment:
    - a single species can stay constant
    - reactions can be used to simulate exponential decay
    - reactions can be used to simulate linear growth
    - two species can correctly react with each other
    """
    simulation = create_simulation(tmp_path)
    cell = simulation.simulation_geometry.compartments['cell']

    # Species that should stay constant
    fixed = simulation.add_species('fixed', valence=0)
    cell.initialize_species(fixed, 1.1 * u.mmol / u.L)
    cell.add_diffusion(fixed, 1 * u.um**2 / u.ms)

    # Species that should decay exponentially over time
    decay = simulation.add_species('decay', valence=0)
    cell.initialize_species(decay, 1.2 * u.mmol / u.L)
    cell.add_reaction(reactants=[decay], products=[], k_f=1 / u.s, k_r=0 * u.mmol / (u.L * u.s))

    # Species that should grow linearly over time
    growth = simulation.add_species('growth', valence=0)
    cell.initialize_species(growth, 1.3 * u.mmol / u.L)
    cell.add_reaction(reactants=[growth], products=[], k_f=0 / u.s, k_r=1 * u.mmol / (u.L * u.s))

    # Two species that react with each other
    reactant_1 = simulation.add_species('reactant_1', valence=0)
    reactant_2 = simulation.add_species('reactant_2', valence=0)
    product = simulation.add_species('product', valence=0)
    cell.initialize_species(reactant_1, 1.4 * u.mmol / u.L)
    cell.initialize_species(reactant_2, 1.5 * u.mmol / u.L)
    cell.add_reaction(
        reactants=[reactant_1, reactant_2], products=[product],
        k_f=10 * u.L / (u.mmol * u.s), k_r=10 / u.s
    )

    # Run the simulation
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=10 * u.ms)

    # Test point values
    result_loader = ecsim.ResultLoader(simulation.result_directory)
    assert len(result_loader) == 101

    point = (0.5, 0.5, 0.5)
    point_values = xr.concat(
        [result_loader.load_point_values(i, points=point) for i in range(len(result_loader))],
        dim='time'
    )

    assert point_values.sel(species="fixed", time=0) == pytest.approx(1.1)
    assert point_values.sel(species="fixed", time=1000) == pytest.approx(1.1)

    assert point_values.sel(species="decay", time=0) == pytest.approx(1.2)
    assert 0.2 < point_values.sel(species="decay", time=1000) < 1.2

    assert point_values.sel(species="growth", time=0) == pytest.approx(1.3)
    assert point_values.sel(species="growth", time=1000) == pytest.approx(2.3)

    assert point_values.sel(species="reactant_1", time=0) == pytest.approx(1.4)
    assert point_values.sel(species="reactant_1", time=1000) < 1.4

    assert point_values.sel(species="reactant_2", time=0) == pytest.approx(1.5)
    assert point_values.sel(species="reactant_2", time=1000) < 1.5
    assert all(r1 < r2 for r1, r2 in zip(point_values.sel(species="reactant_1"), point_values.sel(species="reactant_2")))

    assert point_values.sel(species="product", time=0) == pytest.approx(0)
    assert point_values.sel(species="product", time=1000) > 0
    assert all(p < r1 for p, r1 in zip(point_values.sel(species="product"), point_values.sel(species="reactant_1")))
    assert all(p < r2 for p, r2 in zip(point_values.sel(species="product"), point_values.sel(species="reactant_2")))


    # Test substance values
    total_substance = xr.concat(
        [result_loader.load_total_substance(i) for i in range(len(result_loader))],
        dim="time"
    )
    # Select the region corresponding to the compartment 'cell'
    region = "cell"

    fixed_results = total_substance.sel(species="fixed", region=region)
    assert len(fixed_results) == 101
    assert fixed_results.sel(time=0) == pytest.approx(1.1)
    assert fixed_results.sel(time=1000) == pytest.approx(1.1)

    decay_results = total_substance.sel(species="decay", region=region)
    assert decay_results.sel(time=0) == pytest.approx(1.2)
    assert 0.2 < decay_results.sel(time=1000) < 1.2

    growth_results = total_substance.sel(species="growth", region=region)
    assert growth_results.sel(time=0) == pytest.approx(1.3)
    assert growth_results.sel(time=1000) == pytest.approx(2.3)

    reactant_1_results = total_substance.sel(species="reactant_1", region=region)
    assert reactant_1_results.sel(time=0) == pytest.approx(1.4)
    assert reactant_1_results.sel(time=1000) < 1.4

    reactant_2_results = total_substance.sel(species="reactant_2", region=region)
    assert reactant_2_results.sel(time=0) == pytest.approx(1.5)
    assert reactant_2_results.sel(time=1000) < 1.5
    assert all(r1 < r2 for r1, r2 in zip(reactant_1_results, reactant_2_results))

    product_results = total_substance.sel(species="product", region=region)
    assert product_results.sel(time=0) == pytest.approx(0)
    assert product_results.sel(time=1000) > 0
    assert all(p < r1 for p, r1 in zip(product_results, reactant_1_results))
    assert all(p < r2 for p, r2 in zip(product_results, reactant_2_results))

    if visualize:
        # Create a single figure with two side-by-side panels sharing the same y-axis.
        species = ['fixed', 'decay', 'growth', 'reactant_1', 'reactant_2', 'product']
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, gridspec_kw={'wspace': 0})

        # Extract time axis from coordinates (in ms, convert to s)
        time = point_values.coords['time'].values / 1000

        # Left panel: Concentration [mM] (point values)
        for s in species:
            ax1.plot(time, point_values.sel(species=s), label=s)
        ax1.set_xlabel("Time [s]")
        ax1.set_title('Concentration [mM]')
        ax1.grid(True)
        ax1.legend()

        # Right panel: Substance [amol] (total substance)
        for s in species:
            ax2.plot(time, total_substance.sel(species=s, region=region), label=s)
        ax2.set_xlabel("Time [s]")
        ax2.set_title('Substance [amol]')
        ax2.grid(True)

        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_single_compartment_dynamics(tmpdir, visualize=True)
