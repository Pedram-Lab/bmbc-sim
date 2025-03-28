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
    """Create a simple test simulation with three regions that are sorted into two
    compartments."
    """
    box = occ.Box((0, 0, 0), (1, 1, 1)).mat('cell').bc('reflective')

    geo = occ.OCCGeometry(box)
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    mesh.ngmesh.SetBCName(0, 'left')
    mesh.ngmesh.SetBCName(1, 'right')

    simulation = ecsim.Simulation('test_simulation', result_root=tmp_path)
    simulation.add_geometry(mesh)
    simulation.add_recorder(recorder.PointValues(10 * u.ms, points=[(0.5, 0.5, 0.5)]))
    simulation.add_recorder(recorder.CompartmentSubstance(10 * u.ms))

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
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms)

    # Test point values
    values, time = get_point_values(simulation.result_directory)
    fixed_results = values['fixed']
    assert len(fixed_results) == 101
    assert fixed_results[0] == pytest.approx(1.1)
    assert fixed_results[-1] == pytest.approx(1.1)

    decay_results = values['decay']
    assert decay_results[0] == pytest.approx(1.2)
    assert 0.2 < decay_results[-1] < 1.2

    growth_results = values['growth']
    assert growth_results[0] == pytest.approx(1.3)
    assert growth_results[-1] == pytest.approx(2.3)

    reactant_1_results = values['reactant_1']
    assert reactant_1_results[0] == pytest.approx(1.4)
    assert reactant_1_results[-1] < 1.4

    reactant_2_results = values['reactant_2']
    assert reactant_2_results[0] == pytest.approx(1.5)
    assert reactant_2_results[-1] < 1.5
    assert all(r1 < r2 for r1, r2 in zip(reactant_1_results, reactant_2_results))

    product_results = values['product']
    assert product_results[0] == pytest.approx(0)
    assert product_results[-1] > 0
    assert all(p < r1 for p, r1 in zip(product_results, reactant_1_results))
    assert all(p < r2 for p, r2 in zip(product_results, reactant_2_results))


    # Test substance values
    values, time = get_substance_values(simulation.result_directory)
    fixed_results = values['fixed']
    assert len(fixed_results) == 101
    assert fixed_results[0] == pytest.approx(1.1)
    assert fixed_results[-1] == pytest.approx(1.1)

    decay_results = values['decay']
    assert decay_results[0] == pytest.approx(1.2)
    assert 0.2 < decay_results[-1] < 1.2

    growth_results = values['growth']
    assert growth_results[0] == pytest.approx(1.3)
    assert growth_results[-1] == pytest.approx(2.3)

    reactant_1_results = values['reactant_1']
    assert reactant_1_results[0] == pytest.approx(1.4)
    assert reactant_1_results[-1] < 1.4

    reactant_2_results = values['reactant_2']
    assert reactant_2_results[0] == pytest.approx(1.5)
    assert reactant_2_results[-1] < 1.5
    assert all(r1 < r2 for r1, r2 in zip(reactant_1_results, reactant_2_results))

    product_results = values['product']
    assert product_results[0] == pytest.approx(0)
    assert product_results[-1] > 0
    assert all(p < r1 for p, r1 in zip(product_results, reactant_1_results))
    assert all(p < r2 for p, r2 in zip(product_results, reactant_2_results))

    if visualize:
        species = ['fixed', 'decay', 'growth', 'reactant_1', 'reactant_2', 'product']
        plt.figure()
        for s in species:
            plt.plot(time / 1000, values[s].T, label=s)
        plt.xlabel("Time [s]")
        plt.ylabel("Concentration [mM]")
        plt.title('Value in the center of the cell')
        plt.legend()
        plt.show()

        plt.figure()
        for s in species:
            plt.plot(time / 1000, values[s].T, label=s)
        plt.xlabel("Time [s]")
        plt.ylabel("Substance [amol]")
        plt.title('Value over the whole cell')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_single_compartment_dynamics(tmpdir, visualize=True)
