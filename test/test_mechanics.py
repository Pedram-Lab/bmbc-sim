import tempfile

import ngsolve as ngs
from netgen import occ
import astropy.units as u

import bmbcsim
from bmbcsim.units import mM


def create_box_mesh():
    """Create a simple box geometry."""
    box = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    box.mat("cell").bc("side")
    geo = occ.OCCGeometry(box)
    return ngs.Mesh(geo.GenerateMesh(maxh=0.2))


def test_mechanics_solver_setup(tmp_path):
    """Test that the mechanics solver can be set up with elasticity parameters."""
    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(
        "mechanics_test", mesh, result_root=tmp_path, mechanics=True
    )

    cell = simulation.simulation_geometry.compartments["cell"]

    # Add a species with diffusion
    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.1 * u.um**2 / u.ms)

    # Add elasticity parameters
    cell.add_elasticity(youngs_modulus=1.0 * u.kPa)

    # Run for a few steps - this should work without error
    simulation.run(end_time=1 * u.ms, time_step=0.1 * u.ms, record_interval=1 * u.ms)


def test_mechanics_with_custom_poisson_ratio(tmp_path):
    """Test that custom Poisson ratio can be set."""
    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(
        "mechanics_test", mesh, result_root=tmp_path, mechanics=True
    )

    cell = simulation.simulation_geometry.compartments["cell"]

    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.1 * u.um**2 / u.ms)

    # Add elasticity with custom Poisson ratio
    cell.add_elasticity(youngs_modulus=10.0 * u.kPa, poisson_ratio=0.45)

    simulation.run(end_time=1 * u.ms, time_step=0.1 * u.ms, record_interval=1 * u.ms)


def test_mechanics_missing_elasticity_raises(tmp_path):
    """Test that missing elasticity parameters raise an error."""
    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(
        "mechanics_test", mesh, result_root=tmp_path, mechanics=True
    )

    cell = simulation.simulation_geometry.compartments["cell"]

    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.1 * u.um**2 / u.ms)

    # Don't add elasticity - should raise an error
    try:
        simulation.run(end_time=1 * u.ms, time_step=0.1 * u.ms)
        assert False, "Expected ValueError for missing elasticity"
    except ValueError as e:
        assert "Elasticity not defined" in str(e)


def test_mechanics_with_driving_species(tmp_path):
    """Test that a species can drive mechanical contraction."""
    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(
        "mechanics_driving_test", mesh, result_root=tmp_path, mechanics=True
    )

    cell = simulation.simulation_geometry.compartments["cell"]

    # Add a species that will drive contraction
    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.1 * u.um**2 / u.ms)

    # Add elasticity and driving species
    cell.add_elasticity(youngs_modulus=1.0 * u.kPa)
    cell.add_driving_species(ca, coupling_strength=0.1 * u.kPa / mM)

    # Run simulation - should work without error
    simulation.run(end_time=1 * u.ms, time_step=0.1 * u.ms, record_interval=1 * u.ms)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_mechanics_solver_setup(tmpdir)
        print("test_mechanics_solver_setup passed")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_mechanics_with_custom_poisson_ratio(tmpdir)
        print("test_mechanics_with_custom_poisson_ratio passed")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_mechanics_missing_elasticity_raises(tmpdir)
        print("test_mechanics_missing_elasticity_raises passed")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_mechanics_with_driving_species(tmpdir)
        print("test_mechanics_with_driving_species passed")
