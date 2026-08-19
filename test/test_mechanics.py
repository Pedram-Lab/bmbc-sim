import tempfile

import pytest
import ngsolve as ngs
from netgen import occ
import astropy.units as u
import xarray as xr

import bmbcsim
from bmbcsim.units import mM
from bmbcsim.simulation import transport
from bmbcsim.simulation.fem_details import MechanicSolver, neo_hooke


def create_box_mesh():
    """Create a simple box geometry."""
    box = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
    box.mat("cell").bc("side")
    box.faces[0].bc("influx")
    geo = occ.OCCGeometry(box)
    return ngs.Mesh(geo.GenerateMesh(maxh=0.2))


def test_neo_hooke_vanishes_in_reference_state():
    """The stored energy density must be 0 at F = I, not an arbitrary offset.

    The offset cancels out of every force, so it is invisible in the solution --
    but it inflates the magnitude of the energy the Newton line search compares
    against, and with it the round-off of that comparison. Left in, it swamped
    the real energy decrease on fine meshes and stalled the solve.
    """
    mesh = create_box_mesh()
    for mu, lam in [(1.0, 1.5), (385.0, 1430.0), (0.01, 0.007)]:
        density = neo_hooke(ngs.Id(3), ngs.CF(mu), ngs.CF(lam))
        assert abs(ngs.Integrate(density, mesh)) < 1e-12 * max(mu, lam)


def test_mechanics_solver_setup(tmp_path):
    """Test that the mechanics solver can be set up with elasticity parameters."""
    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(mesh, result_directory=tmp_path, mechanics=True)

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
    simulation = bmbcsim.Simulation(mesh, result_directory=tmp_path, mechanics=True)

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
    simulation = bmbcsim.Simulation(mesh, result_directory=tmp_path, mechanics=True)

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
    simulation = bmbcsim.Simulation(mesh, result_directory=tmp_path, mechanics=True)

    cell = simulation.simulation_geometry.compartments["cell"]

    # Add a species that will drive contraction
    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.1 * u.um**2 / u.ms)

    # Add elasticity and driving species
    cell.add_elasticity(youngs_modulus=1.0 * u.kPa)
    cell.add_driving_species(ca, coupling_strength=0.1 * u.kPa / mM)

    # Run simulation
    simulation.run(end_time=1 * u.ms, time_step=0.1 * u.ms, record_interval=1 * u.ms)

    # Load results and verify concentration change due to volume contraction
    result_loader = bmbcsim.ResultLoader(simulation.result_directory)
    points = [(0.5, 0.5, 0.5)]
    point_values = xr.concat(
        [result_loader.load_point_values(i, points=points) for i in range(len(result_loader))],
        dim="time",
    )

    ca_values = point_values.sel(species="ca").isel(point=0)

    # Initial concentration should be 1.0 mM
    assert ca_values.isel(time=0) == pytest.approx(1.0)

    # Final concentration should be higher due to volume contraction
    # (chemical pressure drives contraction, concentration increases to conserve mass)
    assert ca_values.isel(time=-1) == pytest.approx(1.089, rel=1e-2)


def test_mechanics_takes_fast_path_when_warm_started(tmp_path, monkeypatch):
    """A step warm-started at equilibrium must converge without load stepping.

    Regression guard for a Newton threshold scaled to the residual the solve
    happens to start from: a warm start is already near equilibrium, so the
    threshold shrank along with the thing it was meant to bound, converged states
    were reported as failures, and every step paid for the load-stepping
    fallback. Load stepping could not rescue them either -- a smaller increment
    tightens the threshold in exactly the same way -- so this eventually
    surfaced as a spurious "no stable equilibrium exists" on stiffer meshes.
    """
    fallbacks = []
    fall_back = MechanicSolver._solve_by_load_stepping

    def counted(self, *args):
        fallbacks.append(len(fallbacks))
        return fall_back(self, *args)

    monkeypatch.setattr(MechanicSolver, "_solve_by_load_stepping", counted)

    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(mesh, result_directory=tmp_path, mechanics=True)

    cell = simulation.simulation_geometry.compartments["cell"]
    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 1.0 * mM)
    cell.add_diffusion(ca, 0.1 * u.um**2 / u.ms)
    cell.add_elasticity(youngs_modulus=1.0 * u.kPa)
    cell.add_driving_species(ca, coupling_strength=0.1 * u.kPa / mM)

    simulation.run(end_time=1 * u.ms, time_step=0.1 * u.ms, record_interval=1 * u.ms)

    assert not fallbacks, f"load stepping was triggered on {len(fallbacks)} steps"


def test_mechanics_with_dynamic_species(tmp_path):
    """Test that a species can drive mechanical contraction."""
    mesh = create_box_mesh()
    simulation = bmbcsim.Simulation(mesh, result_directory=tmp_path, mechanics=True)

    cell = simulation.simulation_geometry.compartments["cell"]
    influx_bnd = simulation.simulation_geometry.membranes["influx"]

    # Add a species that will drive contraction
    ca = simulation.add_species("ca")
    cell.initialize_species(ca, 0.0 * mM)
    cell.add_diffusion(ca, 5.0 * u.um**2 / u.ms)

    # Add a source term to increase concentration over time (step function at t=1ms)
    base_flux = 1.0 * u.amol / u.ms
    spike = lambda t: 0.0 if t < 1.0 * u.ms else 1.0
    t = transport.GeneralFlux(flux=base_flux, temporal=spike)
    influx_bnd.add_transport(species=ca, transport=t, source=None, target=cell)

    # Add elasticity and driving species
    cell.add_elasticity(youngs_modulus=1.0 * u.kPa)
    cell.add_driving_species(ca, coupling_strength=0.1 * u.kPa / mM)

    # Run simulation
    simulation.run(end_time=2 * u.ms, time_step=0.1 * u.ms, record_interval=1 * u.ms)

    # Load results and verify concentration change due to volume contraction
    result_loader = bmbcsim.ResultLoader(simulation.result_directory)
    points = [(0.5, 0.5, 0.5)]
    point_values = xr.concat(
        [result_loader.load_point_values(i, points=points) for i in range(len(result_loader))],
        dim="time",
    )

    ca_values = point_values.sel(species="ca").isel(point=0)

    # Initial concentration should be 0.0 mM
    assert ca_values.isel(time=0) == pytest.approx(0.0)

    # Final concentration should be higher due to volume contraction
    # (smaller as in the previous test since influx scales with boundary area)
    assert ca_values.isel(time=-1) == pytest.approx(1.040, rel=1e-2)


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


def test_tangential_spring_penalty_is_nonzero():
    """The tangential spring must actually resist in-plane sliding.

    ``specialcf.tangential(3)`` is an edge quantity and is identically zero on
    the facets of a 3D mesh, so using it silently deletes the shear resistance
    of the compliant embedding, leaving the boundary anchored only along its
    normal. Guard the projection form used instead.
    """
    mesh = create_box_mesh()
    n = ngs.specialcf.normal(3)
    displacement = ngs.CF((1, 0, 0))

    projected = (ngs.InnerProduct(displacement, displacement)
                 - ngs.InnerProduct(displacement, n) ** 2)
    surface_area = ngs.Integrate(ngs.CF(1) * ngs.ds, mesh)

    # A unit x-displacement is tangential on the four faces normal to y and z.
    assert ngs.Integrate(projected * ngs.ds, mesh) == pytest.approx(4 / 6 * surface_area)
    assert ngs.Integrate(ngs.InnerProduct(displacement, ngs.specialcf.tangential(3)) ** 2
                         * ngs.ds, mesh) == 0.0
