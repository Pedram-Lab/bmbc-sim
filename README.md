# Biomechanical and Biochemical Simulator (BMBC-Sim)
:construction: **Note:** This project is still in active development. Expect incomplete features and breaking changes. :construction:

The Biomechanical and Biochemical Simulator (BMBC-Sim) is a multi-physics simulation framework for simulating coupled mechanical and chemical processes in biological models using [NGSolve](https://ngsolve.org/).
It is designed to easily prototype hypotheses, reproduce literature scenarios, or explore custom geometries with tightly coupled mechanical and chemical dynamics.

## Quick Start
```bash
uv sync            # create the development environment
uv run pytest      # optional: verify the install
uv run python scripts/demo.py  # run a demo simulation
```

BMBC-Sim relies on [uv](https://docs.astral.sh/uv/) for reproducible environments. Once `uv sync` completes, the virtual environment is available in your IDE.

## Working With Simulations
Setting up a simulation is straightforward as BMBC-Sim provides an easy-to-use API for defining geometries, species, diffusion, reactions, and more.
```python
import astropy.units as u
import bmbcsim

mesh = bmbcsim.geometry.create_sphere_geometry(radius=10 * u.um, mesh_size=1 * u.um)
sim = bmbcsim.Simulation("demo", mesh, result_root="results")
cell = sim.simulation_geometry.compartments["sphere"]


ca = sim.add_species("ca")
cell.add_diffusion(ca, diffusivity=0.2 * u.um**2 / u.ms)
buffer = sim.add_species("buffer")
cell.add_diffusion(buffer, diffusivity=0.1 * u.um**2 / u.ms)
ca_buffer = sim.add_species("ca_buffer")
cell.add_diffusion(ca_buffer, diffusivity=0.05 * u.um**2 / u.ms)
cell.add_reaction(
    reactants=[ca, buffer], products=[ca_buffer],
    k_f=0.1 / (u.ms * u.um**3), k_r=0.05 / u.ms,
)

sim.run(end_time=10.0 * u.ms, time_step=0.1 * u.ms, record_interval=1.0 * u.ms)
```
The `ResultLoader` API then lets you post-process outputs into pandas/xarray structures.
All results are saved as VTK files for easy visualization in [Paraview](https://www.paraview.org/) or similar tools.

## Repository Structure
- `src/bmbcsim/` – core simulation engine, geometry builders, and utilities.
- `scripts/` – curated scenarios (Paszek, Sala, Tour, and more) ready to run.
- `scripts/tutorials/` – bite-sized notebooks and scripts that demonstrate specific techniques (point sampling, tortuosity, deformation, etc.).
- `test/` – pytest suite covering units, multi-compartment setups, electrostatics, and geometry validation.
