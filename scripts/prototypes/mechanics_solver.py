"""
Example script demonstrating mechanics-driven deformation using calcium in the ECS.

This script loads a tissue geometry from a VTK file, creates a mesh with cells
and extracellular space (ECS), initializes calcium in the ECS, and uses the
calcium concentration to drive mechanical deformation.
"""
import astropy.units as u

import bmbcsim
from bmbcsim.geometry.tissue_geometry import TissueGeometry
from bmbcsim.units import mM


# Load the tissue geometry from the VTK file
tissue = TissueGeometry.from_file("data/tissue_geometry.vtk")
print(f"Loaded tissue with {len(tissue.cells)} cells")

# Process the geometry: decimate for faster meshing and shrink cells to create ECS
ECS_FRACTION = 0.3  # Fraction of space to be ECS
tissue = tissue.shrink_cells((1.0 - ECS_FRACTION) ** (1/3))
tissue = tissue.scale(0.3)
print(tissue.bounding_box())

# Set the bounding box and create a mesh
min_coords, max_coords = (-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)
tissue = tissue.keep_cells_within(min_coords=min_coords, max_coords=max_coords)
print(f"Tissue has {len(tissue.cells)} cells after bounding box filtering")

# Convert to NGSolve mesh - cells are named "cell", ECS is named "ecs"
# Boundaries are: left, right, top, bottom, front, back (exterior)
# and "membrane" (interface between cells and ECS)
print("Generating mesh (this may take a moment)...")
mesh = tissue.to_ngs_mesh(
    mesh_size=5.0,  # Coarser mesh for faster generation
    min_coords=min_coords,
    max_coords=max_coords,
    cell_names="cell",
    cell_bnd_names="membrane",
)
print(f"Created mesh with {mesh.nv} vertices and {mesh.ne} elements")

# Create simulation with mechanics enabled
sim = bmbcsim.Simulation(
    "tissue_mechanics",
    mesh,
    result_root="results",
    mechanics=True,
)

# Get compartments from the simulation geometry
ecs = sim.simulation_geometry.compartments["ecs"]
cell = sim.simulation_geometry.compartments["cell"]

# Add calcium species
ca = sim.add_species("ca")

# Initialize calcium concentration in ECS to 0 mM
ecs.initialize_species(ca, 0.0 * mM)

# Add diffusion for calcium in both compartments
ecs.add_diffusion(ca, diffusivity=0.2 * u.um**2 / u.ms)
cell.add_diffusion(ca, diffusivity=0.1 * u.um**2 / u.ms)

# Add elasticity parameters to both compartments
# ECS is softer than cells
ecs.add_elasticity(youngs_modulus=0.1 * u.kPa, poisson_ratio=0.3)
cell.add_elasticity(youngs_modulus=1.0 * u.kPa, poisson_ratio=0.3)

# Set calcium as the driving species in ECS
# Positive coupling means higher [Ca] causes expansion
ecs.add_driving_species(ca, coupling_strength=0.2 * u.kPa / mM)

# Add a reaction that produces calcium in the ecs at a constant rate
ecs.add_reaction(reactants=[], products=[ca], k_f=1.0 / u.s, k_r=0.0 / u.s)

# Run the simulation
print("Running simulation...")
sim.run(
    end_time=1.0 * u.s,
    time_step=20 * u.ms,
    record_interval=1.0 * u.ms,
)

print(f"Simulation complete. Results saved to: {sim.result_directory}")

# Optional: Load and analyze results
result_loader = bmbcsim.ResultLoader(sim.result_directory)
print(f"Recorded {len(result_loader)} snapshots")
