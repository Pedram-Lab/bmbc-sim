"""
Example script demonstrating mechanics-driven deformation using calcium in the ECS.

This script loads a tissue geometry from a VTK file, creates a mesh with cells
and extracellular space (ECS), initializes calcium in the ECS, and uses the
calcium concentration to drive mechanical deformation.
"""
import numpy as np
import astropy.units as u

import bmbcsim
from bmbcsim.geometry.tissue_geometry import TissueGeometry
from bmbcsim.units import mM


# Load the tissue geometry from the VTK file
tissue = TissueGeometry.from_file("data/tissue_geometry.vtk")
print(f"Loaded tissue with {len(tissue.cells)} cells")

# Decimate to reduce triangle count (prevents bus errors from complex geometry)
tissue = tissue.decimate(0.9)

# Process the geometry: shrink cells to create ECS
ECS_FRACTION = 0.5  # Fraction of space to be ECS
tissue = tissue.shrink_cells((1.0 - ECS_FRACTION) ** (1/3))
tissue = tissue.scale(0.5)

# Set the bounding box and create a mesh
min_coords, max_coords = np.array((-10.0, -10.0, -10.0)), np.array((10.0, 10.0, 10.0))
tissue = tissue.keep_cells_within(min_coords=min_coords, max_coords=max_coords)
print(f"Tissue has {len(tissue.cells)} cells after bounding box filtering")

# Convert to NGSolve mesh - each cell gets a unique name "cell_0", "cell_1", etc.
# ECS is named "ecs"
# Boundaries are: left, right, top, bottom, front, back (exterior)
# and "membrane" (interface between cells and ECS)
print("Generating mesh (this may take a moment)...")
n_cells = len(tissue.cells)
cell_names = [f"cell_{i}" for i in range(n_cells)]
mesh = tissue.to_ngs_mesh(
    mesh_size=10.0,  # Coarser mesh for faster generation
    min_coords=min_coords,
    max_coords=max_coords,
    cell_names=cell_names,
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
cells = [sim.simulation_geometry.compartments[name] for name in cell_names]

# Add calcium species
ca = sim.add_species("ca")

# Initialize calcium concentration in ECS to 0 mM
ecs.initialize_species(ca, 0.0 * mM)

# Add diffusion for calcium in ECS
ecs.add_diffusion(ca, diffusivity=0.2 * u.um**2 / u.ms)

# Add elasticity to ECS (softer than cells)
ecs.add_elasticity(youngs_modulus=0.1 * u.kPa, poisson_ratio=0.3)

# Set calcium as the driving species in ECS
# Positive coupling means higher [Ca] causes expansion
ecs.add_driving_species(ca, coupling_strength=0.2 * u.kPa / mM)

# Add a reaction that produces calcium in the ECS at a constant rate
ecs.add_reaction(reactants=[], products=[ca], k_f=1.0 / u.s, k_r=0.0 / u.s)

# Configure each cell compartment
for cell in cells:
    cell.initialize_species(ca, 0.0 * mM)
    cell.add_diffusion(ca, diffusivity=0.1 * u.um**2 / u.ms)
    cell.add_elasticity(youngs_modulus=1.0 * u.kPa, poisson_ratio=0.3)

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
