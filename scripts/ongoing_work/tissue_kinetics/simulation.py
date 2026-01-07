import math
import numpy as np

from astropy import units as u
from astropy import constants as const

# Optional visualization (you can comment out Draw if not using the viewer)
from ngsolve.webgui import Draw
import ngsolve as ngs

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.geometry import TissueGeometry

# =========== 0) Simulation parameters ===========
CA_ECS = 1.3 * u.mmol / u.L               # [Ca2+] ECS at rest (1.3 mM)
CA_CELL0 = 0.0 * u.mmol / u.L             # Initial [Ca2+] in cell (0 mM)
DIFFUSIVITY_ECS = 0.6 * u.um**2 / u.ms    # Diffusivity in ECS
DIFFUSIVITY_CYTO = 0.22 * u.um**2 / u.ms  # Diffusivity in cytosol (adjust if needed)

CHANNEL_CURRENT = 0.5 * u.pA
N_CHANNELS = 88820
TIME_CONSTANT = 1.0 / 10.0 / u.ms         # τ = 1 ms^{-1}
END_TIME = 0.5 * u.s
TIME_STEP = 1 * u.ms

# Index of the permeable cell(s)
PERMEABLE_CELLS = [29]

# ================================================================
# 1) Load and post-process geometry from VTK
# ================================================================
geometry = TissueGeometry.from_file("data/tissue_geometry.vtk")
print("Cells after from_file:", len(geometry.cells))

# --- 1a) compute the typical "diameter" of each cell ---
cell_diameters = []
for i, cell in enumerate(geometry.cells):
    # bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    bmin = np.array(cell.bounds[::2])   # [xmin, ymin, zmin]
    bmax = np.array(cell.bounds[1::2])  # [xmax, ymax, zmax]
    size = bmax - bmin                  # lengths along each axis
    diam = float(size.max())            # diameter ~ largest extent
    cell_diameters.append(diam)

cell_diameters = np.array(cell_diameters)
median_diam = float(np.median(cell_diameters))
print(f"Median cell diameter (original units): {median_diam}")

# --- 1b) scale so that the median is ~4 µm ---
TARGET_CELL_DIAM = 4  # µm
scale_factor = TARGET_CELL_DIAM / median_diam
print("Scale factor to get ~4 µm cells:", scale_factor)

geometry = geometry.scale(scale_factor)
geometry = geometry.decimate(factor=0.5)
geometry = geometry.smooth(n_iter=10)
geometry = geometry.decimate(factor=0.5)

# --- 1c) bounding box after scaling ---
minc, maxc = geometry.bounding_box()
size = maxc - minc
print("After cell-size-based scale, bbox:")
print("  min:", minc)
print("  max:", maxc)
print("  size:", size)

# --- 1d) translate so that the domain starts at (0,0,0) ---
for cell in geometry.cells:
    cell.points -= minc  # minc is [xmin, ymin, zmin]

minc2, maxc2 = geometry.bounding_box()
size2 = maxc2 - minc2
print("After translate, bbox:")
print("  min:", minc2)
print("  max:", maxc2)
print("  size:", size2)
print("Max domain size after scaling (in µm):", float(size2.max()))
print("Target median cell diameter (in µm):", TARGET_CELL_DIAM)

# --- 1e) open ECS by shrinking the cells ---
ECS_RATIO = 0.2  # volumetric fraction to open ECS gaps
geometry = geometry.shrink_cells(1 - ECS_RATIO, jitter=0.0)
print("Cells after shrink:", len(geometry.cells))

# --- 1f) crop a block of ~20 x 20 x 1 µm around the center ---
minc3, maxc3 = geometry.bounding_box()
size3 = maxc3 - minc3
center = 0.5 * (minc3 + maxc3)
print("BBox before clipping:")
print("  min:", minc3)
print("  max:", maxc3)
print("  size:", size3)
print("  center:", center)

# We want a box of ~20 x 20 x 1 µm (x, y, z)
BOX_SIZE_X = 20.0  # µm
BOX_SIZE_Y = 20.0  # µm
BOX_SIZE_Z = 1.0   # µm

box_size = np.array([BOX_SIZE_X, BOX_SIZE_Y, BOX_SIZE_Z])
half_box = box_size / 2.0

# Define limits of the block centered at 'center', clipped to global bounds
min_box = np.maximum(minc3, center - half_box)
max_box = np.minimum(maxc3, center + half_box)

print("Desired clipping box:")
print("  min_box:", min_box)
print("  max_box:", max_box)
print("  box_size (approx):", max_box - min_box)

geometry = geometry.keep_cells_within(
    min_coords=min_box,
    max_coords=max_box,
    inside_threshold=0.1
)

n_cells = len(geometry.cells)
print("Cells after keep_cells_within (~20 x 20 x 1 µm box):", n_cells)

if n_cells == 0:
    raise RuntimeError(
        "No cells remain after keep_cells_within. "
        "Increase one of BOX_SIZE_X/Y/Z or relax inside_threshold."
    )

# ================================================================
# 1g) Cell and membrane names: all impermeable except specified ones
# ================================================================
cell_names = [f"cell_{i}" for i in range(n_cells)]
bnd_names = ["impermeable"] * n_cells
for idx in PERMEABLE_CELLS:
    bnd_names[idx] = "permeable"

# ================================================================
# 1h) Generate NGSolve mesh with unique material and BC names
# ================================================================
tissue_mesh: ngs.Mesh = geometry.to_ngs_mesh(
    mesh_size=5.0,
    min_coords=min_box,
    max_coords=max_box,
    projection_tol=0.02,
    cell_names=cell_names,
    cell_bnd_names=bnd_names,
)

# Visualization (optional)
Draw(tissue_mesh)
print(f"Create mesh with {tissue_mesh.ne} elements and {tissue_mesh.nv} vertices.")

# ================================================================
# 2) Set up simulation
# ================================================================
sim = bmbcsim.Simulation(
    mesh=tissue_mesh,
    name="tissue_kinetics",
    result_root="results"
)
geo = sim.simulation_geometry

# Compartments and membranes created by to_ngs_mesh:
# - 'ecs' (the volume of the box minus the cells)
# - 'cell_i' for each cell
# - membranes with BC name: 'impermeable', 'permeable'
ecs = geo.compartments["ecs"]
target_cell = geo.compartments[cell_names[PERMEABLE_CELLS[0]]]
mem_perm = geo.membranes["permeable"]

print(f"Target cell index: {PERMEABLE_CELLS[0]}")
print(f"Target cell volume: {target_cell.volume:.2f} µm^3")
print(f"Target cell surface area: {mem_perm.area:.2f} µm^2")

total_cell_volume = sum(geo.compartments[f"cell_{i}"].volume for i in range(n_cells))
ecs_volume = ecs.volume
print(f"Total volume: {total_cell_volume + ecs_volume:.2f} µm^3")

# ================================================================
# 3) Species and initialization
# ================================================================
ca = sim.add_species("Ca")

# Permeable cell at 0 mM (if already initialized, ignore error)
target_cell.initialize_species(ca, CA_CELL0)
# Before adding diffusion
ecs.initialize_species(ca, CA_ECS)

# ================================================================
# 4) Diffusion
# ================================================================
ecs.add_diffusion(ca, DIFFUSIVITY_ECS)
target_cell.add_diffusion(ca, DIFFUSIVITY_CYTO)

# ================================================================
# 5) Transport across 'permeable' membrane
# ================================================================
# Q = N * I / (2 F)  (factor 2 for Ca2+)
const_F = const.e.si * const.N_A  # Faraday constant [C/mol]
Q = N_CHANNELS * CHANNEL_CURRENT / (2 * const_F)  # [mol/s] across the "permeable" membrane

# Temporal pulse: flux(t) = Q * (t*τ) * exp(-t*τ)
pulse = lambda t: (t * TIME_CONSTANT) * math.exp(-t * TIME_CONSTANT)
flux = transport.GeneralFlux(flux=Q, temporal=pulse)

# Direction: ECS -> permeable cell
mem_perm.add_transport(ca, flux, ecs, target_cell)

# ================================================================
# 6) Transport from external reservoir into ECS
# ================================================================
# Simple passive flux to maintain ECS concentration
flux = transport.Passive(1 * u.um ** 3 / u.ms, CA_ECS)

# Direction: ECS -> permeable cell
for bnd in ["top", "bottom", "left", "right", "front", "back"]:
    boundary = geo.membranes[bnd]
    boundary.add_transport(ca, flux, None, ecs)

# ================================================================
# 7) Run simulation
# ================================================================
sim.run(end_time=END_TIME, time_step=TIME_STEP, n_threads=4)
print("Simulation completed.")
