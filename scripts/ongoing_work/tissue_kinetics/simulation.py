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
from bmbcsim.simulation import coefficient_fields as cf

# =========== 0) Simulation parameters ===========
CA_ECS = 1.3 * u.mmol / u.L               # [Ca2+] ECS at rest (1.3 mM)
DIFFUSIVITY_ECS = 0.7 * u.um**2 / u.ms    # Free-solution diffusivity for Ca2+

# Boundary condition parameters (Robin BC with tortuosity)
TORTUOSITY = 1.6
# Permeability reduced by tortuosity factor λ²
BOUNDARY_PERMEABILITY = (1 / TORTUOSITY**2) * u.um**3 / u.ms  # ≈0.39 µm³/ms

# Synapse parameters - scaled for 20×20×1 µm = 400 µm³
# 1 synapse/µm³ → 400 synapses
N_SYNAPSES = 400                          # Total synaptic contacts
N_CHANNELS_PER_SYNAPSE = 35               # NMDARs per synapse
SYNAPSE_DIAMETER = 0.25 * u.um            # Patch diameter
F_ACTIVE = 0.15                           # Fraction of active synapses
I_CHANNEL = 0.5 * u.pA                    # Single channel current (tunable J₀)

# NMDAR kinetics (biexponential - parameters chosen to match pulse from [Rusakov & Fine, 2003])
TAU1 = 10 * u.ms                          # Slow decay time constant
TAU2 = 3 * u.ms                           # Fast rise time constant

# Stimulation protocol (5 pulses at 100 Hz)
PULSE_TIMES = [300, 310, 320, 330, 340] * u.ms  # Pulse times in ms

# Simulation timing
END_TIME = 1.0 * u.s
TIME_STEP = 1.0 * u.ms

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
ECS_RATIO = 0.1  # volumetric fraction to open ECS gaps, yields ~16% ECS
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
# 1g) Cell and membrane names: all cells share "membrane" boundary
# ================================================================
cell_names = [f"cell_{i}" for i in range(n_cells)]
bnd_names = [f"membrane_{i}" for i in range(n_cells)]

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
# - 'membrane' (shared boundary of all cells with ECS)
ecs = geo.compartments["ecs"]
cells = [geo.compartments[f"cell_{i}"] for i in range(n_cells)]
membranes = [geo.membranes[f"membrane_{i}"] for i in range(n_cells)]

total_cell_volume = sum(cell.volume for cell in cells)
total_volume = ecs.volume + total_cell_volume
total_membrane_area = sum(membrane.area for membrane in membranes)
print(f"Total volume: {total_volume:.2f} µm^3")
print(f"ECS volume: {ecs.volume:.2f} µm^3")
print(f"ECS volume fraction: {ecs.volume / total_volume * 100:.2f}%")
print(f"Total membrane area: {total_membrane_area:.2f} µm^2")

# ================================================================
# 3) Species and initialization
# ================================================================
ca = sim.add_species("Ca")
ecs.initialize_species(ca, CA_ECS)

# Initialize cells at 0 to make system well-defined
for cell in cells:
    cell.initialize_species(ca, 0.0 * u.mmol / u.L)

# ================================================================
# 4) Diffusion
# ================================================================
ecs.add_diffusion(ca, DIFFUSIVITY_ECS)

# Add diffusion to cells (required for well-posed FEM system)
DIFFUSIVITY_CYTO = 0.22 * u.um**2 / u.ms
for cell in cells:
    cell.add_diffusion(ca, DIFFUSIVITY_CYTO)

# ================================================================
# 5) Ca2+ sink at distributed synapse patches
# ================================================================
# Q = N * I / (2 F)  (factor 2 for Ca2+)
const_F = const.e.si * const.N_A  # Faraday constant [C/mol]
Q_per_synapse = N_CHANNELS_PER_SYNAPSE * I_CHANNEL / (2 * const_F)

# Number of active synapses (15% of total), distributed randomly across cells
total_active_synapses = int(N_SYNAPSES * F_ACTIVE)
base_synapses_per_cell = total_active_synapses // n_cells
remainder = total_active_synapses % n_cells

# Randomly select which cells get an extra synapse
rng = np.random.default_rng(seed=42)
cells_with_extra = rng.choice(n_cells, size=remainder, replace=False)
synapses_per_cell = np.full(n_cells, base_synapses_per_cell)
synapses_per_cell[cells_with_extra] += 1

print(f"Active synapses: {synapses_per_cell.sum()} (of {N_SYNAPSES} total)")

# Biexponential NMDAR waveform with multi-pulse stimulation
def nmdar_waveform(t):
    """
    Biexponential NMDAR current: J(t) = e^(-t/τ₁) - e^(-t/τ₂)
    Superposition of 5 pulses at 100 Hz (0, 10, 20, 30, 40 ms)
    """
    total = 0.0

    for t_pulse in PULSE_TIMES:
        dt = t - t_pulse
        if dt >= 0 * u.ms:
            total += math.exp(-dt / TAU1) - math.exp(-dt / TAU2)

    return total

# Ca2+ flux from ECS through membrane (sink - no target compartment)
# Distributed synapse patches using LocalizedPeaks
# peak_width ≈ diameter/6 for effective 3σ coverage
for i, (membrane, cell) in enumerate(zip(membranes, cells)):
    n_syn = synapses_per_cell[i]
    if n_syn == 0:
        continue

    synapse_distribution = cf.LocalizedPeaks(
        seed=0,
        num_peaks=n_syn,
        peak_value=Q_per_synapse,
        background_value=0.0 * u.mol / u.s,
        peak_width=SYNAPSE_DIAMETER / 6.0,
        total=n_syn * Q_per_synapse
    )
    synapse_flux = transport.GeneralFlux(flux=synapse_distribution, temporal=nmdar_waveform)
    membrane.add_transport(ca, synapse_flux, ecs, cell)

# ================================================================
# 6) Robin BC: transport from external reservoir into ECS
# ================================================================
# Permeability accounts for tortuosity: P_eff = D_eff / L_char
# where D_eff = D / λ² and λ = 1.6
boundary_flux = transport.Passive(BOUNDARY_PERMEABILITY, CA_ECS)

for bnd in ["top", "bottom", "left", "right", "front", "back"]:
    boundary = geo.membranes[bnd]
    boundary.add_transport(ca, boundary_flux, None, ecs)

# ================================================================
# 7) Run simulation
# ================================================================
sim.run(end_time=END_TIME, time_step=TIME_STEP, record_interval=10 * TIME_STEP, n_threads=4)
print("Simulation completed.")
