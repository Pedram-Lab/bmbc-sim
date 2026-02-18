"""
This script recreates the simulation from Rusakov "The Role of Perisynaptic
Glial Sheaths in Glutamate Spillover and Extracellular Ca2+ Depletion" (2001).
Specifically, we will recreate the simulation of Ca-depletion for presynaptic,
AP-driven calcium influx (Figure 4, top row).
"""
import math

import numpy as np
import astropy.units as u
import astropy.constants as const

import bmbcsim
from bmbcsim.simulation import transport

# Geometry parameters
TOTAL_SIZE = 2 * u.um        # Guessed
SYNAPSE_RADIUS = 0.1 * u.um  # Fig. 4
CLEFT_SIZE = 30 * u.nm       # Sec. "Ca2 diffusion in a calyx-type synapse"
GLIA_DISTANCE = 30 * u.nm    # Guessed
GLIA_WIDTH = 50 * u.nm       # Sec. "Glial sheath and glutamate transporter density"
GLIA_COVERAGE = 0.5          # Varied
TORTUOSITY = 1.4             # Sec. "Synaptic geometry"
POROSITY = 0.12              # Sec. "Synaptic geometry"

# Calcium parameters
CA_RESTING = 1.3 * u.mmol / u.L       # Sec. "Presynaptic calcium influx"
DIFFUSIVITY = 0.4 * u.um**2 / u.ms    # Fig. 4

# Simulation parameters
MESH_SIZE = 0.1 * u.um
PRE_OR_POST_SYNAPTIC = "pre"  # Varied
if PRE_OR_POST_SYNAPTIC == "pre":
    TIME_STEP = 0.2 * u.us
    END_TIME = 1.5 * u.ms
    RECORD_INTERVAL = 10 * u.us
elif PRE_OR_POST_SYNAPTIC == "post":
    TIME_STEP = 0.2 * u.us
    END_TIME = 80 * u.ms
    RECORD_INTERVAL = 0.5 * u.ms
else:
    raise ValueError(f"Invalid value for PRE_OR_POST_SYNAPTIC: {PRE_OR_POST_SYNAPTIC}")

# Create the geometry
angle = float(np.arccos(1 - 2 * GLIA_COVERAGE)) * u.rad
mesh = bmbcsim.create_rusakov_geometry(
    total_size=TOTAL_SIZE,
    synapse_radius=SYNAPSE_RADIUS,
    cleft_size=CLEFT_SIZE,
    glia_distance=GLIA_DISTANCE,
    glia_width=GLIA_WIDTH,
    glial_coverage_angle=angle,
    mesh_size=MESH_SIZE,
)

# Initialize the simulation and all geometry components
simulation = bmbcsim.Simulation(mesh=mesh, name="rusakov", result_root="results")
geo = simulation.simulation_geometry
synapse_ecs = geo.compartments["synapse_ecs"]
neuropil = geo.compartments["neuropil"]
synapse_boundary = geo.membranes["synapse_boundary"]

if PRE_OR_POST_SYNAPTIC == "pre":
    synapse = geo.compartments["presynapse"]
    synaptic_membrane = geo.membranes["presynaptic_membrane"]
elif PRE_OR_POST_SYNAPTIC == "post":
    synapse = geo.compartments["postsynapse"]
    synaptic_membrane = geo.membranes["postsynaptic_membrane"]

# Add calcium and diffusion
ca = simulation.add_species("Ca")
synapse_ecs.initialize_species(ca, CA_RESTING)
neuropil.initialize_species(ca, CA_RESTING)

# Add diffusion in different compartments
synapse_ecs.add_diffusion(ca, DIFFUSIVITY)
synapse.add_diffusion(ca, DIFFUSIVITY)
neuropil.add_diffusion(ca, DIFFUSIVITY / TORTUOSITY**2)
neuropil.add_porosity(POROSITY)

# Add transport across the neuropil boundary
t = transport.Transparent(
    source_diffusivity=DIFFUSIVITY / TORTUOSITY**2,
    target_diffusivity=DIFFUSIVITY,
)
synapse_boundary.add_transport(ca, t, neuropil, synapse_ecs)

# Add the channel flux (either pre- or post-synaptic)
const_F = const.e.si * const.N_A
if PRE_OR_POST_SYNAPTIC == "pre":
    # Compute the channel flux on the presynaptic membrane
    CHANNEL_CURRENT = 0.5 * u.pA     # Sec. "Presynaptic calcium influx"
    TIME_CONSTANT = 10 / u.ms        # Sec. "Presynaptic calcium influx"
    M50 = 39                         # Fig. 4 (number of channels required for 50% depletion)

    # flux(t) = Q * (t*τ) * exp(-t*τ)
    Q = M50 * CHANNEL_CURRENT / (2 * const_F)
    spike = lambda t: (t * TIME_CONSTANT) * math.exp(-t * TIME_CONSTANT)

elif PRE_OR_POST_SYNAPTIC == "post":
    # Compute the channel flux on the postsynaptic membrane
    TAU_1 = 80 * u.ms           # Sec. "Postsynaptic calcium influx"
    TAU_2 = 3 * u.ms            # Sec. "Postsynaptic calcium influx"
    J50 = 29 * u.pA / u.um**2   # Fig. 4 (current density required for 50% depletion)

    # flux(t) = J * (exp(-t/τ_1) - exp(-t/τ_2))
    Q = J50 * synaptic_membrane.area / (2 * const_F)
    spike = lambda t: math.exp(-t / TAU_1) - math.exp(-t / TAU_2)

flux = transport.GeneralFlux(flux=Q, temporal=spike)
synaptic_membrane.add_transport(ca, flux, synapse_ecs, synapse)

# Run the simulation
simulation.run(
    end_time=END_TIME,
    time_step=TIME_STEP,
    record_interval=RECORD_INTERVAL,
    n_threads=4
)
