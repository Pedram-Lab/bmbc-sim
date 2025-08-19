"""
This script recreates the simulation from [Rusakov 2001]. Specifically, we
will recreate the simulation of Ca-depletion for presynaptic, AP-driven
calcium influx (Figure 4, top row).
"""
import math

import numpy as np
import astropy.units as u
import astropy.constants as const

import ecsim
from ecsim.simulation import transport

# Geometry parameters
TOTAL_SIZE = 2 * u.um        # Guessed
SYNAPSE_RADIUS = 0.1 * u.um  # Fig. 4
CLEFT_SIZE = 30 * u.nm       # Sec. "Ca2 diffusion in a calyx-type synapse"
GLIA_DISTANCE = 30 * u.nm    # Guessed
GLIA_WIDTH = 50 * u.nm       # Sec. "Glial sheath and glutamate transporter density"
GLIA_COVERAGE = 0.5          # Varied
TORTUOSITY = 1.4             # Sec. "Synaptic geometry"
POROSITY = 0.12              # Sec. "Synaptic geometry"

# Ca parameters
CA_RESTING = 1.3 * u.mmol / u.L       # Sec. "Presynaptic calcium influx"
CHANNEL_CURRENT = 0.5 * u.pA          # Sec. "Presynaptic calcium influx"
DIFFUSIVITY = 0.4 * u.um**2 / u.ms    # Fig. 4
TIME_CONSTANT = 10 / u.ms             # Sec. "Presynaptic calcium influx"
N_CHANNELS = 39                       # Fig. 4

# Simulation parameters
MESH_SIZE = 0.1 * u.um
TIME_STEP = 1.0 * u.us
END_TIME = 1.5 * u.ms

# Create the geometry
angle = float(np.arccos(1 - 2 * GLIA_COVERAGE)) * u.rad
mesh = ecsim.create_rusakov_geometry(
    total_size=TOTAL_SIZE,
    synapse_radius=SYNAPSE_RADIUS,
    cleft_size=CLEFT_SIZE,
    glia_distance=GLIA_DISTANCE,
    glia_width=GLIA_WIDTH,
    glial_coverage_angle=angle,
    mesh_size=MESH_SIZE,
)

# Initialize the simulation and all geometry components
simulation = ecsim.Simulation(mesh=mesh, name="rusakov", result_root="results")
geo = simulation.simulation_geometry
synapse_ecs = geo.compartments["synapse_ecs"]
neuropil = geo.compartments["neuropil"]
presynaptic_membrane = geo.membranes["presynaptic_membrane"]

# Add calcium and diffusion
ca = simulation.add_species("Ca")
synapse_ecs.add_diffusion(ca, DIFFUSIVITY)
neuropil.add_diffusion(ca, DIFFUSIVITY / TORTUOSITY**2)

# Compute the channel flux on the presynaptic membrane
Q = N_CHANNELS * CHANNEL_CURRENT / (2 * presynaptic_membrane.area)
t = transport.GeneralFlux(lambda t: Q * math.exp(-t * TIME_CONSTANT))
presynaptic_membrane.add_transport(ca, t, neuropil, synapse_ecs)

# Run the simulation
simulation.run(end_time=END_TIME, time_step=TIME_STEP, n_threads=4)


# dist = to_simulation_units(SYNAPSE_RADIUS + GLIA_DISTANCE / 2, 'length')
# eval_points = np.array([
#     [0, 0, 0],          # 1: center
#     [dist, 0, 0],       # 2: inside glia, near cleft
#     [0, 0, dist],       # 3: inside glia, far from cleft
# ])
# eval_synapse = PointEvaluator(mesh, eval_points)
# eval_points = np.array([
#     [0, 0, -2 * dist],  # 4: outside glia (below)
#     [0, 0, 2 * dist],   # 5: outside glia (above)
# ])
# eval_neuropil = PointEvaluator(mesh, eval_points)
