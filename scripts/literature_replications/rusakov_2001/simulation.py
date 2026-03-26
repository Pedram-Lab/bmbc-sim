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
TOTAL_SIZE = 5 * u.um        # Guessed
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

# Presynaptic channel flux parameters
CHANNEL_CURRENT = 0.5 * u.pA     # Sec. "Presynaptic calcium influx"
TIME_CONSTANT = 10 / u.ms        # Sec. "Presynaptic calcium influx"
M50 = 36                         # Fig. 4 (number of channels required for 50% depletion)

# Postsynaptic channel flux parameters
TAU_1 = 80 * u.ms           # Sec. "Postsynaptic calcium influx"
TAU_2 = 3 * u.ms            # Sec. "Postsynaptic calcium influx"
J50 = 29 * u.pA / u.um**2   # Fig. 4 (current density required for 50% depletion)


def run_simulation(
    # Output
    simulation_name="rusakov",
    result_root="results",
    # Mode
    pre_or_post_synaptic="pre",
    # Geometry
    total_size=TOTAL_SIZE,
    synapse_radius=SYNAPSE_RADIUS,
    cleft_size=CLEFT_SIZE,
    glia_distance=GLIA_DISTANCE,
    glia_width=GLIA_WIDTH,
    glia_coverage=GLIA_COVERAGE,
    # Physics
    ca_resting=CA_RESTING,
    diffusivity=DIFFUSIVITY,
    tortuosity=TORTUOSITY,
    porosity=POROSITY,
    # Presynaptic channel flux
    channel_current=CHANNEL_CURRENT,
    time_constant=TIME_CONSTANT,
    m50=M50,
    # Postsynaptic channel flux
    tau_1=TAU_1,
    tau_2=TAU_2,
    j50=J50,
    # Simulation control
    mesh_size=MESH_SIZE,
    time_step=None,
    end_time=None,
    record_interval=None,
    # Performance
    n_threads=4,
):
    # Default timing depends on mode
    if pre_or_post_synaptic == "pre":
        time_step = time_step or 0.2 * u.us
        end_time = end_time or 1.5 * u.ms
        record_interval = record_interval or 10 * u.us
    elif pre_or_post_synaptic == "post":
        time_step = time_step or 0.2 * u.us
        end_time = end_time or 80 * u.ms
        record_interval = record_interval or 0.5 * u.ms
    else:
        raise ValueError(f"Invalid value for pre_or_post_synaptic: {pre_or_post_synaptic}")

    # Create the geometry
    angle = float(np.arccos(1 - 2 * glia_coverage)) * u.rad
    mesh = bmbcsim.create_rusakov_geometry(
        total_size=total_size,
        synapse_radius=synapse_radius,
        cleft_size=cleft_size,
        glia_distance=glia_distance,
        glia_width=glia_width,
        glial_coverage_angle=angle,
        mesh_size=mesh_size,
    )

    # Initialize the simulation and all geometry components
    simulation = bmbcsim.Simulation(mesh=mesh, name=simulation_name, result_root=result_root)
    geo = simulation.simulation_geometry
    synapse_ecs = geo.compartments["synapse_ecs"]
    neuropil = geo.compartments["neuropil"]
    synapse_boundary = geo.membranes["synapse_boundary"]

    if pre_or_post_synaptic == "pre":
        synapse = geo.compartments["presynapse"]
        synaptic_membrane = geo.membranes["presynaptic_membrane"]
    elif pre_or_post_synaptic == "post":
        synapse = geo.compartments["postsynapse"]
        synaptic_membrane = geo.membranes["postsynaptic_membrane"]

    # Add calcium and diffusion
    ca = simulation.add_species("Ca")
    synapse_ecs.initialize_species(ca, ca_resting)
    neuropil.initialize_species(ca, ca_resting)

    # Add diffusion in different compartments
    synapse_ecs.add_diffusion(ca, diffusivity)
    synapse.add_diffusion(ca, diffusivity)
    neuropil.add_diffusion(ca, diffusivity / tortuosity**2)
    neuropil.add_porosity(porosity)

    # Add transport across the neuropil boundary
    t = transport.Transparent(
        source_diffusivity=diffusivity / tortuosity**2,
        target_diffusivity=diffusivity,
    )
    synapse_boundary.add_transport(ca, t, neuropil, synapse_ecs)

    # Add the channel flux (either pre- or post-synaptic)
    const_F = const.e.si * const.N_A
    if pre_or_post_synaptic == "pre":
        Q = m50 * channel_current / (2 * const_F)
        spike = lambda t: (t * time_constant) * math.exp(-t * time_constant)
    elif pre_or_post_synaptic == "post":
        Q = j50 * synaptic_membrane.area / (2 * const_F)
        spike = lambda t: math.exp(-t / tau_1) - math.exp(-t / tau_2)

    flux = transport.GeneralFlux(flux=Q, temporal=spike)
    synaptic_membrane.add_transport(ca, flux, synapse_ecs, synapse)

    # Run the simulation
    simulation.run(
        end_time=end_time,
        time_step=time_step,
        record_interval=record_interval,
        n_threads=n_threads,
    )


if __name__ == "__main__":
    run_simulation()
