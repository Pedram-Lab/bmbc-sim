"""This script recreates the geometry of [Tour, Tsien; 2007]."""
from collections import namedtuple

import astropy.units as u

import ecsim
from ecsim.geometry import create_ca_depletion_mesh
from ecsim.simulation import transport
from ecsim.units import mM, uM


# Dimensions
side = 3.0 * u.um
ecs_height = 0.1 * u.um
cytosol_height = 3.0 * u.um
channel_radius = 50 * u.nm

mesh = create_ca_depletion_mesh(
    side_length_x=side,
    side_length_y=side,
    cytosol_height=cytosol_height,
    ecs_height=ecs_height,
    channel_radius=channel_radius,
    mesh_size=100 * u.nm,
    channel_mesh_size=50 * u.nm
)

simulation = ecsim.Simulation('tsien', mesh, result_root='results')
geometry = simulation.simulation_geometry
print(f"Compartments: {geometry.compartment_names}")
print(f"Membranes: {geometry.membrane_names}")
geometry.visualize(resolve_regions=False)

# Get the compartments and membranes
ecs = geometry.compartments['ecs']
cytosol = geometry.compartments['cytosol']
channel = geometry.membranes['channel']
ecs_top = geometry.membranes['ecs_top']
print(f"ecs volume: {ecs.volume}")
print(f"cytosol volume: {cytosol.volume}")
print(f"Channel area: {channel.area}")

# Define 3 buffering scenarios: BAPTA and EGTA with both high and low initial concentrations
BufferSpec = namedtuple('BufferSpec',
                        ['name', 'initial_concentration', 'diffusivity', 'kf', 'kr'])
buffer_spec = BufferSpec('BAPTA', 1 * mM, 95 * u.um**2 / u.s, 450 / (uM * u.s), 80 / u.s)
# buffer_spec = BufferSpec('EGTA', 40 * mM, 113 * u.um**2 / u.s, 2.7 / (uM * u.s), 0.5 / u.s)
# buffer_spec = BufferSpec('EGTA, 4.5 * mM, 113 * u.um**2 / u.s, 2.7 / (uM * u.s), 0.5 / u.s)

# Add species to the simulation
ca = simulation.add_species('Ca', valence=2)
buffer = simulation.add_species(buffer_spec.name, valence=-2)
ca_buffer = simulation.add_species('Ca_' + buffer_spec.name, valence=0)

# Initial conditions
ecs.initialize_species(ca, value=15 * mM)
cytosol.initialize_species(ca, value=0.1 * uM)

cytosol.initialize_species(buffer, value=buffer_spec.initial_concentration)
cytosol.initialize_species(ca_buffer, value=0 * uM)

# Add diffusion to the species
ecs.add_diffusion(species=ca, diffusivity=600 * u.um**2 / u.s)
cytosol.add_diffusion(species=ca, diffusivity=220 * u.um**2 / u.s)

cytosol.add_diffusion(species=buffer, diffusivity=buffer_spec.diffusivity)
cytosol.add_diffusion(species=ca_buffer, diffusivity=buffer_spec.diffusivity)

# Add reaction
cytosol.add_reaction(
    reactants=[ca, buffer], products=[ca_buffer],
    k_f=buffer_spec.kf, k_r=buffer_spec.kr
)

# Add transport
rate_channel = 10 * u.um / u.ms
permeability = rate_channel * channel.area
channel.add_transport(
    species=ca,
    transport=transport.Passive(permeability=permeability),
    source=ecs,
    target=cytosol
)
ecs_top.add_transport(
    species=ca,
    transport=transport.Passive(permeability=permeability, outside_concentration=15 * mM),
    source=ecs,
    target=None
)

# Run the simulation
simulation.run(end_time=20 * u.ms, time_step=1 * u.us, n_threads=8, record_interval=1 * u.ms)
