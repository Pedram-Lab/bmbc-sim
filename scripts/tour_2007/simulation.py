"""
This script simulates the mathematical model described in the paper:
[Tour, O., … Tsien, R.Y. (2007). Calcium Green FlAsH as a genetically targeted
small-molecule calcium indicator]. The model describes calcium transport from the
extracellular space into the cytosol through a cluster of calcium channels. In
the cytosol, calcium binds to one of the following buffers: BAPTA (1 mM), or
EGTA (4.5 mM or 40 mM). To simulate each buffer, make sure to use the
corresponding diffusion constants and reaction rate parameters.
"""
from collections import namedtuple

import astropy.units as u
from ngsolve.webgui import Draw

import ecsim
from ecsim.simulation import transport
from ecsim.units import mM, uM
from ecsim.geometry import create_ca_depletion_mesh

# Buffer configuration
BufferSpec = namedtuple('BufferSpec', ['name', 'initial_concentration', 'diffusivity', 'kf', 'kr'])

# Define the buffer you want to simulate here:
buffer_spec = BufferSpec(
    name='EGTA_low',
    initial_concentration=4.5 * mM,
    diffusivity=113 * u.um**2 / u.s,
    kf=2.7 / (uM * u.s),
    kr=0.5 / u.s
)
# buffer_spec = BufferSpec(
#     name='EGTA_high',
#     initial_concentration=40 * mM,
#     diffusivity=113 * u.um**2 / u.s,
#     kf=2.7 / (uM * u.s),
#     kr=0.5 / u.s
# )
# buffer_spec = BufferSpec(
#     name='BAPTA',
#     initial_concentration=1 * mM,
#     diffusivity=95 * u.um**2 / u.s,
#     kf=450 / (uM * u.s),
#     kr=80 / u.s
# )

# Geometry
side = 3.0 * u.um
cytosol_height = 3.0 * u.um
ecs_height = 0.1 * u.um
channel_radius = 50 * u.nm
mesh_size = 100 * u.nm

mesh = create_ca_depletion_mesh(
    side_length_x=side,
    side_length_y=side,
    cytosol_height=cytosol_height,
    ecs_height=ecs_height,
    channel_radius=channel_radius,
    mesh_size=mesh_size,
    channel_mesh_size=channel_radius
)
Draw(mesh)


# Initialize simulation
simulation = ecsim.Simulation(f"tour_{buffer_spec.name.lower()}", mesh, result_root='results')
geometry = simulation.simulation_geometry

ecs = geometry.compartments['ecs']
cytosol = geometry.compartments['cytosol']
channel = geometry.membranes['channel']
ecs_top = geometry.membranes['ecs_top']


# Species
ca = simulation.add_species('Ca', valence=2)
buffer = simulation.add_species(buffer_spec.name, valence=-2)
ca_buffer = simulation.add_species(f"Ca_{buffer_spec.name}", valence=0)


# Initial conditions
ecs.initialize_species(ca, value=15 * mM)
cytosol.initialize_species(ca, value=0.1 * uM)
cytosol.initialize_species(buffer, value=buffer_spec.initial_concentration)
cytosol.initialize_species(ca_buffer, value=0 * uM)


# Diffusion
ecs.add_diffusion(ca, diffusivity=600 * u.um**2 / u.s)
cytosol.add_diffusion(ca, diffusivity=220 * u.um**2 / u.s)
cytosol.add_diffusion(buffer, diffusivity=buffer_spec.diffusivity)
cytosol.add_diffusion(ca_buffer, diffusivity=buffer_spec.diffusivity)


# Reaction Ca + Buffer <=> CaBuffer
cytosol.add_reaction(
    reactants=[ca, buffer],
    products=[ca_buffer],
    k_f=buffer_spec.kf,
    k_r=buffer_spec.kr
)


# Transport
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


# Run simulation
simulation.run(
    end_time=20 * u.ms,
    time_step=1 * u.us,
    record_interval=1 * u.ms,
    n_threads=4
)
