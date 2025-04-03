"""This script recreates the geometry of [Tour, Tsien; 2007] using float-based microns."""
import astropy.units as u

import ecsim
from ecsim.geometry import create_ca_depletion_mesh


# Dimensions
side = 3.0 * u.um
ecs_height = 0.1 * u.um
cytosol_height = 3.0 * u.um
channel_radius = 50 * u.nm
M = u.mol / u.L
uM = u.umol / u.L

mesh = create_ca_depletion_mesh(
    side_length_x=side,
    side_length_y=side,
    cytosol_height=cytosol_height,
    ecs_height=ecs_height,
    channel_radius=channel_radius,
    mesh_size=100 * u.nm
)

simulation = ecsim.Simulation('tsien', result_root='results')
geometry = simulation.setup_geometry(mesh)
print(f"Compartments: {geometry.compartment_names}")
print(f"Membranes: {geometry.membrane_names}")
# geometry.visualize(resolve_regions=False)

# Get the compartments and membranes
ecs = geometry.compartments['ecs']
cytosol = geometry.compartments['cytosol']
channel = geometry.membranes['channel']
print(f"ecs volume: {ecs.volume}")
print(f"cytosol volume: {cytosol.volume}")
print(f"Channel area: {channel.area}")

# Add species to the simulation
# ...
# Copy the structure from sala_simulation
# but with the logic of ca_depletion_model_dynamic
