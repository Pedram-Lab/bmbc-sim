"""This script recreates the geometry of [Tour, Tsien; 2007] using float-based microns."""
import astropy.units as u

import ecsim
from ecsim.geometry import create_ca_depletion_mesh
from ecsim.simulation import recorder, transport


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
ca = simulation.add_species('Ca', valence=2)

egta = simulation.add_species('EGTA', valence=-2)
ca_egta = simulation.add_species('Ca_EGTA', valence=0)

bapta = simulation.add_species('BAPTA', valence=-2)
ca_bapta = simulation.add_species('Ca_BAPTA', valence=0)

# Initial conditions
ca_ecs_0 = 15 * u.mmol / u.L
ca_cyt_0 = 0.1 * u.umol / u.L

bapta_0 = 1 * u.mmol / u.L
ca_bapta_0 = 0 * u.mmol / u.L

# egta_low_0 = 4.5 * u.mmol / u.L
# ca_egta_low_0 = 0 * u.mmol / u.L

# egta_high_0 = 40 * u.mmol / u.L
# ca_egta_high_0 = 0 * u.mmol / u.L

ecs.initialize_species(ca, value=ca_ecs_0)
cytosol.initialize_species(ca, value=ca_cyt_0)

cytosol.initialize_species(bapta, value=bapta_0)
cytosol.initialize_species(ca_bapta, value=ca_bapta_0)

# cytosol.initialize_species(egta, value=egta_low_0)
# cytosol.initialize_species(ca_egta, value=ca_egta_low_0)

# cytosol.initialize_species(egta, value=egta_high_0)
# cytosol.initialize_species(ca_egta, value=ca_egta_high_0)

# Add diffusion to the species 
ecs.add_diffusion(species=ca, diffusivity=600 * u.um**2 / u.s)
cytosol.add_diffusion(species=ca, diffusivity=220 * u.um**2 / u.s)

cytosol.add_diffusion(species=bapta, diffusivity=95 * u.um**2 / u.s)
cytosol.add_diffusion(species=ca_bapta, diffusivity=95 * u.um**2 / u.s)

# cytosol.add_diffusion(species=egta, diffusivity =113 * u.um**2 / u.s)
# cytosol.add_diffusion(species=ca_bapta, diffusivity =113 * u.um**2 / u.s)

#Add forward and reverse reaction rates for the buffers
kf_bapta = 450 / (uM * u.s)
kr_bapta = 80 / u.s

kf_egta = 2.7 / (uM * u.s)
kr_egta = 0.5 / u.s

# Add reaction
cytosol.add_reaction(reactants=[ca, bapta], products=[ca_bapta], k_f=kf_bapta, k_r=kr_bapta)
# cytosol.add_reaction(reactants=[ca_cyt, egta], products=[ca_egta], k_f=kf_egta, k_r=kr_egta)

rate_channel = 1 * u.mmol / (u.L * u.s) # Total flux 

print(f"rate_channel: {rate_channel}")

channel.add_transport(
    species=ca, 
    transport=transport.Channel(rate_channel),
    source=ecs,
    target=cytosol
)


# rate_channel = 1 * u.mmol / (u.L * u.s)

#rate_channel = (1 * u.mmol / (u.L * u.s)).to(u.umol / (u.um**3 * u.s))

# rate_channel = ((1 * u.mmol / (u.L * u.s)) * (1 * u.um)).to(u.umol / (u.um**2 * u.s))
# print(f"rate_channel = {rate_channel}")

# # Flujo original en mmol / (L·s)
# flux_concentration = 1 * u.mmol / (u.L * u.s)

# # Convierte a umol / (um³·s)
# flux_volumetric = flux_concentration.to(u.umol / (u.um**3 * u.s))

# # Supongamos que el canal tiene un "espesor" de 1 um, para pasarlo a flujo superficial
# flux_thickness = 1 * u.um  # equivalente a volumen/área

# # Ahora sí, flujo por unidad de área
# rate_channel = (flux_volumetric * flux_thickness).to(u.umol / (u.um**2 * u.s))

# print(f"rate_channel: {rate_channel}")

# channel.add_transport(
#     species=ca_ecs, 
#     transport=transport.Channel(rate_channel),
#     source=ecs,
#     target=cytosol
# )


# # Define el flujo total
# flux_total = (1 * u.mmol / (u.L * u.s)).to(u.umol / (u.s))  # umol/s

# # Área del canal
# channel_radius = 50 * u.nm
# channel_area = (u.pi * channel_radius**2).to(u.um**2)

# # Flujo por unidad de área (lo que espera Channel)
# rate_channel = (flux_total / channel_area).to(u.umol / (u.um**2 * u.s))
# print(f"rate_channel: {rate_channel}")

# channel.add_transport(
#     species=ca_ecs, 
#     transport=transport.
#     Channel(rate_channel),
#     source=ecs,
#     target=cytosol
# )

# rate=1 * u.mmol / (u.L * u.s)

# Add transpor coefficients

# ...
# Copy the structure from sala_simulation
# but with the logic of ca_depletion_model_dynamic
