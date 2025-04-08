"""This script recreates the geometry of [Tour, Tsien; 2007] using float-based microns."""
import astropy.units as u
import numpy as np
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


#print(f"mesh: {mesh}")

simulation = ecsim.Simulation('tsien', result_root='results')
geometry = simulation.setup_geometry(mesh)
print(f"Compartments: {geometry.compartment_names}")
print(f"Membranes: {geometry.membrane_names}")
geometry.visualize(resolve_regions=False)

# Get the compartments and membranes
ecs = geometry.compartments['ecs']
cytosol = geometry.compartments['cytosol']
channel = geometry.membranes['channel']
print(f"ecs volume: {ecs.volume}")
print(f"cytosol volume: {cytosol.volume}")
print(f"Channel area: {channel.area}")

# Add species to the simulation
ca = simulation.add_species('Ca', valence=2)

# egta = simulation.add_species('EGTA', valence=-2)
# ca_egta = simulation.add_species('Ca_EGTA', valence=0)

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
cytosol.add_reaction(
    reactants=[ca, bapta], products=[ca_bapta],
    k_f=kf_bapta, k_r=kr_bapta
)
# cytosol.add_reaction(reactants=[ca_cyt, egta], products=[ca_egta], k_f=kf_egta, k_r=kr_egta)

#rate_channel = 10 * u.um / u.ms  # Total flux
rate_channel = 10 * u.um / u.s  # Total flux
permeability = rate_channel * channel.area
channel.add_transport(
    species=ca, 
    transport=transport.Passive(permeability=permeability),
    source=ecs,
    target=cytosol
)

# Add recorders to capture simulation data
# xs = np.linspace(0, 600, 100)
# points = [[x, 0, 2.995] for x in xs]
# xs = np.linspace(0, 3.0, 100)  # en micras
xs = np.linspace(0, 0.6, 300)  # en micras
points = [[float(x), 0.0, 2.995] for x in xs]
simulation.add_recorder(recorder.FullSnapshot(1 * u.ms))
simulation.add_recorder(recorder.CompartmentSubstance(1 * u.ms))
simulation.add_recorder(recorder.PointValues(20 * u.ms, points))

# Run the simulation
simulation.run(end_time=20 * u.ms, time_step=1 * u.us)