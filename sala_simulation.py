"""This script recreates the simulation of [Sala, Hernández-Cruz; 1990]."""
import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import astropy.units as u

import ecsim
from ecsim.simulation import recorder, transport


# Create a spherical cell
cell = occ.Sphere((0, 0, 0), 20).mat('cell').bc('membrane')

geo = occ.OCCGeometry(cell)
mesh = ngs.Mesh(geo.GenerateMesh(maxh=5))
# Draw(mesh)

# Initialize the simulation
simulation = ecsim.Simulation('sala', result_root='results')
geometry = simulation.add_geometry(mesh)
# geometry.visualize()

# Get the compartments and membranes
cell = geometry.compartments['cell']
membrane = geometry.membranes['membrane']
M = u.mol / u.L
uM = u.umol / u.L

# Add species to the simulation
ca = simulation.add_species('Ca', valence=2)
b1 = simulation.add_species('Buffer_1', valence=-2)
b2 = simulation.add_species('Buffer_2', valence=-2)
b3 = simulation.add_species('Buffer_3', valence=-2)
ca_b1 = simulation.add_species('Ca_Buffer_1', valence=0)
ca_b2 = simulation.add_species('Ca_Buffer_2', valence=0)
ca_b3 = simulation.add_species('Ca_Buffer_3', valence=0)

# Initial conditions
ca_0 = 0.05 * uM
b1_0 = 100 * uM
b2_0 = 600 * uM
b3_0 = 100 * uM
cell.initialize_species(ca, value=ca_0)
cell.initialize_species(b1, value=b1_0)
cell.initialize_species(b2, value=b2_0)
cell.initialize_species(b3, value=b3_0)

# Assume that the buffers are in equilibrium with the calcium initially
kd_b1 = 1 * uM
kd_b2 = 0.4 * uM
kd_b3 = 0.2 * uM
ca_b1_0 = (ca_0 * b1_0) / (ca_0 + kd_b1)
ca_b2_0 = (ca_0 * b2_0) / (ca_0 + kd_b2)
ca_b3_0 = (ca_0 * b3_0) / (ca_0 + kd_b3)
cell.initialize_species(ca_b1, value=ca_b1_0)
cell.initialize_species(ca_b2, value=ca_b2_0)
cell.initialize_species(ca_b3, value=ca_b3_0)

# Add diffusion to the species (B2 is not diffusing)
cell.add_diffusion(species=ca, diffusivity=6e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=b1, diffusivity=0.5e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=ca_b1, diffusivity=0.5e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=b3, diffusivity=2.5e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=ca_b3, diffusivity=2.5e-6 * u.cm**2 / u.s)

# Compute forward and reverse reaction rates for the buffers
kf_b1 = 1e8 / (M * u.s)
kr_b1 = kf_b1 * kd_b1

kf_b2 = 5e5 / (M * u.s)
kr_b2 = kf_b2 * kd_b2

kf_b3 = 1e8 / (M * u.s)
kr_b3 = kf_b3 * kd_b3

# Add reactions
cell.add_reaction(
    reactants=[ca, b1], products=[ca_b1],
    k_f=kf_b1, k_r=kr_b1
)
cell.add_reaction(
    reactants=[ca, b2], products=[ca_b2],
    k_f=kf_b2, k_r=kr_b2
)
cell.add_reaction(
    reactants=[ca, b3], products=[ca_b3],
    k_f=kf_b3, k_r=kr_b3
)

# Compute transport coefficients
u_max = 2 * u.pmol / (u.cm**2 * u.s)
# TODO: only divide by volume because area comes from integration anyway?
v_max = u_max * membrane.area / cell.volume
k_m = 0.83 * uM


# # Add transport mechanisms to the membranes
# # Time dependent influx
# current = 5 * u.nA
# pulse_duration = 100 * u.ms
# membrane.add_transport(
#     species=ca,
#     transport=transport.Linear(permeability=1 * u.umol / (u.L * u.s)),
#     source=cell,
#     target=3 * u.umol / u.L
# )
# # MM-type pump out of the cell (dependent on inside concentration)
# membrane.add_transport(
#     species=ca,
#     transport=transport.MichaelisMenten(v_max=v_max, k_m=k_m),
#     source=cell,
#     target=3 * u.umol / u.L
# )
# # MM-type leak into the cell (dependent on outside concentration)
# ca_0 = 0.05 * u.umol / u.L  # outside concentration
# membrane.add_transport(
#     species=ca,
#     transport=transport.MichaelisMenten(v_max=v_max, k_m=k_m),
#     source=cell,
#     target=3 * u.umol / u.L
# )


# Add recorders to capture simulation data
points = [[x, 0, 0] for x in [0.25, 5.25, 10.25, 19.75]]
simulation.add_recorder(recorder.FullSnapshot(100 * u.ms))
simulation.add_recorder(recorder.CompartmentSubstance(100 * u.ms))
simulation.add_recorder(recorder.PointValues(0.1 * u.ms, points))

# Run the simulation
simulation.run(end_time=2 * u.s, time_step=0.01 * u.ms)
