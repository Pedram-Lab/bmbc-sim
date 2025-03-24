"""This script recreates the simulation of [Sala, Hernández-Cruz; 1990]."""
import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import astropy.units as u

import ecsim
from ecsim.evaluation.point_recorder import PointValues
from ecsim.evaluation.total_substance_recorder import CompartmentSubstance
from ecsim.evaluation.vtk_recorder import FullSnapshot
from ecsim.simulation.geometry import transport


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

# Add species to the simulation
ca = simulation.add_species('Ca', valence=2)
b_1 = simulation.add_species('Buffer_1', valence=-2)
b_2 = simulation.add_species('Buffer_2', valence=-2)
b_3 = simulation.add_species('Buffer_3', valence=-2)
ca_b_1 = simulation.add_species('Ca_Buffer_1', valence=0)
ca_b_2 = simulation.add_species('Ca_Buffer_2', valence=0)
ca_b_3 = simulation.add_species('Ca_Buffer_3', valence=0)

# Initial conditions
cell.initialize_species(ca, value=0.05 * u.umol / u.L)
cell.initialize_species(b_1, value=100 * u.umol / u.L)
cell.initialize_species(b_2, value=600 * u.umol / u.L)
cell.initialize_species(b_3, value=100 * u.umol / u.L)

# Add diffusion to the species (B2 is not diffusing)
cell.add_diffusion(species=ca, diffusivity=6e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=b_1, diffusivity=0.5e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=ca_b_1, diffusivity=0.5e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=b_3, diffusivity=2.5e-6 * u.cm**2 / u.s)
cell.add_diffusion(species=ca_b_3, diffusivity=2.5e-6 * u.cm**2 / u.s)

# Compute forward and reverse reaction rates for the buffers
k_f_1 = 1e8 * u.L / (u.mol * u.s)
k_d_1 = 1 * u.umol / u.L
k_r_1 = k_f_1 * k_d_1

k_f_2 = 5e5 * u.L / (u.mol * u.s)
k_d_2 = 0.4 * u.umol / u.L
k_r_2 = k_f_2 * k_d_2

k_f_3 = 1e8 * u.L / (u.mol * u.s)
k_d_3 = 0.2 * u.umol / u.L
k_r_3 = k_f_3 * k_d_3

# Add reactions
cell.add_reaction(
    reactants=[ca, b_1], products=[ca_b_1],
    k_f=k_f_1, k_r=k_r_1
)
cell.add_reaction(
    reactants=[ca, b_2], products=[ca_b_2],
    k_f=k_f_2, k_r=k_r_2
)
cell.add_reaction(
    reactants=[ca, b_3], products=[ca_b_3],
    k_f=k_f_3, k_r=k_r_3
)

# Compute transport coefficients
cell.volume
membrane.area
u_max = 2 * u.pmol / (u.cm**2 * u.s)
# TODO: only divide by volume because area comes from integration anyway?
v_max = u_max * membrane.area / cell.volume
k_m = 0.83 * u.umol / u.L


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
simulation.add_recorder(FullSnapshot(100 * u.ms))
simulation.add_recorder(CompartmentSubstance(100 * u.ms))
simulation.add_recorder(PointValues(0.1 * u.ms, points))

# Run the simulation
simulation.simulate_until(end_time=2 * u.s, time_step=0.01 * u.ms)
