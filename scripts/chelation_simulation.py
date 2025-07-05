"""
This code simulates chemical interactions among three species: B1 (immobile),
B2 (mobile), and Ca (diffusing). The simulation is performed within a two-region
geometry, where the concentrations of the chemical species are distributed
unevenly across the regions.
"""
import astropy.units as u  # Physical units
from ngsolve.webgui import Draw  # Mesh visualization

import ecsim  # Simulation framework
from ecsim.geometry import create_dish_geometry  # Geometry generator


# Initial and target Ca concentrations
CA_FREE = 1 * u.mmol / u.L  # free calcium

# Define dish geometry dimensions
DISH_HEIGHT = 1 * u.um
SIDELENGTH = 0.5 * u.um
SUBSTRATE_HEIGHT = 0.5 * u.um

# Create and visualize 3D mesh
mesh = create_dish_geometry(
    dish_height=DISH_HEIGHT,
    slice_width=SIDELENGTH,
    slice_depth=0.5 * u.um,
    mesh_size=SIDELENGTH / 10,
    substrate_height=SUBSTRATE_HEIGHT
)
Draw(mesh)
print("Material names in mesh:", mesh.GetMaterials())

# Initialize simulation and link geometry
simulation = ecsim.Simulation('chelation', mesh, result_root='results', electrostatics=False)
geometry = simulation.simulation_geometry

# Access compartments and membrane
dish = geometry.compartments['dish']
outside = geometry.membranes['side']

dish.add_relative_permittivity(80)

# Add Ca species and set diffusion
ca = simulation.add_species('ca', valence=2)
dish.initialize_species(ca, CA_FREE)
#dish.initialize_species(ca, {'free':CA_FREE, 'substrate': 0.09 * u.mmol / u.L})
dish.add_diffusion(ca, 600 * u.um**2 / u.s)

# Buffer 1 parameters
buffer_tot = 2.0 * u.mmol / u.L  # Total buffer
buffer_kd = 10.0 * u.umol / u.L  # Dissociation constant
kf = 1.0e8 / (u.mol / u.L * u.s)  # Forward rate
kr = kf * buffer_kd  # Reverse rate

# Compute initial free buffer and complex
free_buffer_init = buffer_tot * (buffer_kd / (buffer_kd + CA_FREE))
ca_b_init = buffer_tot - free_buffer_init
Ca_Total = CA_FREE + ca_b_init

# Add buffer species (non-diffusive)
buffer = simulation.add_species('immobile_buffer', valence=-1)
dish.add_diffusion(buffer, 0 * u.um**2 / u.s)
dish.initialize_species(buffer, {'free': 0 * u.mmol / u.L, 'substrate': buffer_tot})

# Add complex species (non-diffusive)
cab_complex = simulation.add_species('immobile_complex', valence=0)
dish.initialize_species(cab_complex, {'free': 0 * u.mmol / u.L, 'substrate': 0 * u.mmol / u.L})
dish.add_diffusion(cab_complex, 0 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + buffer ↔ complex
dish.add_reaction(reactants=[ca, buffer], products=[cab_complex], k_f=kf, k_r=kr)

# Buffer2 parameters
buffer_tot_2 = 1.0 * u.mmol / u.L  # Total buffer
buffer_kd_2 = 10.0 * u.umol / u.L  # Dissociation constant
kf_2 = 1e8 / (u.mol / u.L * u.s)  # Forward rate
kr_2 = kf_2 * buffer_kd_2  # Reverse rate

# Compute initial free buffer and complex
free_buffer_init_2 = buffer_tot_2 * (buffer_kd_2 / (buffer_kd_2 + CA_FREE))
ca_b_init_2 = buffer_tot_2 - free_buffer_init_2

# Add buffer species (diffusive)
buffer_2 = simulation.add_species('mobile_buffer', valence=-1)
dish.add_diffusion(buffer_2, 50 * u.um**2 / u.s)
#dish.initialize_species(buffer_2, {'free': buffer_tot_2, 'substrate': 0 * u.mmol / u.L})
dish.initialize_species(buffer_2, {'free': buffer_tot_2, 'substrate': buffer_tot_2})

# Add complex species (diffusive)
cab_complex_2 = simulation.add_species('mobile_complex', valence=0)
dish.initialize_species(cab_complex_2, {'free': 0 * u.mmol / u.L, 'substrate': 0 * u.mmol / u.L})
dish.add_diffusion(cab_complex_2, 50 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + buffer ↔ complex
dish.add_reaction(reactants=[ca, buffer_2], products=[cab_complex_2], k_f=kf_2, k_r=kr_2)


# Run simulation
simulation.run(
    end_time=4 * u.ms,
    time_step=1 * u.us,
    record_interval=100 * u.us,
    n_threads=4
)
