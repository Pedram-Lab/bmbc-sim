"""
This code simulates chemical interactions among three species: B1 (immobile),
B2 (mobile), and Ca (diffusing). The simulation is performed within a two-region
geometry, where the concentrations of the chemical species are distributed
unevenly across the regions. The simulation is designed to run under two scenarios:
1. Pure chelation only -> set "electrostatics = False"
2. Chelation combined with electrostatics -> set "electrostatics = True"
"""
import astropy.units as u  # Physical units
from ngsolve.webgui import Draw  # Mesh visualization

import ecsim  # Simulation framework
from ecsim.geometry import create_cube_geometry  # Geometry generator


# Initial Ca concentration
CA_FREE = 1 * u.mmol / u.L  # free calcium

# Define dish geometry dimensions
CUBE_HEIGHT = 1 * u.um
SIDELENGTH = 0.5 * u.um
SUBSTRATE_HEIGHT = 0.5 * u.um


# Create and visualize 3D mesh
mesh = create_cube_geometry(
    cube_height=CUBE_HEIGHT,
    slice_width=SIDELENGTH,
    slice_depth=0.5 * u.um,
    mesh_size=SIDELENGTH / 20,
    substrate_height=SUBSTRATE_HEIGHT
)
Draw(mesh)


# Initialize simulation and link geometry
simulation = ecsim.Simulation('chelation', mesh, result_root='results', electrostatics=True)
geometry = simulation.simulation_geometry


# Access compartments and membrane
cube = geometry.compartments['cube']
outside = geometry.membranes['side']


cube.add_relative_permittivity(80)


# Add Ca species and set diffusion
ca = simulation.add_species('ca', valence=2)
cube.initialize_species(ca, CA_FREE)
cube.add_diffusion(ca, 600 * u.um**2 / u.s)


# Immobile buffer parameters
total_immobile_buffer = 1.0 * u.mmol / u.L  # Total buffer
immobile_buffer_kd = 10.0 * u.umol / u.L  # Dissociation constant
immobile_buffer_kf = 1.0e8 / (u.mol / u.L * u.s)  # Forward rate
immobile_buffer_kr = immobile_buffer_kf * immobile_buffer_kd   # Reverse rate


# Add buffer species (non-diffusive)
immobile_buffer = simulation.add_species('immobile_buffer', valence=-2)
cube.add_diffusion(immobile_buffer, 0 * u.um**2 / u.s)
cube.initialize_species(immobile_buffer, {'top': 0 * u.mmol / u.L, 'bottom': total_immobile_buffer})


# Add complex species (non-diffusive)
immobile_complex = simulation.add_species('immobile_complex', valence=0)
cube.initialize_species(immobile_complex, {'top': 0 * u.mmol / u.L, 'bottom': 0 * u.mmol / u.L})
cube.add_diffusion(immobile_complex, 0 * u.um**2 / u.s)


#Add reversible binding reaction: Ca + buffer <-> complex
cube.add_reaction(reactants=[ca, immobile_buffer], products=[immobile_complex], k_f=immobile_buffer_kf, k_r=immobile_buffer_kr)


# Mobile buffer parameters 
total_mobile_buffer = 0.5 * u.mmol / u.L  # Total buffer
mobile_buffer_kd = 10.0 * u.umol / u.L  # Dissociation constant
mobile_buffer_kf = 1e8 / (u.mol / u.L * u.s)  # Forward rate
mobile_buffer_kr = mobile_buffer_kf * mobile_buffer_kd  # Reverse rate


# Add buffer species (diffusive)
mobile_buffer = simulation.add_species('mobile_buffer', valence=-2)
cube.add_diffusion(mobile_buffer, 50 * u.um**2 / u.s)
cube.initialize_species(mobile_buffer, {'top': total_mobile_buffer, 'bottom': total_mobile_buffer})


# Add complex species (diffusive)
mobile_complex = simulation.add_species('mobile_complex', valence=0)
cube.initialize_species(mobile_complex, {'top': 0 * u.mmol / u.L, 'bottom': 0 * u.mmol / u.L})
cube.add_diffusion(mobile_complex, 50 * u.um**2 / u.s)


# Add reversible binding reaction: Ca + buffer <-> complex
cube.add_reaction(reactants=[ca, mobile_buffer], products=[mobile_complex], k_f=mobile_buffer_kf, k_r=mobile_buffer_kr)


# Run simulation
simulation.run(
    end_time=4 * u.ms,
    time_step=1 * u.us,
    record_interval=100 * u.us,
    n_threads=4
)
