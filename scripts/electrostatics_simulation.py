"""
This code simulates electrostatic interactions among three chemical species: B1
(immobile), B2 (mobile), and Ca (diffusing). The simulation is performed within
a two-region geometry, where the concentrations of the chemical species are
distributed unevenly across the regions.
"""
import astropy.units as u  # Physical units
from ngsolve.webgui import Draw  # Mesh visualization

import ecsim  # Simulation framework
from ecsim.geometry import create_cube_geometry  # Geometry generator

# Initial and target Ca concentrations
CA_INIT = 1 * u.mmol / u.L

# Define cube geometry dimensions
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
print("Material names in mesh:", mesh.GetMaterials())

# Initialize simulation and link geometry
simulation = ecsim.Simulation('electrostatics', mesh, result_root='results', electrostatics=True)
geometry = simulation.simulation_geometry

# Access compartments and membrane
cube = geometry.compartments['cube']
outside = geometry.membranes['side']

cube.add_relative_permittivity(80)

# Add Ca species and set diffusion
ca = simulation.add_species('ca', valence=2)
cube.initialize_species(ca, CA_INIT)
cube.add_diffusion(ca, 600 * u.um**2 / u.s)

# Immobile buffer parameters
immobile_total_buffer = 1.0 * u.mmol / u.L  # Total buffer

# Add buffer species (non-diffusive)
immobile_buffer = simulation.add_species('immobile_buffer', valence=-2)
cube.add_diffusion(immobile_buffer, 0 * u.um**2 / u.s)
cube.initialize_species(immobile_buffer, {'top': 0 * u.mmol / u.L, 'bottom': immobile_total_buffer})

# Mobile buffer parameters
mobile_total_buffer = 0.5 * u.mmol / u.L  # Total buffer

# Add buffer species (diffusive)
mobile_buffer = simulation.add_species('mobile_buffer', valence=-2)
cube.add_diffusion(mobile_buffer, 50 * u.um**2 / u.s)
cube.initialize_species(mobile_buffer, {'top': mobile_total_buffer, 'bottom': mobile_total_buffer})

# Run simulation
simulation.run(
    end_time=4 * u.ms,
    time_step=1 * u.us,
    record_interval=100 * u.us,
    n_threads=4
)
