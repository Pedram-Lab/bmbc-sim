"""
This code simulates electrostatic interactions among three chemical species: B1
(immobile), B2 (mobile), and Ca (diffusing). The simulation is performed within
a two-region geometry, where the concentrations of the chemical species are
distributed unevenly across the regions.
"""
import astropy.units as u  # Physical units
from ngsolve.webgui import Draw  # Mesh visualization

import ecsim  # Simulation framework
from ecsim.geometry import create_dish_geometry  # Geometry generator


# Initial and target Ca concentrations
CA_INIT = 1 * u.mmol / u.L

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
simulation = ecsim.Simulation('electrostatics', mesh, result_root='results', electrostatics=True)
geometry = simulation.simulation_geometry

# Access compartments and membrane
dish = geometry.compartments['dish']
outside = geometry.membranes['side']

dish.add_relative_permittivity(80)

# Add Ca species and set diffusion
ca = simulation.add_species('ca', valence=2)
dish.initialize_species(ca, CA_INIT)
dish.add_diffusion(ca, 600 * u.um**2 / u.s)

# Buffer 1 parameters
buffer_tot = 2.0 * u.mmol / u.L  # Total buffer

# Add buffer species (non-diffusive)
buffer = simulation.add_species('buffer', valence=-1)
dish.add_diffusion(buffer, 0 * u.um**2 / u.s)
dish.initialize_species(buffer, {'free': 0 * u.mmol / u.L, 'substrate': buffer_tot})

# Buffer2 parameters
buffer_tot_2 = 2.0 * u.mmol / u.L  # Total buffer

# Add buffer species (diffusive)
buffer_2 = simulation.add_species('buffer_2', valence=-1)
dish.add_diffusion(buffer_2, 50 * u.um**2 / u.s)
dish.initialize_species(buffer_2, {'free': buffer_tot_2, 'substrate': 0 * u.mmol / u.L})

# Run simulation
simulation.run(end_time=1 * u.ms, time_step=10 * u.ns, output_interval=2 * u.us)
