"""This code simulates a rapid calcium dilution in a dish-like environment (Tony
experiment), considering:
- Calcium diffusion
- Reversible binding to a buffer
- Controlled calcium removal
- Visualization and recording of spatial and temporal behavior
"""

import astropy.units as u  # Physical units
from ngsolve.webgui import Draw  # Mesh visualization
import ecsim  # Simulation framework
from ecsim.simulation import transport  # Tools for transport
from ecsim.geometry import create_dish_geometry  # Geometry generator

# Initial and target Ca concentrations
CA_INIT = 4 * u.mmol / u.L
CA_DILUTED = 0.5 * u.mmol / u.L

# Define dish geometry dimensions
DISH_HEIGHT = 1.5 * u.mm
SIDELENGTH = 300 * u.um
SUBSTRATE_HEIGHT = 300 * u.um

# Create and visualize 3D mesh
mesh = create_dish_geometry(
    dish_height=DISH_HEIGHT,
    slice_width=SIDELENGTH,
    slice_depth=10 * u.um,
    mesh_size=SIDELENGTH / 10,
    substrate_height=SUBSTRATE_HEIGHT
)
Draw(mesh)
print("Material names in mesh:", mesh.GetMaterials())

# Initialize simulation and link geometry
simulation = ecsim.Simulation('tony', mesh, result_root='results')
geometry = simulation.simulation_geometry

# Access compartments and membrane
dish = geometry.compartments['dish']
outside = geometry.membranes['side']


# Add Ca species and set diffusion
ca = simulation.add_species('ca', valence=0)
dish.initialize_species(ca, CA_INIT)
dish.add_diffusion(ca, 600 * u.um**2 / u.s)

# Buffer parameters
buffer_tot = 1.0 * u.mmol / u.L  # Total buffer
buffer_kd = 0.05 * u.mmol / u.L  # Dissociation constant
kf = 0.001 / (u.umol / u.L * u.s)  # Forward rate
kr = kf * buffer_kd  # Reverse rate

# Compute initial free buffer and complex
free_buffer_init = buffer_tot * (buffer_kd / (buffer_kd + CA_INIT))
ca_b_init = buffer_tot - free_buffer_init

# Add buffer species (non-diffusive)
buffer = simulation.add_species('buffer', valence=0)
dish.add_diffusion(buffer, 0 * u.um**2 / u.s)
dish.initialize_species(buffer, {'free': 0 * u.mmol / u.L, 'substrate': free_buffer_init})

# Add complex species (non-diffusive)
cab_complex = simulation.add_species('complex', valence=0)
dish.initialize_species(cab_complex, {'free': 0 * u.mmol / u.L, 'substrate': ca_b_init})
dish.add_diffusion(cab_complex, 0 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + buffer ↔ complex
dish.add_reaction(reactants=[ca, buffer], products=[cab_complex],
                  k_f=kf, k_r=kr)

# Compute Ca to remove for dilution
substance_to_remove = (CA_INIT - CA_DILUTED) * dish.volume
dilution_start = 1 * u.min
dilution_end = 1 * u.min + 10 * u.s
flux_rate = substance_to_remove / (dilution_end - dilution_start)


# Time-dependent efflux function
def efflux(t):
    """Efflux function for Ca removal during dilution period."""
    if dilution_start <= t < dilution_end:
        return flux_rate
    else:
        return 0 * u.amol / u.s


# Apply Ca efflux through the membrane
tran = transport.GeneralFlux(flux=efflux)
outside.add_transport(ca, transport=tran, source=dish, target=None)

# Run simulation
simulation.run(end_time=5 * u.min, time_step=0.01 * u.s, output_interval=10 * u.s)
