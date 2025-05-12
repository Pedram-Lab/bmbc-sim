"""This code simulates a rapid calcium dilution in a dish-like environment (Tony
experiment), considering:
- Calcium diffusion
- Reversible binding to a buffer
- Controlled calcium removal
- Visualization and recording of spatial and temporal behavior
"""

import astropy.units as u  # Physical units
from ngsolve.webgui import Draw  # Mesh visualization
import numpy as np
import ecsim  # Simulation framework
from ecsim.simulation import recorder, transport  # Tools for data recording and transport
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
simulation = ecsim.Simulation('tony', result_root='results', electrostatics=True)
geometry = simulation.setup_geometry(mesh)

# Access compartments and membrane
dish = geometry.compartments['dish']
outside = geometry.membranes['side']

dish.add_relative_permittivity(80)


# Add Ca species and set diffusion
ca = simulation.add_species('ca', valence=2)
dish.initialize_species(ca, CA_INIT)
dish.add_diffusion(ca, 600 * u.um**2 / u.s)

# Buffer parameters
buffer_tot = 1.0 * u.mmol / u.L  # Total buffer

# Add buffer species (non-diffusive)
buffer = simulation.add_species('buffer', valence=-2)
dish.add_diffusion(buffer, 0 * u.um**2 / u.s)
dish.initialize_species(buffer, {'free': 0 * u.mmol / u.L, 'substrate': buffer_tot})

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

# Define recording points and run simulation
simulation.add_recorder(recorder.FullSnapshot(20 * u.s))
simulation.add_recorder(recorder.PointValues(20 * u.s, points=[(300, 5, 100), (300, 5, 1400)]))
simulation.run(end_time=10 * u.min, time_step=0.1 * u.s, n_threads=4)
