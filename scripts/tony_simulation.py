"""A simulation of Tony's experiments with his Ca-indicator."""
import astropy.units as u
from ngsolve.webgui import Draw
import numpy as np

import ecsim
from ecsim.simulation import recorder, transport
from ecsim.geometry import create_dish_geometry


# Ca parameters
CA_INIT = 4 * u.mmol / u.L
CA_DILUTED = 0.5 * u.mmol / u.L

# Create a geometry for the simulation
DISH_HEIGHT = 1.5 * u.mm
SIDELENGTH = 300 * u.um
SUBSTRATE_HEIGHT = 300 * u.um

mesh = create_dish_geometry(
    dish_height=DISH_HEIGHT,
    slice_width=SIDELENGTH,
    slice_depth=10 * u.um,
    mesh_size=SIDELENGTH / 10,
    substrate_height=SUBSTRATE_HEIGHT
)
Draw(mesh)
print("Material names in mesh:", mesh.GetMaterials())

# Set up simulation environment
simulation = ecsim.Simulation('tony', result_root='results')
geometry = simulation.setup_geometry(mesh)

dish = geometry.compartments['dish'] 
outside = geometry.membranes['side']

ca = simulation.add_species('ca', valence=0)

dish.initialize_species(ca, CA_INIT)
dish.add_diffusion(ca, 600 * u.um**2 / u.s)

# Compute Ca to be removed
substance_to_remove = (CA_INIT - CA_DILUTED) * dish.volume
dilution_start = 1 * u.min
dilution_end = 1 * u.min + 10 * u.s
flux_rate = substance_to_remove / (dilution_end - dilution_start)


def efflux(t):
    if dilution_start <= t < dilution_end:
        return flux_rate
    else:
        return 0 * u.amol / u.s


t = transport.Channel(flux=efflux)
outside.add_transport(ca, transport=t, source=dish, target=None)

# Add evaluation and run the simulation
points = [[150.0, 150.0, float(z)] for z in np.linspace(0, 10, 1500)]
simulation.add_recorder(recorder.FullSnapshot(10 * u.s))

simulation.run(end_time=5 * u.min, time_step=0.1 * u.s)
