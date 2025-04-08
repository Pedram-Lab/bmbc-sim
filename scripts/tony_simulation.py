"""File to simulate Tony's experiments."""
import astropy.units as u
from ngsolve.webgui import Draw

from ecsim.geometry import create_dish_geometry


# Create a geometry for the simulation
DISH_HEIGHT = 1.5 * u.mm
SIDELENGTH = 300 * u.um
SUBSTRATE_HEIGHT = 300 * u.um

mesh = create_dish_geometry(
    dish_height=DISH_HEIGHT,
    sidelength=SIDELENGTH,
    mesh_size=SIDELENGTH / 10,
    substrate_height=SUBSTRATE_HEIGHT
)
Draw(mesh)
