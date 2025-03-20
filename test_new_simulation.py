import logging
import os

import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import astropy.units as u

import ecsim

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)s %(message)s"
)

left = occ.Box((0, 0, 0), (1, 1, 1)).mat('ecm:left').bc('reflective')
middle = occ.Box((1, 0, 0), (2, 1, 1)).mat('ecm:right').bc('reflective')
right = occ.Box((2, 0, 0), (3, 1, 1)).mat('cell').bc('reflective')

geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))

mesh.ngmesh.SetBCName(5, 'clamped')
mesh.ngmesh.SetBCName(1, 'left_membrane')
mesh.ngmesh.SetBCName(6, 'right_membrane')

geometry_description = ecsim.SimulationGeometry(mesh)
# geometry_description.visualize(resolve_regions=True)

simulation = ecsim.Simulation(geometry_description)

ca = simulation.add_species(ecsim.ChemicalSpecies('Ca', valence=2))

simulation.add_diffusion(
    species=ca,
    compartment='cell',
    diffusivity=600 * u.nm**2 / u.ms,
)

simulation.add_diffusion(
    species=ca,
    compartment='ecm',
    diffusivity={'left': 1000 * u.nm**2 / u.ms, 'right': 500 * u.nm**2 / u.ms},
)

simulation.simulate_for(
    time_step=0.1 * u.ms,
    n_steps=100,
)
