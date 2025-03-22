import logging
import os

import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import astropy.units as u

import ecsim
from ecsim.evaluation.vtk_recorder import Snapshot

logging.basicConfig(
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)s %(message)s"
)

left = occ.Box((0, 0, 0), (1, 1, 1)).mat('ecm:left').bc('reflective')
middle = occ.Box((1, 0, 0), (2, 1, 1)).mat('ecm:right').bc('reflective')
right = occ.Box((2, 0, 0), (3, 1, 1)).mat('cell').bc('reflective')

geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.5))

mesh.ngmesh.SetBCName(5, 'clamped')
mesh.ngmesh.SetBCName(1, 'left_membrane')
mesh.ngmesh.SetBCName(6, 'right_membrane')

print("mesh materials: ", mesh.GetMaterials())
print("mesh boundaries: ", mesh.GetBoundaries())

Draw(mesh)

geometry = ecsim.SimulationGeometry(mesh)
ecm = geometry.compartments['ecm']
cell = geometry.compartments['cell']
# geometry_description.visualize(resolve_regions=True)

simulation = ecsim.Simulation(geometry, 'results')
ca = simulation.add_species('Ca', valence=2)

simulation.add_recorder(Snapshot(100 * u.us))

ecm.initialize_species(ca, value={'left': 2.0 * u.mmol / u.L, 'right': 3.0 * u.mmol / u.L})
cell.initialize_species(ca, value=0.5 * u.mmol / u.L)


cell.add_diffusion(
    species=ca,
    diffusivity=600 * u.nm**2 / u.ms,
)
ecm.add_diffusion(
    species=ca,
    diffusivity={'left': 1000 * u.nm**2 / u.ms, 'right': 500 * u.nm**2 / u.ms},
)

simulation.simulate_for(
    n_steps=100,
    time_step=10 * u.us
)
