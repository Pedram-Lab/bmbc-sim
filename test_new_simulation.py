import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import astropy.units as u

import ecsim
from ecsim.evaluation.vtk_recorder import FullSnapshot


left = occ.Box((0, 0, 0), (1, 1, 1)).mat('ecm:left').bc('reflective')
middle = occ.Box((1, 0, 0), (2, 1, 1)).mat('ecm:right').bc('reflective')
right = occ.Box((2, 0, 0), (3, 1, 1)).mat('cell').bc('reflective')

geo = occ.OCCGeometry(occ.Glue([left, middle, right]))
mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.5))

mesh.ngmesh.SetBCName(5, 'clamped')
mesh.ngmesh.SetBCName(1, 'left_membrane')
mesh.ngmesh.SetBCName(6, 'right_membrane')

Draw(mesh)

simulation = ecsim.Simulation('debug', result_root='results')
geometry = simulation.add_geometry(mesh)
ecm = geometry.compartments['ecm']
cell = geometry.compartments['cell']
# geometry_description.visualize(resolve_regions=True)

ca = simulation.add_species('Ca', valence=2)

simulation.add_recorder(FullSnapshot(100 * u.us))

cell.initialize_species(ca, value=0.5 * u.mmol / u.L)
cell.add_diffusion(
    species=ca,
    diffusivity=1000 * u.nm**2 / u.ms,
)

ecm.initialize_species(ca, value={'left': 2.0 * u.mmol / u.L, 'right': 3.0 * u.mmol / u.L})
ecm.add_diffusion(
    species=ca,
    diffusivity={'left': 0 * u.nm**2 / u.ms, 'right': 10 * u.um**2 / u.ms},
)

simulation.simulate_for(
    n_steps=100,
    time_step=10 * u.us
)
