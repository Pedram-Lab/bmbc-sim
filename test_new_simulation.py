import ngsolve as ngs
from netgen import occ
from ngsolve.webgui import Draw
import astropy.units as u

import ecsim
from ecsim.simulation import recorder, transport


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
clamped_membrane = geometry.membranes['clamped']
cell_ecs_membrane = geometry.membranes['right_membrane']
# geometry.visualize(resolve_regions=False)

ca = simulation.add_species('Ca', valence=2)
buf = simulation.add_species('Buffer', valence=-2)
ca_buf = simulation.add_species('Ca_Buffer', valence=0)

line = [[i / 10, 0.5, 0.5] for i in range(31)]
simulation.add_recorder(recorder.FullSnapshot(100 * u.us))
simulation.add_recorder(recorder.CompartmentSubstance(100 * u.us))
simulation.add_recorder(recorder.PointValues(0.5 * u.ms, line))

cell.initialize_species(ca, value=0.5 * u.mmol / u.L)
cell.initialize_species(buf, value=0.5 * u.mmol / u.L)
for species in [ca, buf, ca_buf]:
    cell.add_diffusion(
        species=species,
        diffusivity=1 * u.um**2 / u.ms,
    )

cell.add_reaction(
    reactants=[ca, buf],
    products=[ca_buf],
    k_f=1 * u.mmol / (u.L * u.ms),
    k_r=1 / u.ms
)

ecm.initialize_species(ca, value={'left': 2.0 * u.mmol / u.L, 'right': 3.0 * u.mmol / u.L})
ecm.add_diffusion(
    species=ca,
    diffusivity={'left': 1 * u.nm**2 / u.ms, 'right': 10 * u.um**2 / u.ms},
)

clamped_membrane.add_transport(
    species=ca,
    transport=transport.Linear(permeability=1 * u.mmol / (u.L * u.ms)),
    source=ecm,
    target=3 * u.mmol / u.L
)
cell_ecs_membrane.add_transport(
    species=buf,
    transport=transport.Linear(permeability=10 * u.mmol / (u.L * u.ms)),
    source=cell,
    target=ecm
)

simulation.run(end_time=1 * u.ms, time_step=10 * u.us)
