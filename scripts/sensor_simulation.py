import astropy.units as u
from ngsolve.webgui import Draw
import numpy as np

import ecsim
from ecsim.simulation import recorder
from ecsim.geometry import create_sensor_geometry


# Physical parameters
SIDE_LENGTH = 200 * u.um  # 200x200x200 µm cube
COMPARTMENT_RATIO = 0.5   # Divide the cube in half along the x axis
#SPHERE_POSITION_X = 50 * u.um  # Centered in the first compartment
SPHERE_POSITION_X = 150 * u.um  # Centered in the first compartment
SPHERE_RADIUS = 30 * u.um      # Sphere radius
MESH_SIZE = 5 * u.um          # Mesh size

# Create the mesh
mesh = create_sensor_geometry(
    side_length=SIDE_LENGTH,
    compartment_ratio=COMPARTMENT_RATIO,
    sphere_position_x=SPHERE_POSITION_X,
    sphere_radius=SPHERE_RADIUS,
    mesh_size=MESH_SIZE
)

# Visualize mesh
Draw(mesh)
print("Materials:", mesh.GetMaterials())

# Create simulation
simulation = ecsim.Simulation('sensor', result_root='results')
geometry = simulation.setup_geometry(mesh)

# Compartments
cube = geometry.compartments['cube']


# Add species - Ca
ca = simulation.add_species("ca", valence=2)
CA_INIT = 0.5 * u.mmol / u.L
cube.initialize_species(ca, {'left': CA_INIT, 'right': CA_INIT, 'sphere': CA_INIT})
cube.add_diffusion(ca, 600 * u.um**2 / u.s)


# Buffer 1 parameters
buffer_tot = 1.5 * u.mmol / u.L  # Total buffer
buffer_kd = 0.05 * u.mmol / u.L  # Dissociation constant
kf = 0.01 / (u.umol / u.L * u.s)  # Forward rate
kr = kf * buffer_kd  # Reverse rate

# Compute initial free buffer and complex
free_buffer_init = buffer_tot * (buffer_kd / (buffer_kd + CA_INIT))
ca_b_init = buffer_tot - free_buffer_init

# Add buffer species (non-diffusive)
buffer = simulation.add_species('buffer', valence=-1)
cube.add_diffusion(buffer, 0 * u.um**2 / u.s)
cube.initialize_species(buffer, {'left': free_buffer_init, 'right': 0 * u.mmol / u.L, 'sphere': 0 * u.mmol / u.L})

# Add complex species (non-diffusive)
cab_complex = simulation.add_species('complex', valence=0)
cube.initialize_species(cab_complex, {'left': ca_b_init, 'right': 0 * u.mmol / u.L, 'sphere': 0 * u.mmol / u.L})
cube.add_diffusion(cab_complex, 0 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + buffer ↔ complex
cube.add_reaction(reactants=[ca, buffer], products=[cab_complex], k_f=kf, k_r=kr)

# SENSOR 
# Sensor parameters
sensor_tot = 0.5 * u.mmol / u.L  # Total sensor
sensor_kd = 0.01 * u.mmol / u.L  # Dissociation constant
kf_sensor = 0.001 / (u.umol / u.L * u.s)  # Forward rate
kr_sensor = kf_sensor * sensor_kd  # Reverse rate

# Compute initial free sensor and complex
free_sensor_init = sensor_tot * (sensor_kd / (sensor_kd + CA_INIT))
sensor_b_init = sensor_tot - free_sensor_init

# Add sensor species (non-diffusive)
sensor = simulation.add_species('sensor', valence=-1)
cube.add_diffusion(sensor, 0 * u.um**2 / u.s)
cube.initialize_species(sensor, {'left': 0 * u.mmol / u.L, 'right': 0 * u.mmol / u.L, 'sphere': free_sensor_init})

# Add sensor - complex species (non-diffusive)
sensor_complex = simulation.add_species('sensor_complex', valence=0)
cube.initialize_species(sensor_complex, {'left': 0 * u.mmol / u.L, 'right': 0 * u.mmol / u.L, 'sphere': sensor_b_init})
cube.add_diffusion(sensor_complex, 0 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + sensor ↔ complex
# cube.add_reaction(reactants=[ca, sensor], products=[sensor_complex], k_f=kf_sensor, k_r=kr_sensor)

# Simulation 
# recording_points = {
#     'sphere': [[50.0, 100.0, float(z)] for z in np.linspace(70.0, 130.0, 50)],
#     'left': [[20.0, 100.0, float(z)] for z in np.linspace(0.0, 200.0, 50)],
#     'right': [[180.0, 100.0, float(z)] for z in np.linspace(0.0, 200.0, 50)]
# }

# Register full snapshot every 5 seconds
simulation.add_recorder(recorder.FullSnapshot(0.1 * u.s))

# Simulate
simulation.run(end_time=5 * u.s, time_step=0.01 * u.s)

# # Points inside the sphere (aligned along z, at the center of x and y of the sphere)
# sphere_points = [[50.0, 100.0, float(z)] for z in np.linspace(70.0, 130.0, 50)]

# # Points in the left compartment (aligned in z, far from the sphere)
# left_points = [[20.0, 100.0, float(z)] for z in np.linspace(0.0, 200.0, 50)]

# # Points in the right compartment
# right_points = [[180.0, 100.0, float(z)] for z in np.linspace(0.0, 200.0, 50)]

# # Register species at each set of points
# for species in [ca, buffer, cab_complex, sensor, sensor_complex]:
#     simulation.add_recorder(recorder.FullSnapshot(species=species, points=sphere_points, label=f"{species.name}_sphere"))
#     simulation.add_recorder(recorder.FullSnapshot(species=species, points=left_points, label=f"{species.name}_left"))
#     simulation.add_recorder(recorder.FullSnapshot(species=species, points=right_points, label=f"{species.name}_right"))
# simulation.add_recorder(recorder.FullSnapshot(5 * u.s))
# simulation.run(end_time=1 * u.min, time_step=0.05 * u.s)

# cube.initialize_species(ca, 1 * u.umol / u.L)
# right.initialize_species(ca, 0 * u.umol / u.L)
# sphere.initialize_species(ca, 5 * u.umol / u.L)

# # Diffusion in all compartments
# D_ca = 500 * u.um**2 / u.s
# left.add_diffusion(ca, D_ca)
# right.add_diffusion(ca, D_ca)
# sphere.add_diffusion(ca, D_ca)

# Register data and run simulation



# import astropy.units as u
# from ngsolve.webgui import Draw

# from ecsim import Simulation
# from ecsim.geometry import create_sensor_geometry
# from ecsim.simulation import recorder

# # Parámetros físicos
# SIDE_LENGTH = 200 * u.um
# COMPARTMENT_RATIO = 0.5
# SPHERE_POSITION_X = 35 * u.um
# SPHERE_RADIUS = 20 * u.um
# MESH_SIZE = 20 * u.um

# # Crear la malla
# mesh = create_sensor_geometry(
#     side_length=SIDE_LENGTH,
#     compartment_ratio=COMPARTMENT_RATIO,
#     sphere_position_x=SPHERE_POSITION_X,
#     sphere_radius=SPHERE_RADIUS,
#     mesh_size=MESH_SIZE
# )

# Draw(mesh)
# print("Materiales en mesh:", mesh.GetMaterials())

# # Crear simulación
# sim = Simulation("sensor_sim", result_root="results")

# # Mapear materiales a nombres de compartimentos
# geometry = sim.setup_geometry(mesh, material_map={
#     "cube:left": "left",
#     "cube:right": "right",
#     "cube:sphere": "sphere"
# })

# print("Compartimentos disponibles:", geometry.compartments.keys())

# # Acceder a los compartimentos
# left = geometry.compartments["left"]
# right = geometry.compartments["right"]
# sphere = geometry.compartments["sphere"]

# # Agregar especie difusiva
# ca = sim.add_species("ca", valence=2)
# left.initialize_species(ca, 1 * u.umol / u.L)
# right.initialize_species(ca, 0 * u.umol / u.L)
# sphere.initialize_species(ca, 5 * u.umol / u.L)

# # Difusión
# D_ca = 500 * u.um**2 / u.s
# left.add_diffusion(ca, D_ca)
# right.add_diffusion(ca, D_ca)
# sphere.add_diffusion(ca, D_ca)

# # Registrar resultados y correr simulación
# sim.add_recorder(recorder.FullSnapshot(5 * u.s))
# sim.run(end_time=1 * u.min, time_step=0.05 * u.s)


# import astropy.units as u
# from ngsolve.webgui import Draw

# from ecsim import Simulation
# from ecsim.geometry import create_sensor_geometry
# from ecsim.simulation import recorder

# # Parámetros físicos
# SIDE_LENGTH = 200 * u.um
# COMPARTMENT_RATIO = 0.5
# SPHERE_POSITION_X = 35 * u.um
# SPHERE_RADIUS = 20 * u.um
# MESH_SIZE = 20 * u.um

# # Crear la malla
# mesh = create_sensor_geometry(
#     side_length=SIDE_LENGTH,
#     compartment_ratio=COMPARTMENT_RATIO,
#     sphere_position_x=SPHERE_POSITION_X,
#     sphere_radius=SPHERE_RADIUS,
#     mesh_size=MESH_SIZE
# )

# Draw(mesh)
# print("Materiales en mesh:", mesh.GetMaterials())

# # Crear simulación
# sim = Simulation("sensor_sim", result_root="results")
# geometry = sim.setup_geometry(mesh)

# # Registrar manualmente los compartimentos usando materiales del mesh
# for mat in mesh.GetMaterials():
#     geometry.register_compartment(mat)

# # Mostrar los compartimentos ahora disponibles
# print("Compartimentos registrados:", geometry.compartments.keys())

# # Acceder a los compartimentos
# left = geometry.compartments["cube:left"]
# right = geometry.compartments["cube:right"]
# sphere = geometry.compartments["cube:sphere"]

# # Agregar especie difusiva
# ca = sim.add_species("ca", valence=2)
# left.initialize_species(ca, 1 * u.umol / u.L)
# right.initialize_species(ca, 0 * u.umol / u.L)
# sphere.initialize_species(ca, 5 * u.umol / u.L)

# # Difusión
# D_ca = 500 * u.um**2 / u.s
# left.add_diffusion(ca, D_ca)
# right.add_diffusion(ca, D_ca)
# sphere.add_diffusion(ca, D_ca)

# # Registrar resultados y correr simulación
# sim.add_recorder(recorder.FullSnapshot(5 * u.s))
# sim.run(end_time=1 * u.min, time_step=0.05 * u.s)



# import astropy.units as u
# from ngsolve.webgui import Draw

# from ecsim.geometry import create_sensor_geometry
# from ecsim import Simulation

# # Parámetros seguros
# SIDE_LENGTH = 200 * u.um
# COMPARTMENT_RATIO = 0.5
# SPHERE_POSITION_X = 35 * u.um
# SPHERE_RADIUS = 20 * u.um  # Más pequeño para evitar colisiones
# MESH_SIZE = 20 * u.um

# # Crear la malla
# mesh = create_sensor_geometry(
#     side_length=SIDE_LENGTH,
#     compartment_ratio=COMPARTMENT_RATIO,
#     sphere_position_x=SPHERE_POSITION_X,
#     sphere_radius=SPHERE_RADIUS,
#     mesh_size=MESH_SIZE
# )

# Draw(mesh)
# print("Materiales en mesh:", mesh.GetMaterials())

# # Crear simulación y geometría SIN merge automático
# sim = Simulation("sensor_sim", result_root="results")
# geometry = sim.setup_geometry(mesh, merge=False)

# # Registrar manualmente los compartimentos
# geometry.register_compartment("cube:left")
# geometry.register_compartment("cube:right")
# geometry.register_compartment("cube:sphere")

# # Acceder normalmente
# left = geometry.compartments["cube:left"]
# right = geometry.compartments["cube:right"]
# sphere = geometry.compartments["cube:sphere"]



# # Simulación
# sim = Simulation("sensor_debug", result_root="results")
# geometry = sim.setup_geometry(mesh)
# print("Compartimentos disponibles:", geometry.compartments.keys())

# import astropy.units as u
# from ngsolve.webgui import Draw

# from ecsim import Simulation
# from ecsim.geometry import create_sensor_geometry
# from ecsim.simulation import recorder

# # Parámetros físicos
# SIDE_LENGTH = 200 * u.um  # Cubo de 200x200x200 µm
# COMPARTMENT_RATIO = 0.5   # Divide el cubo a la mitad en el eje x
# SPHERE_POSITION_X = 35 * u.um  # Posición centrada en el primer compartimento
# SPHERE_RADIUS = 30 * u.um      # Radio de la esfera
# MESH_SIZE = 20 * u.um          # Tamaño de la malla

# # Crear la malla
# mesh = create_sensor_geometry(
#     side_length=SIDE_LENGTH,
#     compartment_ratio=COMPARTMENT_RATIO,
#     sphere_position_x=SPHERE_POSITION_X,
#     sphere_radius=SPHERE_RADIUS,
#     mesh_size=MESH_SIZE
# )

# # Visualizar malla
# Draw(mesh)
# print("Materiales:", mesh.GetMaterials())

# # Crear simulación
# sim = Simulation("sensor_sim", result_root="results")
# geometry = sim.setup_geometry(mesh)

# # Compartimentos
# left = geometry.compartments["cube:left"]
# right = geometry.compartments["cube:right"]
# sphere = geometry.compartments["cube:sphere"]

# # Agregar una especie difusiva
# ca = sim.add_species("ca", valence=2)
# left.initialize_species(ca, 1 * u.umol / u.L)
# right.initialize_species(ca, 0 * u.umol / u.L)
# sphere.initialize_species(ca, 5 * u.umol / u.L)

# # Difusión en todos los compartimentos
# D_ca = 500 * u.um**2 / u.s
# left.add_diffusion(ca, D_ca)
# right.add_diffusion(ca, D_ca)
# sphere.add_diffusion(ca, D_ca)

# # Registrar datos y correr simulación
# sim.add_recorder(recorder.FullSnapshot(5 * u.s))
# sim.run(end_time=1 * u.min, time_step=0.05 * u.s)



#########

# """This code simulates a rapid calcium dilution in a dish-like environment (Tony
# experiment), considering:
# - Calcium diffusion
# - Reversible binding to a buffer
# - Controlled calcium removal
# - Visualization and recording of spatial and temporal behavior
# """

# import astropy.units as u  # Physical units
# from ngsolve.webgui import Draw  # Mesh visualization
# import numpy as np
# import ecsim  # Simulation framework
# from ecsim.simulation import recorder, transport  # Tools for data recording and transport
# from ecsim.geometry import create_sensor_geometry  # Geometry generator

# mesh = create_sensor_geometry(
#     total_length=1 * u.mm,
#     side_length_y=500 * u.um,
#     side_length_z=500 * u.um,
#     compartment_ratio=0.7,
#     sphere_radius=100 * u.um,
#     mesh_size=50 * u.um
# )
# Draw(mesh)
# print("Materiales:", mesh.GetMaterials())

# # Initialize simulation and link geometry
# simulation = ecsim.Simulation('sensor', result_root='results')
# geometry = simulation.setup_geometry(mesh)
# comp1 = geometry.compartments["compartment1"]
# comp2 = geometry.compartments["compartment2"]
# sphere1 = geometry.compartments["sphere1"]
# sphere2 = geometry.compartments["sphere2"]

#aDD 

# # Initial and target Ca concentrations
# CA_INIT = 4 * u.mmol / u.L
# CA_DILUTED = 0.5 * u.mmol / u.L

# # Define dish geometry dimensions
# DISH_HEIGHT = 1.5 * u.mm
# SIDELENGTH = 300 * u.um
# SUBSTRATE_HEIGHT = 300 * u.um

# # Create and visualize 3D mesh
# mesh = create_dish_geometry(
#     dish_height=DISH_HEIGHT,
#     slice_width=SIDELENGTH,
#     slice_depth=10 * u.um,
#     mesh_size=SIDELENGTH / 10,
#     substrate_height=SUBSTRATE_HEIGHT
# )
# Draw(mesh)
# print("Material names in mesh:", mesh.GetMaterials())

# # Initialize simulation and link geometry
# simulation = ecsim.Simulation('tony', result_root='results')
# geometry = simulation.setup_geometry(mesh)

# # Access compartments and membrane
# dish = geometry.compartments['dish']
# outside = geometry.membranes['side']


# # Add Ca species and set diffusion
# ca = simulation.add_species('ca', valence=0)
# dish.initialize_species(ca, CA_INIT)
# dish.add_diffusion(ca, 600 * u.um**2 / u.s)

# # Buffer parameters
# buffer_tot = 1.0 * u.mmol / u.L  # Total buffer
# buffer_kd = 0.05 * u.mmol / u.L  # Dissociation constant
# kf = 0.001 / (u.umol / u.L * u.s)  # Forward rate
# kr = kf * buffer_kd  # Reverse rate

# # Compute initial free buffer and complex
# free_buffer_init = buffer_tot * (buffer_kd / (buffer_kd + CA_INIT))
# ca_b_init = buffer_tot - free_buffer_init

# # Add buffer species (non-diffusive)
# buffer = simulation.add_species('buffer', valence=0)
# dish.add_diffusion(buffer, 0 * u.um**2 / u.s)
# dish.initialize_species(buffer, {'free': 0 * u.mmol / u.L, 'substrate': free_buffer_init})

# # Add complex species (non-diffusive)
# cab_complex = simulation.add_species('complex', valence=0)
# dish.initialize_species(cab_complex, {'free': 0 * u.mmol / u.L, 'substrate': ca_b_init})
# dish.add_diffusion(cab_complex, 0 * u.um**2 / u.s)

# # Add reversible binding reaction: Ca + buffer ↔ complex
# dish.add_reaction(reactants=[ca, buffer], products=[cab_complex],
#                   k_f=kf, k_r=kr)

# # Compute Ca to remove for dilution
# substance_to_remove = (CA_INIT - CA_DILUTED) * dish.volume
# dilution_start = 1 * u.min
# dilution_end = 1 * u.min + 10 * u.s
# flux_rate = substance_to_remove / (dilution_end - dilution_start)


# # Time-dependent efflux function
# def efflux(t):
#     """Efflux function for Ca removal during dilution period."""
#     if dilution_start <= t < dilution_end:
#         return flux_rate
#     else:
#         return 0 * u.amol / u.s


# # Apply Ca efflux through the membrane
# tran = transport.GeneralFlux(flux=efflux)
# outside.add_transport(ca, transport=tran, source=dish, target=None)

# # Define recording points and run simulation
# points = [[150.0, 150.0, float(z)] for z in np.linspace(0, 10, 1500)]
# simulation.add_recorder(recorder.FullSnapshot(10 * u.s))
# simulation.run(end_time=5 * u.min, time_step=0.01 * u.s)
