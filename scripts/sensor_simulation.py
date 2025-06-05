import astropy.units as u
from ngsolve.webgui import Draw

import ecsim
from ecsim.geometry import create_sensor_geometry
from ecsim.units import mM, uM


# Physical parameters
SIDE_LENGTH = 200 * u.um  # 200x200x200 µm cube
COMPARTMENT_RATIO = 0.5   # Divide the cube in half along the x axis
#SPHERE_POSITION_X = 50 * u.um  # Centered in the first compartment
SPHERE_POSITION_X = 150 * u.um  # Centered in the second compartment
SPHERE_RADIUS = 30 * u.um      # Sphere radius
MESH_SIZE = 5 * u.um          # Mesh size

SENSOR_ACTIVE = False  # Switch if sensor actively bind Ca or not

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
simulation = ecsim.Simulation('sensor', mesh, result_root='results')

# Compartments
cube = simulation.simulation_geometry.compartments['cube']

# Add species - Ca
ca = simulation.add_species("ca", valence=2)
CA_INIT = 0.5 * mM
cube.initialize_species(ca, {'left': CA_INIT, 'right': CA_INIT, 'sphere': CA_INIT})
cube.add_diffusion(ca, 600 * u.um**2 / u.s)

# Buffer 1 parameters
buffer_tot = 1.5 * mM  # Total buffer
buffer_kd = 0.05 * mM  # Dissociation constant
kf = 0.01 / (uM * u.s)  # Forward rate
kr = kf * buffer_kd  # Reverse rate

# Compute initial free buffer and complex
free_buffer_init = buffer_tot * (buffer_kd / (buffer_kd + CA_INIT))
ca_b_init = buffer_tot - free_buffer_init

# Add buffer species (non-diffusive)
buffer = simulation.add_species('buffer', valence=-1)
cube.add_diffusion(buffer, 0 * u.um**2 / u.s)
cube.initialize_species(buffer, {'left': free_buffer_init, 'right': 0 * mM, 'sphere': 0 * mM})

# Add complex species (non-diffusive)
cab_complex = simulation.add_species('complex', valence=0)
cube.initialize_species(cab_complex, {'left': ca_b_init, 'right': 0 * mM, 'sphere': 0 * mM})
cube.add_diffusion(cab_complex, 0 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + buffer ↔ complex
cube.add_reaction(reactants=[ca, buffer], products=[cab_complex], k_f=kf, k_r=kr)

# SENSOR 
# Sensor parameters
sensor_tot = 0.5 * mM  # Total sensor
sensor_kd = 0.01 * mM  # Dissociation constant
kf_sensor = 0.001 / (uM * u.s)  # Forward rate
kr_sensor = kf_sensor * sensor_kd  # Reverse rate

# Compute initial free sensor and complex
free_sensor_init = sensor_tot * (sensor_kd / (sensor_kd + CA_INIT))
sensor_b_init = sensor_tot - free_sensor_init

# Add sensor species (non-diffusive)
sensor = simulation.add_species('sensor', valence=-1)
cube.add_diffusion(sensor, 0 * u.um**2 / u.s)
cube.initialize_species(sensor, {'left': 0 * mM, 'right': 0 * mM, 'sphere': free_sensor_init})

# Add sensor - complex species (non-diffusive)
sensor_complex = simulation.add_species('sensor_complex', valence=0)
cube.initialize_species(sensor_complex, {'left': 0 * mM, 'right': 0 * mM, 'sphere': sensor_b_init})
cube.add_diffusion(sensor_complex, 0 * u.um**2 / u.s)

if SENSOR_ACTIVE:
    # Add reversible binding reaction: Ca + sensor ↔ complex
    cube.add_reaction(reactants=[ca, sensor], products=[sensor_complex], k_f=kf_sensor, k_r=kr_sensor)

# Simulate
simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=10 * u.ms)
