import astropy.units as u

import ecsim
from ecsim.geometry import create_sensor_geometry
from ecsim.units import mM, uM


### Simulation Parameters
SENSOR_ACTIVE = False  # Switch if sensor actively bind Ca or not

# Geometry parameters
SIDE_LENGTH = 200 * u.um  # 200x200x200 µm cube
COMPARTMENT_RATIO = 0.5   # Divide the cube in half along the x axis
#SPHERE_POSITION_X = 50 * u.um  # Centered in the first compartment
SPHERE_POSITION_X = 150 * u.um  # Centered in the second compartment
SPHERE_RADIUS = 30 * u.um      # Sphere radius
MESH_SIZE = 5 * u.um          # Mesh size

# Calcium parameters
CA_INIT = 0.5 * mM

# Buffer parameters
BUFFER_INIT = 1.5 * mM  # Total buffer
BUFFER_KD = 0.05 * mM  # Dissociation constant
KF_B = 0.01 / (uM * u.s)  # Forward rate
KR_B = KF_B * BUFFER_KD  # Reverse rate

# Sensor parameters
SENSOR_INIT = 0.5 * mM  # Total sensor
SENSOR_KD = 0.01 * mM  # Dissociation constant
KF_S = 0.001 / (uM * u.s)  # Forward rate
KR_S = KF_S * SENSOR_KD  # Reverse rate


### Simulation setup
# Create the mesh
mesh = create_sensor_geometry(
    side_length=SIDE_LENGTH,
    compartment_ratio=COMPARTMENT_RATIO,
    sphere_position_x=SPHERE_POSITION_X,
    sphere_radius=SPHERE_RADIUS,
    mesh_size=MESH_SIZE
)

# Create simulation
simulation = ecsim.Simulation('sensor', mesh, result_root='results')

# Compartments
cube = simulation.simulation_geometry.compartments['cube']

# Add species - Ca
ca = simulation.add_species("ca", valence=2)
cube.initialize_species(ca, CA_INIT)
cube.add_diffusion(ca, 600 * u.um**2 / u.s)

# Add buffer species (non-diffusive)
buffer = simulation.add_species('buffer', valence=-1)
cube.add_diffusion(buffer, 0 * u.um**2 / u.s)
cube.initialize_species(buffer, {'left': BUFFER_INIT, 'right': 0 * mM, 'sphere': 0 * mM})

# Add buffer complex species (non-diffusive)
ca_buffer = simulation.add_species('ca_buffer', valence=0)
cube.initialize_species(ca_buffer, 0 * mM)
cube.add_diffusion(ca_buffer, 0 * u.um**2 / u.s)

# Add reversible binding reaction: Ca + buffer ↔ complex
cube.add_reaction(reactants=[ca, buffer], products=[ca_buffer], k_f=KF_B, k_r=KR_B)

# Add sensor species (non-diffusive)
sensor = simulation.add_species('sensor', valence=-1)
cube.add_diffusion(sensor, 0 * u.um**2 / u.s)
cube.initialize_species(sensor, {'left': 0 * mM, 'right': 0 * mM, 'sphere': SENSOR_INIT})

# Add sensor complex species (non-diffusive)
ca_sensor = simulation.add_species('ca_sensor', valence=0)
cube.initialize_species(ca_sensor, 0 * mM)
cube.add_diffusion(ca_sensor, 0 * u.um**2 / u.s)

if SENSOR_ACTIVE:
    # Add reversible binding reaction: Ca + sensor ↔ complex
    cube.add_reaction(reactants=[ca, sensor], products=[ca_sensor], k_f=KF_S, k_r=KR_S)


### Simulate
simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=10 * u.ms)
