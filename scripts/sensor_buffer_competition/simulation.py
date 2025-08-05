"""
Simulation of chemical interactions among:
- Immobile buffer (B)
- Immobile sensor (S)
- Diffusing calcium (Ca)

Two-region geometry: top and bottom.
You can configure electrostatics and species initial concentrations per compartment.
"""
import astropy.units as u
from ngsolve.webgui import Draw
import ecsim
from ecsim.geometry import create_box_geometry
from ecsim.simulation import transport


# ===========================
# USER PARAMETERS
# ===========================

# Geometry
CA_FREE = 1 * u.mmol / u.L
CUBE_HEIGHT = 1 * u.um
SIDELENGTH = 0.5 * u.um
SUBSTRATE_HEIGHT = 0.5 * u.um

# Initial concentrations per compartment
BUFFER_INITIAL = {
    'top': 0 * u.mmol / u.L,
    'bottom': 2.0 * u.mmol / u.L,
}
SENSOR_INITIAL = 10 * u.umol / u.L

# Reaction constants
BUFFER_KD = 1.0 * u.umol / u.L
BUFFER_KF = 1.0e8 / (u.mol / u.L * u.s)
BUFFER_KR = BUFFER_KF * BUFFER_KD

SENSOR_KD = 1.0 * u.mmol / u.L
SENSOR_KF = 1.0e8 / (u.mol / u.L * u.s)
SENSOR_KR = SENSOR_KF * SENSOR_KD


# ===========================
# GEOMETRY SETUP
# ===========================

mesh = create_box_geometry(
    dimensions=(SIDELENGTH, SIDELENGTH, CUBE_HEIGHT),
    mesh_size=SIDELENGTH / 20,
    split=SUBSTRATE_HEIGHT,
    compartments=True,
)
Draw(mesh)

simulation = ecsim.Simulation('sensor_buffer_competition', mesh, result_root='results')
geometry = simulation.simulation_geometry

compartments = geometry.compartments
interface = geometry.membranes['interface']


# ===========================
# ADD CALCIUM SPECIES
# ===========================
ca = simulation.add_species('ca', valence=2)
for comp in compartments.values():
    comp.initialize_species(ca, CA_FREE)
    comp.add_diffusion(ca, 600 * u.um**2 / u.s)


# ===========================
# MOBILE BUFFER
# ===========================
buffer = simulation.add_species('immobile_buffer')
buffer_complex = simulation.add_species('immobile_buffer_complex')

for name, comp in compartments.items():
    # Initialize buffer and complex per compartment
    comp.add_diffusion(buffer, 2.5e-6 * u.cm**2 / u.s)
    comp.initialize_species(buffer, BUFFER_INITIAL[name])
    comp.add_diffusion(buffer_complex, 2.5e-6 * u.cm**2 / u.s)
    comp.initialize_species(buffer_complex, 0 * u.mmol / u.L)

    # Reaction: Ca + buffer <-> buffer_complex
    comp.add_reaction(
        reactants=[ca, buffer],
        products=[buffer_complex],
        k_f=BUFFER_KF,
        k_r=BUFFER_KR
    )


# ===========================
# MOBILE SENSOR
# ===========================
sensor = simulation.add_species('immobile_sensor', valence=-2)
sensor_complex = simulation.add_species('immobile_sensor_complex', valence=0)

for name, comp in compartments.items():
    # Initialize sensor and complex per compartment
    comp.add_diffusion(sensor, 2.5e-6 * u.cm**2 / u.s)
    comp.initialize_species(sensor, SENSOR_INITIAL)
    comp.add_diffusion(sensor_complex, 2.5e-6 * u.cm**2 / u.s)
    comp.initialize_species(sensor_complex, 0 * u.mmol / u.L)

    # Reaction: Ca + sensor <-> sensor_complex
    comp.add_reaction(
        reactants=[ca, sensor],
        products=[sensor_complex],
        k_f=SENSOR_KF,
        k_r=SENSOR_KR
    )


# ===========================
# TRANSPORT
# ===========================
t = transport.Passive(permeability=1 * u.um**3 / u.ms)
interface.add_transport(species=ca, transport=t,
                        source=compartments["top"], target=compartments["bottom"])


# ===========================
# RUN SIMULATION
# ===========================
simulation.run(
    end_time=5 * u.ms,
    time_step=5 * u.us,
    record_interval=100 * u.us,
    n_threads=4
)
