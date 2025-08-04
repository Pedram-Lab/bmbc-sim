"""
This code simulates chemical interactions among three species: B1 (immobile),
B2 (mobile), and Ca (diffusing). The simulation is performed within a two-region
geometry, where the concentrations of the chemical species are distributed
unevenly across the regions. The simulation is designed to run under different scenarios:
1. Pure chelation only -> set "electrostatics = False"
2. Pure electrostatics only -> set "chelation = False"
3. Chelation combined with electrostatics -> set both to True
"""
import astropy.units as u
from ngsolve.webgui import Draw

import ecsim
import ecsim.geometry as geo


# Switches for electrostatics and chelation
ELECTROSTATICS = True
CHELATION = True

# Define geometry dimensions
BOX_HEIGHT = 1 * u.um
SIDELENGTH = 0.5 * u.um
SPLIT = 0.5 * u.um

# Create and visualize 3D mesh
mesh = geo.create_box_geometry(
    dimensions=(SIDELENGTH, SIDELENGTH, BOX_HEIGHT),
    mesh_size=SIDELENGTH / 20,
    split=SPLIT
)
Draw(mesh)

# Initialize simulation and link geometry
simulation_name = []
if CHELATION:
    simulation_name.append("chelation")
if ELECTROSTATICS:
    simulation_name.append("electrostatics")
if not simulation_name:
    simulation_name.append("no_interaction")
simulation = ecsim.Simulation(
    "_".join(simulation_name), mesh, result_root="results", electrostatics=ELECTROSTATICS
)
geometry = simulation.simulation_geometry

# Access compartments
box = geometry.compartments['box']

if ELECTROSTATICS:
    box.add_relative_permittivity(80)

# Add Ca species
total_ca = 1 * u.mmol / u.L

ca = simulation.add_species('ca', valence=2)
box.initialize_species(ca, total_ca)
box.add_diffusion(ca, 600 * u.um**2 / u.s)

# Add non-diffusive buffer species
total_immobile_buffer = 1.0 * u.mmol / u.L
immobile_buffer_kd = 10.0 * u.umol / u.L  # Dissociation constant
immobile_buffer_kf = 1.0e8 / (u.mol / u.L * u.s)  # Forward rate
immobile_buffer_kr = immobile_buffer_kf * immobile_buffer_kd   # Reverse rate

immobile_buffer = simulation.add_species('immobile_buffer', valence=-2)
box.add_diffusion(immobile_buffer, 0 * u.um**2 / u.s)
box.initialize_species(immobile_buffer, {'top': 0 * u.mmol / u.L, 'bottom': total_immobile_buffer})

# Add diffusive buffer species
total_mobile_buffer = 0.5 * u.mmol / u.L
mobile_buffer_kd = 10.0 * u.umol / u.L  # Dissociation constant
mobile_buffer_kf = 1e8 / (u.mol / u.L * u.s)  # Forward rate
mobile_buffer_kr = mobile_buffer_kf * mobile_buffer_kd  # Reverse rate

mobile_buffer = simulation.add_species('mobile_buffer', valence=-2)
box.add_diffusion(mobile_buffer, 50 * u.um**2 / u.s)
box.initialize_species(mobile_buffer, total_mobile_buffer)

if CHELATION:
    # Add reversible binding reaction: Ca + immobile_buffer <-> immobile_complex
    immobile_complex = simulation.add_species('immobile_complex', valence=0)
    box.initialize_species(immobile_complex, {'top': 0 * u.mmol / u.L, 'bottom': 0 * u.mmol / u.L})
    box.add_diffusion(immobile_complex, 0 * u.um**2 / u.s)

    box.add_reaction(reactants=[ca, immobile_buffer], products=[immobile_complex],
                    k_f=immobile_buffer_kf, k_r=immobile_buffer_kr)

    # Add reversible binding reaction: Ca + mobile_buffer <-> mobile_complex
    mobile_complex = simulation.add_species('mobile_complex', valence=0)
    box.initialize_species(mobile_complex, {'top': 0 * u.mmol / u.L, 'bottom': 0 * u.mmol / u.L})
    box.add_diffusion(mobile_complex, 50 * u.um**2 / u.s)

    box.add_reaction(reactants=[ca, mobile_buffer], products=[mobile_complex],
                    k_f=mobile_buffer_kf, k_r=mobile_buffer_kr)

# Run simulation
simulation.run(
    end_time=4 * u.ms,
    time_step=1 * u.us,
    record_interval=100 * u.us,
    n_threads=4
)
