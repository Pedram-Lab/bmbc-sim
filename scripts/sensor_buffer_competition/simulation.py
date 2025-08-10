"""
Simulation of chemical interactions among:
- Immobile buffer (B)
- Immobile sensor (S)
- Diffusing calcium (Ca)

Two-region geometry: top and bottom.
You can configure electrostatics and species initial concentrations per compartment.
"""
import argparse

import astropy.units as u
from ngsolve.webgui import Draw

import ecsim
from ecsim.geometry import create_box_geometry
from ecsim.simulation import transport


def run_simulation(buffer_conc, buffer_kd):
    """Run simulation with specific buffer concentration and KD."""
    # Create simulation name based on parameters
    sim_name = f"sensor_buffer_competition_conc{buffer_conc:.0e}_kd{buffer_kd:.0e}"

    # Geometry parameters
    ca_free = 1 * u.mmol / u.L
    cube_height = 1 * u.um
    sidelength = 0.5 * u.um
    substrate_height = 0.5 * u.um

    # Initial concentrations per compartment
    buffer_initial = {
        'top': 0 * u.mmol / u.L,
        'bottom': buffer_conc * u.mmol / u.L,
    }
    sensor_initial = 10 * u.umol / u.L

    # Buffer reaction constants
    buffer_kd = buffer_kd * u.mmol / u.L
    buffer_kf = 1.0e8 / (u.mol / u.L * u.s)
    buffer_kr = buffer_kf * buffer_kd

    # Sensor reaction constants
    sensor_kd = 1.0 * u.mmol / u.L
    sensor_kf = 1.0e8 / (u.mol / u.L * u.s)
    sensor_kr = sensor_kf * sensor_kd

    # Geometry setup
    mesh = create_box_geometry(
        dimensions=(sidelength, sidelength, cube_height),
        mesh_size=sidelength / 20,
        split=substrate_height,
        compartments=True,
    )
    Draw(mesh)

    simulation = ecsim.Simulation(sim_name, mesh, result_root='results')
    geometry = simulation.simulation_geometry

    compartments = geometry.compartments
    interface = geometry.membranes['interface']

    # Add calcium species
    ca = simulation.add_species('ca', valence=2)
    for comp in compartments.values():
        comp.initialize_species(ca, ca_free)
        comp.add_diffusion(ca, 600 * u.um**2 / u.s)

    # Mobile buffer
    buffer = simulation.add_species('buffer')
    buffer_complex = simulation.add_species('buffer_complex')

    for name, comp in compartments.items():
        # Initialize buffer and complex per compartment
        comp.add_diffusion(buffer, 2.5e-6 * u.cm**2 / u.s)
        comp.initialize_species(buffer, buffer_initial[name])
        comp.add_diffusion(buffer_complex, 2.5e-6 * u.cm**2 / u.s)
        comp.initialize_species(buffer_complex, 0 * u.mmol / u.L)

        # Reaction: Ca + buffer <-> buffer_complex
        comp.add_reaction(
            reactants=[ca, buffer],
            products=[buffer_complex],
            k_f=buffer_kf,
            k_r=buffer_kr
        )

    # Mobile sensor
    sensor = simulation.add_species('sensor')
    sensor_complex = simulation.add_species('sensor_complex')

    for name, comp in compartments.items():
        # Initialize sensor and complex per compartment
        comp.add_diffusion(sensor, 2.5e-6 * u.cm**2 / u.s)
        comp.initialize_species(sensor, sensor_initial)
        comp.add_diffusion(sensor_complex, 2.5e-6 * u.cm**2 / u.s)
        comp.initialize_species(sensor_complex, 0 * u.mmol / u.L)

        # Reaction: Ca + sensor <-> sensor_complex
        comp.add_reaction(
            reactants=[ca, sensor],
            products=[sensor_complex],
            k_f=sensor_kf,
            k_r=sensor_kr
        )

    # Transport
    t = transport.Passive(permeability=lambda t: (0 if t < 1 * u.ms else 10) * u.um**3 / u.ms)
    interface.add_transport(species=ca, transport=t,
                            source=compartments["top"], target=compartments["bottom"])

    # Run simulation
    simulation.run(
        end_time=4 * u.ms,
        time_step=5 * u.us,
        record_interval=100 * u.us,
        n_threads=4
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run buffer simulation with specific parameters')
    parser.add_argument('--buffer_conc', type=float, required=True,
                        help='Buffer concentration in mM')
    parser.add_argument('--buffer_kd', type=float, required=True,
                        help='Buffer KD value in uM')

    args = parser.parse_args()
    run_simulation(args.buffer_conc, args.buffer_kd)
