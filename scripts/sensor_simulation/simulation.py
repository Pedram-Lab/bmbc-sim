import argparse

import astropy.units as u

import ecsim
from ecsim.geometry import create_sensor_geometry
from ecsim.units import mM, uM, nM


def run_sensor_simulation(
    sensor_active=False,
    sensor_left=True,
    side_length=200 * u.um,
    compartment_ratio=0.5,
    sphere_position_x=None,
    sphere_radius=30 * u.um,
    mesh_size=5 * u.um,
    ca_init=0.05 * uM,
    buffer_init=600 * uM,
    buffer_kd=400 * nM,
    kf_b=50 / (mM * u.s),
    sensor_init=100 * uM,
    sensor_kd=420 * nM,
    kf_s=100 / (uM * u.s),
):
    """Run a simulation of a cube where one side is occupied by a buffer, and
    there is a spherical region with a sensor. The sensor region can be either
    left or right.
    Default values are taken from [Sala, Hernández-Cruz; 1990].
    """
    if sphere_position_x is None:
        sphere_position_x = 50 * u.um if sensor_left else 150 * u.um

    kr_b = kf_b * buffer_kd  # Reverse rate
    kr_s = kf_s * sensor_kd  # Reverse rate

    # Create the mesh
    mesh = create_sensor_geometry(
        side_length=side_length,
        compartment_ratio=compartment_ratio,
        sphere_position_x=sphere_position_x,
        sphere_radius=sphere_radius,
        mesh_size=mesh_size
    )

    # Create simulation
    simulation = ecsim.Simulation('sensor', mesh, result_root='results')

    # Compartments
    cube = simulation.simulation_geometry.compartments['cube']

    # Add species - Ca
    ca = simulation.add_species("ca", valence=2)
    cube.initialize_species(ca, ca_init)
    cube.add_diffusion(ca, 600 * u.um**2 / u.s)

    # Add buffer species (non-diffusive)
    buffer = simulation.add_species('buffer', valence=-1)
    cube.add_diffusion(buffer, 0 * u.um**2 / u.s)
    if sensor_left:
        cube.initialize_species(buffer, {'left': 0 * mM, 'right': buffer_init, 'sphere': 0 * mM})
    else:
        cube.initialize_species(buffer, {'left': 0 * mM, 'right': buffer_init, 'sphere': buffer_init})

    # Add buffer complex species (non-diffusive)
    ca_buffer = simulation.add_species('ca_buffer', valence=0)
    cube.initialize_species(ca_buffer, 0 * mM)
    cube.add_diffusion(ca_buffer, 0 * u.um**2 / u.s)

    # Add reversible binding reaction: Ca + buffer ↔ complex
    cube.add_reaction(reactants=[ca, buffer], products=[ca_buffer], k_f=kf_b, k_r=kr_b)

    # Add sensor species (non-diffusive)
    sensor = simulation.add_species('sensor', valence=-1)
    cube.add_diffusion(sensor, 0 * u.um**2 / u.s)
    cube.initialize_species(sensor, {'left': 0 * mM, 'right': 0 * mM, 'sphere': sensor_init})

    # Add sensor complex species (non-diffusive)
    ca_sensor = simulation.add_species('ca_sensor', valence=0)
    cube.initialize_species(ca_sensor, 0 * mM)
    cube.add_diffusion(ca_sensor, 0 * u.um**2 / u.s)

    if sensor_active:
        # Add reversible binding reaction: Ca + sensor ↔ complex
        cube.add_reaction(reactants=[ca, sensor], products=[ca_sensor], k_f=kf_s, k_r=kr_s)

    # Simulate
    simulation.run(end_time=1 * u.s, time_step=1 * u.ms, record_interval=100 * u.ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the sensor simulation.")
    parser.add_argument(
        "--buffer_kd",
        type=float,
        default=400e-6,
        help="Buffer dissociation constant in mM.",
    )
    parser.add_argument(
        "--sensor_kd",
        type=float,
        default=420e-6,
        help="Sensor dissociation constant in mM.",
    )

    args = parser.parse_args()

    # Convert command-line arguments to appropriate units
    run_sensor_simulation(
        buffer_kd=args.buffer_kd * mM,
        sensor_kd=args.sensor_kd * mM,
    )
