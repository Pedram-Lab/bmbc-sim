"""A Ca2+ sensor competing with a buffer for calcium.

A cube whose right half is filled with an immobile buffer, plus a spherical
region holding an immobile sensor. The sphere sits either in the buffered half
(default) or in the clean half, which is what ``configs/sensor_left.yaml``
switches. Comparing the two shows how much the buffer distorts what the sensor
reports. Default values are taken from [Sala, Hernández-Cruz; 1990].

    uv run scripts/ongoing_work/sensor_simulation/simulation.py
    uv run scripts/ongoing_work/sensor_simulation/simulation.py --config-name sensor_left
    uv run scripts/ongoing_work/sensor_simulation/simulation.py buffer.kd="1 uM"

``sweep.py`` fans the same Config/run pair out over a grid of affinities.
"""
import astropy.units as u

import bmbcsim
from bmbcsim.geometry import create_sensor_geometry
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)


class Geometry(ConfigGroup):
    """Cube split into a "left" and "right" half, with a sphere inside one of them."""

    side_length: Quantity("um") = "200 um"
    compartment_ratio: float = 0.5
    # None -> centered in whichever half the sensor is in (see run()).
    sphere_position_x: Quantity("um") | None = None
    sphere_radius: Quantity("um") = "30 um"
    mesh_size: Quantity("um") = "5 um"


class Binder(ConfigGroup):
    """A Ca2+-binding species (buffer or sensor), parametrized by its affinity."""

    initial: Quantity("uM")
    kd: Quantity("nM")
    kf: Quantity("1 / (mM s)")

    @property
    def kr(self) -> u.Quantity:
        """Reverse rate, derived from ``Kd = kr / kf``."""
        return self.kf * self.kd


class Config(SimulationConfig):
    """Full config for the sensor/buffer competition in a cube."""

    simulation_name: str = "sensor"
    # Whether the sensor binds Ca at all; False is the "unperturbed" control.
    sensor_active: bool = True
    # Which half of the cube the sensor sphere sits in.
    sensor_left: bool = False
    geometry: Geometry = Geometry()
    # Calcium
    ca_init: Quantity("uM") = "0.05 uM"
    ca_diffusivity: Quantity("um2 / s") = "600 um2 / s"
    # Binders (both immobile)
    buffer: Binder = Binder(initial="600 uM", kd="400 nM", kf="50 / (mM s)")
    sensor: Binder = Binder(initial="100 uM", kd="420 nM", kf="100 / (uM s)")
    # Timing
    end_time: Quantity("s") = "1 s"
    time_step: Quantity("s") = "1 ms"
    record_interval: Quantity("s") = "100 ms"


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom = cfg.geometry
    immobile = 0 * u.um**2 / u.s
    zero = 0 * u.mmol / u.L

    # Center the sphere in whichever half the sensor is meant to be in.
    sphere_position_x = geom.sphere_position_x
    if sphere_position_x is None:
        sphere_position_x = geom.side_length * (0.25 if cfg.sensor_left else 0.75)

    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.simulation_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    mesh = create_sensor_geometry(
        side_length=geom.side_length,
        compartment_ratio=geom.compartment_ratio,
        sphere_position_x=sphere_position_x,
        sphere_radius=geom.sphere_radius,
        mesh_size=geom.mesh_size,
    )

    simulation = bmbcsim.Simulation(mesh, result_directory=result_dir)
    cube = simulation.simulation_geometry.compartments["cube"]

    # Calcium: the only diffusing species
    ca = simulation.add_species("ca", valence=2)
    cube.initialize_species(ca, cfg.ca_init)
    cube.add_diffusion(ca, cfg.ca_diffusivity)

    # Buffer, in the right half only. The sphere counts as buffered unless the
    # sensor was moved into the clean (left) half.
    buffer = simulation.add_species("buffer", valence=-1)
    cube.add_diffusion(buffer, immobile)
    cube.initialize_species(buffer, {
        "left": zero,
        "right": cfg.buffer.initial,
        "sphere": zero if cfg.sensor_left else cfg.buffer.initial,
    })

    ca_buffer = simulation.add_species("ca_buffer", valence=0)
    cube.initialize_species(ca_buffer, zero)
    cube.add_diffusion(ca_buffer, immobile)
    cube.add_reaction(
        reactants=[ca, buffer], products=[ca_buffer],
        k_f=cfg.buffer.kf, k_r=cfg.buffer.kr,
    )

    # Sensor, in the sphere only
    sensor = simulation.add_species("sensor", valence=-1)
    cube.add_diffusion(sensor, immobile)
    cube.initialize_species(
        sensor, {"left": zero, "right": zero, "sphere": cfg.sensor.initial}
    )

    ca_sensor = simulation.add_species("ca_sensor", valence=0)
    cube.initialize_species(ca_sensor, zero)
    cube.add_diffusion(ca_sensor, immobile)
    if cfg.sensor_active:
        cube.add_reaction(
            reactants=[ca, sensor], products=[ca_sensor],
            k_f=cfg.sensor.kf, k_r=cfg.sensor.kr,
        )

    simulation.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval,
        n_threads=cfg.n_threads,
    )


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
