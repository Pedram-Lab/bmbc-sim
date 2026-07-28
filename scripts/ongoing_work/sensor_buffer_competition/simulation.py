"""Sensor/buffer competition for calcium across a substrate interface.

Two-region box (top / bottom). A mobile buffer sits in the bottom region only, a
mobile sensor everywhere, and calcium crosses the interface once transport
switches on. The question is how badly the buffer distorts what the sensor
reports, as a function of the buffer's concentration and affinity.

    uv run scripts/ongoing_work/sensor_buffer_competition/simulation.py
    uv run scripts/ongoing_work/sensor_buffer_competition/simulation.py buffer.concentration="1 mM"

``sweep.py`` scans concentration x Kd; ``plot_parameter_sweep_heatmaps.py`` turns
that sweep into heatmaps.
"""
import astropy.units as u

import bmbcsim
from bmbcsim.geometry import create_box_geometry
from bmbcsim.simulation import transport
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)


class Geometry(ConfigGroup):
    """Box split into a "bottom" (substrate) and "top" compartment."""

    sidelength: Quantity("um") = "0.5 um"
    cube_height: Quantity("um") = "1 um"
    substrate_height: Quantity("um") = "0.5 um"
    mesh_size_factor: float = 20  # mesh size = sidelength / this


class Binder(ConfigGroup):
    """A Ca2+-binding species, parametrized by its affinity Kd = kr / kf."""

    concentration: Quantity("mM")
    kd: Quantity("mM")
    kf: Quantity("1 / (M s)") = "1.0e8 / (M s)"
    diffusivity: Quantity("cm2 / s") = "2.5e-6 cm2 / s"

    @property
    def kr(self) -> u.Quantity:
        """Reverse rate, derived from ``Kd = kr / kf``."""
        return self.kf * self.kd


class Config(SimulationConfig):
    """Full config for the sensor/buffer competition experiment."""

    simulation_name: str = "sensor_buffer_competition"
    geometry: Geometry = Geometry()
    ca_free: Quantity("mM") = "1 mM"
    # Buffer: bottom region only. Concentration and Kd are the swept axes.
    buffer: Binder = Binder(concentration="1 mM", kd="1 mM")
    sensor: Binder = Binder(concentration="10 uM", kd="1.0 mM")
    # Transport across the interface, switched on at `transport_onset`
    interface_permeability: Quantity("um3 / ms") = "10 um3 / ms"
    transport_onset: Quantity("ms") = "1 ms"
    # Timing
    end_time: Quantity("ms") = "4 ms"
    time_step: Quantity("ms") = "5 us"
    record_interval: Quantity("ms") = "100 us"


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom = cfg.geometry
    zero = 0 * u.mmol / u.L

    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.simulation_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    mesh = create_box_geometry(
        dimensions=(geom.sidelength, geom.sidelength, geom.cube_height),
        mesh_size=geom.sidelength / geom.mesh_size_factor,
        split=geom.substrate_height,
        compartments=True,
    )

    simulation = bmbcsim.Simulation(mesh, result_directory=result_dir)
    geometry = simulation.simulation_geometry
    compartments = geometry.compartments
    interface = geometry.membranes["interface"]

    # Calcium, present everywhere at the same concentration
    ca = simulation.add_species("ca", valence=2)
    for comp in compartments.values():
        comp.initialize_species(ca, cfg.ca_free)
        comp.add_diffusion(ca, 600 * u.um**2 / u.s)

    # Buffer (bottom only) and sensor (everywhere), both mobile and both binding
    # Ca via X + Ca <-> X_complex.
    for label, binder, initial in (
        ("buffer", cfg.buffer, {"top": zero, "bottom": cfg.buffer.concentration}),
        ("sensor", cfg.sensor, None),
    ):
        free = simulation.add_species(label)
        complex_species = simulation.add_species(f"{label}_complex")
        for name, comp in compartments.items():
            comp.add_diffusion(free, binder.diffusivity)
            comp.initialize_species(
                free, initial[name] if initial is not None else binder.concentration
            )
            comp.add_diffusion(complex_species, binder.diffusivity)
            comp.initialize_species(complex_species, zero)
            comp.add_reaction(
                reactants=[ca, free],
                products=[complex_species],
                k_f=binder.kf,
                k_r=binder.kr,
            )

    # Transport across the interface, off until `transport_onset`
    spike = lambda t: 0.0 if t < cfg.transport_onset else 1.0
    interface.add_transport(
        species=ca,
        transport=transport.Passive(
            permeability=cfg.interface_permeability, temporal=spike
        ),
        source=compartments["top"],
        target=compartments["bottom"],
    )

    simulation.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval,
        n_threads=cfg.n_threads,
    )


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
