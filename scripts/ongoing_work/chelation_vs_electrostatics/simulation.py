"""Chelation vs. electrostatics as mechanisms of Ca2+ redistribution.

Three species in a two-region box: an immobile buffer B1 (bottom half only), a
mobile buffer B2 (everywhere) and diffusing Ca. Either mechanism can be switched
off, which is what the three configs in ``configs/`` do:

    uv run scripts/ongoing_work/chelation_vs_electrostatics/simulation.py          # both
    uv run scripts/ongoing_work/chelation_vs_electrostatics/simulation.py --config-name chelation
    uv run scripts/ongoing_work/chelation_vs_electrostatics/simulation.py --config-name electrostatics

The result directory is named after the enabled mechanisms (see
:meth:`Config.derived_postfix`), which is how ``visualization.py`` finds it.
"""
import astropy.units as u

import bmbcsim
import bmbcsim.geometry as geo
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)


class Geometry(ConfigGroup):
    """Box split into a "top" and a "bottom" region at height ``split``."""

    sidelength: Quantity("um") = "0.5 um"
    height: Quantity("um") = "1 um"
    split: Quantity("um") = "0.5 um"
    mesh_size_factor: float = 20  # mesh size = sidelength / this


class Buffer(ConfigGroup):
    """A Ca2+ buffer, parametrized by its affinity Kd = kr / kf."""

    total: Quantity("mM")
    kd: Quantity("uM")
    kf: Quantity("1 / (M s)")
    diffusivity: Quantity("um2 / s")

    @property
    def kr(self) -> u.Quantity:
        """Reverse rate, derived from ``Kd = kr / kf``."""
        return self.kf * self.kd


class Config(SimulationConfig):
    """Full config for the chelation/electrostatics comparison."""

    simulation_name: str = "chelation_vs_electrostatics"
    # Mechanisms. Both off is a valid control run (pure diffusion).
    electrostatics: bool = True
    chelation: bool = True
    geometry: Geometry = Geometry()
    relative_permittivity: float = 80
    # Calcium
    total_ca: Quantity("mM") = "1 mM"
    ca_diffusivity: Quantity("um2 / s") = "600 um2 / s"
    # Buffers: the immobile one starts in the bottom region only, the mobile one
    # is spread evenly.
    immobile_buffer: Buffer = Buffer(
        total="1.0 mM", kd="10.0 uM", kf="1e8 / (M s)", diffusivity="0 um2 / s"
    )
    mobile_buffer: Buffer = Buffer(
        total="0.5 mM", kd="10.0 uM", kf="1e8 / (M s)", diffusivity="50 um2 / s"
    )
    # Timing
    end_time: Quantity("ms") = "4 ms"
    time_step: Quantity("ms") = "1 us"
    record_interval: Quantity("ms") = "100 us"

    def derived_postfix(self) -> str:
        """Name the run after the mechanisms it has switched on -- they *are* the
        variant. ``visualization.py`` gets the same name from ``Config.run_name``,
        so this is the one place the naming is defined.
        """
        parts = [
            name for name, enabled in
            (("chelation", self.chelation), ("electrostatics", self.electrostatics))
            if enabled
        ]
        return "_".join(parts or ["no_interaction"])


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom = cfg.geometry
    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.run_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    mesh = geo.create_box_geometry(
        dimensions=(geom.sidelength, geom.sidelength, geom.height),
        mesh_size=geom.sidelength / geom.mesh_size_factor,
        split=geom.split,
    )

    simulation = bmbcsim.Simulation(
        mesh, result_directory=result_dir, electrostatics=cfg.electrostatics
    )
    box = simulation.simulation_geometry.compartments["box"]

    if cfg.electrostatics:
        box.add_relative_permittivity(cfg.relative_permittivity)

    # Add Ca species
    ca = simulation.add_species("ca", valence=2)
    box.initialize_species(ca, cfg.total_ca)
    box.add_diffusion(ca, cfg.ca_diffusivity)

    # Buffers. The immobile one is confined to the bottom region; without
    # chelation neither binds Ca, and only their charge matters.
    immobile = simulation.add_species("immobile_buffer", valence=-2)
    box.add_diffusion(immobile, cfg.immobile_buffer.diffusivity)
    box.initialize_species(
        immobile, {"top": 0 * u.mmol / u.L, "bottom": cfg.immobile_buffer.total}
    )

    mobile = simulation.add_species("mobile_buffer", valence=-2)
    box.add_diffusion(mobile, cfg.mobile_buffer.diffusivity)
    box.initialize_species(mobile, cfg.mobile_buffer.total)

    if cfg.chelation:
        for name, free, buffer in (
            ("immobile_complex", immobile, cfg.immobile_buffer),
            ("mobile_complex", mobile, cfg.mobile_buffer),
        ):
            complex_species = simulation.add_species(name, valence=0)
            box.initialize_species(complex_species, 0 * u.mmol / u.L)
            box.add_diffusion(complex_species, buffer.diffusivity)
            box.add_reaction(
                reactants=[ca, free],
                products=[complex_species],
                k_f=buffer.kf,
                k_r=buffer.kr,
            )

    simulation.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval,
        n_threads=cfg.n_threads,
    )


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
