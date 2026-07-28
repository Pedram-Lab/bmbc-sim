"""Sala & Hernández-Cruz, "Calcium diffusion modeling in a spherical neuron.
Relevance of buffering properties" (1990).

Calcium enters a spherical neuron through voltage-gated and leak channels and
interacts with three cytosolic buffers:

* B1 - intrinsic natural buffers (mobile, fast, low capacity)
* B2 - mimics the endoplasmic reticulum (immobile, slow, high capacity)
* B3 - calcium sensor (mobile, fast, high affinity)

Calcium is removed again by SERCA pumps and the sodium-calcium exchanger (NCX),
both lumped into one Michaelis-Menten extrusion term.

The ``Config`` below is the single source of truth for parameters, defaults and
units; ``run`` reads straight off it. Hydra composes the YAML in ``configs/``:

    uv run scripts/literature_replications/sala_1990/simulation.py
    uv run scripts/literature_replications/sala_1990/simulation.py --config-name quick
    uv run scripts/literature_replications/sala_1990/simulation.py b2.total="300 uM"
"""
import astropy.units as u

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)

# Faraday constant (charge per mole); 2 electrons per Ca2+.
FARADAY = 96485.3321 * u.s * u.A / u.mol


class Buffer(ConfigGroup):
    """One cytosolic Ca2+ buffer, parametrized by its affinity Kd = kr / kf.

    Kd is what the paper tabulates, so kr is the derived quantity. A zero
    diffusivity means the buffer (and its complex) is immobile.
    """

    total: Quantity("uM")
    kd: Quantity("uM")
    kf: Quantity("1 / (M s)")
    diffusivity: Quantity("cm2 / s") = "0 cm2 / s"

    @property
    def kr(self) -> u.Quantity:
        """Reverse rate, derived from ``Kd = kr / kf``."""
        return self.kf * self.kd


class Config(SimulationConfig):
    """Full config for the Sala & Hernández-Cruz replication."""

    simulation_name: str = "sala"
    # Geometry
    radius: Quantity("um") = "20 um"
    mesh_size: Quantity("um") = "2 um"
    # Calcium
    ca_init: Quantity("uM") = "0.05 uM"
    ca_diffusivity: Quantity("cm2 / s") = "6e-6 cm2 / s"
    # Buffers (all initialized at equilibrium with ca_init)
    b1: Buffer = Buffer(total="100 uM", kd="1 uM", kf="1e8 / (M s)",
                        diffusivity="0.5e-6 cm2 / s")
    b2: Buffer = Buffer(total="600 uM", kd="0.4 uM", kf="5e5 / (M s)")  # immobile
    b3: Buffer = Buffer(total="100 uM", kd="0.2 uM", kf="1e8 / (M s)",
                        diffusivity="2.5e-6 cm2 / s")
    # Michaelis-Menten extrusion (SERCA + NCX lumped together)
    u_max: Quantity("pmol / (cm2 s)") = "2 pmol / (cm2 s)"
    km: Quantity("uM") = "0.83 uM"
    # Voltage-gated influx: a step of `channel_current` switched off at `t_off`
    channel_current: Quantity("nA") = "5 nA"
    t_off: Quantity("ms") = "100 ms"
    # Timing
    end_time: Quantity("s") = "2 s"
    time_step: Quantity("s") = "0.1 ms"
    record_interval: Quantity("s") = "1 ms"


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.run_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    # Create a spherical cell
    mesh = bmbcsim.geometry.create_sphere_geometry(
        radius=cfg.radius, mesh_size=cfg.mesh_size
    )

    simulation = bmbcsim.Simulation(mesh, result_directory=result_dir)
    geometry = simulation.simulation_geometry
    cell = geometry.compartments["sphere"]
    membrane = geometry.membranes["boundary"]

    # Calcium
    ca = simulation.add_species("Ca", valence=2)
    cell.initialize_species(ca, value=cfg.ca_init)
    cell.add_diffusion(species=ca, diffusivity=cfg.ca_diffusivity)

    # Buffers: free buffer + complex start at equilibrium with ca_init, and each
    # pair binds Ca via Ca + B <-> CaB. Species with zero diffusivity get no
    # diffusion term at all (B2 mimics the immobile ER).
    for name, buffer in (("Buffer_1", cfg.b1), ("Buffer_2", cfg.b2), ("Buffer_3", cfg.b3)):
        complex_0 = (cfg.ca_init * buffer.total) / (cfg.ca_init + buffer.kd)

        free = simulation.add_species(name, valence=-2)
        bound = simulation.add_species(f"Ca_{name}", valence=0)
        cell.initialize_species(free, value=buffer.total - complex_0)
        cell.initialize_species(bound, value=complex_0)

        if buffer.diffusivity != 0:
            cell.add_diffusion(species=free, diffusivity=buffer.diffusivity)
            cell.add_diffusion(species=bound, diffusivity=buffer.diffusivity)

        cell.add_reaction(
            reactants=[ca, free], products=[bound], k_f=buffer.kf, k_r=buffer.kr
        )

    # Transport across the cell membrane
    v_max = cfg.u_max * membrane.area
    channel_current = cfg.channel_current / (2 * FARADAY)
    spike = lambda t: 1.0 if t < cfg.t_off else 0.0

    # Time-dependent influx through voltage-gated channels
    membrane.add_transport(
        species=ca,
        transport=transport.GeneralFlux(flux=channel_current, temporal=spike),
        source=None,
        target=cell,
    )
    # MM-type extrusion out of the cell (depends on the inside concentration)
    membrane.add_transport(
        species=ca,
        transport=transport.Active(v_max=v_max, km=cfg.km),
        source=cell,
        target=None,
    )
    # MM-type leak into the cell. The outside concentration is fixed at ca_init,
    # so this balances the extrusion term exactly at rest.
    leak_flux = v_max * cfg.ca_init / (cfg.km + cfg.ca_init)
    membrane.add_transport(
        species=ca,
        transport=transport.GeneralFlux(leak_flux),
        source=None,
        target=cell,
    )

    simulation.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval,
        n_threads=cfg.n_threads,
    )


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
