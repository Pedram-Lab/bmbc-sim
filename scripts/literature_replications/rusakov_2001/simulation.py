"""Rusakov, "The Role of Perisynaptic Glial Sheaths in Glutamate Spillover and
Extracellular Ca2+ Depletion" (2001).

Recreates the Ca-depletion simulation for AP-driven calcium influx (Figure 4, top
row), either presynaptic or postsynaptic:

    uv run scripts/literature_replications/rusakov_2001/simulation.py
    uv run scripts/literature_replications/rusakov_2001/simulation.py --config-name post
    uv run scripts/literature_replications/rusakov_2001/simulation.py geometry.glia_coverage=0.9

Comments name the section or figure each value comes from; "guessed" and "tuned"
mark the ones the paper does not state. ``tune_channels.py`` re-tunes ``m50`` /
``j50`` against the 50%-depletion target.
"""
import math
from typing import Literal

import astropy.constants as const
import astropy.units as u
import numpy as np

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)


class Geometry(ConfigGroup):
    """Synaptic cleft with a partial glial sheath, embedded in neuropil."""

    total_size: Quantity("um") = "5 um"          # guessed
    synapse_radius: Quantity("um") = "0.1 um"    # Fig. 4
    cleft_size: Quantity("nm") = "30 nm"         # Sec. "Ca2 diffusion in a calyx-type synapse"
    glia_distance: Quantity("nm") = "30 nm"      # guessed
    glia_width: Quantity("nm") = "50 nm"         # Sec. "Glial sheath and glutamate transporter density"
    glia_coverage: float = 0.5                   # varied
    mesh_size: Quantity("um") = "0.1 um"


class Physics(ConfigGroup):
    """Calcium in the cleft, the synapse interior and the porous neuropil."""

    ca_resting: Quantity("mM") = "1.3 mM"            # Sec. "Presynaptic calcium influx"
    diffusivity: Quantity("um2 / ms") = "0.4 um2 / ms"  # Fig. 4
    tortuosity: float = 1.4                          # Sec. "Synaptic geometry"
    porosity: float = 0.12                           # Sec. "Synaptic geometry"


class Presynaptic(ConfigGroup):
    """Presynaptic influx: ``m50`` channels, each carrying ``channel_current``."""

    channel_current: Quantity("pA") = "0.5 pA"   # Sec. "Presynaptic calcium influx"
    time_constant: Quantity("1 / ms") = "10 / ms"  # Sec. "Presynaptic calcium influx"
    # Number of open channels, tuned to match 50% depletion in Fig. 4 for the
    # "50% glial coverage" case (see tune_channels.py)
    m50: int = 24


class Postsynaptic(ConfigGroup):
    """Postsynaptic influx: a current density over the postsynaptic membrane."""

    tau_1: Quantity("ms") = "80 ms"   # Sec. "Postsynaptic calcium influx"
    tau_2: Quantity("ms") = "3 ms"    # Sec. "Postsynaptic calcium influx"
    # Tuned to match 50% depletion in Fig. 4 for the "50% glial coverage" case
    j50: Quantity("pA / um2") = "88 pA / um2"


class Config(SimulationConfig):
    """Full config for the Rusakov replication."""

    simulation_name: str = "rusakov"
    # Which side of the cleft the calcium comes from; also picks the default timing.
    mode: Literal["pre", "post"] = "pre"
    geometry: Geometry = Geometry()
    physics: Physics = Physics()
    presynaptic: Presynaptic = Presynaptic()
    postsynaptic: Postsynaptic = Postsynaptic()
    # Timing. None -> the default for `mode` (see MODE_TIMING): the postsynaptic
    # influx lasts ~50x longer than the presynaptic one, so they need different
    # steps and horizons.
    time_step: Quantity("ms") | None = None
    end_time: Quantity("ms") | None = None
    record_interval: Quantity("ms") | None = None


# Per-mode timing defaults: (time_step, end_time, record_interval).
MODE_TIMING = {
    "pre": ("0.2 us", "1.5 ms", "10 us"),
    "post": ("1.0 us", "80 ms", "0.5 ms"),
}


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom, phys = cfg.geometry, cfg.physics
    default_step, default_end, default_record = MODE_TIMING[cfg.mode]
    time_step = cfg.time_step if cfg.time_step is not None else u.Quantity(default_step)
    end_time = cfg.end_time if cfg.end_time is not None else u.Quantity(default_end)
    record_interval = (
        cfg.record_interval if cfg.record_interval is not None else u.Quantity(default_record)
    )

    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.simulation_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    # Create the geometry
    angle = float(np.arccos(1 - 2 * geom.glia_coverage)) * u.rad
    mesh = bmbcsim.create_rusakov_geometry(
        total_size=geom.total_size,
        synapse_radius=geom.synapse_radius,
        cleft_size=geom.cleft_size,
        glia_distance=geom.glia_distance,
        glia_width=geom.glia_width,
        glial_coverage_angle=angle,
        mesh_size=geom.mesh_size,
    )

    # Initialize the simulation and all geometry components
    simulation = bmbcsim.Simulation(mesh=mesh, result_directory=result_dir)
    geo = simulation.simulation_geometry
    synapse_ecs = geo.compartments["synapse_ecs"]
    neuropil = geo.compartments["neuropil"]
    synapse_boundary = geo.membranes["synapse_boundary"]

    if cfg.mode == "pre":
        synapse = geo.compartments["presynapse"]
        synaptic_membrane = geo.membranes["presynaptic_membrane"]
    else:
        synapse = geo.compartments["postsynapse"]
        synaptic_membrane = geo.membranes["postsynaptic_membrane"]

    # Add calcium and diffusion
    ca = simulation.add_species("Ca")
    synapse_ecs.initialize_species(ca, phys.ca_resting)
    neuropil.initialize_species(ca, phys.ca_resting)

    # Add diffusion in different compartments
    synapse_ecs.add_diffusion(ca, phys.diffusivity / phys.tortuosity**2)
    synapse.add_diffusion(ca, phys.diffusivity)
    neuropil.add_diffusion(ca, phys.diffusivity / phys.tortuosity**2)
    neuropil.add_porosity(phys.porosity)

    # Add transport across the neuropil boundary
    t = transport.Transparent(
        source_diffusivity=phys.diffusivity / phys.tortuosity**2,
        target_diffusivity=phys.diffusivity,
    )
    synapse_boundary.add_transport(ca, t, neuropil, synapse_ecs)

    # Add the channel flux (either pre- or post-synaptic)
    faraday = const.e.si * const.N_A
    if cfg.mode == "pre":
        pre = cfg.presynaptic
        flux_amplitude = pre.m50 * pre.channel_current / (2 * faraday)
        spike = lambda t: (t * pre.time_constant) * math.exp(-t * pre.time_constant)
    else:
        post = cfg.postsynaptic
        flux_amplitude = post.j50 * synaptic_membrane.area / (2 * faraday)
        spike = lambda t: math.exp(-t / post.tau_1) - math.exp(-t / post.tau_2)

    flux = transport.GeneralFlux(flux=flux_amplitude, temporal=spike)
    synaptic_membrane.add_transport(ca, flux, synapse_ecs, synapse)

    simulation.run(
        end_time=end_time,
        time_step=time_step,
        record_interval=record_interval,
        n_threads=cfg.n_threads,
    )


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
