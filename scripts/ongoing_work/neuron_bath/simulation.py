"""Neuron-bath simulation: one spherical cell taking up Ca2+ from a large bath.

The bath is the tissue-kinetics box without the tissue: a single cell in its center
absorbs Ca2+ through its whole membrane during an action-potential train, with the
tissue-kinetics channel law (influx linear in the Ca2+ at the membrane, zero at
``uptake.depletion``, full at ``uptake.saturation``). All other parameters are the
tissue-kinetics defaults.

    uv run scripts/ongoing_work/neuron_bath/simulation.py diffusion.ca_ecs="1.2 mM"
"""
import math

import numpy as np
from astropy import constants as const
from astropy import units as u
from netgen import occ
import ngsolve as ngs

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.config import (
    SimulationConfig,
    ConfigGroup,
    Quantity,
    dump_resolved,
    run_from_cli,
)


class Geometry(ConfigGroup):
    """Box-shaped bath with a spherical cell in its center."""

    box: list[Quantity("um")] = ["14 um", "14 um", "7 um"]
    cell_diameter: Quantity("um") = "2 um"
    mesh_size: Quantity("um") = "1 um"
    cell_mesh_size: Quantity("um") = "0.2 um"   # on the membrane and inside the cell


class Diffusion(ConfigGroup):
    """Bath/cytosol Ca2+ diffusion and the reservoir boundary condition."""

    ca_ecs: Quantity("mM") = "2.0 mM"
    diffusivity_ecs: Quantity("um2 / ms") = "0.7 um2 / ms"
    diffusivity_cyto: Quantity("um2 / ms") = "0.22 um2 / ms"
    tortuosity: float = 1.6
    boundary_permeability: Quantity("um / ms") | None = None  # from tortuosity if None


class Uptake(ConfigGroup):
    """Ca2+ influx into the cell: NMDAR-like channels, spread over the whole membrane."""

    # The tissue has about one active synapse per 10 um2 of membrane, i.e. about one
    # on this cell (12.6 um2).
    n_synapses: int = 1
    n_channels_per_synapse: int = 35
    i_channel: Quantity("pA") = "0.5 pA"
    tau1: Quantity("ms") = "10 ms"
    tau2: Quantity("ms") = "3 ms"
    # Influx law (ProportionalFlux): zero at bath Ca <= depletion, full at >= saturation,
    # linear in between. The same law for every bath concentration.
    depletion: Quantity("mM") = "0.47 mM"
    saturation: Quantity("mM") = "2.0 mM"
    # Action-potential train
    first_pulse: Quantity("ms") = "1 s"
    n_pulses: int = 20
    frequency: Quantity("Hz") = "20 Hz"
    # Linear extrusion: cell Ca decays as exp(-t / extrusion_time). Not from the tissue
    # kinetics (which has no extrusion); fit to the decay of the measured traces.
    extrusion_time: Quantity("ms") = "500 ms"


class Config(SimulationConfig):
    """Full config for the neuron-bath simulation."""

    simulation_name: str = "neuron_bath"
    end_time: Quantity("s") = "4 s"
    time_step: Quantity("s") = "1 ms"
    record_interval_factor: int = 10      # 10 ms records
    geometry: Geometry = Geometry()
    diffusion: Diffusion = Diffusion()
    uptake: Uptake = Uptake()


def create_mesh(geom: Geometry) -> ngs.Mesh:
    """Box [0, box] (material "ecs", boundary "wall") around a sphere ("cell", "membrane")."""
    box_size = [s.to_value(u.um) for s in geom.box]
    box = occ.Box((0, 0, 0), tuple(box_size))
    box.bc("wall")
    cell = occ.Sphere(tuple(s / 2 for s in box_size), geom.cell_diameter.to_value(u.um) / 2)
    cell.mat("cell")
    cell.bc("membrane")
    cell.maxh = geom.cell_mesh_size.to_value(u.um)
    ecs = box - cell
    ecs.mat("ecs")
    geo = occ.OCCGeometry(occ.Glue([ecs, cell]))
    return ngs.Mesh(geo.GenerateMesh(maxh=geom.mesh_size.to_value(u.um)))


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom, diff, upt = cfg.geometry, cfg.diffusion, cfg.uptake
    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.run_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    mesh = create_mesh(geom)
    print(f"  Mesh has {mesh.ne} elements and {mesh.nv} vertices")
    sim = bmbcsim.Simulation(mesh=mesh, result_directory=result_dir)
    geo = sim.simulation_geometry
    ecs, cell = geo.compartments["ecs"], geo.compartments["cell"]
    membrane, wall = geo.membranes["membrane"], geo.membranes["wall"]
    print(f"  Bath volume: {ecs.volume:.2f}, cell volume: {cell.volume:.3f}, "
          f"membrane area: {membrane.area:.3f}")

    ca = sim.add_species("Ca")
    ecs.initialize_species(ca, diff.ca_ecs)
    cell.initialize_species(ca, 0.0 * u.mmol / u.L)
    ecs.add_diffusion(ca, diff.diffusivity_ecs)
    cell.add_diffusion(ca, diff.diffusivity_cyto)

    # --- Ca2+ uptake, uniform over the membrane: Q = N * I / (2 F)  (factor 2 for Ca2+) ---
    faraday = const.e.si * const.N_A
    q_max = upt.n_synapses * upt.n_channels_per_synapse * upt.i_channel / (2 * faraday)
    pulse_times = upt.first_pulse + np.arange(upt.n_pulses) / upt.frequency

    def nmdar_waveform(t):
        """J(t) = e^(-t/tau1) - e^(-t/tau2), superposition of pulses."""
        dts = [t - t_pulse for t_pulse in pulse_times if t >= t_pulse]
        return sum(math.exp(-dt / upt.tau1) - math.exp(-dt / upt.tau2) for dt in dts)

    uptake = transport.ProportionalFlux(
        flux=q_max,
        saturation=upt.saturation,
        depletion=upt.depletion,
        temporal=nmdar_waveform,
    )
    membrane.add_transport(ca, uptake, ecs, cell)
    # ponytail: extruded Ca is removed instead of returned to the bath (< 0.01 mM there);
    # needs a one-sided linear transport class to send it back.
    extrusion = transport.Passive(cell.volume / upt.extrusion_time, 0.0 * u.mmol / u.L)
    membrane.add_transport(ca, extrusion, cell, None)

    # --- Robin BC: transport from external reservoir into the bath ---
    boundary_permeability = diff.boundary_permeability
    if boundary_permeability is None:
        l_char = max(geom.box) / 2.0
        boundary_permeability = diff.diffusivity_ecs / diff.tortuosity**2 / l_char
    wall.add_transport(
        ca, transport.Passive(boundary_permeability * wall.area, diff.ca_ecs), None, ecs
    )

    print("Running simulation...")
    sim.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval_factor * cfg.time_step,
        n_threads=cfg.n_threads,
    )
    print("Simulation complete.")


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
