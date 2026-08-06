"""Buffered diffusion in an elongated box.

Ca2+ crosses one end face of a long, thin box and diffuses down the long (y)
axis. ``scenario`` selects which of four variants of that experiment to run,
using the Ca + ECM <-> ECM_Ca chemistry from
``scripts/ongoing_work/tissue_kinetics/simulation.py`` for the buffer. Comparing
them shows how buffering slows the apparent speed of diffusion, and that it
slows uptake and release differently.

Run the scenarios, then compare them with ``evaluate.py``:

    uv run scripts/ongoing_work/buffered_diffusion/simulation.py           # nobuffer
    uv run scripts/ongoing_work/buffered_diffusion/simulation.py scenario=depleted
    uv run scripts/ongoing_work/buffered_diffusion/simulation.py scenario=saturated
    uv run scripts/ongoing_work/buffered_diffusion/simulation.py scenario=replenishment

(equivalently ``--config-name <scenario>``, which is what
``scripts/run_all_simulations.sh`` uses).

The box mesh is built directly with netgen.occ (the same approach
``bmbcsim.geometry.create_box_geometry`` uses internally). We do not use
``create_box_geometry`` itself because its single-compartment output uses the
material name "box:top", which the single-region assembly path cannot resolve
(it looks up "top"); building our own mesh with the colon-free material "box"
avoids that and lets us name the source/far end faces explicitly.
"""
from typing import Literal

import astropy.units as u
import ngsolve as ngs
import numpy as np
from netgen import occ

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.units import to_simulation_units
from bmbcsim.config import (
    ConfigGroup,
    Quantity,
    SimulationConfig,
    dump_resolved,
    run_from_cli,
)


class Box(ConfigGroup):
    """Box dimensions (full lengths). The long axis is y."""

    width_x: Quantity("um") = "4.0 um"
    length_y: Quantity("um") = "60.0 um"
    height_z: Quantity("um") = "2.0 um"
    mesh_size: Quantity("um") = "1.0 um"


class Buffer(ConfigGroup):
    """Immobile ECM buffer, present in every scenario but "nobuffer". kf and total
    match the tissue sim; Kd matches the Ca reservoir so the buffer is
    half-saturated near the source."""

    ecm_total: Quantity("mM") = "2.0 mM"
    ecm_kf: Quantity("1 / (mM s)") = "10.0 / (mM s)"
    kd: Quantity("mM") = "1.3 mM"

    @property
    def ecm_kr(self) -> u.Quantity:
        """Reverse rate, derived from ``Kd = ecm_kr / ecm_kf``."""
        return (self.kd * self.ecm_kf).to(1 / u.ms)


class Config(SimulationConfig):
    """Full config for the buffered-diffusion experiment."""

    simulation_name: str = "buffered_diffusion"
    # The experiment's only variant, and its run-name postfix:
    #   nobuffer      -- no buffer; Ca starts at 0 and flows in. The free-diffusion
    #                    reference evaluate.py measures the others against.
    #   depleted      -- buffer present and fully unbound; Ca starts at 0 and flows in.
    #   saturated     -- buffer in equilibrium with Ca at its Kd (so half-bound);
    #                    Ca flows in on top of that baseline.
    #   replenishment -- as saturated, but the flux is exactly reversed: Ca is drawn
    #                    out of the source face and the buffer releases Ca to refill it.
    scenario: Literal["nobuffer", "depleted", "saturated", "replenishment"] = "nobuffer"
    box: Box = Box()
    buffer: Buffer = Buffer()
    ca_source: Quantity("mM") = "1.3 mM"
    diffusivity: Quantity("um2 / ms") = "0.7 um2 / ms"
    # Constant Ca flux across the source face (out of the box for "replenishment", in
    # for every other scenario). A concentration-independent flux
    # (GeneralFlux) is used rather than a fixed-concentration reservoir because
    # membrane transport is integrated explicitly (fem_details.transport_step),
    # so a stiff Robin/Passive reservoir would be numerically unstable. If None,
    # the density is derived so the no-buffer surface concentration reaches
    # ~ca_source by end_time: q = ca_source * sqrt(pi*D) / (2*sqrt(end_time)).
    source_flux_density: Quantity("mM um / ms") | None = None
    # Timing (mirrors the tissue sim)
    end_time: Quantity("s") = "1.0 s"
    time_step: Quantity("s") = "1.0 ms"
    record_interval: Quantity("s") = "10.0 ms"

    @property
    def has_buffer(self) -> bool:
        """Whether the ECM buffer species exist at all."""
        return self.scenario != "nobuffer"

    @property
    def initial_ca(self) -> u.Quantity:
        """Uniform initial [Ca]. The scenarios that start in buffer equilibrium sit at
        the buffer's Kd (half-bound); the others start empty."""
        equilibrated = self.scenario in ("saturated", "replenishment")
        return self.buffer.kd if equilibrated else 0.0 * u.mmol / u.L

    def derived_postfix(self) -> str:
        """The scenario *is* the variant: evaluate.py compares the runs."""
        return self.scenario


def make_box_mesh(box: Box) -> ngs.Mesh:
    """Build an elongated box mesh, source face at y=0, far face at y=Ly.

    One compartment "box"; exterior faces named "source" (y=0), "far" (y=Ly)
    and "side" (the four long faces).
    """
    lx = to_simulation_units(box.width_x, "length")
    ly = to_simulation_units(box.length_y, "length")
    lz = to_simulation_units(box.height_z, "length")

    occ_box = occ.Box(occ.Pnt(-lx / 2, 0, 0), occ.Pnt(lx / 2, ly, lz))
    occ_box.mat("box")
    # occ.Box face order: 0:x-min 1:x-max 2:y-min 3:y-max 4:z-min 5:z-max
    occ_box.faces[2].bc("source")
    occ_box.faces[3].bc("far")
    for f in (0, 1, 4, 5):
        occ_box.faces[f].bc("side")

    geo = occ.OCCGeometry(occ_box)
    return ngs.Mesh(geo.GenerateMesh(maxh=to_simulation_units(box.mesh_size, "length")))


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    buffer_cfg = cfg.buffer
    initial_ca = cfg.initial_ca
    efflux = cfg.scenario == "replenishment"
    print(f"=== Running {cfg.run_name} ===")

    source_flux_density = cfg.source_flux_density
    if source_flux_density is None:
        # Surface concentration of constant-flux diffusion: C(0,t) = 2 q sqrt(t)
        # / sqrt(pi D). Choose q so C(0, end_time) ~ ca_source.
        source_flux_density = (
            cfg.ca_source * np.sqrt(np.pi * cfg.diffusivity) / (2 * np.sqrt(cfg.end_time))
        ).to((u.mmol / u.L) * u.um / u.ms)
    print(f"  Source flux density: {source_flux_density:.4g}"
          f"{' (drawn out)' if efflux else ''}")

    if efflux:
        # GeneralFlux is concentration-independent, so too strong an efflux keeps
        # drawing Ca out after the source face is empty, driving [Ca] negative. The
        # worst case is the same drawdown with no buffer to replenish it:
        # C(0,t) = initial_ca - 2 q sqrt(t) / sqrt(pi D).
        drawdown = (
            2 * source_flux_density * np.sqrt(cfg.end_time) / np.sqrt(np.pi * cfg.diffusivity)
        ).to(u.mmol / u.L)
        # The derived default reaches ca_source == kd == initial_ca exactly at
        # end_time, i.e. touches zero at the last step; hence the float slack.
        assert drawdown <= 1.000001 * initial_ca, (
            f"efflux would draw the source face down by {drawdown:.4g} from its initial "
            f"{initial_ca:.4g}, i.e. below zero: lower source_flux_density or end_time"
        )

    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.run_name)
    dump_resolved(cfg, result_dir)
    print(f"  Results and config -> {result_dir}")

    # ================================================================
    # 1) Geometry and simulation
    # ================================================================
    mesh = make_box_mesh(cfg.box)
    print(f"  Mesh has {mesh.ne} elements and {mesh.nv} vertices")

    sim = bmbcsim.Simulation(mesh=mesh, result_directory=result_dir)
    box = sim.simulation_geometry.compartments["box"]
    print(f"  Box volume: {box.volume:.1f}")

    # ================================================================
    # 2) Species, initialization and diffusion
    # ================================================================
    ca = sim.add_species("Ca")
    box.initialize_species(ca, initial_ca)
    box.add_diffusion(ca, cfg.diffusivity)
    print(f"  Initial [Ca]: {initial_ca:.4g}")

    if cfg.has_buffer:
        # The buffer starts in equilibrium with initial_ca, i.e. Langmuir-bound:
        # ECM_Ca = ecm_total * Ca / (Ca + Kd). "depleted" (Ca = 0) therefore starts
        # fully unbound, "saturated"/"replenishment" (Ca = Kd) half-bound.
        # No add_diffusion -> immobile buffer.
        bound = buffer_cfg.ecm_total * initial_ca / (initial_ca + buffer_cfg.kd)
        print(
            f"  Buffer: total={buffer_cfg.ecm_total}, kf={buffer_cfg.ecm_kf}, "
            f"kr={buffer_cfg.ecm_kr:.4g}, Kd={buffer_cfg.kd} (matches reservoir), "
            f"initially bound={bound:.4g}"
        )

        ecm = sim.add_species("ECM")
        ecm_ca = sim.add_species("ECM_Ca")
        box.initialize_species(ecm, buffer_cfg.ecm_total - bound)
        box.initialize_species(ecm_ca, bound)
        box.add_reaction(
            reactants=[ca, ecm],
            products=[ecm_ca],
            k_f=buffer_cfg.ecm_kf,
            k_r=buffer_cfg.ecm_kr,
        )

    # ================================================================
    # 3) Constant Ca flux across the source end ("source", y = 0)
    # ================================================================
    source = sim.simulation_geometry.membranes["source"]
    source_flux = transport.GeneralFlux(source_flux_density * source.area)
    if efflux:
        source.add_transport(ca, source_flux, box, None)  # Ca is drawn out of the box
    else:
        source.add_transport(ca, source_flux, None, box)

    # ================================================================
    # 4) Run
    # ================================================================
    sim.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval,
        n_threads=cfg.n_threads,
    )
    print(f"=== Done ({cfg.run_name}) ===\n")


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
