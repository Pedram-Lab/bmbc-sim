"""Buffered diffusion in an elongated box.

Ca2+ enters from one end of a long, thin box and diffuses down the long (y)
axis. The same experiment runs once without a buffer and once with an immobile
buffer using the Ca + ECM <-> ECM_Ca chemistry from
``scripts/ongoing_work/tissue_kinetics/simulation.py``. Comparing the two shows
how buffering slows the apparent speed of diffusion.

Run both conditions, then analyze them with ``evaluate.py``:

    uv run scripts/ongoing_work/buffered_diffusion/simulation.py
    uv run scripts/ongoing_work/buffered_diffusion/simulation.py --config-name buffer

The box mesh is built directly with netgen.occ (the same approach
``bmbcsim.geometry.create_box_geometry`` uses internally). We do not use
``create_box_geometry`` itself because its single-compartment output uses the
material name "box:top", which the single-region assembly path cannot resolve
(it looks up "top"); building our own mesh with the colon-free material "box"
avoids that and lets us name the source/far end faces explicitly.
"""
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
    """Immobile ECM buffer. kf and total match the tissue sim; Kd matches the Ca
    reservoir so the buffer is half-saturated near the source."""

    enabled: bool = False
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
    box: Box = Box()
    buffer: Buffer = Buffer()
    ca_source: Quantity("mM") = "1.3 mM"
    diffusivity: Quantity("um2 / ms") = "0.7 um2 / ms"
    # Constant Ca influx at the source face. A concentration-independent flux
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
    # One result directory per condition, so evaluate.py can find both by name.
    label = "buffer" if buffer_cfg.enabled else "nobuffer"
    name = f"{cfg.simulation_name}_{label}"
    print(f"=== Running buffered_diffusion ({label}) ===")

    source_flux_density = cfg.source_flux_density
    if source_flux_density is None:
        # Surface concentration of constant-flux diffusion: C(0,t) = 2 q sqrt(t)
        # / sqrt(pi D). Choose q so C(0, end_time) ~ ca_source.
        source_flux_density = (
            cfg.ca_source * np.sqrt(np.pi * cfg.diffusivity) / (2 * np.sqrt(cfg.end_time))
        ).to((u.mmol / u.L) * u.um / u.ms)
    print(f"  Source flux density: {source_flux_density:.4g}")

    result_dir = bmbcsim.timestamped_directory(cfg.result_root, name)
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
    box.initialize_species(ca, 0.0 * u.mmol / u.L)
    box.add_diffusion(ca, cfg.diffusivity)

    if buffer_cfg.enabled:
        print(
            f"  Buffer: total={buffer_cfg.ecm_total}, kf={buffer_cfg.ecm_kf}, "
            f"kr={buffer_cfg.ecm_kr:.4g}, Kd={buffer_cfg.kd} (matches reservoir)"
        )

        # Ca starts at 0, so the buffer starts fully unbound at equilibrium:
        # ECM = ecm_total, ECM_Ca = 0. No add_diffusion -> immobile buffer.
        ecm = sim.add_species("ECM")
        ecm_ca = sim.add_species("ECM_Ca")
        box.initialize_species(ecm, buffer_cfg.ecm_total)
        box.initialize_species(ecm_ca, 0.0 * u.mmol / u.L)
        box.add_reaction(
            reactants=[ca, ecm],
            products=[ecm_ca],
            k_f=buffer_cfg.ecm_kf,
            k_r=buffer_cfg.ecm_kr,
        )

    # ================================================================
    # 3) Constant Ca influx at the source end ("source", y = 0)
    # ================================================================
    source = sim.simulation_geometry.membranes["source"]
    source_flux = transport.GeneralFlux(source_flux_density * source.area)
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
    print(f"=== Done ({label}) ===\n")


if __name__ == "__main__":
    run_from_cli(Config, run, __file__)
