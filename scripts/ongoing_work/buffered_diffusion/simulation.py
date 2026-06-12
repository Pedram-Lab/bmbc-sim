"""Buffered diffusion in an elongated box.

Ca2+ enters from one end of a long, thin box held at a fixed reservoir
concentration and diffuses down the long (y) axis. The same experiment is run
once without a buffer and once with an immobile buffer using the Ca + ECM <->
ECM_Ca chemistry from ``scripts/ongoing_work/tissue_kinetics/simulation.py``.
Comparing the two shows how buffering slows the apparent speed of diffusion.

Run this script to produce both result folders, then analyze them with
``evaluate.py``.

The box mesh is built directly with netgen.occ (the same approach
``bmbcsim.geometry.create_box_geometry`` uses internally). We do not use
``create_box_geometry`` itself because its single-compartment output uses the
material name "box:top", which the single-region assembly path cannot resolve
(it looks up "top"); building our own mesh with the colon-free material "box"
avoids that and lets us name the source/far end faces explicitly.
"""
import numpy as np
import astropy.units as u
from netgen import occ
import ngsolve as ngs

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.units import to_simulation_units

# Box dimensions (full lengths). Long axis is y; the source reservoir is the
# face at y = 0, the far (reflecting) wall is at y = BOX_LENGTH_Y.
BOX_WIDTH_X = 4.0 * u.um
BOX_LENGTH_Y = 60.0 * u.um
BOX_HEIGHT_Z = 2.0 * u.um


def make_box_mesh(mesh_size):
    """Build an elongated box mesh, source face at y=0, far face at y=Ly.

    One compartment "box"; exterior faces named "source" (y=0), "far" (y=Ly)
    and "side" (the four long faces).
    """
    lx = to_simulation_units(BOX_WIDTH_X, "length")
    ly = to_simulation_units(BOX_LENGTH_Y, "length")
    lz = to_simulation_units(BOX_HEIGHT_Z, "length")

    box = occ.Box(occ.Pnt(-lx / 2, 0, 0), occ.Pnt(lx / 2, ly, lz))
    box.mat("box")
    # occ.Box face order: 0:x-min 1:x-max 2:y-min 3:y-max 4:z-min 5:z-max
    box.faces[2].bc("source")
    box.faces[3].bc("far")
    for f in (0, 1, 4, 5):
        box.faces[f].bc("side")

    geo = occ.OCCGeometry(box)
    return ngs.Mesh(geo.GenerateMesh(maxh=to_simulation_units(mesh_size, "length")))


def run_simulation(
    *,
    with_buffer,
    result_root="results",
    # ECS / diffusion
    ca_source=1.3 * u.mmol / u.L,
    diffusivity=0.7 * u.um**2 / u.ms,
    # Buffer (immobile ECM); kf and total match the tissue sim, Kd matches the
    # Ca reservoir so the buffer is half-saturated near the source.
    ecm_total=2.0 * u.mmol / u.L,
    ecm_kf=10.0 * u.L / (u.mmol * u.s),
    kd=1.3 * u.mmol / u.L,
    mesh_size=1.0 * u.um,
    # Constant Ca influx at the source face. A concentration-independent flux
    # (GeneralFlux) is used rather than a fixed-concentration reservoir because
    # membrane transport is integrated explicitly (fem_details.transport_step),
    # so a stiff Robin/Passive reservoir would be numerically unstable. If None,
    # the density is derived so the no-buffer surface concentration reaches
    # ~ca_source by end_time: q = ca_source * sqrt(pi*D) / (2*sqrt(end_time)).
    source_flux_density=None,
    # Timing (mirrors the tissue sim)
    end_time=1.0 * u.s,
    time_step=1.0 * u.ms,
    record_interval=10.0 * u.ms,
    n_threads=4,
):
    label = "buffer" if with_buffer else "nobuffer"
    print(f"=== Running buffered_diffusion ({label}) ===")

    if source_flux_density is None:
        # Surface concentration of constant-flux diffusion: C(0,t) = 2 q sqrt(t)
        # / sqrt(pi D). Choose q so C(0, end_time) ~ ca_source.
        source_flux_density = (
            ca_source * np.sqrt(np.pi * diffusivity) / (2 * np.sqrt(end_time))
        ).to((u.mmol / u.L) * u.um / u.ms)
    print(f"  Source flux density: {source_flux_density:.4g}")

    # ================================================================
    # 1) Geometry and simulation
    # ================================================================
    mesh = make_box_mesh(mesh_size)
    print(f"  Mesh has {mesh.ne} elements and {mesh.nv} vertices")

    sim = bmbcsim.Simulation(
        name=f"buffered_diffusion_{label}",
        mesh=mesh,
        result_root=result_root,
    )
    box = sim.simulation_geometry.compartments["box"]
    print(f"  Box volume: {box.volume:.1f}")

    # ================================================================
    # 2) Species, initialization and diffusion
    # ================================================================
    ca = sim.add_species("Ca")
    box.initialize_species(ca, 0.0 * u.mmol / u.L)
    box.add_diffusion(ca, diffusivity)

    if with_buffer:
        # Kd = k_r / k_f  =>  k_r = Kd * k_f  (pattern from buffer_sweep.py).
        ecm_kr = (kd * ecm_kf).to(1 / u.ms)
        print(
            f"  Buffer: total={ecm_total}, kf={ecm_kf}, kr={ecm_kr:.4g}, "
            f"Kd={kd} (matches reservoir)"
        )

        # Ca starts at 0, so the buffer starts fully unbound at equilibrium:
        # ECM = ecm_total, ECM_Ca = 0. No add_diffusion -> immobile buffer.
        ecm = sim.add_species("ECM")
        ecm_ca = sim.add_species("ECM_Ca")
        box.initialize_species(ecm, ecm_total)
        box.initialize_species(ecm_ca, 0.0 * u.mmol / u.L)
        box.add_reaction(
            reactants=[ca, ecm],
            products=[ecm_ca],
            k_f=ecm_kf,
            k_r=ecm_kr,
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
        end_time=end_time,
        time_step=time_step,
        record_interval=record_interval,
        n_threads=n_threads,
    )
    print(f"=== Done ({label}) ===\n")


if __name__ == "__main__":
    run_simulation(with_buffer=False)
    run_simulation(with_buffer=True)
