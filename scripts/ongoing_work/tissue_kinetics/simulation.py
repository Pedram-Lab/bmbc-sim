import math
from pathlib import Path
from typing import Optional

import numpy as np

from astropy import units as u
from astropy import constants as const

import ngsolve as ngs

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.geometry import TissueGeometry
from bmbcsim.simulation import coefficient_fields as cf
from bmbcsim.config import SimConfig, SimGroup, Quantity


# ======================================================================
# Config: the single source of truth for parameters, defaults and units.
# ``run_simulation`` reads straight off this; nothing is declared twice.
# ======================================================================
class Geometry(SimGroup):
    """Mesh / cell-packing geometry."""

    target_cell_diam: float = 4.0
    ecs_ratio: float = 0.1
    # ECS ratio at which the cell set is cropped/numbered (step 1d); set to the
    # largest ecs_ratio in a comparison sweep. See step 1d.
    reference_ecs_ratio: float = 0.19
    box_size_x: float = 20.0
    box_size_y: float = 20.0
    box_size_z: float = 1.0
    mesh_size: float = 5.0


class Diffusion(SimGroup):
    """ECS/cytosol Ca2+ diffusion and the reservoir boundary condition."""

    ca_ecs: Quantity("mmol / L") = "1.3 mmol / L"
    diffusivity_ecs: Quantity("um2 / ms") = "0.7 um2 / ms"
    diffusivity_cyto: Quantity("um2 / ms") = "0.22 um2 / ms"
    tortuosity: float = 1.6
    boundary_permeability: Optional[Quantity("um / ms")] = None  # derived from tortuosity if None
    depletion: Quantity("mmol / L") = "0.47 mmol / L"


class Synapse(SimGroup):
    """Distributed NMDAR synapse patches (Ca2+ sink)."""

    n_synapses: int = 400
    n_channels_per_synapse: int = 35
    synapse_diameter: Quantity("um") = "0.25 um"
    f_active: float = 0.15
    i_channel: Quantity("pA") = "0.5 pA"
    tau1: Quantity("ms") = "10 ms"
    tau2: Quantity("ms") = "3 ms"
    pulse_times: list[Quantity("ms")] = ["300 ms", "310 ms", "320 ms", "330 ms", "340 ms"]


class ECM(SimGroup):
    """Extracellular-matrix Ca2+ buffer (Ca + ECM <-> ECM_Ca)."""

    enabled: bool = False
    ecm_total: Quantity("mmol / L") = "2.0 mmol / L"
    ecm_kf: Quantity("L / (mmol s)") = "10.0 L / (mmol s)"
    ecm_kr: Quantity("1 / ms") = "0.1 / ms"


class Mechanics(SimGroup):
    """ECS/cell elasticity and ECM_Ca-driven contraction (implies ECM)."""

    enabled: bool = False
    ecs_youngs_modulus: Quantity("kPa") = "0.5 kPa"
    ecs_poisson_ratio: float = 0.3
    cell_youngs_modulus: Quantity("kPa") = "1.0 kPa"
    cell_poisson_ratio: float = 0.4
    ecm_ca_coupling: Quantity("kPa L / mmol") = "0.1 kPa L / mmol"  # = 0.1 kPa / (mmol/L)


class Config(SimConfig):
    """Full config for the tissue-kinetics simulation."""

    simulation_name: str = "tissue_kinetics"
    # Random seed for synapse distribution (also the sweep replicate index)
    seed: int = 42
    # Timing
    end_time: Quantity("s") = "1.0 s"
    time_step: Quantity("s") = "1.0 ms"
    record_interval_factor: int = 10
    # Subsystems
    geometry: Geometry = Geometry()
    diffusion: Diffusion = Diffusion()
    synapse: Synapse = Synapse()
    ecm: ECM = ECM()
    mechanics: Mechanics = Mechanics()


def _on_outer_box(min_box, max_box, eps_rel=1e-6):
    """Return a predicate flagging DOFs on any of the 6 simulation-box faces."""
    min_box = np.asarray(min_box, dtype=float)
    max_box = np.asarray(max_box, dtype=float)
    eps = eps_rel * float((max_box - min_box).max())

    def predicate(coords):
        on_face = np.zeros(len(coords), dtype=bool)
        for axis in range(3):
            on_face |= np.isclose(coords[:, axis], min_box[axis], atol=eps)
            on_face |= np.isclose(coords[:, axis], max_box[axis], atol=eps)
        return on_face

    return predicate


def run_simulation(cfg: Config) -> None:
    geom, diff, syn, ecm_cfg, mech = (
        cfg.geometry, cfg.diffusion, cfg.synapse, cfg.ecm, cfg.mechanics
    )
    # Mechanics implies ECM (needs ECM_Ca as driving species)
    with_mechanics = mech.enabled
    with_ecm = ecm_cfg.enabled or with_mechanics
    # Derived parameter: boundary permeability from tortuosity if not set
    boundary_permeability = diff.boundary_permeability
    if boundary_permeability is None:
        d_eff = diff.diffusivity_ecs / diff.tortuosity**2
        l_char = max(geom.box_size_x, geom.box_size_y, geom.box_size_z) / 2.0 * u.um
        boundary_permeability = d_eff / l_char

    # ================================================================
    # 1) Load and post-process geometry from VTK
    # ================================================================
    print("Loading geometry...")
    geometry = TissueGeometry.from_file("data/tissue_geometry.vtk")
    print(f"  Cells after from_file: {len(geometry.cells)}")

    # --- 1a) compute the typical "diameter" of each cell ---
    cell_diameters = []
    for i, cell in enumerate(geometry.cells):
        bmin = np.array(cell.bounds[::2])
        bmax = np.array(cell.bounds[1::2])
        size = bmax - bmin
        diam = float(size.max())
        cell_diameters.append(diam)

    cell_diameters = np.array(cell_diameters)
    median_diam = float(np.median(cell_diameters))
    print(f"  Median cell diameter (original units): {median_diam:.3f}")

    # --- 1b) scale so that the median matches target_cell_diam ---
    scale_factor = geom.target_cell_diam / median_diam
    print(f"  Scale factor to get ~{geom.target_cell_diam} um cells: {scale_factor:.3f}")

    geometry = geometry.scale(scale_factor)
    geometry = geometry.decimate(factor=0.5)
    geometry = geometry.smooth(n_iter=10)
    geometry = geometry.decimate(factor=0.5)

    # --- 1c) translate so that the domain starts at (0,0,0) ---
    minc, _ = geometry.bounding_box()
    for cell in geometry.cells:
        cell.points -= minc

    # --- 1d) crop at a fixed reference shrink so the cell set is ecs_ratio-independent ---
    # Cropping at reference_ecs_ratio (not the run's ecs_ratio) keeps the kept set,
    # its numbering (step 1f) and synapse seeding (step 5) identical across ratios,
    # so (seed, synapse_idx) is the same synapse everywhere; step 1e then scales to
    # the run's ecs_ratio. shrink_cells/LocalizedPeaks are both centroid-radial, so
    # only cell size (hence ECS volume) changes. Cropping shrunk (vs full-size)
    # cells also avoids over-packing the box and pinching the ECS apart.
    geometry = geometry.shrink_cells(1 - geom.reference_ecs_ratio, jitter=0.0)

    minc3, maxc3 = geometry.bounding_box()
    center = 0.5 * (minc3 + maxc3)
    box_size = np.array([geom.box_size_x, geom.box_size_y, geom.box_size_z])
    half_box = box_size / 2.0
    min_box = np.maximum(minc3, center - half_box)
    max_box = np.minimum(maxc3, center + half_box)

    geometry = geometry.keep_cells_within(
        min_coords=min_box,
        max_coords=max_box,
        inside_threshold=0.1
    )
    n_cells = len(geometry.cells)
    print(f"  Cells after keep_cells_within (reference ecs={geom.reference_ecs_ratio}): "
          f"{n_cells}")

    # --- 1e) scale the fixed cell set to this run's ECS ratio (grows it for smaller ratios) ---
    geometry = geometry.shrink_cells(
        (1 - geom.ecs_ratio) / (1 - geom.reference_ecs_ratio), jitter=0.0
    )
    print(f"  Cells scaled to ecs_ratio={geom.ecs_ratio}: {n_cells} cells")

    if n_cells == 0:
        raise RuntimeError(
            "No cells remain after keep_cells_within. "
            "Increase one of box_size_x/y/z or relax inside_threshold."
        )

    # ================================================================
    # 1f) Cell and membrane names
    # ================================================================
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    bnd_names = [f"membrane_{i}" for i in range(n_cells)]

    # ================================================================
    # 1g) Generate NGSolve mesh
    # ================================================================
    print("Building mesh...")
    tissue_mesh: ngs.Mesh = geometry.to_ngs_mesh(
        mesh_size=geom.mesh_size,
        min_coords=min_box,
        max_coords=max_box,
        projection_tol=0.02,
        cell_names=cell_names,
        cell_bnd_names=bnd_names,
    )
    print(f"  Mesh has {tissue_mesh.ne} elements and {tissue_mesh.nv} vertices")

    # ECS must be a single connected region (to_ngs_mesh names a disconnected one
    # "ecs:region_0", "ecs:region_1", ...). Fail fast: it's unphysical, and the
    # seed-independent geometry would break every seed identically.
    ecs_regions = [m for m in set(tissue_mesh.GetMaterials())
                   if m == "ecs" or m.startswith("ecs:")]
    if len(ecs_regions) != 1:
        raise RuntimeError(
            f"ECS split into {len(ecs_regions)} disconnected regions "
            f"{sorted(ecs_regions)} at ecs_ratio={geom.ecs_ratio}; raise "
            f"reference_ecs_ratio (now {geom.reference_ecs_ratio}) toward it."
        )

    # ================================================================
    # 2) Set up simulation
    # ================================================================
    print("Setting up simulation...")
    sim = bmbcsim.Simulation(
        mesh=tissue_mesh,
        name=cfg.simulation_name,
        result_root=cfg.result_root,
        mechanics=with_mechanics,
    )
    geo = sim.simulation_geometry

    ecs = geo.compartments["ecs"]
    cells = [geo.compartments[f"cell_{i}"] for i in range(n_cells)]
    membranes = [geo.membranes[f"membrane_{i}"] for i in range(n_cells)]

    total_cell_volume = sum(cell.volume for cell in cells)
    total_volume = ecs.volume + total_cell_volume
    total_membrane_area = sum(membrane.area for membrane in membranes)
    print(f"  Total volume: {total_volume:.2f} um^3")
    print(f"  ECS volume: {ecs.volume:.2f} um^3")
    print(f"  ECS volume fraction: {ecs.volume / total_volume * 100:.2f}%")
    print(f"  Total membrane area: {total_membrane_area:.2f} um^2")

    # ================================================================
    # 2b) Mechanical properties (optional)
    # ================================================================
    if with_mechanics:
        ecs.add_elasticity(
            youngs_modulus=mech.ecs_youngs_modulus,
            poisson_ratio=mech.ecs_poisson_ratio,
        )
        for cell in cells:
            cell.add_elasticity(
                youngs_modulus=mech.cell_youngs_modulus,
                poisson_ratio=mech.cell_poisson_ratio,
            )

    # ================================================================
    # 3) Species and initialization
    # ================================================================
    ca = sim.add_species("Ca")
    ecs.initialize_species(ca, diff.ca_ecs)

    if with_ecm:
        ecm_k = (ecm_cfg.ecm_kf / ecm_cfg.ecm_kr).decompose()
        ecm_ca_equilibrium = ecm_k * diff.ca_ecs * ecm_cfg.ecm_total / (1 + ecm_k * diff.ca_ecs)
        ecm_concentration = ecm_cfg.ecm_total - ecm_ca_equilibrium

        ecm = sim.add_species("ECM")
        ecm_ca = sim.add_species("ECM_Ca")
        ecs.initialize_species(ecm, ecm_concentration)
        ecs.initialize_species(ecm_ca, ecm_ca_equilibrium)

        if with_mechanics:
            ecs.add_driving_species(
                ecm_ca, mech.ecm_ca_coupling, baseline=ecm_ca_equilibrium
            )

    for cell in cells:
        cell.initialize_species(ca, 0.0 * u.mmol / u.L)

    # ================================================================
    # 4) Diffusion and reactions
    # ================================================================
    ecs.add_diffusion(ca, diff.diffusivity_ecs)

    for cell in cells:
        cell.add_diffusion(ca, diff.diffusivity_cyto)

    if with_ecm:
        ecs.add_reaction(
            reactants=[ca, ecm],
            products=[ecm_ca],
            k_f=ecm_cfg.ecm_kf,
            k_r=ecm_cfg.ecm_kr,
        )

    # ================================================================
    # 5) Ca2+ sink at distributed synapse patches
    # ================================================================
    # Q = N * I / (2 F)  (factor 2 for Ca2+)
    const_F = const.e.si * const.N_A
    Q_per_synapse = syn.n_channels_per_synapse * syn.i_channel / (2 * const_F)

    # Number of active synapses, distributed randomly across cells
    total_active_synapses = int(syn.n_synapses * syn.f_active)
    base_synapses_per_cell = total_active_synapses // n_cells
    remainder = total_active_synapses % n_cells

    rng = np.random.default_rng(seed=cfg.seed)
    cells_with_extra = rng.choice(n_cells, size=remainder, replace=False)
    synapses_per_cell = np.full(n_cells, base_synapses_per_cell)
    synapses_per_cell[cells_with_extra] += 1

    print(f"  Active synapses: {synapses_per_cell.sum()} (of {syn.n_synapses} total)")

    # Biexponential NMDAR waveform with multi-pulse stimulation
    def nmdar_waveform(t):
        """J(t) = e^(-t/tau1) - e^(-t/tau2), superposition of pulses."""
        total = 0.0
        for t_pulse in syn.pulse_times:
            dt = t - t_pulse
            if dt >= 0 * u.ms:
                total += math.exp(-dt / syn.tau1) - math.exp(-dt / syn.tau2)
        return total

    # Skip membrane DOFs that lie on the outer simulation box: synapses there
    # would straddle the simulation boundary rather than the cell membrane.
    exclude_outer_box = _on_outer_box(min_box, max_box)

    # Distributed synapse patches using LocalizedPeaks
    for i, (membrane, cell) in enumerate(zip(membranes, cells)):
        n_syn = synapses_per_cell[i]
        if n_syn == 0:
            continue

        synapse_distribution = cf.LocalizedPeaks(
            seed=int(rng.integers(0, 2**31)),
            num_peaks=n_syn,
            peak_value=Q_per_synapse,
            background_value=0.0 * u.mol / u.s,
            peak_width=syn.synapse_diameter / 6.0,
            total=n_syn * Q_per_synapse,
            exclude_predicate=exclude_outer_box,
        )
        synapse_flux = transport.ProportionalFlux(
            flux=synapse_distribution,
            saturation=diff.ca_ecs,
            depletion=diff.depletion,
            temporal=nmdar_waveform,
        )
        membrane.add_transport(ca, synapse_flux, ecs, cell)

    # ================================================================
    # 6) Robin BC: transport from external reservoir into ECS
    # ================================================================
    for bnd in ["top", "bottom", "left", "right", "front", "back"]:
        boundary = geo.membranes[bnd]
        boundary_flux = transport.Passive(
            boundary_permeability * boundary.area, diff.ca_ecs
        )
        boundary.add_transport(ca, boundary_flux, None, ecs)

    # ================================================================
    # 7) Run simulation
    # ================================================================
    print("Running simulation...")
    sim.run(
        end_time=cfg.end_time,
        time_step=cfg.time_step,
        record_interval=cfg.record_interval_factor * cfg.time_step,
        n_threads=cfg.n_threads,
    )
    print("Simulation complete.")


def run(cfg: Config) -> None:
    """Public entry: run the simulation from a validated config."""
    run_simulation(cfg)


_CONFIGS = str(Path(__file__).resolve().parent / "configs")

# Register the default config as Hydra's schema so CLI overrides of any (nested)
# field work without a "+". Generated from Config itself -- units serialize to
# strings, so defaults are declared exactly once, in the Config class above.
ConfigStore.instance().store(name="tissue_kinetics", node=Config().model_dump())


@hydra.main(version_base=None, config_path=_CONFIGS, config_name="tissue_kinetics")
def main(dcfg) -> None:
    run(Config(**OmegaConf.to_container(dcfg, resolve=True)))


if __name__ == "__main__":
    main()
