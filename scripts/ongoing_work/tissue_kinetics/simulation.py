"""Tissue-kinetics simulation: Ca2+ in the ECS of a packed-cell tissue block.

The ``Config`` below is the single source of truth for parameters, defaults and
units -- ``run`` reads straight off it, nothing is declared twice. Hydra composes
the YAML (``configs/``); ``sweep.py`` fans the same ``Config``/``run`` pair out
over a parameter grid.

    uv run scripts/ongoing_work/tissue_kinetics/simulation.py mechanics.enabled=true
"""
import json
import math

import numpy as np

from astropy import units as u
from astropy import constants as const

import ngsolve as ngs

import bmbcsim
from bmbcsim.meshing.tissue_mesher import load_tissue_mesh
from bmbcsim.simulation import transport
from bmbcsim.simulation import coefficient_fields as cf
from make_geometry import geometry_file
from bmbcsim.config import (
    SimulationConfig,
    ConfigGroup,
    Quantity,
    dump_resolved,
    run_from_cli,
)


class Geometry(ConfigGroup):
    """Tissue mesh from make_geometry.py, identified by (mesh_dir, seed, ecs_ratio).

    The cell set and numbering depend only on the seed, so the meshes of one seed
    at different ecs_ratio have the same cells. The box is set by the mesh.
    """

    # data/cells (round cells) or data/tubes (tube-shaped cells)
    mesh_dir: str = "data/cells"
    seed: int = 0
    # Target ECS volume fraction (a whole percentage); the measured one is in geometry.json
    ecs_ratio: float = 0.1
    # Uniform ``ngs.Mesh.Refine()`` passes after meshing: each halves h (8x elements)
    # on the *same* geometry, which maxh cannot guarantee once the cell-surface
    # triangulation dominates h. For convergence tests.
    n_refine: int = 0


class Diffusion(ConfigGroup):
    """ECS/cytosol Ca2+ diffusion and the reservoir boundary condition."""

    ca_ecs: Quantity("mM") = "1.3 mM"
    diffusivity_ecs: Quantity("um2 / ms") = "0.7 um2 / ms"
    diffusivity_cyto: Quantity("um2 / ms") = "0.22 um2 / ms"
    tortuosity: float = 1.6
    boundary_permeability: Quantity("um / ms") | None = None  # from tortuosity if None
    depletion: Quantity("mM") = "0.47 mM"


class Synapse(ConfigGroup):
    """Distributed NMDAR synapse patches (Ca2+ sink)."""

    n_synapses: int = 1372   # 1 per um^3 of the 14 x 14 x 7 box, f_active of them carry flux
    n_channels_per_synapse: int = 35
    synapse_diameter: Quantity("um") = "0.25 um"
    f_active: float = 0.15
    i_channel: Quantity("pA") = "0.5 pA"
    tau1: Quantity("ms") = "10 ms"
    tau2: Quantity("ms") = "3 ms"
    pulse_times: list[Quantity("ms")] = ["100 ms", "110 ms", "120 ms", "130 ms", "140 ms"]
    # Indices into synapse_centers.npy that carry flux; None = all of them. The other
    # synapses are still placed and saved (so indices stay comparable across runs), they
    # just stay silent: a sparse stimulus for solitary-synapse experiments.
    active: list[int] | None = None
    # Convergence-test mode: no synapse patches; instead the whole membrane of the
    # cell nearest the box center absorbs the total active-synapse flux uniformly,
    # so the sink does not move with the mesh the way snapped synapse centers do.
    uniform_cell: bool = False


class ECM(ConfigGroup):
    """Extracellular-matrix Ca2+ buffer (Ca + ECM <-> ECM_Ca)."""

    enabled: bool = False
    # Mean ECM concentration over the ECS; the total amount is the same whether
    # the buffer is uniform or localized.
    ecm_total: Quantity("mM") = "2.0 mM"
    # None: uniform buffer. Otherwise the buffer is a sum of Gaussians (this
    # standard deviation) centered on the synapses, confined to the ECS, scaled
    # so its ECS mean is ecm_total. Keep it at a few mesh edge lengths.
    local_width: Quantity("um") | None = None
    # Indices into synapse_centers.npy (all active synapses, cell order) that get
    # the localized buffer; None = all of them. ecm_total is unchanged, so fewer
    # targets means more buffer per target.
    local_target: list[int] | None = None
    # Parametrized by (kf, Kd) rather than (kf, kr): the on-rate kf is a property of
    # the molecules, while Kd = kr / kf is the affinity the buffer sweeps scan, so kr
    # is the one to derive. Defaults are unchanged (kr = 100 / s = 0.1 / ms).
    ecm_kf: Quantity("1 / (mM s)") = "10.0 / (mM s)"
    ecm_kd: Quantity("mM") = "10.0 mM"

    @property
    def ecm_kr(self) -> u.Quantity:
        """Reverse rate, derived from ``Kd = ecm_kr / ecm_kf``."""
        return self.ecm_kf * self.ecm_kd


class Mechanics(ConfigGroup):
    """ECS/cell elasticity and ECM_Ca-driven contraction (implies ECM)."""

    enabled: bool = False
    ecs_youngs_modulus: Quantity("kPa") = "0.5 kPa"
    ecs_poisson_ratio: float = 0.3
    cell_youngs_modulus: Quantity("kPa") = "1.0 kPa"
    cell_poisson_ratio: float = 0.4
    # Fractional shrinkage of the ECS's stress-free volume per mM of bound Ca:
    # the gel condenses as Ca binds, so ECM_Ca depletion makes it swell. Named
    # for the sign it carries -- `add_driving_species` measures *swelling*, so
    # this is passed negated, and calling it a "coupling" hid that flip.
    # (Was a pressure, kPa/mM; the solver now takes a target volume instead,
    # which removes the ~0.64 mu limit point that capped expansion.)
    ecm_ca_condensation: Quantity("1 / mM") = "0.25 / mM"
    # Embedding springs on the block's exterior relative to E / (2 L) (a 2 L layer of the
    # same tissue; L = mesh diagonal / sqrt(3)); 0 = free block, large = rigid box.
    boundary_stiffness: float = 1.0


class Config(SimulationConfig):
    """Full config for the tissue-kinetics simulation."""

    simulation_name: str = "tissue_kinetics"
    # Random seed for synapse distribution (also the sweep replicate index)
    seed: int = 42
    # Timing
    end_time: Quantity("s") = "0.5 s"
    time_step: Quantity("s") = "0.5 ms"   # 1 ms under-resolves the synapse peaks (depletion biased low)
    record_interval_factor: int = 4       # 2 ms records
    # Subsystems
    geometry: Geometry = Geometry()
    diffusion: Diffusion = Diffusion()
    synapse: Synapse = Synapse()
    ecm: ECM = ECM()
    mechanics: Mechanics = Mechanics()


def _check_indices(values, n, name):
    """Distinct synapse indices in [0, n) as an int array, else a ValueError naming `name`."""
    idx = np.asarray(values, dtype=int)
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n or len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} must be distinct indices in [0, {n}), got {list(values)}")
    return idx


def _on_outer_box(min_box, max_box, eps_rel=1e-6):
    """Return a predicate flagging points on or outside the simulation box."""
    eps = eps_rel * float(np.max(np.subtract(max_box, min_box)))

    def predicate(coords):
        return ((coords <= min_box + eps) | (coords >= max_box - eps)).any(axis=1)

    return predicate


def _boundary_vertices(mesh: ngs.Mesh, name: str) -> np.ndarray:
    """Coordinates of the mesh vertices on the named boundary, shape (n, 3)."""
    coords = np.array(mesh.ngmesh.Coordinates())
    idx = {v.nr for el in mesh.Boundaries(name).Elements() for v in el.vertices}
    return coords[sorted(idx)]


def _sample_triangles(tris: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points (n, 3) drawn uniformly by area on the triangles tris (M, 3, 3)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    k = rng.choice(len(tris), size=n, p=area / area.sum())
    s, t = rng.random((2, n))
    s = np.sqrt(s)
    return (1 - s)[:, None] * a[k] + (s * (1 - t))[:, None] * b[k] + (s * t)[:, None] * c[k]


def _volume_centroid(mesh: ngs.Mesh, material: str) -> np.ndarray:
    """Volume centroid of the named region (does not depend on how the mesh resolves it)."""
    region = mesh.Materials(material)
    return np.array([ngs.Integrate(x, mesh, definedon=region) for x in (ngs.x, ngs.y, ngs.z)]) \
        / ngs.Integrate(1, mesh, definedon=region)


def run(cfg: Config) -> None:
    """Run the simulation from a validated config."""
    geom, diff, syn, ecm_cfg, mech = (
        cfg.geometry, cfg.diffusion, cfg.synapse, cfg.ecm, cfg.mechanics
    )
    # Mechanics implies ECM (needs ECM_Ca as driving species)
    with_mechanics = mech.enabled
    with_ecm = ecm_cfg.enabled or with_mechanics
    # Claim the result directory and record the config that produced it before
    # loading the geometry, so a run that fails on a missing mesh is debuggable.
    result_dir = bmbcsim.timestamped_directory(cfg.result_root, cfg.run_name)
    dump_resolved(cfg, result_dir)
    print(f"Results and config -> {result_dir}")

    # --- Load the tissue mesh (make_geometry.py) ---
    path = geometry_file(geom.mesh_dir, geom.seed, geom.ecs_ratio)
    print(f"Loading geometry {path}...")
    tissue_mesh, meta = load_tissue_mesh(path)
    with open(result_dir / "geometry.json", "w") as f:
        json.dump({"file": path, **{k: np.asarray(v).tolist() for k, v in meta.items()
                                    if not k.startswith("ref_")}}, f, indent=1)
    n_cells = len(tissue_mesh.GetMaterials()) - 1
    coords = np.array(tissue_mesh.ngmesh.Coordinates())
    min_box, max_box = coords.min(axis=0), coords.max(axis=0)
    print(f"  {n_cells} cells in a {' x '.join(f'{s:g}' for s in max_box - min_box)} um box, "
          f"ECS fraction {meta['ecs_fraction']:.4f} (target {meta['ecs_ratio']}, gap w = {meta['w']:.4f} um)")
    for _ in range(geom.n_refine):
        tissue_mesh.Refine()
    print(f"  Mesh has {tissue_mesh.ne} elements and {tissue_mesh.nv} vertices "
          f"(n_refine={geom.n_refine})")
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    bnd_names = [f"membrane_{i}" for i in range(n_cells)]

    # Derived parameter: boundary permeability from tortuosity if not set
    boundary_permeability = diff.boundary_permeability
    if boundary_permeability is None:
        l_char = float(np.max(max_box - min_box)) / 2.0 * u.um
        boundary_permeability = diff.diffusivity_ecs / diff.tortuosity**2 / l_char

    # --- Set up simulation ---
    print("Setting up simulation...")
    sim = bmbcsim.Simulation(
        mesh=tissue_mesh,
        result_directory=result_dir,
        mechanics=with_mechanics,
        boundary_stiffness=mech.boundary_stiffness,
    )
    geo = sim.simulation_geometry

    ecs = geo.compartments["ecs"]
    cells = [geo.compartments[name] for name in cell_names]
    membranes = [geo.membranes[name] for name in bnd_names]

    total_volume = ecs.volume + sum(cell.volume for cell in cells)
    print(f"  Total volume: {total_volume:.2f} um^3")
    print(f"  ECS volume: {ecs.volume:.2f} um^3")
    print(f"  ECS volume fraction: {ecs.volume / total_volume * 100:.2f}%")
    print(f"  Total membrane area: {sum(m.area for m in membranes):.2f} um^2")

    # Mechanical properties (optional)
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

    # --- Species and initialization ---
    ca = sim.add_species("Ca")
    ecs.initialize_species(ca, diff.ca_ecs)
    for cell in cells:
        cell.initialize_species(ca, 0.0 * u.mmol / u.L)

    # --- Diffusion ---
    ecs.add_diffusion(ca, diff.diffusivity_ecs)
    for cell in cells:
        cell.add_diffusion(ca, diff.diffusivity_cyto)

    # --- Ca2+ sink at distributed synapse patches ---
    # Q = N * I / (2 F)  (factor 2 for Ca2+)
    faraday = const.e.si * const.N_A
    q_per_synapse = syn.n_channels_per_synapse * syn.i_channel / (2 * faraday)

    # Active synapses, distributed randomly (but reproducibly) across cells
    rng = np.random.default_rng(seed=cfg.seed)
    base, remainder = divmod(int(syn.n_synapses * syn.f_active), n_cells)
    synapses_per_cell = np.full(n_cells, base)
    synapses_per_cell[rng.choice(n_cells, size=remainder, replace=False)] += 1
    print(f"  Active synapses: {synapses_per_cell.sum()} (of {syn.n_synapses} total)")

    def nmdar_waveform(t):
        """J(t) = e^(-t/tau1) - e^(-t/tau2), superposition of pulses."""
        dts = [t - t_pulse for t_pulse in syn.pulse_times if t >= t_pulse]
        return sum(math.exp(-dt / syn.tau1) - math.exp(-dt / syn.tau2) for dt in dts)

    # Skip membrane points on or outside the simulation box: synapses there
    # would straddle the simulation boundary rather than the cell membrane.
    exclude_outer_box = _on_outer_box(min_box, max_box)

    if syn.uniform_cell and ecm_cfg.local_width is not None:
        raise ValueError("ecm.local_width needs synapse patches; unset synapse.uniform_cell")

    if syn.uniform_cell:
        centroids = np.array([_volume_centroid(tissue_mesh, c) for c in cell_names])
        i_center = int(np.argmin(np.linalg.norm(centroids - 0.5 * (min_box + max_box), axis=1)))
        print(f"  Uniform sink on cell_{i_center} (whole membrane, "
              f"{membranes[i_center].area:.2f} um^2)")
        uniform_flux = transport.ProportionalFlux(
            flux=synapses_per_cell.sum() * q_per_synapse,
            saturation=diff.ca_ecs,
            depletion=diff.depletion,
            temporal=nmdar_waveform,
        )
        membranes[i_center].add_transport(ca, uniform_flux, ecs, cells[i_center])
        synapses_per_cell[:] = 0  # skip the patch loop below

    # Synapse centers: anchors drawn uniformly by area (seeded rng) on the cell's raw
    # Voronoi faces, which make_geometry.py stores and which do not depend on the gap w,
    # each snapped to the nearest free membrane vertex (so the narrow flux Gaussian sits
    # on a DOF). The cells of one geometry seed are the same at every ecs_ratio, so
    # (seed, synapse index) is the same membrane spot across ratios up to the gap and
    # the mesh size.
    if "ref_faces" not in meta:
        raise ValueError(f"{path} has no reference faces for the synapses; regenerate it with make_geometry.py")
    ref_faces, ref_cell = meta["ref_faces"].reshape(-1, 3, 3), meta["ref_cell"]
    synapse_centers = {}  # cell index -> (n_syn, 3) centers
    max_snap = 0.0
    for i, n_syn in enumerate(synapses_per_cell):
        if n_syn == 0:
            continue
        anchors = _sample_triangles(ref_faces[ref_cell == i], n_syn, rng)
        vertices = _boundary_vertices(tissue_mesh, bnd_names[i])
        dist = np.linalg.norm(anchors[:, None] - vertices[None], axis=2)
        dist[:, exclude_outer_box(vertices)] = np.inf
        picked = []
        for row in dist:
            row[picked] = np.inf
            picked.append(int(np.argmin(row)))
            max_snap = max(max_snap, row[picked[-1]])
        synapse_centers[i] = vertices[picked]
    all_centers = (np.concatenate(list(synapse_centers.values()))
                   if synapse_centers else np.empty((0, 3)))
    np.save(result_dir / "synapse_centers.npy", all_centers)
    print(f"  Synapse centers on membrane vertices, max snap {max_snap:.3f} um "
          f"-> {result_dir / 'synapse_centers.npy'}")

    flux_centers = synapse_centers
    if syn.active is not None:
        idx = _check_indices(syn.active, len(all_centers), "synapse.active")
        offsets = np.cumsum([0] + [len(c) for c in synapse_centers.values()])
        flux_centers = {}
        for (i, centers), lo in zip(synapse_centers.items(), offsets):
            sel = idx[(idx >= lo) & (idx < lo + len(centers))] - lo
            if len(sel):
                flux_centers[i] = centers[sel]
        print(f"  Active synapses with flux: {len(idx)}/{len(all_centers)} "
              f"(cells {sorted(flux_centers)})")

    for i, centers in flux_centers.items():
        synapse_distribution = cf.GaussianPeaks(
            centers,
            peak_width=syn.synapse_diameter / 6.0,
            peak_value=q_per_synapse,
            total=len(centers) * q_per_synapse,
        )
        synapse_flux = transport.ProportionalFlux(
            flux=synapse_distribution,
            saturation=diff.ca_ecs,
            depletion=diff.depletion,
            temporal=nmdar_waveform,
        )
        membranes[i].add_transport(ca, synapse_flux, ecs, cells[i])

    # --- ECM buffer (uniform, or Gaussians on the synapse centers) ---
    if with_ecm:
        ecm_k = (ecm_cfg.ecm_kf / ecm_cfg.ecm_kr).decompose()
        ecm_ca_equilibrium = ecm_k * diff.ca_ecs * ecm_cfg.ecm_total / (1 + ecm_k * diff.ca_ecs)
        ecm_init, ecm_ca_init = ecm_cfg.ecm_total - ecm_ca_equilibrium, ecm_ca_equilibrium

        if ecm_cfg.local_width is not None:
            buffer_centers = all_centers
            if ecm_cfg.local_target is not None:
                buffer_centers = all_centers[
                    _check_indices(ecm_cfg.local_target, len(all_centers), "ecm.local_target")]
            print(f"  Localized ECM on {len(buffer_centers)}/{len(all_centers)} synapses, "
                  f"sigma {ecm_cfg.local_width}")

            # Same shape for ECM and ECM_Ca (Ca is uniform at t=0, so the bound
            # fraction is too); `total` fixes the ECS mean to the uniform value.
            def localized(mean_conc):
                return cf.GaussianPeaks(
                    buffer_centers,
                    peak_width=ecm_cfg.local_width,
                    peak_value=mean_conc,
                    total=mean_conc * ecs.volume * u.um**3,
                )
            ecm_init, ecm_ca_init = localized(ecm_init), localized(ecm_ca_init)

        ecm = sim.add_species("ECM")
        ecm_ca = sim.add_species("ECM_Ca")
        ecs.initialize_species(ecm, ecm_init)
        ecs.initialize_species(ecm_ca, ecm_ca_init)
        ecs.add_reaction(
            reactants=[ca, ecm],
            products=[ecm_ca],
            k_f=ecm_cfg.ecm_kf,
            k_r=ecm_cfg.ecm_kr,
        )
        if with_mechanics:
            # Baseline = initial ECM_Ca field, so a localized buffer is stress-free at t=0.
            ecs.add_driving_species(ecm_ca, -mech.ecm_ca_condensation, baseline=ecm_ca_init)

    # --- Robin BC: transport from external reservoir into ECS ---
    for bnd in ["top", "bottom", "left", "right", "front", "back"]:
        boundary = geo.membranes[bnd]
        boundary_flux = transport.Passive(
            boundary_permeability * boundary.area, diff.ca_ecs
        )
        boundary.add_transport(ca, boundary_flux, None, ecs)

    # --- Run simulation ---
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
