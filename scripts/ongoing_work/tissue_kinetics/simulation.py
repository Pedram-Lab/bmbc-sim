import math
import numpy as np

from astropy import units as u
from astropy import constants as const

import ngsolve as ngs

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.geometry import TissueGeometry
from bmbcsim.simulation import coefficient_fields as cf


def run_simulation(
    # Simulation identity / output
    simulation_name="tissue_kinetics",
    result_root="results",
    # Feature switches
    with_ecm=False,
    with_mechanics=False,
    # Random seed for synapse distribution
    seed=42,
    # ECS / diffusion
    ca_ecs=1.3 * u.mmol / u.L,
    diffusivity_ecs=0.7 * u.um**2 / u.ms,
    tortuosity=1.6,
    boundary_permeability=None,  # derived from tortuosity if None
    # Synapse parameters - scaled for 20x20x1 um = 400 um^3
    n_synapses=400,
    n_channels_per_synapse=35,
    synapse_diameter=0.25 * u.um,
    f_active=0.15,
    i_channel=0.5 * u.pA,
    # NMDAR kinetics (biexponential)
    tau1=10 * u.ms,
    tau2=3 * u.ms,
    pulse_times=None,  # defaults to [300, 310, 320, 330, 340] * u.ms
    # Simulation timing
    end_time=1.0 * u.s,
    time_step=1.0 * u.ms,
    record_interval_factor=10,
    # Geometry processing
    target_cell_diam=4.0,
    ecs_ratio=0.1,
    box_size_x=20.0,
    box_size_y=20.0,
    box_size_z=1.0,
    mesh_size=5.0,
    diffusivity_cyto=0.22 * u.um**2 / u.ms,
    depletion=0.47 * u.mmol / u.L,
    # ECM reaction parameters (used when with_ecm or with_mechanics)
    ecm_total=2.0 * u.mmol / u.L,
    ecm_kf=10.0 * u.L / (u.mmol * u.s),
    ecm_kr=0.1 / u.ms,
    # Mechanics parameters (used when with_mechanics)
    ecs_youngs_modulus=0.5 * u.kPa,
    ecs_poisson_ratio=0.3,
    cell_youngs_modulus=1.0 * u.kPa,
    cell_poisson_ratio=0.4,
    ecm_ca_coupling=0.1 * u.kPa / (u.mmol / u.L),
    # Performance
    n_threads=4,
):
    # Mechanics implies ECM (needs ECM_Ca as driving species)
    if with_mechanics:
        with_ecm = True
    # Defaults for mutable / derived parameters
    if pulse_times is None:
        pulse_times = [300, 310, 320, 330, 340] * u.ms
    if boundary_permeability is None:
        d_eff = diffusivity_ecs / tortuosity**2
        l_char = max(box_size_x, box_size_y, box_size_z) / 2.0 * u.um
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
    scale_factor = target_cell_diam / median_diam
    print(f"  Scale factor to get ~{target_cell_diam} um cells: {scale_factor:.3f}")

    geometry = geometry.scale(scale_factor)
    geometry = geometry.decimate(factor=0.5)
    geometry = geometry.smooth(n_iter=10)
    geometry = geometry.decimate(factor=0.5)

    # --- 1c) translate so that the domain starts at (0,0,0) ---
    minc, _ = geometry.bounding_box()
    for cell in geometry.cells:
        cell.points -= minc

    # --- 1d) open ECS by shrinking the cells ---
    geometry = geometry.shrink_cells(1 - ecs_ratio, jitter=0.0)
    print(f"  Cells after shrink: {len(geometry.cells)}")

    # --- 1e) crop a block around the center ---
    minc3, maxc3 = geometry.bounding_box()
    center = 0.5 * (minc3 + maxc3)
    box_size = np.array([box_size_x, box_size_y, box_size_z])
    half_box = box_size / 2.0
    min_box = np.maximum(minc3, center - half_box)
    max_box = np.minimum(maxc3, center + half_box)

    geometry = geometry.keep_cells_within(
        min_coords=min_box,
        max_coords=max_box,
        inside_threshold=0.1
    )

    n_cells = len(geometry.cells)
    print(f"  Cells after keep_cells_within: {n_cells}")

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
        mesh_size=mesh_size,
        min_coords=min_box,
        max_coords=max_box,
        projection_tol=0.02,
        cell_names=cell_names,
        cell_bnd_names=bnd_names,
    )
    print(f"  Mesh has {tissue_mesh.ne} elements and {tissue_mesh.nv} vertices")

    # ================================================================
    # 2) Set up simulation
    # ================================================================
    print("Setting up simulation...")
    sim = bmbcsim.Simulation(
        mesh=tissue_mesh,
        name=simulation_name,
        result_root=result_root,
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
            youngs_modulus=ecs_youngs_modulus,
            poisson_ratio=ecs_poisson_ratio,
        )
        for cell in cells:
            cell.add_elasticity(
                youngs_modulus=cell_youngs_modulus,
                poisson_ratio=cell_poisson_ratio,
            )

    # ================================================================
    # 3) Species and initialization
    # ================================================================
    ca = sim.add_species("Ca")
    ecs.initialize_species(ca, ca_ecs)

    if with_ecm:
        ecm_k = (ecm_kf / ecm_kr).decompose()
        ecm_ca_equilibrium = ecm_k * ca_ecs * ecm_total / (1 + ecm_k * ca_ecs)
        ecm_concentration = ecm_total - ecm_ca_equilibrium

        ecm = sim.add_species("ECM")
        ecm_ca = sim.add_species("ECM_Ca")
        ecs.initialize_species(ecm, ecm_concentration)
        ecs.initialize_species(ecm_ca, ecm_ca_equilibrium)

        if with_mechanics:
            ecs.add_driving_species(
                ecm_ca, ecm_ca_coupling, baseline=ecm_ca_equilibrium
            )

    for cell in cells:
        cell.initialize_species(ca, 0.0 * u.mmol / u.L)

    # ================================================================
    # 4) Diffusion and reactions
    # ================================================================
    ecs.add_diffusion(ca, diffusivity_ecs)

    for cell in cells:
        cell.add_diffusion(ca, diffusivity_cyto)

    if with_ecm:
        ecs.add_reaction(
            reactants=[ca, ecm],
            products=[ecm_ca],
            k_f=ecm_kf,
            k_r=ecm_kr,
        )

    # ================================================================
    # 5) Ca2+ sink at distributed synapse patches
    # ================================================================
    # Q = N * I / (2 F)  (factor 2 for Ca2+)
    const_F = const.e.si * const.N_A
    Q_per_synapse = n_channels_per_synapse * i_channel / (2 * const_F)

    # Number of active synapses, distributed randomly across cells
    total_active_synapses = int(n_synapses * f_active)
    base_synapses_per_cell = total_active_synapses // n_cells
    remainder = total_active_synapses % n_cells

    rng = np.random.default_rng(seed=seed)
    cells_with_extra = rng.choice(n_cells, size=remainder, replace=False)
    synapses_per_cell = np.full(n_cells, base_synapses_per_cell)
    synapses_per_cell[cells_with_extra] += 1

    print(f"  Active synapses: {synapses_per_cell.sum()} (of {n_synapses} total)")

    # Biexponential NMDAR waveform with multi-pulse stimulation
    def nmdar_waveform(t):
        """J(t) = e^(-t/tau1) - e^(-t/tau2), superposition of pulses."""
        total = 0.0
        for t_pulse in pulse_times:
            dt = t - t_pulse
            if dt >= 0 * u.ms:
                total += math.exp(-dt / tau1) - math.exp(-dt / tau2)
        return total

    # Distributed synapse patches using LocalizedPeaks
    for i, (membrane, cell) in enumerate(zip(membranes, cells)):
        n_syn = synapses_per_cell[i]
        if n_syn == 0:
            continue

        synapse_distribution = cf.LocalizedPeaks(
            seed=0,
            num_peaks=n_syn,
            peak_value=Q_per_synapse,
            background_value=0.0 * u.mol / u.s,
            peak_width=synapse_diameter / 6.0,
            total=n_syn * Q_per_synapse,
        )
        synapse_flux = transport.ProportionalFlux(
            flux=synapse_distribution,
            saturation=ca_ecs,
            depletion=depletion,
            temporal=nmdar_waveform,
        )
        membrane.add_transport(ca, synapse_flux, ecs, cell)

    # ================================================================
    # 6) Robin BC: transport from external reservoir into ECS
    # ================================================================
    for bnd in ["top", "bottom", "left", "right", "front", "back"]:
        boundary = geo.membranes[bnd]
        boundary_flux = transport.Passive(
            boundary_permeability * boundary.area, ca_ecs
        )
        boundary.add_transport(ca, boundary_flux, None, ecs)

    # ================================================================
    # 7) Run simulation
    # ================================================================
    print("Running simulation...")
    sim.run(
        end_time=end_time,
        time_step=time_step,
        record_interval=record_interval_factor * time_step,
        n_threads=n_threads,
    )
    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation()
