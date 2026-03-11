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
    # Performance
    n_threads=4,
):
    # Defaults for mutable / derived parameters
    if pulse_times is None:
        pulse_times = [300, 310, 320, 330, 340] * u.ms
    if boundary_permeability is None:
        boundary_permeability = (1 / tortuosity**2) * u.um**3 / u.ms

    # ================================================================
    # 1) Load and post-process geometry from VTK
    # ================================================================
    geometry = TissueGeometry.from_file("data/tissue_geometry.vtk")
    print("Cells after from_file:", len(geometry.cells))

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
    print(f"Median cell diameter (original units): {median_diam}")

    # --- 1b) scale so that the median is ~4 um ---
    scale_factor = target_cell_diam / median_diam
    print("Scale factor to get ~4 um cells:", scale_factor)

    geometry = geometry.scale(scale_factor)
    geometry = geometry.decimate(factor=0.5)
    geometry = geometry.smooth(n_iter=10)
    geometry = geometry.decimate(factor=0.5)

    # --- 1c) bounding box after scaling ---
    minc, maxc = geometry.bounding_box()
    size = maxc - minc
    print("After cell-size-based scale, bbox:")
    print("  min:", minc)
    print("  max:", maxc)
    print("  size:", size)

    # --- 1d) translate so that the domain starts at (0,0,0) ---
    for cell in geometry.cells:
        cell.points -= minc

    minc2, maxc2 = geometry.bounding_box()
    size2 = maxc2 - minc2
    print("After translate, bbox:")
    print("  min:", minc2)
    print("  max:", maxc2)
    print("  size:", size2)
    print("Max domain size after scaling (in um):", float(size2.max()))
    print("Target median cell diameter (in um):", target_cell_diam)

    # --- 1e) open ECS by shrinking the cells ---
    geometry = geometry.shrink_cells(1 - ecs_ratio, jitter=0.0)
    print("Cells after shrink:", len(geometry.cells))

    # --- 1f) crop a block around the center ---
    minc3, maxc3 = geometry.bounding_box()
    size3 = maxc3 - minc3
    center = 0.5 * (minc3 + maxc3)
    print("BBox before clipping:")
    print("  min:", minc3)
    print("  max:", maxc3)
    print("  size:", size3)
    print("  center:", center)

    box_size = np.array([box_size_x, box_size_y, box_size_z])
    half_box = box_size / 2.0

    min_box = np.maximum(minc3, center - half_box)
    max_box = np.minimum(maxc3, center + half_box)

    print("Desired clipping box:")
    print("  min_box:", min_box)
    print("  max_box:", max_box)
    print("  box_size (approx):", max_box - min_box)

    geometry = geometry.keep_cells_within(
        min_coords=min_box,
        max_coords=max_box,
        inside_threshold=0.1
    )

    n_cells = len(geometry.cells)
    print("Cells after keep_cells_within:", n_cells)

    if n_cells == 0:
        raise RuntimeError(
            "No cells remain after keep_cells_within. "
            "Increase one of box_size_x/y/z or relax inside_threshold."
        )

    # ================================================================
    # 1g) Cell and membrane names
    # ================================================================
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    bnd_names = [f"membrane_{i}" for i in range(n_cells)]

    # ================================================================
    # 1h) Generate NGSolve mesh
    # ================================================================
    tissue_mesh: ngs.Mesh = geometry.to_ngs_mesh(
        mesh_size=mesh_size,
        min_coords=min_box,
        max_coords=max_box,
        projection_tol=0.02,
        cell_names=cell_names,
        cell_bnd_names=bnd_names,
    )
    print(f"Create mesh with {tissue_mesh.ne} elements and {tissue_mesh.nv} vertices.")

    # ================================================================
    # 2) Set up simulation
    # ================================================================
    sim = bmbcsim.Simulation(
        mesh=tissue_mesh,
        name=simulation_name,
        result_root=result_root,
    )
    geo = sim.simulation_geometry

    ecs = geo.compartments["ecs"]
    cells = [geo.compartments[f"cell_{i}"] for i in range(n_cells)]
    membranes = [geo.membranes[f"membrane_{i}"] for i in range(n_cells)]

    total_cell_volume = sum(cell.volume for cell in cells)
    total_volume = ecs.volume + total_cell_volume
    total_membrane_area = sum(membrane.area for membrane in membranes)
    print(f"Total volume: {total_volume:.2f} um^3")
    print(f"ECS volume: {ecs.volume:.2f} um^3")
    print(f"ECS volume fraction: {ecs.volume / total_volume * 100:.2f}%")
    print(f"Total membrane area: {total_membrane_area:.2f} um^2")

    # ================================================================
    # 3) Species and initialization
    # ================================================================
    ca = sim.add_species("Ca")
    ecs.initialize_species(ca, ca_ecs)

    for cell in cells:
        cell.initialize_species(ca, 0.0 * u.mmol / u.L)

    # ================================================================
    # 4) Diffusion
    # ================================================================
    ecs.add_diffusion(ca, diffusivity_ecs)

    for cell in cells:
        cell.add_diffusion(ca, diffusivity_cyto)

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

    print(f"Active synapses: {synapses_per_cell.sum()} (of {n_synapses} total)")

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
    boundary_flux = transport.Passive(boundary_permeability, ca_ecs)

    for bnd in ["top", "bottom", "left", "right", "front", "back"]:
        boundary = geo.membranes[bnd]
        boundary.add_transport(ca, boundary_flux, None, ecs)

    # ================================================================
    # 7) Run simulation
    # ================================================================
    sim.run(
        end_time=end_time,
        time_step=time_step,
        record_interval=record_interval_factor * time_step,
        n_threads=n_threads,
    )
    print("Simulation completed.")


if __name__ == "__main__":
    run_simulation()
