"""Plot deformation magnitude over time for mechanics simulations."""

import matplotlib.pyplot as plt
import numpy as np

from bmbcsim.simulation.result_io import ResultLoader

# ============ Configuration ============
SIMULATION_NAME = "tissue_kinetics_with_mechanics"
RESULTS_ROOT = "results"
MAX_TRAJECTORIES = 1000  # Limit number of individual trajectories to plot
# =======================================

# Load simulation results
loader = ResultLoader.find(simulation_name=SIMULATION_NAME, results_root=RESULTS_ROOT)
n_snapshots = len(loader)
print(f"Loading {n_snapshots} snapshots...")

# Get mesh info from first snapshot
mesh = loader.load_snapshot(0)
n_points = mesh.points.shape[0]
print(f"Mesh has {n_points} points")

# Collect deformation magnitudes over time
times = []
deformation_magnitudes = []

for step in range(n_snapshots):
    mesh = loader.load_snapshot(step)
    ts = loader.load_total_substance(step)

    time = float(ts.coords['time'].values[0])
    deformation = mesh.point_data['deformation']
    magnitude = np.linalg.norm(deformation, axis=1)

    times.append(time)
    deformation_magnitudes.append(magnitude)

times = np.array(times)
deformation_magnitudes = np.array(deformation_magnitudes)  # Shape: (n_snapshots, n_points)

print(f"Deformation data shape: {deformation_magnitudes.shape}")

# Subsample points for plotting individual trajectories
if n_points > MAX_TRAJECTORIES:
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(n_points, size=MAX_TRAJECTORIES, replace=False)
else:
    sample_indices = np.arange(n_points)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual trajectories in gray with low alpha
for idx in sample_indices:
    ax.plot(times, deformation_magnitudes[:, idx], color='gray', alpha=0.05, linewidth=0.5)

# Plot mean trajectory
mean_deformation = np.mean(deformation_magnitudes, axis=1)
ax.plot(times, mean_deformation, color='blue', linewidth=2, label='Mean')

# Plot percentiles
p5 = np.percentile(deformation_magnitudes, 5, axis=1)
p95 = np.percentile(deformation_magnitudes, 95, axis=1)
ax.fill_between(times, p5, p95, color='blue', alpha=0.2, label='5-95% range')

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Deformation magnitude (µm)")
ax.set_title(f"Mesh deformation over time - {SIMULATION_NAME}")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
