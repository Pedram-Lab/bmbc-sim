"""Plot species concentration statistics in ECS over time across multiple simulations."""

import matplotlib.pyplot as plt
import numpy as np

from bmbcsim.simulation.result_io import ResultLoader

# ============ Configuration ============
SIMULATION_NAMES = [
    "tissue_kinetics",
    "tissue_kinetics_with_ecm",
]
RESULTS_ROOT = "results"
SPECIES_NAME = "Ca"  # Species to plot
CENTILE = 5
# =======================================

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10.colors

for i, sim_name in enumerate(SIMULATION_NAMES):
    loader = ResultLoader.find(simulation_name=sim_name, results_root=RESULTS_ROOT)
    regions = loader.load_regions()
    ecs_mask = np.array(regions["ecs"], dtype=bool)

    times = []
    lows = []
    medians = []
    means = []
    highs = []

    for step in range(len(loader)):
        mesh = loader.load_snapshot(step)
        time_series = loader.load_total_substance(step)
        values = np.array(mesh[SPECIES_NAME])[ecs_mask]

        times.append(float(time_series["time"]))
        lows.append(np.quantile(values, CENTILE / 100.0))
        medians.append(np.median(values))
        means.append(np.mean(values))
        highs.append(np.quantile(values, 1 - CENTILE / 100.0))

    times = np.array(times)
    color = colors[i % len(colors)]

    # Shaded area for low-high range
    ax.fill_between(times, lows, highs, alpha=0.2, color=color)

    # Mean as solid line, median as dashed
    ax.plot(times, means, linewidth=2, linestyle="-", color=color, label=f"{sim_name} mean")
    ax.plot(times, medians, linewidth=2, linestyle="--", color=color, label=f"{sim_name} median")

ax.set_xlabel("Time (ms)")
ax.set_ylabel(f"{SPECIES_NAME} in ECS (mM)")
ax.set_title(f"{SPECIES_NAME} concentration in ECS over time ({CENTILE}-{100-CENTILE}% range shaded)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
