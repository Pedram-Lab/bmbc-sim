"""Compare the three tour_2007 buffer runs, taking the latest run of each.

The figure is written to <run directory>/plots/ of the *first* buffer in EXPERIMENTS
(EGTA 4.5 mM) -- a comparison belongs to no single run, so one of them has to host it.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import bmbcsim

def get_radial_profile(buffer_name, *, z, species_of_interest="Ca", n_points=50):
    """Load the concentration profile for a specific buffer and species
    evaluated radially outward from the channel cluster.

    :param buffer_name: Name of the buffer used in the simulation (egta or bapta).
    :param species_of_interest: The species to extract from the simulation.
    :param n_points: Number of points to evaluate in the radial profile.
    :return: The run's loader, the distances and values of the specified species at
        those distances, and the time at which the data was recorded.
    """
    sim_name = "tour_" + buffer_name.lower()
    loader = bmbcsim.ResultLoader.find(simulation_name=sim_name, results_root="results")
    last_step = len(loader) - 1
    distances = np.linspace(0.0, 0.6, n_points)
    points = [(0, d, z) for d in distances]
    ds = loader.load_point_values(last_step, points)

    if species_of_interest not in ds.coords['species']:
        raise ValueError(f"Species '{species_of_interest}' not found in {sim_name}")
    sim_values = ds.sel(species=species_of_interest).values.flatten()
    return loader, distances, sim_values, ds.coords['time'].values[0]


figsize = bmbcsim.plot_style("pedramlab")
EXPERIMENTS = [
    ("EGTA_low", "EGTA 4.5 mM"),
    ("EGTA_high", "EGTA 40 mM"),
    ("BAPTA", "BAPTA 1 mM"),
]
Z = 2.95

# Plot all buffers
plt.figure(figsize=figsize)
loaders = []
for name, label in EXPERIMENTS:
    try:
        loader, dist, values, time = get_radial_profile(name, z=Z)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping {name} (no valid data found)")
        continue
    loaders.append(loader)
    plt.plot(dist * 1000, values, label=label, marker='o')
plt.xlabel("Distance from the channel cluster (nm)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_i$ (mM)")
plt.title(f"Tour et al. 2007, evaluation at z={Z:.2f} µm, t={time:.2f} ms")
plt.grid(True)
plt.legend()
plt.tight_layout()


# === Save next to the first run that was found ===
if not loaders:
    raise SystemExit("No tour runs found in results/ -- nothing to plot.")
plot_path = os.path.join(loaders[0].plot_dir, "buffer_comparison.pdf")
plt.savefig(plot_path, format="pdf")
print(f"Figure written to {plot_path}")
plt.show()
