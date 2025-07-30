import datetime

import matplotlib.pyplot as plt
import numpy as np

import ecsim

def get_radial_profile(buffer_name, *, z, species_of_interest="Ca", n_points=50):
    """Load the concentration profile for a specific buffer and species
    evaluated radially outward from the channel cluster.

    :param buffer_name: Name of the buffer used in the simulation (egta or bapta).
    :param species_of_interest: The species to extract from the simulation.
    :param n_points: Number of points to evaluate in the radial profile.
    :return: Distances and values of the specified species at those distances
        as well as the time at which the data was recorded.
    """
    sim_name = "tour_" + buffer_name.lower()
    loader = ecsim.ResultLoader.find(simulation_name=sim_name, results_root="results")
    last_step = len(loader) - 1
    distances = np.linspace(0.0, 0.6, n_points)
    points = [(0, d, z) for d in distances]
    ds = loader.load_point_values(last_step, points)

    if species_of_interest not in ds.coords['species']:
        raise ValueError(f"Species '{species_of_interest}' not found in {sim_name}")
    sim_values = ds.sel(species=species_of_interest).values.flatten()
    return distances, sim_values, ds.coords['time'].values[0]


figsize = ecsim.plot_style("pedramlab")
EXPERIMENTS = [
    ("EGTA_low", "EGTA 4.5 mM"),
    ("EGTA_high", "EGTA 40 mM"),
    ("BAPTA", "BAPTA 1 mM"),
]
Z = 2.95

# Plot all buffers
plt.figure(figsize=figsize)
for name, label in EXPERIMENTS:
    try:
        dist, values, time = get_radial_profile(name, z=Z)
    except (ValueError, RuntimeError) as e:
        print(f"Skipping {name} (no valid data found)")
        continue
    plt.plot(dist * 1000, values, label=label, marker='o')
plt.xlabel("Distance from the channel cluster (nm)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_i$ (mM)")
plt.title(f"Tour et al. 2007, evaluation at z={Z:.2f} µm, t={time:.2f} ms")
plt.grid(True)
plt.legend()
plt.tight_layout()


# === Save with date and time ===
now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
FILENAME = f"tour_2007_visualization_{now}.pdf"
plt.savefig(FILENAME, format="pdf")
plt.show()
