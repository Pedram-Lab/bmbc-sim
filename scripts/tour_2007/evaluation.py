import os
import pyvista as pv
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import datetime  

from ecsim.simulation.result_io.result_loader import ResultLoader

custom_theme = {
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 9,
    'legend.edgecolor': 'black',
    'legend.frameon': False,
    'lines.linewidth': 0.5,
    'font.family': ['Arial', 'sans-serif'],
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
}

# Apply custom theme
for key, value in custom_theme.items():
    plt.rcParams[key] = value

fig_width = 5.36  # inches
fig_height = 3.27  # inches

# Helper function to load radial concentration profile
def get_radial_profile(result_path, species_of_interest="Ca", n_points=500):
    loader = ResultLoader(result_path)
    step = len(loader) - 1
    distances = np.linspace(0.0, 0.6, n_points)
    points = [(0, d, 2.4) for d in distances]
    ds = loader.load_point_values(step, points)

    if species_of_interest not in ds.coords['species']:
        raise ValueError(f"Species '{species_of_interest}' not found in {result_path}")
    values = ds.sel(species=species_of_interest).values.flatten()
    return distances, values, ds.coords['time'].values[0]  # also returns the time


# === Load simulation data ===
#path1 = "results/tsien_bapta_2025-07-16-104253"
#path2 = "results/tsien_egta_40_2025-07-16-150535"
path3 = "results/tsien_egta_2025-07-24-070926"

#dist1, values1, time1 = get_radial_profile(path1)
#dist2, values2, time2 = get_radial_profile(path2)
dist3, values3, time2 = get_radial_profile(path3)

# === Plot ===
plt.figure(figsize=(fig_width, fig_height))
#plt.plot(dist1 * 1000, values1, label=f"BAPTA 1 mM", linestyle='--')
#plt.plot(dist2 * 1000, values2, label=f"EGTA 40 mM", linestyle='--')
plt.plot(dist3 * 1000, values3, label=f"EGTA 4.5 mM", linestyle='--')
plt.xlabel("Distance from the channel cluster (nm)")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_i$ (mM)")
plt.title("Tsien simulation, evaluation point (0, (0-0.6), 2.4)")
plt.grid(True)
plt.legend()
plt.tight_layout()


# === Save with date and time ===
now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
filename = f"tsien_simulation_visualization_{now}.pdf"
plt.savefig(filename, format="pdf")
plt.show()