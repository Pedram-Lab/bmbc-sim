import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
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

# === Load results ===
loader = ResultLoader.find(simulation_name="sala", results_root="results")

# === Load total substance by snapshot ===
total_substance = xr.concat(
    [loader.load_total_substance(i) for i in range(len(loader))],
    dim="time"
)
total_substance["time"] = [loader.snapshots[i][0].value for i in range(len(loader))]

species_list = total_substance.coords['species'].values
regions = total_substance.coords['region'].values

# === Calculate volumes by region (for average concentration) ===
region_sizes = loader.compute_region_sizes()

# === Plot all species in the first region ===
region = regions[0]
plt.figure(figsize=(fig_width, fig_height))
for species in species_list:
    data = total_substance.sel(region=region, species=species)
    volume = region_sizes[region]
    plt.semilogy(total_substance["time"], data / volume, label=species)
plt.xlabel("Time [s]")
plt.ylabel("Average concentration [mM]")
plt.title(f"Total concentration in region '{region}'")
plt.legend()
plt.tight_layout()
plt.savefig("sala_species_concentrations.pdf", bbox_inches="tight")
plt.show()

# === Define points in Cartesian coordinates (x, y, z) ===
# For example: points along the x axis from near the membrane (20 μm) to the center (0 μm)
distances = [19.75, 10.25, 5.25, 0.25]  # μm from the center
points = [(d, 0, 0) for d in distances]  # points along the x axis

# === Prepare structure to store results ===
species_of_interest = "Ca"
times = []
concentration_data = []

# === Iterate over snapshots ===
for step in range(len(loader)):
    ds = loader.load_point_values(step, points)
    if species_of_interest not in ds.coords['species']:
        raise ValueError(f"Species '{species_of_interest}' not found.")
    values = ds.sel(species=species_of_interest).values.flatten()  # shape: (n_points,)
    concentration_data.append(values)
    times.append(ds.coords['time'].values[0])  # in seconds

# === Convert to numpy arrays ===
concentration_data = np.array(concentration_data)  # shape: (n_steps, n_points)
times = np.array(times)

# === Plot ===
plt.figure(figsize=(fig_width, fig_height))
for i, d in enumerate(distances):
    plt.semilogy(times / 1000, concentration_data[:, i] * 1000, label=f"{d} µm")
plt.xlabel("Time [s]")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]_i$ (μM)")
#plt.title("Calcium concentration at different radial distances")
plt.legend()
# ticks_y = [0.1, 0.3, 1, 3, 10, 30]
# plt.yticks(ticks_y, [str(t) for t in ticks_y])
# plt.tick_params(axis='both')
plt.grid(True)
plt.tight_layout()
plt.savefig("sala_ca_point_profiles.pdf", format="pdf")
plt.show()
