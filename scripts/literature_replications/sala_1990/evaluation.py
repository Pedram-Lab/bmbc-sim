import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import bmbcsim


# === Load results ===
loader = bmbcsim.ResultLoader.find(simulation_name="sala", results_root="results")
figsize = bmbcsim.plot_style("pedramlab")

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
plt.figure(figsize=figsize)
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
distances = [0.25, 5.0, 10.0, 20.0]  # μm from the center
points = [(20.0 - d, 0, 0) for d in distances]  # points along the x axis

# === Prepare structure to store results ===
TARGET_SPECIES = "Ca"
times = []
concentration_data = []

# === Iterate over snapshots ===
for step in range(len(loader)):
    ds = loader.load_point_values(step, points)
    if TARGET_SPECIES not in ds.coords['species']:
        raise ValueError(f"Species '{TARGET_SPECIES}' not found.")
    values = ds.sel(species=TARGET_SPECIES).values.flatten()  # shape: (n_points,)
    concentration_data.append(values)
    times.append(ds.coords['time'].values[0])  # in seconds

# === Convert to numpy arrays ===
concentration_data = np.array(concentration_data)  # shape: (n_steps, n_points)
times = np.array(times)

# === Plot ===
plt.figure(figsize=figsize)
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
