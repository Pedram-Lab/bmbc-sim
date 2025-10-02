import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import bmbcsim


# Load results
loader = bmbcsim.ResultLoader.find(simulation_name="rusakov", results_root="results")
figsize = bmbcsim.plot_style("pedramlab")
plt.rcParams.update({"lines.linewidth": 2})

# Load total substance by snapshot
total_substance = xr.concat(
    [loader.load_total_substance(i) for i in range(len(loader))],
    dim="time"
)
time = total_substance.coords["time"].values

# Calculate volumes by region (for average concentration)
regions = total_substance.coords['region'].values
region_sizes = loader.compute_region_sizes()
interesting_regions = ["synapse_ecs", "presynapse", "neuropil"]

# Plot total mass in all interesting compartments (account for porosity in neuropil)
fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
for region in interesting_regions:
    data = total_substance.sel(region=region, species="Ca")
    volume = region_sizes[region]
    axes[0].plot(time, data / volume, label=region)
axes[0].legend()
axes[0].set_xlabel("Time [ms]")
axes[0].set_ylabel("Average concentration [mM]")

total_ca = (
    total_substance.sel(region="synapse_ecs", species="Ca")
    + total_substance.sel(region="presynapse", species="Ca")
    + 0.12 * total_substance.sel(region="neuropil", species="Ca")
)
mass_conservation_error = np.abs(total_ca - total_ca.isel(time=0)) / total_ca.isel(time=0)
axes[1].semilogy(total_substance["time"], mass_conservation_error)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Mass conservation error")
plt.title("Mass conservation")
plt.tight_layout()
plt.show()

# Plot concentration traces in five points of interest given in the original paper
SYNAPSE_RADIUS = 0.1  # μm
GLIA_DISTANCE = 0.03  # μm
DIST = SYNAPSE_RADIUS + GLIA_DISTANCE / 2
points = [
    [0, 0, 0],          # 1: center
    [DIST, 0, 0],       # 2: inside glia, near cleft
    [0, 0, DIST],       # 3: inside glia, far from cleft
    [0, 0, -2 * DIST],  # 4: outside glia (below)
    [0, 0, 2 * DIST],   # 5: outside glia (above)
]
point_labels = [
    "Center",
    "Inside Glia (Near Cleft)",
    "Inside Glia (Far from Cleft)",
    "Outside Glia (Below)",
    "Outside Glia (Above)",
]

# Load point evaluations
point_values = xr.concat(
    [loader.load_point_values(i, points=points) for i in range(len(loader))],
    dim="time"
)

# Plot
plt.figure(figsize=figsize)
for i, p in enumerate(points):
    plt.plot(time, point_values.sel(species="Ca").isel(point=i))
plt.ylim(0.4, 1.4)
plt.xlabel("Time [ms]")
plt.ylabel(r"$[\mathrm{Ca}^{2+}]$ (mM)")
plt.title("Calcium concentration at different radial distances")
plt.legend(point_labels)
plt.grid(True)
plt.tight_layout()
plt.show()
