import os

import xarray as xr
import matplotlib.pyplot as plt

from ecsim import find_latest_results

# Find the latest folder with test data
latest_folder = find_latest_results("tony", "results")

### Point values
zarr_path = os.path.join(latest_folder, "point_data_0.zarr")
point_data = xr.open_zarr(zarr_path)

species_list = point_data.coords['species'].values
time = point_data.coords['time'].values
points = point_data.coords['point'].values
x_coords = [xyz[0] for xyz in point_data.attrs['point_coordinates']]
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
# Apply the theme to matplotlib globally
for key, value in custom_theme.items():
    plt.rcParams[key] = value
fig_width = 3  # pulgadas
fig_height = 2  # pulgadas

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
point1 = point_data.sel(species='ca').isel(point=0).to_dataarray().squeeze()
point2 = point_data.sel(species='ca').isel(point=1).to_dataarray().squeeze()
plt.plot(time / 1e3, point1, '-x', color='k', label='In buffer region')
plt.plot(time / 1e3, point2, '-o', color='k', markersize=2, label='Far from buffer')
plt.axvline(x=60, linestyle='--', color='k', label='Dilution event')
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mM)")
# plt.title("Calcium concentration over time")
plt.legend()
plt.tight_layout()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
