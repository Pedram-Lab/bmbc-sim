import os

import pyvista as pv
import xarray as xr
import matplotlib.pyplot as plt

from ecsim import find_latest_results

# Find the latest folder with test data
latest_folder = find_latest_results("tsien", "results")

### Full snapshots
# snapshot_file = os.path.join(latest_folder, "snapshots", "snapshot_step00010.vtu")
# data = pv.read(snapshot_file)
# data.plot(scalars='Ca', show_edges=True)


### Compartment substances
zarr_path = os.path.join(latest_folder, "substance_data.zarr")
point_data = xr.open_zarr(zarr_path)

species_list = point_data.coords['species'].values
compartment = point_data.coords['compartment'].values[0]
volume = point_data.attrs['compartment_volume'][0]

plt.figure()
for species in species_list:
    ts = point_data.sel(species=species, compartment=compartment)
    ts_array = ts.to_array().values / volume
    plt.semilogy(ts['time'].values, ts_array.T, label=species)
plt.xlabel("Time [ms]")
plt.ylabel("Average concentration [mM]")
plt.title("Total substance in cell")
plt.legend()
plt.show()

### Point values
zarr_path = os.path.join(latest_folder, "point_data_0.zarr")
point_data = xr.open_zarr(zarr_path)

species_list = point_data.coords['species'].values
time = point_data.coords['time'].values
points = point_data.coords['point'].values
x_coords = [xyz[0] for xyz in point_data.attrs['point_coordinates']]

for species in species_list:
    plt.figure()
    final_concentration = point_data.sel(species=species).isel(time=-1)
    final_concentration = final_concentration.to_dataarray().squeeze()
    plt.plot(x_coords, final_concentration, label=species)
    plt.xlabel("Distance from center [µm]")
    plt.ylabel("Concentration [µM]")
    plt.title(f"Species: {species} at t={time[-1]} ms")
    plt.legend()
plt.show()