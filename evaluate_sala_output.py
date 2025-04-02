import os

import pyvista as pv
import xarray as xr
import matplotlib.pyplot as plt

from ecsim import find_latest_results

# Find the latest folder with test data
latest_folder = find_latest_results("sala", "results")

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

dist = [0.25, 5.25, 10.25, 19.75]
for species in species_list:
    plt.figure()
    for d, point in zip(dist, points):
        ts = point_data.sel(species=species, point=point)
        ts_array = ts.to_array().values * 1000
        plt.semilogy(time, ts_array.T, label=f"Distance {d}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Concentration [µM]")
    plt.ylim(4e-2, 3e1)
    plt.xlim(0, 2000)
    plt.title(f"Species: {species}")
    plt.legend()
plt.show()
