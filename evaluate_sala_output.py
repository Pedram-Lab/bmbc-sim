import os

import pyvista as pv
import xarray as xr
import matplotlib.pyplot as plt

# Find the latest folder with test data
results_dir = "results"
all_folders = [
    d for d in os.listdir(results_dir)
    if d.startswith("sala_") and os.path.isdir(os.path.join(results_dir, d))
]
if not all_folders:
    raise RuntimeError("No debug folders found in the results directory.")

latest_folder = max(all_folders, key=lambda d: os.path.getctime(os.path.join(results_dir, d)))
print("Using folder:", latest_folder)

### Full snapshots
snapshot_file = os.path.join(results_dir, latest_folder, "snapshots", "snapshot_step00010.vtu")
data = pv.read(snapshot_file)
data.plot(scalars='Ca', show_edges=True)


### Compartment substances
zarr_path = os.path.join(results_dir, latest_folder, "substance_data.zarr")
point_data = xr.open_zarr(zarr_path)

species_list = point_data.coords['species'].values
compartment_list = point_data.coords['compartment'].values

for species in species_list:
    plt.figure()
    for compartment in compartment_list:
        ts = point_data.sel(species=species, compartment=compartment)
        ts_array = ts.to_array().values  # convert the Dataset to a DataArray
        plt.plot(ts['time'].values, ts_array.T, label=f"Compartment: {compartment}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Substance [amol]")
    plt.title(f"Species: {species}")
    plt.legend()
plt.show()

### Point values
zarr_path = os.path.join(results_dir, latest_folder, "point_data_0.zarr")
point_data = xr.open_zarr(zarr_path)

species_list = point_data.coords['species'].values
time = point_data.coords['time'].values
points = point_data.coords['point'].values
x_coords = [xyz[0] for xyz in point_data.attrs['point_coordinates']]

for species in species_list:
    plt.figure()
    for point in points:
        ts = point_data.sel(species=species, point=point)
        ts_array = ts.to_array().values
        plt.semilogy(time, ts_array.T, label=f"Point {point}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Concentration [mM]")
    plt.title(f"Species: {species}")
    plt.legend()
plt.show()
