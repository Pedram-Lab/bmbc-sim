import os

import xarray as xr


def get_point_values(result_dir):
    """Read the point values from a simulation."""
    zarr_path = os.path.join(result_dir, "point_data_0.zarr")
    point_data = xr.open_zarr(zarr_path)

    species_list = point_data.coords['species'].values
    time = point_data.coords['time'].values
    point = point_data.coords['point'].values[0]

    values = {}
    for species in species_list:
        ts = point_data.sel(species=species, point=point)
        values[species] = ts.to_array().values.squeeze()

    return values, time


def get_substance_values(result_dir):
    """Read the substance values from a simulation."""
    zarr_path = os.path.join(result_dir, "substance_data.zarr")
    substance_data = xr.open_zarr(zarr_path)

    species_list = substance_data.coords['species'].values
    time = substance_data.coords['time'].values
    compartment = substance_data.coords['compartment'].values[0]

    values = {}
    for species in species_list:
        ts = substance_data.sel(species=species, compartment=compartment)
        values[species] = ts.to_array().values.squeeze()

    return values, time
