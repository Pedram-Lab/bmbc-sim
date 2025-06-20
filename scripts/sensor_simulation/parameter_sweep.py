import subprocess
import os

import numpy as np
import xarray as xr

import ecsim

# Define parameter ranges (in mM)
buffer_kd_values = np.geomspace(1e-3, 10, 5)  # 1uM to 10mM
sensor_kd_values = np.geomspace(1e-3, 10, 5)  # 1uM to 10mM

# Function to run a single simulation
def run_simulation(buffer_kd, sensor_kd):
    """Run a sensor simulation with given buffer and sensor dissociation constants."""
    command = [
        "python",
        "scripts/sensor_simulation.py",
        "--buffer_kd", str(buffer_kd),
        "--sensor_kd", str(sensor_kd),
    ]
    subprocess.run(command, check=True)

# Run simulations sequentially for each combination of parameters
params_list = [(buffer_kd, sensor_kd)
               for buffer_kd in buffer_kd_values
               for sensor_kd in sensor_kd_values]

for param_set in params_list:
    print(f"Running simulation with buffer_kd={param_set[0]} mM and sensor_kd={param_set[1]} mM")
    run_simulation(*param_set)

# The results of the parameter sweep are the last 25 simulations
simulation_results = sorted([
    d for d in os.listdir("results") if d.startswith("sensor_")
])[-len(params_list):]

# Initialize xarray dataset
time_points = np.linspace(0, 1, 11)  # Time points from 0 to 1s in 100ms intervals
regions = ["cube:sphere"]

shape = (len(time_points), len(buffer_kd_values), len(sensor_kd_values), 2)
data_vars = {
    "parameter_sweep": (("time", "buffer_kd", "sensor_kd", "channel"), np.empty(shape)),
}
coords = {
    "time": time_points,
    "buffer_kd": buffer_kd_values,
    "sensor_kd": sensor_kd_values,
    "channel": ["free_ca", "total_ca"],
}
results = xr.Dataset(data_vars=data_vars, coords=coords)

# Collect results
for path, (buffer_kd, sensor_kd) in zip(simulation_results, params_list):
    result_loader = ecsim.ResultLoader(f"results/{path}")

    total_substance = xr.concat(
        [result_loader.load_total_substance(i) for i in range(len(result_loader))],
        dim="time",
    ).sel(region="cube:sphere")

    free_ca = total_substance.sel(species="ca")
    sensed_ca = total_substance.sel(species="ca_sensor")
    total_ca = free_ca + sensed_ca + total_substance.sel(species="ca_buffer")

    # Save results to disk
    results["parameter_sweep"].loc[
        dict(buffer_kd=buffer_kd, sensor_kd=sensor_kd)
    ] = np.array([free_ca.values, total_ca.values]).T

# Save the dataset to disk as a zarr file
results.to_zarr("results/sensor_parameter_sweep.zarr", mode="w")
