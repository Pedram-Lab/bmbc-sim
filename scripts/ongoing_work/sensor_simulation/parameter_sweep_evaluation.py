import xarray as xr
import matplotlib.pyplot as plt


RESULTS_FILE = "results/sensor_parameter_sweep.zarr"
TOTAL_OR_FREE = "free_ca"

# Select a single time step
results = xr.open_zarr(RESULTS_FILE)
data = results["parameter_sweep"].isel(time=10).sel(channel=TOTAL_OR_FREE)

# Plot the heat map
plt.figure(figsize=(8, 6))
plt.imshow(data, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Ca substance in sensor region (amol)")
plt.title(f"Heat map of '{TOTAL_OR_FREE}' at 1s")
plt.xlabel("Buffer Kd (mM)")
plt.ylabel("Sensor Kd (mM)")
plt.xticks(ticks=range(len(data["buffer_kd"])), labels=data["buffer_kd"].values)
plt.yticks(ticks=range(len(data["sensor_kd"])), labels=data["sensor_kd"].values)
plt.show()
