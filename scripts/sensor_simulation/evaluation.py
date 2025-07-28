import xarray as xr
import matplotlib.pyplot as plt

import ecsim


result_loader = ecsim.ResultLoader.find(
    results_root="results",
    simulation_name="sensor",
)

total_substance = xr.concat(
    [result_loader.load_total_substance(i) for i in range(len(result_loader))],
    dim="time",
)

free_ca = total_substance.sel(species="ca")
sensed_ca = total_substance.sel(species="ca_sensor")
total_ca = free_ca + sensed_ca + total_substance.sel(species="ca_buffer")

regions = ["cube:left", "cube:right", "cube:sphere"]
region_sizes = result_loader.compute_region_sizes()

plt.rcParams['lines.linewidth'] = 2
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
for ax, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax.plot(
        free_ca.sel(region=region).time,
        free_ca.sel(region=region) / region_size,
        label="Free Ca",
    )
    ax.plot(
        total_ca.sel(region=region).time,
        total_ca.sel(region=region) / region_size,
        linestyle="--",
        label="Total Ca",
    )
    if region == "cube:sphere":
        ax2 = ax.twinx()
        ax2.plot(
            total_ca.sel(region=region).time,
            total_ca.sel(region=region) / region_size,
            linestyle=":",
            color="tab:green",
        )
        ax2.set_ylabel("Sensed Ca (a.u.)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_ylabel("Average concentration (mM)")
    ax.set_title(f"{region}")
    ax.set_xlabel("Time (ms)")
    ax.grid(True)

axes[0].legend()
plt.subplots_adjust(wspace=0)  # Remove gap between subplots
plt.suptitle("Free Ca and Total Ca over Time by Region")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
