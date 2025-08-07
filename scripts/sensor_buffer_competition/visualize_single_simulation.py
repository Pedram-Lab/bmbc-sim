"""
This script visualizes the results from a specific chelation simulation folder.
It produces four plots:
1. Free Ca²⁺ and total Ca²⁺
2. Mobile buffer and mobile complex
3. Mobile sensor and complex
4. All species together
"""
from datetime import datetime
from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt

import ecsim

# Customizing the plot style
fig_width, fig_height = ecsim.plot_style("pedramlab")


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
simulation_name = "sensor_buffer_competition_conc1000.0_kd1e-06"
result_loader = ecsim.ResultLoader.find(
    results_root=Path("results") / simulation_name,
    simulation_name=simulation_name
)


# Compose file prefix
file_prefix = f"{timestamp}_{simulation_name}"

total_substance = xr.concat(
    [result_loader.load_total_substance(i) for i in range(len(result_loader))],
    dim="time",
)


free_ca = total_substance.sel(species="ca")
buffer = total_substance.sel(species="buffer")
buffer_complex = total_substance.sel(species="buffer_complex")
sensor = total_substance.sel(species="sensor")
sensor_complex = total_substance.sel(species="sensor_complex")
total_ca = free_ca + buffer_complex + sensor_complex

# Regions
compartments = ["top", "bottom"] 

region_sizes = result_loader.compute_region_sizes()

# === Plot 1: Free and total calcium ===
plt.rcParams['lines.linewidth'] = 2
fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax.plot(free_ca.sel(region=region).time, free_ca.sel(region=region) / region_size, label=r"Free $\mathrm{Ca}^{2+}$")
    ax.plot(total_ca.sel(region=region).time, total_ca.sel(region=region) / region_size, linestyle="--", label=r"Total $\mathrm{Ca}^{2+}$")
    ax.set_ylabel("Average concentration (mM)")
    ax.set_title(f"{region}")
    ax.set_xlabel("Time (ms)")
    ax.grid(True)
for ax in axes:
    ax.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle(r"Sensor-Buffer competition $[B]_{\mathrm{bottom}}$ = 1 nM, $Kd_B = 1.0\,\mu M$", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_free_total_ca.pdf", bbox_inches="tight")
plt.show()
plt.close()

# === Plot 2: Buffer and complex ===
fig2, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax2, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax2.plot(buffer.sel(region=region).time, buffer.sel(region=region) / region_size, color="green", label="Buffer")
    ax2.plot(buffer_complex.sel(region=region).time, buffer_complex.sel(region=region) / region_size, linestyle="--", color="red", label="Buffer complex")
    ax2.set_ylabel("Average concentration (mM)")
    ax2.set_title(f"{region}")
    ax2.set_xlabel("Time (ms)")
    ax2.grid(True)
for ax2 in axes:
    ax2.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Sensor-Buffer competition", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_buffer_complex.pdf", bbox_inches="tight")
plt.show()
plt.close()

# === Plot 3: Sensor and complex ===
fig3, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax3, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax3.plot(sensor.sel(region=region).time, sensor.sel(region=region) / region_size, color="purple", label="Sensor")
    ax3.plot(sensor_complex.sel(region=region).time, sensor_complex.sel(region=region) / region_size, linestyle="--", color="orange", label="Sensor complex")
    ax3.set_ylabel("Average concentration (mM)")
    ax3.set_title(f"{region}")
    ax3.set_xlabel("Time (ms)")
    ax3.grid(True)
for ax3 in axes:
    ax3.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle(r"Sensor-Buffer competition $[B]_{\mathrm{bottom}}$ = 1 nM, $Kd_B = 1.0\,\mu M$", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_sensor_complex.pdf", bbox_inches="tight")
plt.show()
plt.close()

# === Plot 4: All species ===
fig4, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax4, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax4.plot(free_ca.sel(region=region).time, free_ca.sel(region=region) / region_size, label=r"Free $\mathrm{Ca}^{2+}$")
    ax4.plot(total_ca.sel(region=region).time, total_ca.sel(region=region) / region_size, linestyle="--", label=r"Total $\mathrm{Ca}^{2+}$")
    ax4.plot(buffer.sel(region=region).time, buffer.sel(region=region) / region_size, label="Buffer")
    ax4.plot(sensor_complex.sel(region=region).time, sensor_complex.sel(region=region) / region_size, linestyle="--", label="Sensor complex")
    ax4.plot(sensor.sel(region=region).time, sensor.sel(region=region) / region_size, label="Sensor")
    ax4.plot(sensor_complex.sel(region=region).time, sensor_complex.sel(region=region) / region_size, linestyle="--", label="Sensor complex")
    ax4.set_ylabel("Average concentration (mM)")
    ax4.set_title(f"{region}")
    ax4.set_xlabel("Time (ms)")
    ax4.grid(True)
for ax4 in axes:
    ax4.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle(r"Sensor-Buffer competition $[B]_{\mathrm{bottom}}$ = 1 nM, $Kd_B = 1.0\,\mu M$", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_all_species.pdf", bbox_inches="tight")
plt.show()
plt.close()

Kd_sensor = 1
estimated_ca_sensor = Kd_sensor * sensor_complex / (sensor - sensor_complex)
# === Plot 5: Compare [Ca] simulated vs estimated ===
fig5, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax5, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax5.plot(estimated_ca_sensor.sel(region=region).time, estimated_ca_sensor.sel(region=region) / region_size, linestyle="--", label=r"Estimated $\mathrm{Ca}^{2+}$ (sensor)")
    ax5.set_ylabel("Average concentration (mM)")
    ax5.set_title(f"{region}")
    ax5.set_xlabel("Time (ms)")
    ax5.grid(True)
for ax5 in axes:
    ax5.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Estimated $[Ca^{2+}]$", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_estimated_ca.pdf", bbox_inches="tight")
plt.show()
plt.close()
