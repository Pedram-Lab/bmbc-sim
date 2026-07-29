"""
This script visualizes the results from a single simulation.py run.
It produces four plots:
1. Free Ca²⁺ and total Ca²⁺
2. Mobile buffer and mobile complex
3. Mobile sensor and complex
4. All species together

Pass a run directory to plot a specific run (e.g. one out of a sweep tree);
without one, the latest simulation.py run in results/ is used. Figures are written
to <run directory>/plots/.

    uv run .../visualize_single_simulation.py
    uv run .../visualize_single_simulation.py results/<stamp>_sensor_buffer_competition
"""
import os
import sys

import astropy.units as u
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import bmbcsim
from bmbcsim.simulation.result_io import result_loader

from simulation import Config

# Customizing the plot style
fig_width, fig_height = bmbcsim.plot_style("pedramlab")

# Parameters of the run being plotted, read off the config rather than repeated
# here. Override them on the Config call to describe a non-default run.
CONFIG = Config()
mM = u.mmol / u.L
BUFFER_CONC = CONFIG.buffer.concentration.to_value(mM)
BUFFER_KD = CONFIG.buffer.kd.to_value(mM)
SENSOR_KD = CONFIG.sensor.kd.to_value(mM)

result_loader = result_loader.ResultLoader.open(
    sys.argv[1] if len(sys.argv) > 1 else None,
    simulation_name=CONFIG.simulation_name,
    results_root="results",
    # time_stamp="2025-08-08-075312",
)
# The run directory is already timestamped, so the figures inside it need no prefix.
PLOT_DIR = result_loader.plot_dir

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
    ax.plot(
        free_ca.sel(region=region).time,
        free_ca.sel(region=region) / region_size,
        label=r"Free $\mathrm{Ca}^{2+}$",
    )
    ax.plot(
        total_ca.sel(region=region).time,
        total_ca.sel(region=region) / region_size,
        linestyle="--",
        label=r"Total $\mathrm{Ca}^{2+}$",
    )
    ax.set_ylabel("Average concentration (mM)")
    ax.set_title(f"{region}")
    ax.set_xlabel("Time (ms)")
    ax.grid(True)
    ax.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle(
    r"Sensor-Buffer competition $[B]_{\mathrm{bottom}}$ = %.0e mM, $Kd_B$ = %.0e mM"
    % (BUFFER_CONC, BUFFER_KD),
    fontsize=9,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "free_total_ca.pdf"), bbox_inches="tight")
plt.show()
plt.close()

# === Plot 2: Buffer and complex ===
fig2, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax2, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax2.plot(
        buffer.sel(region=region).time,
        buffer.sel(region=region) / region_size,
        color="green",
        label="Buffer",
    )
    ax2.plot(
        buffer_complex.sel(region=region).time,
        buffer_complex.sel(region=region) / region_size,
        linestyle="--",
        color="red",
        label="Buffer complex",
    )
    ax2.set_ylabel("Average concentration (mM)")
    ax2.set_title(f"{region}")
    ax2.set_xlabel("Time (ms)")
    ax2.grid(True)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Sensor-Buffer competition", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "buffer_complex.pdf"), bbox_inches="tight")
plt.show()
plt.close()

# === Plot 3: Sensor and complex ===
fig3, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax3, region in zip(axes, compartments):
    region_size = region_sizes[region]
    ax3.plot(
        sensor.sel(region=region).time,
        sensor.sel(region=region) / region_size,
        color="purple",
        label="Sensor",
    )
    ax3.plot(
        sensor_complex.sel(region=region).time,
        sensor_complex.sel(region=region) / region_size,
        linestyle="--",
        color="orange",
        label="Sensor complex",
    )
    ax3.set_ylabel("Average concentration (mM)")
    ax3.set_title(f"{region}")
    ax3.set_xlabel("Time (ms)")
    ax3.grid(True)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle(
    r"Sensor-Buffer competition $[B]_{\mathrm{bottom}}$ = %.0e mM, $Kd_B$ = %.0e mM"
    % (BUFFER_CONC, BUFFER_KD),
    fontsize=9,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "sensor_complex.pdf"), bbox_inches="tight")
plt.show()
plt.close()

# === Plot 4: mass conservation ===
def get_average_concentration(*species):
    """Get average concentration and it's initial value."""
    total_volume = sum(region_sizes.values())
    avg_concentration = 0
    for s in species:
        avg_concentration += (s.sel(region="top") + s.sel(region="bottom")) / total_volume
    return avg_concentration, avg_concentration.isel(time=0)

fig4, ax4 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
avg_ca, initial_ca = get_average_concentration(free_ca, buffer_complex, sensor_complex)
ax4.semilogy(
    avg_ca.time,
    np.abs(avg_ca - initial_ca) / initial_ca,
    label=r"$\mathrm{Ca}^{2+}$",
)
avg_buffer, initial_buffer = get_average_concentration(buffer, buffer_complex)
ax4.semilogy(
    avg_buffer.time,
    np.abs(avg_buffer - initial_buffer) / initial_buffer,
    label=r"Buffer",
)
avg_sensor, initial_sensor = get_average_concentration(sensor, sensor_complex)
ax4.semilogy(
    avg_sensor.time,
    np.abs(avg_sensor - initial_sensor) / initial_sensor,
    label="Sensor",
)
ax4.set_ylabel("Deviation from initial concentration (relative)")
ax4.set_xlabel("Time (ms)")
ax4.grid(True)
ax4.set_ylim(0, 2)
ax4.legend()
plt.subplots_adjust(wspace=0)
plt.suptitle(
    r"Sensor-Buffer competition $[B]_{\mathrm{bottom}}$ = %.0e mM, $Kd_B$ = %.0e mM"
    % (BUFFER_CONC, BUFFER_KD),
    fontsize=9,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "mass_conservation.pdf"), bbox_inches="tight")
plt.show()
plt.close()

# === Plot 5: Compare [Ca] simulated vs estimated ===
estimated_ca = SENSOR_KD * sensor_complex / sensor
fig5, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax5, region in zip(axes, compartments):
    ax5.plot(
        estimated_ca.sel(region=region).time,
        estimated_ca.sel(region=region),
        linestyle="--",
        label=r"Estimated $\mathrm{Ca}^{2+}$ (sensor)",
    )
    ax5.set_ylabel("Average concentration (mM)")
    ax5.set_title(f"{region}")
    ax5.set_xlabel("Time (ms)")
    ax5.grid(True)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Estimated $[Ca^{2+}]$", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "estimated_ca.pdf"), bbox_inches="tight")
plt.show()
plt.close()
