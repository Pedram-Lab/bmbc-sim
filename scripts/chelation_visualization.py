"""
This script visualizes the results from a specific chelation simulation folder.
It produces four plots:
1. Free Ca²⁺ and total Ca²⁺
2. Mobile buffer and mobile complex
3. Immobile buffer and immobile complex
4. All species together
"""

import xarray as xr
import matplotlib.pyplot as plt
import ecsim
from datetime import datetime

# Personalización de estilo
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
for key, value in custom_theme.items():
    plt.rcParams[key] = value

fig_width = 5.36  # inches
fig_height = 3.27  # inches


# Timestamp and simulation name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
simulation_name = "chelation"

# Load results
result_loader = ecsim.ResultLoader.find(
    results_root="results",
    simulation_name=simulation_name,
)

# Compose file prefix
file_prefix = f"{timestamp}_{simulation_name}"

total_substance = xr.concat(
    [result_loader.load_total_substance(i) for i in range(len(result_loader))],
    dim="time",
)

free_ca = total_substance.sel(species="ca")
immobile_buffer = total_substance.sel(species="immobile_buffer")
mobile_buffer = total_substance.sel(species="mobile_buffer")
mobile_complex = total_substance.sel(species="mobile_complex")
immobile_complex = total_substance.sel(species="immobile_complex")
potential = total_substance.sel(species="potential")
total_ca = free_ca + immobile_complex + mobile_complex

# Regiones
regions = ["cube:top", "cube:bottom"]
region_sizes = result_loader.compute_region_sizes()

# === Plot 1: Free and total calcium ===
plt.rcParams['lines.linewidth'] = 2
fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax, region in zip(axes, regions):
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
plt.suptitle("Chelation simulation - Electrostatics: True", fontsize=9) 
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_free_total_ca.pdf", bbox_inches="tight")
plt.show()
plt.close()

# === Plot 2: Mobile buffer and complex ===
fig2, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax2, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax2.plot(mobile_buffer.sel(region=region).time, mobile_buffer.sel(region=region) / region_size, linestyle="--", color="green", label="Mobile buffer")
    ax2.plot(mobile_complex.sel(region=region).time, mobile_complex.sel(region=region) / region_size, linestyle="--", color="red", label="Mobile complex")
    ax2.set_ylabel("Average concentration (mM)")
    ax2.set_title(f"{region}")
    ax2.set_xlabel("Time (ms)")
    ax2.grid(True)
for ax2 in axes:
    ax2.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Chelation simulation - Electrostatics: True", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_mobile_buffer_complex.pdf", bbox_inches="tight")
plt.show()
plt.close()

# === Plot 3: Immobile buffer and complex ===
fig3, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax3, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax3.plot(immobile_buffer.sel(region=region).time, immobile_buffer.sel(region=region) / region_size, linestyle="--", color="purple", label="Immobile buffer")
    ax3.plot(immobile_complex.sel(region=region).time, immobile_complex.sel(region=region) / region_size, linestyle="--", color="orange", label="Immobile complex")
    ax3.set_ylabel("Average concentration (mM)")
    ax3.set_title(f"{region}")
    ax3.set_xlabel("Time (ms)")
    ax3.grid(True)
for ax3 in axes:
    ax3.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Chelation simulation - Electrostatics: True", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_immobile_buffer_complex.pdf", bbox_inches="tight")
plt.show()
plt.close()

# === Plot 4: All species ===
fig4, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax4, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax4.plot(free_ca.sel(region=region).time, free_ca.sel(region=region) / region_size, label=r"Free $\mathrm{Ca}^{2+}$")
    ax4.plot(total_ca.sel(region=region).time, total_ca.sel(region=region) / region_size, linestyle="--", label=r"Total $\mathrm{Ca}^{2+}$")
    ax4.plot(mobile_buffer.sel(region=region).time, mobile_buffer.sel(region=region) / region_size, linestyle="--", label="Mobile buffer")
    ax4.plot(mobile_complex.sel(region=region).time, mobile_complex.sel(region=region) / region_size, linestyle="--", label="Mobile complex")
    ax4.plot(immobile_buffer.sel(region=region).time, immobile_buffer.sel(region=region) / region_size, linestyle="--", label="Immobile buffer")
    ax4.plot(immobile_complex.sel(region=region).time, immobile_complex.sel(region=region) / region_size, linestyle="--", label="Immobile complex")
    ax4.set_ylabel("Average concentration (mM)")
    ax4.set_title(f"{region}")
    ax4.set_xlabel("Time (ms)")
    ax4.grid(True)
for ax4 in axes:
    ax4.set_ylim(0, 2)
axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Chelation simulation - Electrostatics: True", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_all_species.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Plot 5: Potential
plt.rcParams['lines.linewidth'] = 2
fig5, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharex=True)
for ax5, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax5.plot(
        potential.sel(region=region).time,
        potential.sel(region=region) / region_size,
        label="Potential",
    )
    ax5.set_ylabel("Potential")
    ax5.set_title(f"{region}")
    ax5.set_xlabel("Time (ms)")
    ax5.grid(True)

axes[0].legend()
plt.subplots_adjust(wspace=0.3)
plt.suptitle("Chelation simulation - Electrostatics: True", fontsize=9)
#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_prefix}_potential.pdf", bbox_inches="tight")
plt.show()
plt.close()