import xarray as xr
import matplotlib.pyplot as plt
import ecsim

result_loader = ecsim.ResultLoader.find(
    results_root="results",
    simulation_name="electrostatics",
)

total_substance = xr.concat(
    [result_loader.load_total_substance(i) for i in range(len(result_loader))],
    dim="time",
)

# Selection of species
free_ca = total_substance.sel(species="ca")
immobile_buffer = total_substance.sel(species="immobile_buffer")
mobile_buffer = total_substance.sel(species="mobile_buffer")
potential = total_substance.sel(species="potential")
total_ca = free_ca 

# Regions
regions = ["dish:free", "dish:substrate"]
region_sizes = result_loader.compute_region_sizes()

# Plot 1: Total Ca
total_ca_concentration = total_ca.sum(dim="region") / sum(region_sizes.values())
print(total_ca_concentration)
total_ca_concentration.plot()
plt.savefig("chelation_total_Ca.pdf", bbox_inches="tight")
plt.close()

# Plot 2: Free and total Ca
plt.rcParams['lines.linewidth'] = 2
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
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
    ax.set_ylabel("Average concentration (mM)")
    ax.set_title(f"{region}")
    ax.set_xlabel("Time (ms)")
    ax.grid(True)

axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Electrostatics simulation (electrostatics_2025-07-03-164829)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("electrostatics_total_free_ca_electrostatics_2025-07-03-164829.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Plot 3: Buffers and complex
fig2, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
for ax2, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax2.plot(
        free_ca.sel(region=region).time,
        free_ca.sel(region=region) / region_size,
        label="Free Ca",
    )
    ax2.plot(
        total_ca.sel(region=region).time,
        total_ca.sel(region=region) / region_size,
        linestyle="--",
        label="Total Ca",
    )
    ax2.plot(
        mobile_buffer.sel(region=region).time,
        mobile_buffer.sel(region=region) / region_size,
        linestyle="--",
        label="Mobile buffer",
    )
    
    ax2.plot(
        immobile_buffer.sel(region=region).time,
        immobile_buffer.sel(region=region) / region_size,
        linestyle="--",
        label="Immobile buffer",
    )
    
    ax2.set_ylabel("Average concentration (mM)")
    ax2.set_title(f"{region}")
    ax2.set_xlabel("Time (ms)")
    ax2.grid(True)

axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Electrostatics simulation (electrostatics_2025-07-03-164829)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("electrostatics_all_species_electrostatics_2025-07-03-164829.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Plot 4: Potential
plt.rcParams['lines.linewidth'] = 2
fig3, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
for ax3, region in zip(axes, regions):
    region_size = region_sizes[region]
    ax3.plot(
        potential.sel(region=region).time,
        potential.sel(region=region) / region_size,
        label="Free Ca",
    )
   
    ax3.set_ylabel("Potential")
    ax3.set_title(f"{region}")
    ax3.set_xlabel("Time (ms)")
    ax3.grid(True)

axes[0].legend()
plt.subplots_adjust(wspace=0)
plt.suptitle("Electrostatics simulation (electrostatics_2025-07-03-164829)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("electrostatics_potential_electrostatics_2025-07-03-164829.pdf", bbox_inches="tight")
plt.show()
plt.close()
