"""
This script generates heatmaps showing the ratio of estimated calcium to actual calcium
for different buffer concentrations and Kd values in both compartments.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import ecsim

# Define the parameter sweep values (from parameter_sweep.py)
buffer_concs = [1e-6, 1e-3, 1.0, 1e3]  # in mmol/L
buffer_kds = [1e-6, 1e-3, 1.0, 1e3]    # in mmol/L
buffer_conc_names = ["1e-06", "0.001", "1.0", "1000.0"]
buffer_kd_names = ["1e-06", "0.001", "1.0", "1000.0"]
Kd_sensor = 1  # from visualize_single_simulation.py

# Initialize arrays to store results
ratios_top = np.zeros((len(buffer_concs), len(buffer_kds)))
ratios_bottom = np.zeros((len(buffer_concs), len(buffer_kds)))

# Results root directory
results_root = Path(__file__).resolve().parents[2] / "results" / "heatmap_data_plot"

# Loop through all parameter combinations
running_count = 0
for j, conc in enumerate(buffer_concs):
    for i, kd in enumerate(buffer_kds):
        # Construct simulation name pattern
        running_count += 1
        conc_str = buffer_conc_names[j]
        kd_str = buffer_kd_names[i]
        sim_name = f"s{running_count}_sensor_buffer_competition_conc{conc_str}_kd{kd_str}"

        try:
            # Try to load the most recent result for this parameter combination
            result_loader = ecsim.ResultLoader.find(
                simulation_name=sim_name,
                results_root=results_root
            )

            # Load only the last timestep directly
            last_timestep = result_loader.load_total_substance(-1)

            # Extract species
            sensor = last_timestep.sel(species="sensor")
            sensor_complex = last_timestep.sel(species="sensor_complex")
            ca = last_timestep.sel(species="ca")

            # Calculate estimated calcium
            region_sizes = result_loader.compute_region_sizes()
            estimated_ca = Kd_sensor * sensor_complex / sensor

            # Calculate ratios for both compartments
            # (actual calcium is substance -> convert to concentration)
            for compartment, ratio_array in [("top", ratios_top), ("bottom", ratios_bottom)]:
                ca_est = estimated_ca.sel(region=compartment)
                ca_act = ca.sel(region=compartment) / region_sizes[compartment]
                ratio_array[i, j] = (ca_est / ca_act).squeeze().values

        except Exception as e:
            print(f"Error processing {sim_name}: {e}")
            ratios_top[i, j] = np.nan
            ratios_bottom[i, j] = np.nan

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Create logarithmic spacing for the plot
X, Y = np.meshgrid(np.arange(len(buffer_concs)), np.arange(len(buffer_kds)))


def plot_heatmap(data, ax, title):
    """Create a heatmap using pcolormesh."""
    im = ax.pcolormesh(X, Y, data, cmap='viridis', shading='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(buffer_concs)))
    ax.set_yticks(np.arange(len(buffer_kds)))
    ax.set_xticklabels([f"{conc:.0e}" for conc in buffer_concs])
    ax.set_yticklabels([f"{kd:.0e}" for kd in buffer_kds])

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Estimated/Actual Ca²⁺ ratio')

    # Labels and title
    ax.set_xlabel('Buffer concentration (mM)')
    ax.set_ylabel('Buffer Kd (mM)')
    ax.set_title(title)


# Plot heatmaps
plot_heatmap(ratios_top, ax1, 'Top compartment')
plot_heatmap(ratios_bottom, ax2, 'Bottom compartment')

plt.suptitle('Ratio of estimated to actual Ca²⁺ concentration')
plt.tight_layout()
plt.savefig('parameter_sweep_heatmap.pdf', bbox_inches='tight')
plt.show()
