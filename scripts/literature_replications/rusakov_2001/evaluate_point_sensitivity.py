"""Sensitivity analysis for evaluation point placement.

Loads the latest normal simulation (not from a parameter sweep) and evaluates
Ca²⁺ concentration at the baseline points plus small perturbations in x and y.
One figure per evaluation point, showing baseline vs. shifted traces.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import bmbcsim

# Geometry constants (must match current simulation.py values)
SYNAPSE_RADIUS = 0.1   # μm
GLIA_DISTANCE = 0.03   # μm
DIST = SYNAPSE_RADIUS + GLIA_DISTANCE / 2

# Baseline evaluation points
BASELINE_POINTS = [
    [0, 0, 0],
    [DIST, 0, 0],
    [0, 0, DIST],
    [0, 0, -2 * DIST],
    [0, 0, 2 * DIST],
]
POINT_LABELS = [
    "Center",
    "Inside Glia (Near Cleft)",
    "Inside Glia (Far from Cleft)",
    "Outside Glia (Below)",
    "Outside Glia (Above)",
]

# Perturbation magnitude in μm
SHIFT = 0.01  # 10 nm


def make_shifted_points(base_point, shift):
    """Generate shifted variants of a single point in x and y."""
    p = np.array(base_point)
    variants = {}
    for axis, axis_name in [(0, "x"), (1, "y")]:
        for sign, sign_name in [(-1, "-"), (1, "+")]:
            shifted = p.copy()
            shifted[axis] += sign * shift
            variants[f"{axis_name}{sign_name}{shift*1000:.0f}nm"] = shifted.tolist()
    return variants


# Load latest normal simulation
loader = bmbcsim.ResultLoader.find(simulation_name="rusakov", results_root="results")
figsize = bmbcsim.plot_style("pedramlab")

# Load time axis
total_substance = xr.concat(
    [loader.load_total_substance(i) for i in range(len(loader))],
    dim="time",
)
time = total_substance.coords["time"].values

# For each evaluation point, create a figure
for pt_idx, (base_point, label) in enumerate(zip(BASELINE_POINTS, POINT_LABELS)):
    variants = make_shifted_points(base_point, SHIFT)

    # Build all points for a single load call: baseline + 4 shifts
    all_points = [base_point] + [v for v in variants.values()]
    all_labels = ["baseline"] + list(variants.keys())

    point_values = xr.concat(
        [loader.load_point_values(i, points=all_points) for i in range(len(loader))],
        dim="time",
    )
    ca_values = point_values.sel(species="Ca")

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Left panel: x perturbations
    axes[0].plot(time, ca_values.isel(point=0), color="black", label="baseline")
    for i, lbl in enumerate(all_labels[1:], start=1):
        if lbl.startswith("x"):
            color = "tab:blue" if "-" in lbl else "tab:red"
            axes[0].plot(time, ca_values.isel(point=i), color=color, label=lbl)
    axes[0].set_title(f"{label} — x shift")
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel(r"$[\mathrm{Ca}^{2+}]$ (mM)")
    axes[0].legend(fontsize=7)

    # Right panel: y perturbations
    axes[1].plot(time, ca_values.isel(point=0), color="black", label="baseline")
    for i, lbl in enumerate(all_labels[1:], start=1):
        if lbl.startswith("y"):
            color = "tab:blue" if "-" in lbl else "tab:red"
            axes[1].plot(time, ca_values.isel(point=i), color=color, label=lbl)
    axes[1].set_title(f"{label} — y shift")
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel(r"$[\mathrm{Ca}^{2+}]$ (mM)")
    axes[1].legend(fontsize=7)

    fig.suptitle(f"Point sensitivity: {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"point_sensitivity_{pt_idx}_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {label}")

print("Done.")
