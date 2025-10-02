import subprocess
from itertools import product
from multiprocessing import Pool
from pathlib import Path

# Define the parameter sweep values
buffer_concs = [1e-6, 1e-3, 1, 1e3]  # in mmol/L
buffer_kds = [1e-6, 1e-3, 1, 1e3]    # in mmol/L

# Generate all parameter combinations
param_combinations = list(product(buffer_concs, buffer_kds))

# Absolute path to simulation.py script
script_path = Path(__file__).resolve().parent / "simulation.py"


def run_simulation(params):
    buffer_conc, buffer_kd = params

    # Build command
    cmd = [
        "python", str(script_path),
        "--buffer_conc", str(buffer_conc),
        "--buffer_kd", str(buffer_kd)
    ]

    # Show command
    print(f"Running: {cmd}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation for buffer_conc={buffer_conc}, buffer_kd={buffer_kd}")
        print(e)


if __name__ == "__main__":
    # Use all available cores (or adjust the number)
    with Pool(processes=4) as pool:
        pool.map(run_simulation, param_combinations)
