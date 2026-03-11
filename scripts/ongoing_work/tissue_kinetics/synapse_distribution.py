"""Run tissue kinetics simulation over multiple synapse distribution seeds."""

from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

N_SEEDS = 100
N_PROCESSES = 10


def run_seed(args):
    seed, result_root = args
    from simulation import run_simulation

    print(f"Starting simulation with seed={seed}")
    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=str(result_root),
    )
    print(f"Finished simulation with seed={seed}")


if __name__ == "__main__":
    set_start_method("spawn")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    result_root = Path("results") / f"synapse_distribution_{timestamp}"
    result_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {N_SEEDS} seeds with {N_PROCESSES} processes")
    print(f"Results will be stored in: {result_root}")

    args = [(seed, result_root) for seed in range(N_SEEDS)]
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_seed, args)

    print(f"All simulations complete. Results in: {result_root}")
