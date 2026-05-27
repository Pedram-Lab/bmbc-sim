"""Run tissue kinetics simulation over multiple synapse distribution seeds."""

import argparse
from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

N_SEEDS = 100
N_PROCESSES = 10


def run_seed(args):
    seed, result_root, ecs_ratio = args
    from simulation import run_simulation

    print(f"Starting simulation with seed={seed}")
    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=str(result_root),
        ecs_ratio=ecs_ratio,
    )
    print(f"Finished simulation with seed={seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ecs", type=float, default=0.19,
        help="ECS volume fraction (e.g. 0.04, 0.09, 0.142, 0.19). Default: 0.19",
    )
    cli_args = parser.parse_args()
    ecs_ratio = cli_args.ecs

    set_start_method("spawn")

    suffix = int(100 * (ecs_ratio + 0.06))  # 0.06 is the baseline ECS ratio in the geometry
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    result_root = Path("results") / f"synapse_distribution_ecs_{suffix}_{timestamp}"
    result_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {N_SEEDS} seeds with {N_PROCESSES} processes (ecs_ratio={ecs_ratio})")
    print(f"Results will be stored in: {result_root}")

    args = [(seed, result_root, ecs_ratio) for seed in range(N_SEEDS)]
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_seed, args)

    print(f"All simulations complete. Results in: {result_root}")
