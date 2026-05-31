"""Run tissue kinetics simulation over multiple synapse distribution seeds."""
import sys
from pathlib import Path
import argparse
from datetime import datetime
from pathlib import Path

from dask.distributed import Client, as_completed

from bmbcsim.utils import create_cluster

N_SEEDS = 100
N_WORKERS = 10


def run_seed(seed, result_root, ecs_ratio):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simulation import run_simulation

    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=result_root,
        ecs_ratio=ecs_ratio,
    )
    return seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ecs", type=float, default=0.19,
        help="ECS volume fraction (e.g. 0.04, 0.09, 0.142, 0.19). Default: 0.19",
    )
    cli_args = parser.parse_args()
    ecs_ratio = cli_args.ecs

    suffix = int(100 * (ecs_ratio + 0.06))  # 0.06 is the baseline ECS ratio in the geometry
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    result_root = (Path("results") / f"synapse_distribution_ecs_{suffix}_{timestamp}").resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {N_SEEDS} seeds with {N_WORKERS} workers (ecs_ratio={ecs_ratio})")
    print(f"Results will be stored in: {result_root}")

    with create_cluster("local", n_workers=N_WORKERS) as cluster, Client(cluster) as client:
        futures = client.map(
            run_seed,
            list(range(N_SEEDS)),
            result_root=str(result_root),
            ecs_ratio=ecs_ratio,
        )
        for future in as_completed(futures):
            print(f"Finished simulation with seed={future.result()}")

    print(f"All simulations complete. Results in: {result_root}")
