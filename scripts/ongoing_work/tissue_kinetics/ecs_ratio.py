"""Run tissue kinetics simulation over a range of ECS volume fractions."""

from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

ECS_RATIOS = [i / 100 for i in range(4, 20)]  # 10% to 25%
N_PROCESSES = 1


def run_ecs_ratio(args):
    ratio, result_root = args
    from simulation import run_simulation

    pct = int(ratio * 100)
    print(f"Starting simulation with ecs_ratio={ratio:.2f} ({pct}%)")
    run_simulation(
        ecs_ratio=ratio,
        simulation_name=f"tissue_kinetics_ecs{pct:02d}",
        result_root=str(result_root),
    )
    print(f"Finished simulation with ecs_ratio={ratio:.2f} ({pct}%)")


if __name__ == "__main__":
    set_start_method("spawn")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    result_root = Path("results") / f"ecs_ratio_{timestamp}"
    result_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(ECS_RATIOS)} ECS ratios with {N_PROCESSES} processes")
    print(f"Results will be stored in: {result_root}")

    args = [(ratio, result_root) for ratio in ECS_RATIOS]
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_ecs_ratio, args)

    print(f"All simulations complete. Results in: {result_root}")
