"""Run tissue kinetics simulation over a range of ECS volume fractions."""

from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

# Define the range of ECS ratios to simulate and the number of processes to use
# Some values have to be slightly adapted to avoid numerical issues during meshing (e.g., 0.11 -> 0.112)
ECS_RATIOS = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.112, 0.12, 0.134, 0.142, 0.15, 0.16, 0.17, 0.18, 0.19]
N_PROCESSES = 1

# Toggle ECM binding and mechanics (mechanics implies ECM)
WITH_ECM = False
WITH_MECHANICS = False


def run_ecs_ratio(args):
    ratio, result_root, with_ecm, with_mechanics = args
    from simulation import run_simulation

    pct = int(ratio * 100)
    suffix = ""
    if with_mechanics:
        suffix = "_mechanics"
    elif with_ecm:
        suffix = "_ecm"
    print(f"Starting simulation with ecs_ratio={ratio:.2f} ({pct}%){suffix}")
    run_simulation(
        ecs_ratio=ratio,
        simulation_name=f"tissue_kinetics{suffix}_ecs{pct:02d}",
        result_root=str(result_root),
        with_ecm=with_ecm,
        with_mechanics=with_mechanics,
    )
    print(f"Finished simulation with ecs_ratio={ratio:.2f} ({pct}%){suffix}")


if __name__ == "__main__":
    set_start_method("spawn")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    folder_suffix = ""
    if WITH_MECHANICS:
        folder_suffix = "_mechanics"
    elif WITH_ECM:
        folder_suffix = "_ecm"
    result_root = Path("results") / f"ecs_ratio{folder_suffix}_{timestamp}"
    result_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(ECS_RATIOS)} ECS ratios with {N_PROCESSES} processes")
    print(f"Results will be stored in: {result_root}")

    args = [(ratio, result_root, WITH_ECM, WITH_MECHANICS) for ratio in ECS_RATIOS]
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_ecs_ratio, args)

    print(f"All simulations complete. Results in: {result_root}")
