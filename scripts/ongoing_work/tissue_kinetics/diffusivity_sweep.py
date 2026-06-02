"""Run tissue kinetics simulations sweeping over diffusivity and ECS values."""
import sys
from pathlib import Path
from itertools import product

from dask.distributed import Client, as_completed

from bmbcsim.utils import create_cluster

ECS_RATIOS = [0.04, 0.19]
DIFFUSIVITIES = [0.4, 0.7, 1.0]  # um^2 / ms
N_SEEDS_PER_COMBO = 10
N_WORKERS = None


def run_seed(seed, result_root, ecs_ratio, diffusivity):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astropy import units as u
    from simulation import run_simulation

    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=result_root,
        ecs_ratio=ecs_ratio,
        diffusivity_ecs=diffusivity * u.um**2 / u.ms,
    )
    return seed, ecs_ratio, diffusivity


if __name__ == "__main__":
    result_root = Path("results") / "diffusivity-sweep"
    result_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(N_SEEDS_PER_COMBO))
    jobs = [
        (seed, ecs, diff)
        for diff, ecs, seed in product(DIFFUSIVITIES, ECS_RATIOS, seeds)
    ]
    print(
        f"Running {len(jobs)} simulations "
        f"({len(DIFFUSIVITIES)} diffusivities x {len(ECS_RATIOS)} ECS ratios "
        f"x {N_SEEDS_PER_COMBO} seeds) with {N_WORKERS} workers"
    )
    print(f"Results will be stored in: {result_root.resolve()}")

    job_roots = [
        str((result_root / f"diffusivity_{diff}" / f"ecs_{ecs}").resolve())
        for _, ecs, diff in jobs
    ]
    for root in set(job_roots):
        Path(root).mkdir(parents=True, exist_ok=True)

    N_WORKERS = N_WORKERS or len(jobs)

    with create_cluster("local", n_workers=N_WORKERS) as cluster, Client(cluster) as client:
        futures = [
            client.submit(run_seed, seed, root, ecs, diff)
            for (seed, ecs, diff), root in zip(jobs, job_roots)
        ]
        for future in as_completed(futures):
            seed, ecs, diff = future.result()
            print(f"Finished seed={seed} ecs={ecs} diffusivity={diff}")

    print(f"All simulations complete. Results in: {result_root.resolve()}")
