"""Run tissue kinetics simulations sweeping over the binding rate kr.

ECM_total and Kd are held fixed. Kd is the dissociation constant of the
Ca <-> ECM binding reaction, ``Kd = ecm_kr / ecm_kf``. We sweep the reverse
rate ``ecm_kr`` and set ``ecm_kf = ecm_kr / Kd`` so that Kd stays constant while
the absolute reaction speed (kf/kr) scans several orders of magnitude.
"""
import sys
from pathlib import Path
from itertools import product

from dask.distributed import Client, as_completed

from bmbcsim.utils import create_cluster

# Fixed values.
ECM_TOTAL_mM = 2.0
KD_mM = 2.0

# (label, ecm_kr value in 1/s). Sweep kr from 1 to 1e6 in steps of 10.
KR_VALUES = [(f"1e{i}perS", float(10**i)) for i in range(7)]
ECS_RATIOS = [0.04, 0.19]
N_SEEDS_PER_COMBO = 10
N_WORKERS = None


def run_seed(seed, result_root, kr_perS, ecs_ratio):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astropy import units as u
    from simulation import run_simulation

    # Sweep ecm_kr; set ecm_kf = ecm_kr / Kd so that Kd stays fixed.
    kd = KD_mM * u.mmol / u.L
    ecm_kr = kr_perS / u.s
    ecm_kf = ecm_kr / kd
    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=result_root,
        ecs_ratio=ecs_ratio,
        ecm_total=ECM_TOTAL_mM * u.mmol / u.L,
        ecm_kf=ecm_kf,
        ecm_kr=ecm_kr,
        with_ecm=True,
    )
    return seed


if __name__ == "__main__":
    result_root = Path("results") / "buffer-kinetics-sweep"
    result_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(N_SEEDS_PER_COMBO))
    jobs = [
        (seed, kr, ecs)
        for kr, ecs, seed in product(KR_VALUES, ECS_RATIOS, seeds)
    ]
    print(
        f"Running {len(jobs)} simulations "
        f"({len(KR_VALUES)} kr x {len(ECS_RATIOS)} ECS ratios "
        f"x {N_SEEDS_PER_COMBO} seeds; "
        f"ecm_total={ECM_TOTAL_mM} mM, Kd={KD_mM} mM fixed)"
    )
    print(f"Results will be stored in: {result_root.resolve()}")

    job_roots = [
        str(
            (
                result_root
                / f"kr_{kr_label}"
                / f"ecs_{ecs}"
            ).resolve()
        )
        for _, (kr_label, _), ecs in jobs
    ]
    for root in set(job_roots):
        Path(root).mkdir(parents=True, exist_ok=True)

    n_workers = N_WORKERS or len(jobs)

    with create_cluster("local", n_workers=n_workers) as cluster, Client(cluster) as client:
        futures = {}
        for (seed, (kr_label, kr_val), ecs), root in zip(jobs, job_roots):
            future = client.submit(run_seed, seed, root, kr_val, ecs)
            futures[future] = (seed, kr_label, ecs)
        for future in as_completed(futures):
            seed, kr_label, ecs = futures[future]
            future.result()
            print(f"Finished seed={seed} kr={kr_label} ecs={ecs}")

    print(f"All simulations complete. Results in: {result_root.resolve()}")
