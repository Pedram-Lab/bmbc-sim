"""Run tissue kinetics simulations sweeping over ECM_total and Kd values.

Kd is the dissociation constant of the Ca <-> ECM binding reaction. We hold
``ecm_kf`` at its default and set ``ecm_kr = Kd * ecm_kf`` so that the
equilibrium constant ``K = ecm_kf / ecm_kr = 1 / Kd`` scans the requested Kd.
"""
import sys
from pathlib import Path
from itertools import product

from dask.distributed import Client, as_completed

from bmbcsim.utils import create_cluster

# (label, value in mmol/L)
CONCENTRATIONS = [
    ("2e-3uM", 2.0e-3),
    ("2e-2uM", 2.0e-2),
    ("2e-1uM", 2.0e-1),
    ("2e0mM", 2.0),
    ("2e1mM", 2e1),
]
ECM_TOTAL_VALUES = CONCENTRATIONS
KD_VALUES = CONCENTRATIONS
N_SEEDS_PER_COMBO = 10
N_WORKERS = None


def run_seed(seed, result_root, ecm_total_mM, kd_mM):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astropy import units as u
    from simulation import run_simulation

    # Match the default ecm_kf in simulation.run_simulation; vary ecm_kr to set Kd.
    ecm_kf = 10.0 * u.L / (u.mmol * u.s)
    ecm_kr = kd_mM * (u.mmol / u.L) * ecm_kf
    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=result_root,
        ecm_total=ecm_total_mM * u.mmol / u.L,
        ecm_kf=ecm_kf,
        ecm_kr=ecm_kr,
        with_ecm=True,
    )
    return seed


if __name__ == "__main__":
    result_root = Path("results") / "viscosity-sweep"
    result_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(N_SEEDS_PER_COMBO))
    jobs = [
        (seed, ecm_total, kd)
        for ecm_total, kd, seed in product(ECM_TOTAL_VALUES, KD_VALUES, seeds)
    ]
    print(
        f"Running {len(jobs)} simulations "
        f"({len(ECM_TOTAL_VALUES)} ecm_total x {len(KD_VALUES)} kd "
        f"x {N_SEEDS_PER_COMBO} seeds)"
    )
    print(f"Results will be stored in: {result_root.resolve()}")

    job_roots = [
        str((result_root / f"ecm_total_{ecm_label}_kd_{kd_label}").resolve())
        for _, (ecm_label, _), (kd_label, _) in jobs
    ]
    for root in set(job_roots):
        Path(root).mkdir(parents=True, exist_ok=True)

    n_workers = N_WORKERS or len(jobs)

    with create_cluster("local", n_workers=n_workers) as cluster, Client(cluster) as client:
        futures = {}
        for (seed, (ecm_label, ecm_val), (kd_label, kd_val)), root in zip(jobs, job_roots):
            future = client.submit(run_seed, seed, root, ecm_val, kd_val)
            futures[future] = (seed, ecm_label, kd_label)
        for future in as_completed(futures):
            seed, ecm_label, kd_label = futures[future]
            future.result()
            print(f"Finished seed={seed} ecm_total={ecm_label} kd={kd_label}")

    print(f"All simulations complete. Results in: {result_root.resolve()}")
