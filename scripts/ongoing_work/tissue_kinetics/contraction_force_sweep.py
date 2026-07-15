"""Run tissue kinetics simulations with mechanics, sweeping the contraction force.

This builds on the buffer-capacity sweep: we fix the ECM buffer at the Kd =
1.3 mM operating point (ECM_total = 2 mM) and turn mechanics on. The quantity we
scan is ``ecm_ca_coupling`` -- the stress that a bound Ca molecule (the
``ECM_Ca`` species) exerts on the surrounding tissue. In ``run_simulation`` this
enters as ``ecs.add_driving_species(ecm_ca, ecm_ca_coupling, baseline=...)``, so
larger values mean a stronger contraction per unit of bound Ca.

As in the buffer sweep, Kd is the dissociation constant of the Ca <-> ECM
binding reaction; we hold ``ecm_kf`` at its default and set
``ecm_kr = Kd * ecm_kf`` so the equilibrium ``K = ecm_kf / ecm_kr = 1 / Kd``.
"""
import sys
from pathlib import Path
from itertools import product

from dask.distributed import Client, as_completed

from bmbcsim.utils import create_cluster

# Buffer held at the Kd = 1.3 mM operating point from the buffer-capacity sweep.
ECM_TOTAL_mM = 2.0
KD_mM = 1.3

# Contraction force: stress per unit bound Ca, in kPa / (mmol/L).
# (label, value). 0 is a mechanics-on control with no Ca-driven contraction;
# 1e-1 is the run_simulation default.
COUPLING_VALUES = [
    ("0e0", 0.0),
    ("1e-2", 0.0),
    ("3e-2", 3e-2),
    ("1e-1", 1e-1),
]
ECS_RATIOS = [0.04, 0.19]
N_SEEDS_PER_COMBO = 10
N_WORKERS = None


def run_seed(seed, result_root, coupling_kPa_per_mM, ecs_ratio):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from astropy import units as u
    from simulation import run_simulation

    # Match the default ecm_kf in simulation.run_simulation; vary ecm_kr to set Kd.
    ecm_kr = 1e3 / u.s
    ecm_kf = ecm_kr / (KD_mM * u.mmol / u.L)
    run_simulation(
        seed=seed,
        simulation_name=f"tissue_kinetics_seed{seed}",
        result_root=result_root,
        ecs_ratio=ecs_ratio,
        ecm_total=ECM_TOTAL_mM * u.mmol / u.L,
        ecm_kf=ecm_kf,
        ecm_kr=ecm_kr,
        ecm_ca_coupling=coupling_kPa_per_mM * u.kPa / (u.mmol / u.L),
        with_mechanics=True,
    )
    return seed


if __name__ == "__main__":
    result_root = Path("results") / "contraction-force-sweep"
    result_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(N_SEEDS_PER_COMBO))
    jobs = [
        (seed, coupling, ecs)
        for coupling, ecs, seed in product(COUPLING_VALUES, ECS_RATIOS, seeds)
    ]
    print(
        f"Running {len(jobs)} simulations "
        f"({len(COUPLING_VALUES)} coupling x {len(ECS_RATIOS)} ECS ratios "
        f"x {N_SEEDS_PER_COMBO} seeds) at Kd={KD_mM}mM, ECM_total={ECM_TOTAL_mM}mM"
    )
    print(f"Results will be stored in: {result_root.resolve()}")

    job_roots = [
        str((result_root / f"coupling_{coupling_label}kPa_mM" / f"ecs_{ecs}").resolve())
        for _, (coupling_label, _), ecs in jobs
    ]
    for root in set(job_roots):
        Path(root).mkdir(parents=True, exist_ok=True)

    n_workers = N_WORKERS or len(jobs)

    with create_cluster("local", n_workers=n_workers) as cluster, Client(cluster) as client:
        futures = {}
        for (seed, (coupling_label, coupling_val), ecs), root in zip(jobs, job_roots):
            future = client.submit(run_seed, seed, root, coupling_val, ecs)
            futures[future] = (seed, coupling_label, ecs)
        failures = []
        for future in as_completed(futures):
            seed, coupling_label, ecs = futures[future]
            # Isolate failures: one crashing run (e.g. a singular matrix in the
            # mechanics-driven diffusion solve on a fragile low-ECS mesh) must not
            # abort the whole sweep. Log it and let the other runs finish.
            try:
                future.result()
                print(f"Finished seed={seed} coupling={coupling_label} ecs={ecs}")
            except Exception as exc:
                failures.append((seed, coupling_label, ecs, exc))
                print(f"FAILED seed={seed} coupling={coupling_label} ecs={ecs}: {exc!r}")

    if failures:
        print(f"\n{len(failures)} of {len(jobs)} runs failed:")
        for seed, coupling_label, ecs, exc in failures:
            print(f"  seed={seed} coupling={coupling_label} ecs={ecs}: {exc!r}")
    print(f"All simulations complete. Results in: {result_root.resolve()}")
