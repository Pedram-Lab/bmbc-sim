"""Config-driven parameter sweep for the tissue-kinetics simulation.

Replaces the bespoke ``*_sweep.py`` drivers: the base config, the swept axes and
the cluster all come from a YAML config in ``configs/``, and the grid is fanned
out via ``bmbcsim.config.run_sweep`` (local processes or LSF jobs, per-run
failures isolated).

    uv run scripts/ongoing_work/tissue_kinetics/sweep.py                  # contraction_sweep
    uv run scripts/ongoing_work/tissue_kinetics/sweep.py cluster=janelia  # ... on LSF
    uv run scripts/ongoing_work/tissue_kinetics/sweep.py --config-name diffusivity_sweep

Available sweeps (``configs/*_sweep.yaml``), all also overridable on the CLI:

    buffer_capacity_sweep       Kd x ECS ratio x 10 seeds     (120 runs)
    buffer_kinetics_sweep       kr x ECS ratio x 10 seeds     (140 runs)
    contraction_sweep           condensation x ECS ratio x 10 seeds (100 runs, mechanics)
    diffusivity_sweep           D_ecs x ECS ratio x 10 seeds   (60 runs)
    ecs_ratio_sweep             ECS ratio, one seed             (4 runs)
    synapse_distribution_sweep  100 seeds at one ECS ratio    (100 runs)

``sweep.py --check`` validates all of them without running anything.
"""
from bmbcsim.config import sweep_from_cli
from simulation import Config  # same directory (added to sys.path when run directly)

if __name__ == "__main__":
    sweep_from_cli(Config, __file__, config_name="contraction_sweep")
