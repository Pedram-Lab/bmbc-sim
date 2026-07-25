"""Config-driven parameter sweep for the tissue-kinetics simulation.

Replaces the bespoke ``*_sweep.py`` drivers: the base config, the swept axes and
the cluster all come from a YAML config in ``configs/``, and the grid is fanned
out via ``bmbcsim.config.run_sweep`` (local processes or LSF jobs, per-run
failures isolated).

    uv run scripts/ongoing_work/tissue_kinetics/sweep.py                 # contraction_sweep
    uv run scripts/ongoing_work/tissue_kinetics/sweep.py cluster=janelia  # ... on LSF
"""
from bmbcsim.config import sweep_from_cli
from simulation import Config  # same directory (added to sys.path when run directly)

if __name__ == "__main__":
    sweep_from_cli(Config, __file__, config_name="contraction_sweep")
