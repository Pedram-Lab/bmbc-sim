"""Config-driven parameter sweep for the tissue-kinetics simulation.

Replaces the bespoke ``*_sweep.py`` drivers: the base config, the swept axes and
the cluster all come from a YAML config, and the grid is fanned out via
``bmbcsim.config.run_sweep`` (local processes or LSF jobs, per-run failures
isolated).

    uv run scripts/ongoing_work/tissue_kinetics/sweep.py \
        --config-name contraction_sweep
"""
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from bmbcsim.config import ClusterConfig, run_sweep
from simulation import Config  # same directory (added to sys.path when run directly)

_SIM_FILE = Path(__file__).resolve().parent / "simulation.py"
_CONFIGS = str(Path(__file__).resolve().parent / "configs")


@hydra.main(version_base=None, config_path=_CONFIGS, config_name="contraction_sweep")
def main(dcfg) -> None:
    cfg = OmegaConf.to_container(dcfg, resolve=True)
    run_sweep(
        sim_file=_SIM_FILE,
        base_config=Config(**cfg.get("base", {})),
        sweep=cfg["sweep"],
        seeds=cfg.get("seeds", 1),
        cluster=ClusterConfig(**cfg.get("cluster", {})),
        result_root=cfg.get("result_root"),
    )


if __name__ == "__main__":
    main()
