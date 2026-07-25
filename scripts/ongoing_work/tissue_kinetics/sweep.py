"""Config-driven parameter sweep for the tissue-kinetics simulation.

Replaces the bespoke ``*_sweep.py`` drivers: the base config, the swept axes and
the cluster all come from a YAML config in ``configs/``, and the grid is fanned
out via ``bmbcsim.config.run_sweep`` (local processes or LSF jobs, per-run
failures isolated).

    uv run scripts/ongoing_work/tissue_kinetics/sweep.py                 # contraction_sweep
    uv run scripts/ongoing_work/tissue_kinetics/sweep.py cluster=janelia  # ... on LSF
"""
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from bmbcsim.config import ClusterConfig, run_sweep
from simulation import Config  # same directory (added to sys.path when run directly)

_HERE = Path(__file__).resolve().parent
_KNOWN_KEYS = {"base", "sweep", "seeds", "cluster", "result_root"}


@hydra.main(
    version_base=None,
    config_path=str(_HERE / "configs"),
    config_name="contraction_sweep",
)
def main(dcfg) -> None:
    cfg = OmegaConf.to_container(dcfg, resolve=True)
    # Every key below has a fallback, so a typo would silently sweep the wrong
    # grid ("seed: 10" -> 1 seed, "bases:" -> an all-default base config).
    if unknown := set(cfg) - _KNOWN_KEYS:
        raise SystemExit(
            f"unknown key(s) in sweep config: {sorted(unknown)}; "
            f"expected a subset of {sorted(_KNOWN_KEYS)}"
        )
    run_sweep(
        sim_file=_HERE / "simulation.py",
        base_config=Config(**cfg.get("base", {})),
        sweep=cfg["sweep"],
        seeds=cfg.get("seeds", 1),
        cluster=ClusterConfig(**cfg.get("cluster", {})),
        result_root=cfg.get("result_root"),
    )


if __name__ == "__main__":
    main()
