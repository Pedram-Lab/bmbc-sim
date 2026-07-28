"""Config-driven parameter sweep for the sensor simulation.

Replaces ``parameter_sweep.py``, which shelled out to ``python simulation.py
--buffer_kd ...`` one run at a time and then guessed which result directories
were its own ("the last 25 sorted by name"). The base config, the swept axes and
the cluster now come from a YAML config in ``configs/``, the grid is fanned out
via ``bmbcsim.config.run_sweep`` (local processes or LSF jobs, per-run failures
isolated), and the runs land in a single timestamped tree that
``collect_kd_sweep.py`` reads back.

    uv run scripts/ongoing_work/sensor_simulation/sweep.py
    uv run scripts/ongoing_work/sensor_simulation/sweep.py cluster=janelia
    uv run scripts/ongoing_work/sensor_simulation/sweep.py --check   # validate only
"""
from bmbcsim.config import sweep_from_cli
from simulation import Config  # same directory (added to sys.path when run directly)

if __name__ == "__main__":
    sweep_from_cli(Config, __file__, config_name="kd_sweep")
