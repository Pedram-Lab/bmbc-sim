"""Config-driven parameter sweep for the sensor/buffer competition simulation.

Replaces ``parameter_sweep.py``, which shelled out to ``python simulation.py
--buffer_conc ... --buffer_kd ...`` through a ``multiprocessing.Pool`` and left
the runs scattered across ``results/`` under encoded names. The base config, the
swept axes and the cluster now come from a YAML config in ``configs/``, the grid
goes out via ``bmbcsim.config.run_sweep`` (local processes or LSF jobs, per-run
failures isolated) and the runs land in one timestamped tree.

    uv run scripts/ongoing_work/sensor_buffer_competition/sweep.py
    uv run scripts/ongoing_work/sensor_buffer_competition/sweep.py cluster=janelia
    uv run scripts/ongoing_work/sensor_buffer_competition/sweep.py --check   # validate only
"""
from bmbcsim.config import sweep_from_cli
from simulation import Config  # same directory (added to sys.path when run directly)

if __name__ == "__main__":
    sweep_from_cli(Config, __file__, config_name="buffer_sweep")
