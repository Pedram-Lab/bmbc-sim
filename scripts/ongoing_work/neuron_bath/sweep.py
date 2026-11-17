"""Bath-concentration sweep for the neuron-bath simulation (``configs/bath_sweep.yaml``).

    uv run scripts/ongoing_work/neuron_bath/sweep.py
    uv run scripts/ongoing_work/neuron_bath/sweep.py --check   # validate only
"""
from bmbcsim.config import sweep_from_cli
from simulation import Config  # same directory (added to sys.path when run directly)

if __name__ == "__main__":
    sweep_from_cli(Config, __file__, config_name="bath_sweep")
