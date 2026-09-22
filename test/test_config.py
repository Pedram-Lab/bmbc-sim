"""Tests for the config-driven simulation support (units + sweep expansion)."""
import astropy.units as u
import pytest
from pydantic import ValidationError

from bmbcsim.config import BareQuantity, Quantity, SimulationConfig, ConfigGroup, expand_sweep


class _Cfg(SimulationConfig):
    simulation_name: str = "demo"
    seed: int = 0
    ca: Quantity("mmol / L") = "1.3 mmol / L"
    rate: BareQuantity = "0.1 / ms"  # bare: no dimension check
    factor: float = 2.0


class _Group(ConfigGroup):
    coupling: BareQuantity = "0.1 kPa L / mmol"
    ecs_ratio: float = 0.1


class _Nested(SimulationConfig):
    simulation_name: str = "demo"
    seed: int = 0
    mech: _Group = _Group()


def test_quantity_parsing_and_defaults():
    c = _Cfg()
    # Defaults are strings that must still be parsed (validate_default).
    assert isinstance(c.ca, u.Quantity) and c.ca == 1.3 * u.mmol / u.L
    assert c.rate == 0.1 / u.ms
    # Overrides parse too, including domain molar units (uM, nM, ...).
    assert _Cfg(ca="5 uM").ca.to("nmol / L").value == pytest.approx(5000)


def test_dimensionality_check_rejects_wrong_units():
    with pytest.raises(ValidationError):
        _Cfg(ca="5 ms")  # time is not a concentration


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        _Cfg(typo=1)  # extra="forbid" guards against silently-ignored params


def test_expand_sweep_grid_and_dirs():
    base = _Cfg(simulation_name="demo", result_root="results")
    jobs = expand_sweep(
        base,
        sweep={"ca": ["1 mmol / L", "2 mmol / L"], "factor": [1.0, 2.0, 3.0]},
        seeds=2,
        result_root="out",
    )
    # 2 x 3 grid x 2 seeds
    assert len(jobs) == 2 * 3 * 2
    labels, cfg, subdir = jobs[0]
    assert set(labels) == {"ca", "factor", "seed"}
    # Swept values land in a validated config with the right units + per-run dir.
    assert cfg.ca == 1.0 * u.mmol / u.L
    assert cfg.simulation_name == "demo_seed0"
    assert str(subdir) == "out/ca=1-mmol-L/factor=1.0"
    # Every job's result_root points at its own subdir.
    assert all(c.result_root == str(d) for _, c, d in jobs)


def test_expand_sweep_dotted_keys_target_nested_groups():
    base = _Nested()
    jobs = expand_sweep(
        base,
        sweep={
            "mech.ecs_ratio": [0.04, 0.19],
            "mech.coupling": ["0 kPa L / mmol", "0.1 kPa L / mmol"],
        },
        seeds=1,
        result_root="out",
    )
    assert len(jobs) == 4
    _, cfg, subdir = jobs[0]
    # Dotted key reached the nested group field, with unit validation.
    assert cfg.mech.ecs_ratio == 0.04
    assert cfg.mech.coupling.value == 0.0
    # Subdir is labelled by the leaf name, not the dotted path.
    assert str(subdir) == "out/ecs_ratio=0.04/coupling=0-kPa-L-mmol"
    # The base config is not mutated by expansion (deepcopy per job).
    assert base.mech.ecs_ratio == 0.1


def test_expand_sweep_zips_comma_joined_axes():
    jobs = expand_sweep(
        _Cfg(),
        sweep={"ca, factor": [["1 mmol / L", 1.0], ["2 mmol / L", 4.0]], "rate": ["0.1 / ms", "0.2 / ms"]},
        result_root="out",
    )
    assert len(jobs) == 4  # 2 pairs x 2 rates, not 2 x 2 x 2
    labels, cfg, subdir = jobs[0]
    assert (cfg.ca.value, cfg.factor) == (1.0, 1.0)
    assert set(labels) == {"ca", "factor", "rate", "seed"}
    # Zipped axes still get one directory level each.
    assert str(subdir) == "out/ca=1-mmol-L/factor=1.0/rate=0.1-ms"
    with pytest.raises(ValueError, match="zipped axis"):
        expand_sweep(_Cfg(), sweep={"ca, factor": [["1 mmol / L"]]}, result_root="out")


def test_expand_sweep_disambiguates_axes_sharing_a_leaf_name():
    class _TwoGroups(SimulationConfig):
        simulation_name: str = "demo"
        buffer: _Group = _Group()
        sensor: _Group = _Group()

    jobs = expand_sweep(
        _TwoGroups(),
        sweep={"buffer.ecs_ratio": [0.04], "sensor.ecs_ratio": [0.19]},
        result_root="out",
    )
    # Both axes would be labelled "ecs_ratio", making the two directory levels
    # indistinguishable; the full dotted key is used instead.
    _, cfg, subdir = jobs[0]
    assert str(subdir) == "out/buffer.ecs_ratio=0.04/sensor.ecs_ratio=0.19"
    assert (cfg.buffer.ecs_ratio, cfg.sensor.ecs_ratio) == (0.04, 0.19)


def test_run_name_takes_the_postfix_from_the_config_or_the_parameters():
    class _Variant(SimulationConfig):
        simulation_name: str = "demo"
        with_buffer: bool = False

        def derived_postfix(self) -> str:
            return "buffer" if self.with_buffer else "nobuffer"

    class _Plain(SimulationConfig):
        simulation_name: str = "demo"

    # No postfix at all -> the bare simulation name; a derived one -> appended.
    assert _Plain().run_name == "demo"
    assert _Variant().run_name == "demo_nobuffer"
    assert _Variant(with_buffer=True).run_name == "demo_buffer"
    # An explicit postfix (from a config YAML) wins over the derived one, so a
    # config can always name its own run.
    assert _Variant(postfix="quick").run_name == "demo_quick"


def test_janelia_job_script_has_a_project_and_no_memory_directive(tmp_path):
    """Janelia's LSF rejects a job without "-P" (bsub exits 255, stderr swallowed
    by dask-jobqueue) and allocates memory by slot, so "-M" must not be emitted.
    """
    from bmbcsim.utils import create_cluster

    # n_workers=0: build the job script, submit nothing (no LSF needed here).
    cluster = create_cluster(
        "janelia", n_workers=0, n_threads_per_worker=4, log_directory=str(tmp_path)
    )
    try:
        script = cluster.job_script()
    finally:
        cluster.close()

    assert '#BSUB -P "scicompsoft"' in script
    assert "#BSUB -M" not in script
    assert "#BSUB -n 4" in script  # slots are what actually reserve the memory
