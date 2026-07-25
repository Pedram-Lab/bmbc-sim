"""Tests for the config-driven simulation support (units + sweep expansion)."""
import astropy.units as u
import pytest
from pydantic import ValidationError

from bmbcsim.config import Quantity, SimConfig, SimGroup, expand_sweep, quantity


class _Cfg(SimConfig):
    simulation_name: str = "demo"
    seed: int = 0
    ca: quantity("mmol / L") = "1.3 mmol / L"
    rate: Quantity = "0.1 / ms"
    factor: float = 2.0


class _Group(SimGroup):
    coupling: Quantity = "0.1 kPa L / mmol"
    ecs_ratio: float = 0.1


class _Nested(SimConfig):
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
