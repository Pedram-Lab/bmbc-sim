"""Run directories: how they are named, and how a run is found again.

A run directory is ``<simulation_name>[_<postfix>]_<timestamp>``, the postfix
naming the variant (``sala_quick``, ``rusakov_post``). ``ResultLoader.find`` must
therefore find a simulation whatever its postfix, must still let a caller pin one
variant by full name, and must not stray into a *different* experiment whose name
happens to start the same way. Every evaluation script locates its input this way,
so getting it wrong silently plots nothing, or the wrong run.

Results written while the timestamp led (``<timestamp>_<name>``) must keep
working too.
"""
import re

import pytest
import yaml

from bmbcsim import ResultLoader
from bmbcsim.utils import timestamped_directory


@pytest.fixture
def fake_loader(monkeypatch):
    """Skip ResultLoader's real construction: only directory matching is tested."""
    monkeypatch.setattr(ResultLoader, "__init__", lambda self, root: setattr(self, "path", root))


def make_run(root, name, simulation_name=None):
    """Create a run directory, with the dumped config that names its simulation."""
    run = root / name
    run.mkdir()
    if simulation_name is not None:
        (run / "config.yaml").write_text(yaml.safe_dump({"simulation_name": simulation_name}))
    return run


@pytest.mark.parametrize("name", ["2026-07-26-150039_sala", "sala_2026-05-29-185441"])
def test_find_accepts_both_namings(tmp_path, fake_loader, name):
    make_run(tmp_path, name, "sala")
    loader = ResultLoader.find(simulation_name="sala", results_root=str(tmp_path))
    assert loader.path == str(tmp_path / name)


def test_find_ignores_the_postfix(tmp_path, fake_loader):
    # The point of the postfix: "sala" finds the quick variant just as well.
    run = make_run(tmp_path, "2026-07-26-150039_sala_quick", "sala")
    loader = ResultLoader.find(simulation_name="sala", results_root=str(tmp_path))
    assert loader.path == str(run)


def test_find_pins_one_variant_by_full_name(tmp_path, fake_loader):
    quick = make_run(tmp_path, "2026-07-26-150039_sala_quick", "sala")
    make_run(tmp_path, "2026-07-27-150039_sala", "sala")  # newer, but not the one asked for
    loader = ResultLoader.find(simulation_name="sala_quick", results_root=str(tmp_path))
    assert loader.path == str(quick)


def test_find_does_not_cross_into_another_simulation(tmp_path, fake_loader):
    # "sensor_buffer_competition" is a different experiment, not a "sensor" variant:
    # the name alone cannot tell, so the run's own config settles it.
    make_run(tmp_path, "2026-07-26-150039_sensor_buffer_competition", "sensor_buffer_competition")
    with pytest.raises(RuntimeError):
        ResultLoader.find(simulation_name="sensor", results_root=str(tmp_path))


def test_timestamped_directory_never_reuses_a_directory(tmp_path):
    # Same name within the same second (parallel runs of two variants) must not
    # hand both runs the same directory to write snapshot.h5 into.
    # Also: each name must stay findable, i.e. keep the <name>_<timestamp> shape.
    dirs = [timestamped_directory(tmp_path, "sala") for _ in range(3)]
    assert len(set(dirs)) == 3 and all(d.is_dir() for d in dirs)
    assert all(re.fullmatch(r"sala_\d{4}-\d{2}-\d{2}-\d{6}", d.name) for d in dirs)
