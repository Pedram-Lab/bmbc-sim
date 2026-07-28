"""``ResultLoader.find`` must accept both run-directory namings.

``timestamped_directory`` puts the timestamp first; results archived before that
change have it last. Every evaluation script locates its input through ``find``,
so getting this wrong silently finds nothing.
"""
import pytest

from bmbcsim import ResultLoader


@pytest.fixture
def fake_loader(monkeypatch):
    """Skip ResultLoader's real construction: only directory matching is tested."""
    monkeypatch.setattr(ResultLoader, "__init__", lambda self, root: setattr(self, "path", root))


@pytest.mark.parametrize("name", ["2026-07-26-150039_sala", "sala_2026-05-29-185441"])
def test_find_accepts_both_namings(tmp_path, fake_loader, name):
    (tmp_path / name).mkdir()
    loader = ResultLoader.find(simulation_name="sala", results_root=str(tmp_path))
    assert loader.path == str(tmp_path / name)


def test_find_rejects_a_different_simulation(tmp_path, fake_loader):
    # "sala" must not match "sala_extra" from either side of the timestamp.
    (tmp_path / "2026-07-26-150039_sala_extra").mkdir()
    (tmp_path / "sala_extra_2026-05-29-185441").mkdir()
    with pytest.raises(RuntimeError):
        ResultLoader.find(simulation_name="sala", results_root=str(tmp_path))
