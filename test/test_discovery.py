"""Locating runs in a sweep result tree of any shape."""
import pytest

from bmbcsim.simulation.result_io import find_run_dirs, run_labels, run_seed


def make_run(root, relative):
    """Create a run directory (identified by its snapshot file) at root/relative."""
    run = root / relative
    run.mkdir(parents=True)
    (run / "snapshot.h5").touch()
    return run


@pytest.fixture
def sweep(tmp_path):
    """A 2-axis sweep, plus analysis output that must never be mistaken for a run."""
    for kd in ("ecm_kd=0.1-mM", "ecm_kd=1.3-mM"):
        for ecs in ("ecs_ratio=0.04", "ecs_ratio=0.19"):
            for seed in range(3):
                make_run(tmp_path, f"{kd}/{ecs}/2026-07-26-150039_tissue_kinetics_seed{seed}")
            # Provenance file written next to the runs, not inside one.
            (tmp_path / kd / ecs / "tissue_kinetics_seed0.config.yaml").touch()
    (tmp_path / "processed-data" / "ecm_kd=0.1-mM").mkdir(parents=True)
    (tmp_path / "plots").mkdir()
    return tmp_path


def test_finds_every_run_at_full_depth(sweep):
    assert len(find_run_dirs(sweep)) == 12


def test_ignores_analysis_output_dirs(sweep):
    # A stray snapshot under processed-data/ must not be picked up as a run.
    make_run(sweep, "processed-data/ecm_kd=0.1-mM/leftover")
    assert len(find_run_dirs(sweep)) == 12


def test_accepts_a_subtree_or_a_single_run(sweep):
    assert len(find_run_dirs(sweep / "ecm_kd=0.1-mM")) == 6
    assert len(find_run_dirs(sweep / "ecm_kd=0.1-mM" / "ecs_ratio=0.04")) == 3
    run = sweep / "ecm_kd=0.1-mM/ecs_ratio=0.04/2026-07-26-150039_tissue_kinetics_seed0"
    assert find_run_dirs(run) == [run]


def test_empty_for_a_tree_without_runs(tmp_path):
    (tmp_path / "not-a-sweep").mkdir()
    assert find_run_dirs(tmp_path) == []


def test_orders_seeds_numerically_not_lexicographically(tmp_path):
    for seed in (0, 1, 2, 10, 11):
        make_run(tmp_path, f"ecs_ratio=0.04/2026-07-26-150039_sim_seed{seed}")
    found = find_run_dirs(tmp_path)
    assert [run_seed(p) for p in found] == [0, 1, 2, 10, 11]


def test_labels_give_the_grid_coordinates_at_any_depth(sweep):
    run = sweep / "ecm_kd=1.3-mM/ecs_ratio=0.19/2026-07-26-150039_tissue_kinetics_seed2"
    assert run_labels(run, sweep) == {"ecm_kd": "1.3-mM", "ecs_ratio": "0.19"}
    # Relative to a subtree, only the axes below it remain.
    assert run_labels(run, sweep / "ecm_kd=1.3-mM") == {"ecs_ratio": "0.19"}


def test_labels_empty_when_nothing_was_swept(tmp_path):
    run = make_run(tmp_path, "2026-07-26-150039_tissue_kinetics")
    assert run_labels(run, tmp_path) == {}


def test_seed_found_whichever_side_the_timestamp_is_on(tmp_path):
    # The layout puts the timestamp first; archived results have it last.
    assert run_seed(tmp_path / "2026-07-26-150039_tissue_kinetics_seed7") == 7
    assert run_seed(tmp_path / "tissue_kinetics_seed7_2026-07-26-150039") == 7
    assert run_seed(tmp_path / "2026-07-26-150039_tissue_kinetics") is None


def test_older_pvd_results_are_still_found(tmp_path):
    run = tmp_path / "ecs_ratio=0.04/tissue_kinetics_seed0_2026-05-29-185441"
    run.mkdir(parents=True)
    (run / "snapshot.pvd").touch()
    assert find_run_dirs(tmp_path) == [run]
