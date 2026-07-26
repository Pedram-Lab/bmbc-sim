"""Locating simulation runs in a result tree.

A sweep writes one directory level per swept axis and stamps every run with a
timestamp (see :func:`bmbcsim.config.expand_sweep`,
:func:`bmbcsim.timestamped_directory`)::

    results/buffer-capacity-sweep/
        ecm_kd=0.1-mM/
            ecs_ratio=0.04/
                2026-07-26-150039_tissue_kinetics_seed0/    <- a run
                2026-07-26-150039_tissue_kinetics_seed1/
                tissue_kinetics_seed0.config.yaml           <- written before dispatch
            ecs_ratio=0.19/
                ...
        ecm_kd=0.3-mM/
            ...

Both the depth and the run names therefore depend on the sweep: an analysis script
that matches run directories by name or expects a fixed nesting breaks as soon as a
sweep gains an axis or a name changes. :func:`find_run_dirs` identifies runs by the
snapshot file they contain instead, and :func:`run_labels` reads a run's position in
the grid back off the ``<axis>=<value>`` directories above it. Between them, one
analysis script works on every sweep, pointed at a whole sweep root, at one axis
subtree, or at a single run.
"""
import os
import re
from pathlib import Path

# A directory holding one of these is a run; nothing below it is searched. "pvd" is
# the older recorder's output, still present in archived results.
_RUN_MARKERS = ("snapshot.h5", "snapshot.pvd")

# Analysis output written *into* a sweep tree, never a run (see run_sweep_analysis.sh).
NON_RUN_DIRS = frozenset({"processed-data", "plots"})

_SEED_RE = re.compile(r"seed(\d+)")
_LABEL_RE = re.compile(r"^([^=]+)=(.*)$")


def run_seed(run_dir: str | os.PathLike) -> int | None:
    """The replicate index of a run, from ``...seed<N>...`` in its directory name.

    Matches anywhere in the name, so it is indifferent to whether the timestamp is a
    prefix or a suffix.

    :param run_dir: A run directory.
    :returns: The seed, or ``None`` for a run that is not part of a seed series.
    """
    match = _SEED_RE.search(Path(run_dir).name)
    return int(match.group(1)) if match else None


def find_run_dirs(
    root: str | os.PathLike, *, exclude: frozenset[str] = NON_RUN_DIRS
) -> list[Path]:
    """Every simulation run at or beneath ``root``, at any depth.

    A run is identified by the snapshot file it contains, not by its name -- see the
    module docstring for why. ``root`` may be a sweep root, any subtree of one, or a
    single run directory (in which case it is returned by itself).

    :param root: Directory to search.
    :param exclude: Directory names never descended into.
    :returns: Run directories, ordered by location and then by seed (so that a caller
        taking the first ``n`` gets seeds 0..n-1, not the lexicographic order that
        would start 0, 1, 10, 11). Empty if ``root`` holds no runs.
    """
    root = Path(root)
    if any((root / marker).exists() for marker in _RUN_MARKERS):
        return [root]

    runs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        if any(marker in filenames for marker in _RUN_MARKERS):
            runs.append(Path(dirpath))
            dirnames[:] = []  # a run contains no runs
    return sorted(runs, key=lambda p: (str(p.parent), run_seed(p) is None, run_seed(p) or 0, p.name))


def latest_sweep_dir(results_root: str | os.PathLike) -> Path:
    """The most recently modified directory under ``results_root`` that holds runs.

    Sweep directories are timestamped, so "latest" is the one to reach for when a
    script is run without an explicit path. Nothing is inferred from the name: what
    a sweep varied is recorded in its runs' ``config.yaml`` and in the
    ``<axis>=<value>`` directories, not in the sweep's own name.

    :param results_root: Directory holding sweep directories, usually ``results``.
    :returns: The newest one containing at least one run.
    :raises FileNotFoundError: If none of them contains a run.
    """
    candidates = sorted(
        (entry for entry in os.scandir(results_root) if entry.is_dir() and find_run_dirs(entry.path)),
        key=lambda entry: entry.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No directories with simulation results under {results_root}")
    return Path(candidates[-1].path)


def run_labels(run_dir: str | os.PathLike, root: str | os.PathLike) -> dict[str, str]:
    """The swept values of a run, read off the ``<axis>=<value>`` dirs between root and it.

    Recovers a run's coordinates in the sweep grid whatever the number of axes::

        >>> run_labels("sweep/ecm_kd=0.1-mM/ecs_ratio=0.04/<stamp>_sim_seed0", "sweep")
        {'ecm_kd': '0.1-mM', 'ecs_ratio': '0.04'}

    Values are the filesystem-safe slugs the sweep wrote, so a unit-bearing value
    reads ``0.1-mM``, not ``0.1 mM``; the run's own ``config.yaml`` holds the exact
    values. Directories that are not ``<axis>=<value>`` (a timestamped run name, an
    intermediate grouping directory) are skipped.

    :param run_dir: A run directory, below ``root``.
    :param root: The sweep root the labels are relative to.
    :returns: Axis name -> value slug, outermost axis first.
    :raises ValueError: If ``run_dir`` is not below ``root``.
    """
    relative = Path(run_dir).resolve().relative_to(Path(root).resolve())
    return {
        match[1]: match[2]
        for part in relative.parts
        if (match := _LABEL_RE.match(part))
    }
