import logging
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib import font_manager
from dask.distributed import LocalCluster, SpecCluster

from bmbcsim.logging import logger


def _silence_heartbeat_shutdown(record: logging.LogRecord) -> bool:
    """Drop the benign "Failed to communicate with scheduler during heartbeat"
    traceback that dask workers log when the scheduler shuts down a moment
    before their next periodic heartbeat fires. Real worker errors come through
    a different log message ("Unexpected exception during heartbeat") and are
    unaffected.
    """
    return "Failed to communicate with scheduler during heartbeat" not in record.getMessage()


_heartbeat_filter_installed = False


def _install_heartbeat_shutdown_filter() -> None:
    global _heartbeat_filter_installed
    if _heartbeat_filter_installed:
        return
    logging.getLogger("distributed.worker").addFilter(_silence_heartbeat_shutdown)
    _heartbeat_filter_installed = True


def timestamped_directory(root: str | Path, name: str) -> Path:
    """Create and return ``<root>/<name>_<timestamp>/`` to hold one run's output.

    The name leads so that runs of one simulation group together in a shared result
    root; the trailing timestamp keeps repeat runs distinct and sorted.

    :class:`bmbcsim.Simulation` takes a finished directory rather than inventing
    one, so a script can name its output directory up front and write everything
    that describes the run -- a resolved config, the inputs -- there *before*
    building the simulation. That way a crash during setup (meshing, geometry)
    still leaves a directory saying what was being attempted.

    The directory is claimed exclusively (no ``exist_ok``): the timestamp only
    resolves to a second, so two runs of the same name started together -- one
    config per variant, launched in parallel -- would otherwise be handed the
    same directory and overwrite each other's snapshot.h5 mid-run. On a
    collision, wait for the next second and retry, rather than decorating the
    name with a suffix that :meth:`ResultLoader.find` would no longer match.

    :param root: Directory under which the run directory is created.
    :param name: Name of the run; the timestamp keeps repeat runs distinct.
    :return: The created directory.
    """
    while True:
        directory = Path(root) / f"{name}_{datetime.now():%Y-%m-%d-%H%M%S}"
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            sleep(1.0)  # the timestamp's resolution: the next attempt gets a new one


def _sans_family() -> list[str]:
    """The lab's font, or the closest thing this machine actually has installed.

    Arial is what the lab style asks for; on Ubuntu it comes from the
    ttf-mscorefonts-installer package (whose download server is currently dead, hence
    this fallback). Naming a font that is missing makes matplotlib log "findfont: Font
    family 'Arial' not found." *once per text object* -- a later entry in the list does
    not silence it -- so only installed families are named. Liberation Sans is
    metrically identical to Arial, so figures come out the same size either way.
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [f for f in ("Arial", "Liberation Sans", "Nimbus Sans") if f in installed]
    if "Arial" not in installed:
        logger.warning(
            "Arial is not installed; falling back to %s.",
            preferred[0] if preferred else "matplotlib's default sans-serif",
        )
    return [*preferred, "sans-serif"]


def plot_style(style: Literal["default", "pedramlab"]) -> tuple[float, float]:
    """Set the plot style according to the specified style.

    :param style: The style to apply. "default" for no changes, "pedramlab" for custom theme.
    :returns: A tuple (width, height) for the figure size in inches.
    :raises ValueError: If style is not "default" or "pedramlab"
    """
    match style:
        case "default":
            return 6.4, 4.8
        case "pedramlab":
            plt.rcParams.update({
                "font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "legend.fontsize": 9,
                "legend.edgecolor": "black",
                "legend.frameon": False,
                "lines.linewidth": 0.5,
                "font.family": _sans_family(),  # Arial if installed, see above
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            })
            return 5.36, 3.27
        case _:
            raise ValueError(
                f"Unknown style '{style}'. Must be 'default' or 'pedramlab'."
            )


def create_cluster(
    backend: Literal["local", "janelia"],
    *,
    n_workers: int,
    n_threads_per_worker: int = 4,
    **cluster_kwargs: Any,
) -> SpecCluster:
    """Create a Dask cluster for parallel workload execution.

    Provides a single entry point for spinning up either an in-process Dask
    cluster (``"local"``) or a cluster of LSF jobs on the Janelia HHMI compute
    cluster (``"janelia"``). The returned object is usable as a context manager
    and can be passed directly to ``dask.distributed.Client``.

    :param backend: ``"local"`` for an in-process ``LocalCluster``;
        ``"janelia"`` for an ``LSFCluster`` configured for Janelia's LSF scheduler.
    :param n_workers: Number of workers to launch. For ``"local"`` this is the
        number of worker processes; for ``"janelia"`` this is the number of LSF
        jobs requested.
    :param n_threads_per_worker: Threads per worker (``LocalCluster``) or cores
        per LSF job (``LSFCluster``). Defaults to 4.
    :param cluster_kwargs: Extra keyword arguments forwarded to the underlying
        cluster constructor. Override any of the backend's default settings.
    :returns: A Dask cluster instance.
    :raises ValueError: If backend is not ``"local"`` or ``"janelia"``.
    """
    _install_heartbeat_shutdown_filter()
    match backend:
        case "local":
            # One task slot per worker process (``threads_per_worker=1``): each
            # NGSolve simulation runs alone in its own process, with its own
            # memory and its own copy of Netgen's process-global state. Do NOT
            # set this to ``n_threads_per_worker`` -- Dask would then run that
            # many simulations concurrently *inside one process*, multiplying
            # peak memory and sharing global solver state. Internal NGSolve
            # threading is controlled separately via the simulation's own
            # ``n_threads`` argument.
            return LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                processes=True,
                **cluster_kwargs,
            )
        case "janelia":
            from dask_jobqueue.lsf import LSFCluster

            # Janelia allocates memory by slot (15G / slot), so the number of
            # slots (``ncpus``, LSF ``-n``) is the *only* thing that reserves
            # memory: an explicit "#BSUB -M" is ignored by the scheduler, and
            # emitting one just risks bsub rejecting the job. ``memory`` still
            # has to be passed -- dask-jobqueue requires it -- but it is used
            # only for Dask's own per-worker accounting (``--memory-limit``),
            # so it states what the slots give us.
            #
            # ``cores`` is what Dask turns into ``--nthreads`` (= task slots per
            # worker), so it must be 1: one simulation per LSF job. The cores
            # the simulation actually needs are reserved via ``ncpus``; NGSolve's
            # internal threads then run on those reserved cores. Setting
            # ``cores=n_threads_per_worker`` instead would pack that many
            # simulations into a single worker process and exhaust the job's
            # memory reservation.
            defaults: dict[str, Any] = {
                "queue": "local",
                # Billing project ("#BSUB -P"). Janelia's LSF rejects jobs
                # without one -- bsub exits 255 and dask-jobqueue discards its
                # stderr, so the only symptom is every worker failing to start.
                "project": "pedram",
                "cores": 1,
                "processes": 1,
                "ncpus": n_threads_per_worker,
                "memory": f"{15 * n_threads_per_worker}GB",
                "job_directives_skip": ["#BSUB -M"],
                # dask-jobqueue's LSFCluster default is 30 min, which is too
                # short for our NGSolve sweeps and silently produces partial
                # snapshot.h5 files (no data/time) when LSF kills the worker.
                "walltime": "01:00",
                # Write per-worker stdout/stderr to files (adds "#BSUB -o/-e").
                # Without this, LSF emails each worker's output on completion --
                # one email per job. The directory is created automatically.
                "log_directory": "dask-worker-logs",
            }
            defaults.update(cluster_kwargs)
            cluster = LSFCluster(**defaults)
            cluster.scale(jobs=n_workers)
            return cluster
        case _:
            raise ValueError(
                f"Unknown backend '{backend}'. Must be 'local' or 'janelia'."
            )
