from typing import Any, Literal

import matplotlib.pyplot as plt
from dask.distributed import LocalCluster, SpecCluster


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
                "font.family": ["Arial", "sans-serif"],
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
            from dask_jobqueue import LSFCluster

            # Janelia allocates memory by slot (15G / slot).
            #
            # ``cores`` is what Dask turns into ``--nthreads`` (= task slots per
            # worker), so it must be 1: one simulation per LSF job. The cores
            # and memory the simulation actually needs are reserved separately
            # via ``ncpus`` (LSF ``-n``) and ``memory`` (LSF ``-M``); NGSolve's
            # internal threads then run on those reserved cores. Setting
            # ``cores=n_threads_per_worker`` instead would pack that many
            # simulations into a single worker process and exhaust the job's
            # memory reservation.
            defaults: dict[str, Any] = {
                "queue": "local",
                "cores": 1,
                "processes": 1,
                "ncpus": n_threads_per_worker,
                "memory": f"{15 * n_threads_per_worker}GB",
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
