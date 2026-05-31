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
            return LocalCluster(
                n_workers=n_workers,
                threads_per_worker=n_threads_per_worker,
                processes=True,
                **cluster_kwargs,
            )
        case "janelia":
            from dask_jobqueue import LSFCluster

            # Janelia allocates memory by slot (15G / slot)
            defaults: dict[str, Any] = {
                "queue": "local",
                "cores": n_threads_per_worker,
                # One worker per LSF job, so cores == threads on that worker;
                # otherwise dask-jobqueue splits each job into multiple workers.
                "processes": 1,
                "memory": f"{15 * n_threads_per_worker}GB",
            }
            defaults.update(cluster_kwargs)
            cluster = LSFCluster(**defaults)
            cluster.scale(jobs=n_workers)
            return cluster
        case _:
            raise ValueError(
                f"Unknown backend '{backend}'. Must be 'local' or 'janelia'."
            )
