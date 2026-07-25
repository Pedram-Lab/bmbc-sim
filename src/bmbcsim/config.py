"""Config-driven simulation support: unit-aware pydantic types + a Dask sweep runner.

The two things every config-driven experiment needs:

* :data:`Quantity` / :func:`quantity` -- pydantic field types that turn config
  strings like ``"1.3 mmol / L"`` into :class:`astropy.units.Quantity`. Group an
  experiment's parameters into :class:`SimGroup` subclasses under one
  :class:`SimConfig`; each name/default/unit is declared exactly once. Hydra
  composes the YAML and the one-line bridge
  ``Config(**OmegaConf.to_container(dcfg, resolve=True))`` parses + validates it.
* :func:`run_sweep` -- expand a parameter grid over an existing simulation and
  fan it out via :func:`bmbcsim.utils.create_cluster` (local processes or LSF
  jobs), isolating per-run failures. This is the ``contraction_force_sweep.py``
  driver generalized.

An experiment's ``simulation.py`` only has to expose a ``Config`` (subclass of
:class:`SimConfig`) and a ``run(cfg)`` function; :func:`run_sweep` re-imports and
re-validates on each worker, so nothing experiment-specific lives here.
"""
import copy
import re
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Literal

import astropy.units as u
import yaml
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

# Teach astropy's string parser the domain-standard molar units (M, mM, uM, nM,
# ...) so configs can write "400 nM". bmbcsim.units keeps its Quantity aliases;
# this only affects parsing of config strings.
_molar_ns: dict[str, u.UnitBase] = {}
u.def_unit("M", u.mol / u.L, prefixes=True, namespace=_molar_ns)
u.add_enabled_units(list(_molar_ns.values()))


def _to_quantity(v: Any) -> u.Quantity:
    return v if isinstance(v, u.Quantity) else u.Quantity(v)


# A physical quantity parsed from a config string ("1.3 mmol / L") with no
# dimensionality constraint. Serializes back to a string for provenance dumps.
Quantity = Annotated[
    u.Quantity,
    PlainValidator(_to_quantity),
    PlainSerializer(str, return_type=str),
]


def quantity(unit: str | u.UnitBase):
    """Like :data:`Quantity` but rejects values not convertible to ``unit``.

    Use for fields where a wrong dimension would silently corrupt results
    (concentrations, diffusivities, times). ``quantity("mmol / L")`` accepts
    ``"5 uM"`` but rejects ``"5 ms"``.
    """
    ref = u.Unit(unit)

    def _validate(v: Any) -> u.Quantity:
        q = _to_quantity(v)
        if not q.unit.is_equivalent(ref):
            raise ValueError(f"'{q}' is not convertible to {ref} ({ref.physical_type})")
        return q

    return Annotated[u.Quantity, PlainValidator(_validate), PlainSerializer(str, return_type=str)]


class SimGroup(BaseModel):
    """Base for a config *group* (a nested subsystem block: geometry, diffusion, ...).

    Parses unit strings, validates defaults, rejects unknown keys. Group the
    parameters of one experiment into ``SimGroup`` subclasses, then reference them
    from a :class:`SimConfig` -- each parameter (name, default, unit) is declared
    exactly once, in one place.
    """

    # validate_default: defaults are strings ("1.3 mmol / L") that must be parsed
    # too, not just overrides. extra=forbid: an unknown config key is a typo, not
    # a silently-ignored parameter.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True, extra="forbid"
    )


class SimConfig(SimGroup):
    """Top level of an experiment config: cross-cutting fields + nested groups."""

    simulation_name: str
    result_root: str = "results"
    n_threads: int = 4


class ClusterConfig(BaseModel):
    """Dask cluster settings for a sweep (see :func:`bmbcsim.utils.create_cluster`)."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "janelia"] = "local"
    n_workers: int | None = None  # None -> one worker per job
    n_threads_per_worker: int = 4
    extra: dict[str, Any] = {}  # forwarded to the cluster constructor


def dump_resolved(cfg: BaseModel, directory: str | Path, filename: str = "config.yaml") -> Path:
    """Write the fully resolved config (units as strings) next to the results."""
    path = Path(directory) / filename
    path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    return path


def _slug(value: Any) -> str:
    """Filesystem-safe label for a swept value ('0.1 kPa / (mmol/L)' -> '0.1-kPa-mmol-L')."""
    return re.sub(r"[^0-9A-Za-z.+-]+", "-", str(value)).strip("-")


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``d["a"]["b"] = value`` from a dotted key ``"a.b"`` (groups must exist)."""
    *parents, leaf = dotted.split(".")
    for k in parents:
        d = d[k]
    d[leaf] = value


def _run_sim(sim_file: str, config_dict: dict[str, Any]) -> None:
    """Worker entry point: import the experiment module by path, rebuild + run.

    Runs on a Dask worker (a separate process, possibly a different LSF node), so
    it takes a primitive dict rather than a pydantic instance -- the ``Config``
    class need not be importable to unpickle the payload; the module supplies it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sweep_sim", sim_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(module.Config(**config_dict))


def expand_sweep(
    base_config: SimConfig,
    sweep: dict[str, list[Any]],
    seeds: int | list[int] = 1,
    result_root: str | Path | None = None,
) -> list[tuple[dict[str, Any], SimConfig, Path]]:
    """Expand the sweep grid into validated per-run configs (no I/O, no cluster).

    :returns: List of ``(labels, config, subdir)`` -- one per grid point x seed.
        ``labels`` is the swept values + seed; ``config`` is revalidated (units,
        dimensions, unknown keys all raise here); ``subdir`` is its output dir.

    Swept keys may be dotted (``"geometry.ecs_ratio"``) to target a nested group;
    the subdir is labelled by the leaf name (``ecs_ratio=0.04``). The cross-cutting
    fields ``result_root``/``simulation_name``/``seed`` stay top-level.
    """
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    root = Path(result_root) if result_root is not None else Path(base_config.result_root)
    cls = type(base_config)
    base = base_config.model_dump()
    keys = list(sweep)
    has_seed = "seed" in cls.model_fields
    if not has_seed and len(seed_list) > 1:
        raise ValueError(f"{cls.__name__} has no 'seed' field but {len(seed_list)} seeds requested")

    jobs: list[tuple[dict[str, Any], SimConfig, Path]] = []
    for combo in product(*[sweep[k] for k in keys]):
        combo_labels = dict(zip(keys, combo))
        subdir = root / Path(*[f"{k.split('.')[-1]}={_slug(v)}" for k, v in combo_labels.items()])
        for seed in seed_list:
            labels = {**combo_labels, **({"seed": seed} if has_seed else {})}
            name = f"{base_config.simulation_name}_seed{seed}" if has_seed else base_config.simulation_name
            cfg_dict = copy.deepcopy(base)
            for key, value in combo_labels.items():
                _set_dotted(cfg_dict, key, value)
            cfg_dict.update(result_root=str(subdir), simulation_name=name)
            if has_seed:
                cfg_dict["seed"] = seed
            cfg = cls(**cfg_dict)  # revalidates (units, dims, typos)
            jobs.append((labels, cfg, subdir))
    return jobs


def run_sweep(
    *,
    sim_file: str | Path,
    base_config: SimConfig,
    sweep: dict[str, list[Any]],
    seeds: int | list[int] = 1,
    cluster: ClusterConfig | None = None,
    result_root: str | Path | None = None,
) -> list[tuple[dict[str, Any], BaseException]]:
    """Run ``sim_file``'s ``run`` over the Cartesian product of ``sweep`` x ``seeds``.

    Each combination becomes a validated config copy (fail-fast: bad combos raise
    here, before any cluster is started) written to its own result subdirectory,
    then dispatched to a Dask cluster. One crashing run is logged and skipped, not
    allowed to abort the sweep.

    :param sim_file: Path to the experiment ``simulation.py`` (exposes ``Config``/``run``).
    :param base_config: Config holding every non-swept parameter.
    :param sweep: Maps a config field name to the list of values to sweep it over.
    :param seeds: ``n`` (-> ``range(n)``) or an explicit list of seeds.
    :param cluster: Dask cluster settings; defaults to a local cluster.
    :param result_root: Base output dir; defaults to ``base_config.result_root``.
    :returns: List of ``(job_labels, exception)`` for the runs that failed.
    """
    from dask.distributed import Client, as_completed

    from bmbcsim.utils import create_cluster

    cluster = cluster or ClusterConfig()
    # Validate + materialize every job up front so a bad grid fails fast.
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []  # (labels, config_dict)
    for labels, cfg, subdir in expand_sweep(base_config, sweep, seeds, result_root):
        cfg_dict = cfg.model_dump()
        jobs.append((labels, cfg_dict))
        subdir.mkdir(parents=True, exist_ok=True)
        # Provenance: dask workers bypass Hydra's outputs/, so record each
        # resolved config next to where its results will land.
        (subdir / f"{cfg.simulation_name}.config.yaml").write_text(
            yaml.safe_dump(cfg_dict, sort_keys=False)
        )
    root = Path(result_root) if result_root is not None else Path(base_config.result_root)
    n_seeds = seeds if isinstance(seeds, int) else len(seeds)

    print(
        f"Sweeping {len(jobs)} runs: "
        + " x ".join(f"{k}({len(v)})" for k, v in sweep.items())
        + f" x {n_seeds} seeds -> {root.resolve()}"
    )

    n_workers = cluster.n_workers or len(jobs)
    sim_file = str(Path(sim_file).resolve())
    failures: list[tuple[dict[str, Any], BaseException]] = []
    with create_cluster(
        cluster.backend,
        n_workers=n_workers,
        n_threads_per_worker=cluster.n_threads_per_worker,
        **cluster.extra,
    ) as clust, Client(clust) as client:
        futures = {
            client.submit(_run_sim, sim_file, cfg_dict): labels
            for labels, cfg_dict in jobs
        }
        for future in as_completed(futures):
            labels = futures[future]
            try:
                future.result()
                print(f"  done: {labels}")
            except Exception as exc:  # isolate: one failure must not kill the sweep
                failures.append((labels, exc))
                print(f"  FAILED: {labels}: {exc!r}")

    if failures:
        print(f"\n{len(failures)}/{len(jobs)} runs failed:")
        for labels, exc in failures:
            print(f"  {labels}: {exc!r}")
    return failures
