"""Config-driven simulation support: unit-aware pydantic types + a Dask sweep runner.

The things every config-driven experiment needs:

* :func:`Quantity` / :data:`BareQuantity` -- pydantic field types that turn config
  strings like ``"1.3 mM"`` into :class:`astropy.units.Quantity`. Group an
  experiment's parameters into :class:`ConfigGroup` subclasses under one
  :class:`SimulationConfig`; each name/default/unit is declared exactly once.
* :func:`run_from_cli` / :func:`sweep_from_cli` -- the command-line entry points.
  They own all of the Hydra wiring (schema registration, config path, validation,
  output settings), so an experiment script ends at::

      if __name__ == "__main__":
          run_from_cli(Config, run, __file__)

* :func:`check_sweep_configs` -- validate every sweep config of an experiment without
  running anything (``sweep.py --check``). Reached via :func:`sweep_from_cli`.
* :func:`run_sweep` -- expand a parameter grid over an existing simulation and
  fan it out via :func:`bmbcsim.utils.create_cluster` (local processes or LSF
  jobs), isolating per-run failures. This replaces the hand-written per-sweep Dask
  drivers each experiment used to carry.

An experiment's ``simulation.py`` only has to expose a ``Config`` (subclass of
:class:`SimulationConfig`) and a ``run(cfg)`` function; :func:`run_sweep` re-imports
and re-validates on each worker, so nothing experiment-specific lives here.
"""
import copy
import inspect
import re
import sys
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Literal

import astropy.units as u
import hydra
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

# Imported for its side effect: registering the domain-standard molar units (M,
# mM, uM, nM, ...) with astropy's string parser, so configs can write "400 nM".
import bmbcsim.units  # noqa: F401


def _to_quantity(v: Any) -> u.Quantity:
    return v if isinstance(v, u.Quantity) else u.Quantity(v)


# A physical quantity parsed from a config string ("1.3 mM") with NO
# dimensionality check -- the escape hatch for a field whose dimension legitimately
# varies. Prefer Quantity(unit) below, which also checks the dimension. Serializes
# back to a string for provenance dumps.
BareQuantity = Annotated[
    u.Quantity,
    PlainValidator(_to_quantity),
    PlainSerializer(str, return_type=str),
]


def Quantity(unit: str | u.UnitBase):
    """A Quantity field type that rejects values not convertible to ``unit``.

    Preferred over :data:`BareQuantity`: a wrong dimension is a silent corruption,
    so check it. ``Quantity("mM")`` accepts ``"5 uM"`` but rejects ``"5 ms"``.
    Use as a field type: ``ca: Quantity("mM") = "1.3 mM"``.

    A call in an annotation position is not a valid *static* type form, so Pylance
    flags this with ``reportInvalidTypeForm`` (same as pydantic's ``conint`` /
    ``constr``). It works because pydantic resolves annotations at runtime; the
    rule is muted in ``[tool.pyright]`` (see pyproject.toml).
    """
    ref = u.Unit(unit)

    def _validate(v: Any) -> u.Quantity:
        q = _to_quantity(v)
        if not q.unit.is_equivalent(ref):
            raise ValueError(f"'{q}' is not convertible to {ref} ({ref.physical_type})")
        return q

    return Annotated[u.Quantity, PlainValidator(_validate), PlainSerializer(str, return_type=str)]


class ConfigGroup(BaseModel):
    """Base for a config *group* (a nested subsystem block: geometry, diffusion, ...).

    Parses unit strings, validates defaults, rejects unknown keys. Group the
    parameters of one experiment into ``ConfigGroup`` subclasses, then reference them
    from a :class:`SimulationConfig` -- each parameter (name, default, unit) is declared
    exactly once, in one place.
    """

    # validate_default: defaults are strings ("1.3 mM") that must be parsed
    # too, not just overrides. extra=forbid: an unknown config key is a typo, not
    # a silently-ignored parameter.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True, extra="forbid"
    )


class SimulationConfig(ConfigGroup):
    """Top level of an experiment config: cross-cutting fields + nested groups."""

    simulation_name: str
    # Distinguishes variants of one simulation: the run directory is named
    # "<timestamp>_<simulation_name>[_<postfix>]", so evaluation scripts can ask for
    # the latest run of a simulation *class* whatever its postfix (see
    # :meth:`ResultLoader.find`), or pin one variant by passing the full run name.
    postfix: str = ""
    result_root: str = "results"
    n_threads: int = 4

    @property
    def run_name(self) -> str:
        """Name of this run's result directory (without the timestamp)."""
        postfix = self.postfix or self.derived_postfix()
        return f"{self.simulation_name}_{postfix}" if postfix else self.simulation_name

    def derived_postfix(self) -> str:
        """Postfix implied by the parameters, for experiments whose variant *is* a
        parameter (which buffer, which mechanism switched on). Override in the
        experiment's ``Config``; an explicit ``postfix`` wins over it, so a config
        can always name its own run.
        """
        return ""


class ClusterConfig(BaseModel):
    """Dask cluster settings for a sweep (see :func:`bmbcsim.utils.create_cluster`)."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "janelia"] = "local"
    n_workers: int | None = None  # None -> one worker per job
    n_threads_per_worker: int = 4
    extra: dict[str, Any] = {}  # forwarded to the cluster constructor


# Hydra's own settings, injected as CLI overrides by the entry points below so that
# no experiment config has to carry them. Each run records itself in its own result
# directory (see :func:`dump_resolved`, :func:`bmbcsim.timestamped_directory`), so
# Hydra's default outputs/<date>/<time>/ tree is just duplication scattered wherever
# the script happened to be launched from. "run.dir=." creates nothing (the working
# directory already exists) and "output_subdir=null" drops the .hydra/ config copies.
#
# job_logging=none (no handlers) is deliberate -- NOT "disabled", which sets
# disable_existing_loggers=true and switches off bmbcsim's logger, since that is
# created at import time, before Hydra configures logging. The symptom is an empty
# simulation.log in every result directory.
#
# The terminal handler is Hydra's own (hydra/hydra_logging=default puts a "[HYDRA]"
# StreamHandler on the *root* logger, which bmbcsim's DEBUG records reach by
# propagation -- a handler without a level passes everything). Levelling it at WARNING
# keeps bmbcsim's bookkeeping (per-step solver chatter, the compartment/membrane
# inventory, DOF counts) in the run's simulation.log, whose FileHandler is the one that
# wants DEBUG, instead of scrolling it past the tqdm progress bar. Scripts print their
# own progress, so the terminal only gets that plus anything actually wrong. It needs
# the "+" (the key does not exist in Hydra's config) and it assumes that default
# handler, so drop it if you ever select a different hydra/hydra_logging.
_HYDRA_OUTPUT_OVERRIDES = (
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra/job_logging=none",
    "+hydra.hydra_logging.handlers.console.level=WARNING",
)

# Keys a sweep YAML may set; anything else is a typo. Every one has a fallback, so
# without this check "seed: 10" would silently sweep 1 seed instead of 10, and a
# misspelled "base:" would silently sweep an all-default config.
_SWEEP_KEYS = frozenset({"base", "sweep", "seeds", "cluster", "result_root"})

# Where a sweep runs. This describes the machine, not the experiment -- it is the
# same for every sweep in the repo, so the presets live here as Hydra group options
# instead of a configs/cluster/*.yaml copy per experiment. A sweep YAML selects one
# with "defaults: - cluster: local" and the CLI switches it with "cluster=janelia";
# individual fields stay overridable ("cluster.n_workers=40").
#
# Only non-default values are listed: the rest come from ClusterConfig, and the LSF
# knobs (queue, cores, ncpus, memory, log_directory, ...) from create_cluster.
_CLUSTER_PRESETS: dict[str, dict[str, Any]] = {
    "local": {},  # in-process; run_sweep gives it one worker process per job
    "janelia": {
        "backend": "janelia",
        # n_workers stays None: one LSF job per run, so the whole sweep is
        # submitted at once and LSF's own scheduling decides how many run
        # concurrently. Set it to cap the number of jobs in flight.
        "extra": {"walltime": "04:00"},  # create_cluster defaults to 01:00
    },
}

# Registered at import time (not inside sweep_from_cli) so that anything composing a
# sweep config -- a driver, a check script, a notebook -- can resolve "cluster: local".
for _preset, _values in _CLUSTER_PRESETS.items():
    ConfigStore.instance().store(
        group="cluster", name=_preset, node=ClusterConfig(**_values).model_dump()
    )


def _hydra_cli(config_name: str, script: str | Path, job: Callable[[Any], None]) -> None:
    """Compose ``config_name`` from ``<script>/configs`` and hand the result to ``job``.

    Appends :data:`_HYDRA_OUTPUT_OVERRIDES`, skipping any the user overrode explicitly
    (Hydra rejects duplicates). Appended rather than prepended: Hydra's argument parser
    needs all overrides in one contiguous run of positionals, so injecting them in front
    of a flag like "--config-name x" would make the user's own overrides unparseable.
    """
    given = {arg.split("=")[0] for arg in sys.argv[1:]}
    injected = [o for o in _HYDRA_OUTPUT_OVERRIDES if o.split("=")[0] not in given]
    sys.argv = [*sys.argv, *injected]
    config_path = str(Path(script).resolve().parent / "configs")
    hydra.main(version_base=None, config_path=config_path, config_name=config_name)(job)()


def run_from_cli(
    config_cls: type[SimulationConfig],
    run: Callable[[Any], None],
    script: str | Path,
    *,
    config_name: str | None = None,
) -> None:
    """Command-line entry point for one config-driven simulation.

    Registers ``config_cls`` as Hydra's schema so CLI overrides of any nested field
    work without a ``+``, composes the YAML in ``<script>/configs``, validates it
    into a ``config_cls`` and calls ``run`` with it. An experiment's whole entry
    point is therefore::

        if __name__ == "__main__":
            run_from_cli(Config, run, __file__)

    :param config_cls: The experiment's :class:`SimulationConfig` subclass.
    :param run: Called with the validated config.
    :param script: The experiment script, normally ``__file__``; its sibling
        ``configs/`` directory is Hydra's config path.
    :param config_name: Default primary config, and the name YAML variants inherit
        with ``defaults: - <config_name>``. Defaults to ``simulation_name``.
    """
    default_config = config_cls()
    config_name = config_name or default_config.simulation_name
    ConfigStore.instance().store(name=config_name, node=default_config.model_dump())
    _hydra_cli(
        config_name, script,
        lambda dcfg: run(config_cls(**OmegaConf.to_container(dcfg, resolve=True))),
    )


def sweep_from_cli(
    config_cls: type[SimulationConfig], script: str | Path, *, config_name: str
) -> None:
    """Command-line entry point for a :func:`run_sweep` parameter sweep.

    Composes the sweep YAML in ``<script>/configs`` (keys: ``base``, ``sweep``,
    ``seeds``, ``cluster``, ``result_root``) and fans the grid out over the
    simulation that defines ``config_cls``. A sweep driver is therefore::

        if __name__ == "__main__":
            sweep_from_cli(Config, __file__, config_name="contraction_sweep")

    Passing ``--check`` instead validates every sweep config in ``<script>/configs``
    and exits without running anything (see :func:`check_sweep_configs`).

    :param config_cls: The simulated experiment's config class. The module that
        defines it is the ``sim_file`` workers re-import, so it must expose ``run``.
    :param script: The sweep driver, normally ``__file__``.
    :param config_name: Default sweep config in ``<script>/configs``.
    """
    if "--check" in sys.argv:
        # Handled here rather than as a Hydra override: Hydra composes one primary
        # config per run, while a check is only useful across all of them at once.
        sys.argv.remove("--check")
        check_sweep_configs(config_cls, script)
        return

    def job(dcfg) -> None:
        cfg = OmegaConf.to_container(dcfg, resolve=True)
        run_sweep(**_sweep_kwargs(config_cls, cfg))

    _hydra_cli(config_name, script, job)


def _sweep_kwargs(config_cls: type[SimulationConfig], cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a composed sweep config and turn it into :func:`run_sweep` arguments.

    All of the config's own validation happens here: unknown top-level keys, the base
    config's units and dimensions, and the cluster block. Only the grid itself is left
    to :func:`expand_sweep`.

    :raises ValueError: On an unknown top-level key. A plain exception rather than
        ``SystemExit`` so that :func:`check_sweep_configs` can report it alongside the
        pydantic errors and carry on to the next config.
    """
    if unknown := set(cfg) - _SWEEP_KEYS:
        raise ValueError(
            f"unknown key(s) in sweep config: {sorted(unknown)}; "
            f"expected a subset of {sorted(_SWEEP_KEYS)}"
        )
    return dict(
        sim_file=inspect.getfile(config_cls),
        base_config=config_cls(**cfg.get("base", {})),
        sweep=cfg["sweep"],
        seeds=cfg.get("seeds", 1),
        cluster=ClusterConfig(**cfg.get("cluster", {})),
        result_root=cfg.get("result_root"),
    )


def check_sweep_configs(
    config_cls: type[SimulationConfig],
    script: str | Path,
    *,
    pattern: str = "*_sweep.yaml",
) -> dict[str, list[tuple[dict[str, Any], SimulationConfig, Path]]]:
    """Statically validate every sweep config next to ``script``; run nothing.

    Composes each config and expands its grid, which is where a bad unit, a wrong
    dimension, a misspelled key or a dotted path into a nonexistent group raises --
    all of it in a second, instead of after a cluster has started. Prints the run
    count, axes and first output path per config, and reports *every* broken config
    rather than stopping at the first.

    Note this is strictly more than Hydra's ``--cfg job`` does: that prints the
    merged config without ever validating it against ``config_cls``.

    :param config_cls: The simulated experiment's config class.
    :param script: The sweep driver, normally ``__file__``; sweep configs are read
        from its sibling ``configs/`` directory.
    :param pattern: Which files in there are sweep configs (the rest being e.g.
        single-run variants for :func:`run_from_cli`).
    :returns: Expanded jobs per config name, as :func:`expand_sweep` returns them.
    :raises SystemExit: If any config is invalid (so this is usable as a CI check).
    """
    configs = Path(script).resolve().parent / "configs"
    expanded: dict[str, list[tuple[dict[str, Any], SimulationConfig, Path]]] = {}
    failures: dict[str, Exception] = {}
    paths = sorted(configs.glob(pattern))
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        for path in paths:
            try:
                cfg = OmegaConf.to_container(compose(path.stem), resolve=True)
                kwargs = _sweep_kwargs(config_cls, cfg)
                jobs = expand_sweep(
                    kwargs["base_config"], kwargs["sweep"],
                    kwargs["seeds"], kwargs["result_root"],
                )
            except Exception as exc:
                failures[path.stem] = exc
                print(f"{path.stem:28s} INVALID: {exc}")
                continue
            expanded[path.stem] = jobs
            # Full dotted keys, not leaf names: two axes can share a leaf name.
            axes = " x ".join(f"{k}({len(v)})" for k, v in cfg["sweep"].items())
            print(f"{path.stem:28s} {len(jobs):4d} runs  {axes}")
            print(f"{'':28s}      -> {jobs[0][2]}/{jobs[0][1].simulation_name}")

    if failures:
        raise SystemExit(f"{len(failures)} of {len(paths)} sweep configs are invalid")
    return expanded


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
        if k not in d:
            raise ValueError(f"{dotted!r}: no config group {k!r} (have {sorted(d)})")
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
    base_config: SimulationConfig,
    sweep: dict[str, list[Any]],
    seeds: int | list[int] = 1,
    result_root: str | Path | None = None,
) -> list[tuple[dict[str, Any], SimulationConfig, Path]]:
    """Expand the sweep grid into validated per-run configs (no I/O, no cluster).

    :returns: List of ``(labels, config, subdir)`` -- one per grid point x seed.
        ``labels`` is the swept values + seed; ``config`` is revalidated (units,
        dimensions, unknown keys all raise here); ``subdir`` is its output dir.

    Swept keys may be dotted (``"geometry.ecs_ratio"``) to target a nested group;
    the subdir is labelled by the leaf name (``ecs_ratio=0.04``), or by the full
    dotted key when two axes share a leaf name (``buffer.kd`` / ``sensor.kd``),
    which would otherwise give both directory levels the same label. The
    cross-cutting fields ``result_root``/``simulation_name``/``seed`` stay top-level.
    """
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    root = Path(result_root) if result_root is not None else Path(base_config.result_root)
    cls = type(base_config)
    base = base_config.model_dump()
    keys = list(sweep)
    has_seed = "seed" in cls.model_fields
    if not has_seed and len(seed_list) > 1:
        raise ValueError(f"{cls.__name__} has no 'seed' field but {len(seed_list)} seeds requested")

    leaves = [k.split(".")[-1] for k in keys]
    axis_name = {k: (leaf if leaves.count(leaf) == 1 else k) for k, leaf in zip(keys, leaves)}

    jobs: list[tuple[dict[str, Any], SimulationConfig, Path]] = []
    for combo in product(*[sweep[k] for k in keys]):
        combo_labels = dict(zip(keys, combo))
        subdir = root / Path(*[f"{axis_name[k]}={_slug(v)}" for k, v in combo_labels.items()])
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
    base_config: SimulationConfig,
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
    :param result_root: Base output dir; defaults to ``base_config.result_root``. A
        timestamp is prepended to its last component, as for a single run.
    :returns: List of ``(job_labels, exception)`` for the runs that failed.
    """
    from dask.distributed import Client, as_completed

    from bmbcsim.utils import create_cluster, timestamped_directory

    cluster = cluster or ClusterConfig()
    # Stamp the sweep root, so that re-running a sweep collects its own tree instead of
    # merging into the previous run's: results/<sweep> -> results/<timestamp>_<sweep>.
    configured = Path(result_root) if result_root is not None else Path(base_config.result_root)
    root = timestamped_directory(configured.parent, configured.name)
    # Validate + materialize every job up front so a bad grid fails fast.
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []  # (labels, config_dict)
    for labels, cfg, subdir in expand_sweep(base_config, sweep, seeds, root):
        cfg_dict = cfg.model_dump()
        jobs.append((labels, cfg_dict))
        subdir.mkdir(parents=True, exist_ok=True)
        # Provenance: dask workers bypass Hydra's outputs/, so record each
        # resolved config next to where its results will land.
        (subdir / f"{cfg.simulation_name}.config.yaml").write_text(
            yaml.safe_dump(cfg_dict, sort_keys=False)
        )
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
