# Tissue kinetics

Ca2+ depletion in the extracellular space (ECS) of a tissue block: rounded Voronoi
cells, NMDAR synapse patches on the membranes as Ca2+ sinks, and optionally an ECM
buffer and mechanics. Run from the repository root.

## Workflow

1. Generate the tissue meshes. `make_geometry.py` bisects the gap `w` between the
   cells until the ECS volume fraction reaches the target, then writes one mesh per
   (seed, ECS ratio) to `data/cells/tissue_s{seed}_ecs{NN}.vtu`, where `NN` is the
   ECS ratio in percent:

   ```sh
   uv run --extra geometry scripts/ongoing_work/tissue_kinetics/make_geometry.py --ecs-ratio 0.1 0.17 0.25
   ```

   The default box is 14 x 14 x 7 um. One mesh takes about 5 min and
   has about 200k tets. The cells and their numbering depend only on `--seed`, so all
   ratios of one seed have the same cells, only eroded by different gaps. The meshes
   are not in git; regenerate them with the command above.

   `--cells tube` makes tube-shaped (dendrite-like) cells instead, written to
   `data/tubes` with the same file names (defaults `--d 2 --r 0.25 --h 0.25`; about 100 cells and
   450k tets in the default box, 7 min per mesh).

2. Run simulations. `simulation.py` runs one config, `sweep.py` runs a sweep config
   from `configs/`:

   ```sh
   uv run scripts/ongoing_work/tissue_kinetics/sweep.py --config-name ecs_ratio_sweep cluster=janelia
   ```

   A run loads the mesh named by `geometry.mesh_dir` (`data/cells` or `data/tubes`),
   `geometry.seed` and `geometry.ecs_ratio`. If that mesh is missing, the run fails; generate it first
   (step 1). Each run writes `geometry.json` with the mesh file, the measured ECS
   fraction and `w`, next to its `config.yaml`.

3. Evaluate the results with the `evaluate_*`, `plot_*` and `report_*` scripts.

## Conventions

- `geometry.ecs_ratio` is the true ECS volume fraction and must be a whole
  percentage. The standard values are 0.1, 0.17 and 0.25. Before the rounded
  Voronoi geometry, `ecs_ratio` was a cell-shrink parameter: 0.04, 0.1 and 0.19
  gave about 10, 17 and 25 % ECS.
- `seed` (top level) seeds the synapse placement; `geometry.seed` seeds the cells.
- Each synapse sits on the free membrane vertex closest to an anchor drawn uniformly
  by area on the cell's raw Voronoi faces. The mesh stores these faces, and they do
  not depend on the gap `w`. For one `seed`, synapse index `i` is therefore the same
  membrane spot at every ECS ratio, up to the gap and the mesh size.
- The synapse density is 1 per um^3 (`synapse.n_synapses` = 1372 for the default
  box), and the fraction `synapse.f_active` of them carry flux.
- The convergence sweep uses its own 6 x 6 x 1 um meshes in `data/tissue_6x6x1`
  (see the command in `configs/convergence_sweep.yaml`).
