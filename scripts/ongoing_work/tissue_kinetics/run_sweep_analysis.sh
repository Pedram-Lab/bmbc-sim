#!/usr/bin/env bash
# Evaluate spatial metrics on every leaf of a sweep folder (any depth) and
# visualize the results per group, where a "group" is the parent directory of
# a set of leaf CSVs. The sweep layout is auto-detected:
#   2-level (<param>/<ecs>/seeds/)  -> one plot per <param>, combining ECS ratios
#   1-level (<param>/seeds/)        -> one combined plot at the root
#
# Usage: run_sweep_analysis.sh <sweep-folder>

set -euo pipefail
shopt -s nullglob

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") <sweep-folder>" >&2
    exit 1
fi

if [[ ! -d "$1" ]]; then
    echo "Error: '$1' is not a directory" >&2
    exit 1
fi
SWEEP_DIR="$(cd "$1" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATE_PY="$SCRIPT_DIR/evaluate_synapse_distribution_spatial.py"
VISUALIZE_PY="$SCRIPT_DIR/visualize_spatial_metrics.py"

PROCESSED_DIR="$SWEEP_DIR/processed-data"
PLOTS_DIR="$SWEEP_DIR/plots"

# 1. Find every leaf directory (one that directly contains tissue_kinetics_seed*
#    subdirs) and run the evaluator on it. Prune our own output dirs.
mapfile -t leaves < <(
    find "$SWEEP_DIR" \
        \( -path "$PROCESSED_DIR" -o -path "$PLOTS_DIR" \) -prune -o \
        -type d -name 'tissue_kinetics_seed*' -printf '%h\n' \
        | sort -u
)
if [[ ${#leaves[@]} -eq 0 ]]; then
    echo "No leaf directories with tissue_kinetics_seed* subdirs found under $SWEEP_DIR" >&2
    exit 1
fi
echo "Found ${#leaves[@]} leaf director$([[ ${#leaves[@]} -eq 1 ]] && echo y || echo ies) to evaluate."

for leaf in "${leaves[@]}"; do
    rel="${leaf#"$SWEEP_DIR"/}"
    out_csv="$PROCESSED_DIR/$rel.csv"
    mkdir -p "$(dirname "$out_csv")"
    echo "==> Evaluating $rel"
    uv run python "$EVALUATE_PY" "$leaf" --n-workers 10 --out "$out_csv"
done

# 2. For every directory under processed-data/ that has CSVs directly inside,
#    visualize them together. Mirror the path under plots/.
while IFS= read -r group_dir; do
    csvs=("$group_dir"/*.csv)
    [[ ${#csvs[@]} -eq 0 ]] && continue
    rel="${group_dir#"$PROCESSED_DIR"}"
    rel="${rel#/}"
    plot_dir="$PLOTS_DIR${rel:+/$rel}"
    mkdir -p "$plot_dir"
    echo "==> Visualizing ${rel:-<root>} (${#csvs[@]} CSV(s))"
    uv run python "$VISUALIZE_PY" "${csvs[@]}" --color-sources --out-dir "$plot_dir"
done < <(find "$PROCESSED_DIR" -type d | sort)

echo "Done. CSVs: $PROCESSED_DIR  |  Plots: $PLOTS_DIR"
