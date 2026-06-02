"""Visualize and analyze the relationship between min_ca and the spatial metrics.

Loads a spatial_metrics.csv (from evaluate_synapse_distribution_spatial.py) and:
  1. Plots min_ca vs each predictor (d_boundary, d_neighbor, v_local_*) with a
     saturating-exponential fit y = a*(1 - exp(-b*(x-c))) and the pseudo R^2.
  2. Runs a PCA on the standardized predictor matrix and reports component
     loadings + per-PC and cumulative R^2 of min_ca regressed on the PCs.

Saves two PNGs next to the CSV (--out-dir to override) and prints the numbers.
"""

import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress


def _sat_model(x, a, b, c):
    return a * (1.0 - np.exp(-b * (x - c)))


def fit_saturating(x, y):
    """Fit y = a*(1 - exp(-b*(x-c))). Returns dict(popt, r2) or None on failure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span_x = float(x.max() - x.min())
    if span_x <= 0 or len(x) < 4:
        return None
    a0 = float(y.max())
    b0 = 3.0 / span_x
    c0 = float(x.min()) - 0.5 / b0
    try:
        popt, _ = curve_fit(_sat_model, x, y, p0=(a0, b0, c0), maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    y_pred = _sat_model(x, *popt)
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"popt": tuple(float(p) for p in popt), "r2": r2}


def predictor_columns(df):
    return [c for c in df.columns
            if c.startswith("d_") or c.startswith("vfrac_local_")]


_VLOCAL_RE = re.compile(r"^v_local_r(?P<r>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")


def normalize_v_local(df):
    """Convert absolute v_local_r<r> columns into accessible-volume fractions.

    The denominator is the box-clipped sphere volume vol(ball(r) intersect domain),
    stored per radius as ``v_sphere_box_r<r>`` by the evaluator. This makes the
    ratio plateau at the local porosity for large r instead of decaying as the
    sphere overruns the simulation box. If no matching ``v_sphere_box_r<r>`` column
    is present (older CSVs), fall back to a full continuous sphere (4/3)*pi*r^3.

    Renames each ``v_local_r<r>`` to ``vfrac_local_r<r>`` so the semantic shift is
    visible in plots, and drops the consumed ``v_sphere_box_r<r>`` columns.
    """
    rename = {}
    drop = []
    for col in df.columns:
        m = _VLOCAL_RE.match(col)
        if not m:
            continue
        r_str = m.group("r")
        box_col = f"v_sphere_box_r{r_str}"
        if box_col in df.columns:
            denom = df[box_col].replace(0.0, np.nan)
            drop.append(box_col)
        else:
            r = float(r_str)
            denom = (4.0 / 3.0) * math.pi * r ** 3
        df[col] = df[col] / denom
        rename[col] = f"vfrac_local_r{r_str}"
    df.rename(columns=rename, inplace=True)
    if drop:
        df.drop(columns=drop, inplace=True)
    return df


def _unique_path_labels(paths):
    """Shortest trailing-path label per file that is unique within `paths`.

    Builds each label from path components (parents + filename stem),
    drops trailing components shared by all paths, then takes the shortest
    unique suffix. Examples:
      ecm_10.csv, ecm_25.csv (same dir)           -> "ecm_10", "ecm_25"
      runA/spatial_metrics.csv, runB/...csv       -> "runA",   "runB"
      a/b/c/sm.csv, x/y/c/sm.csv                  -> "b",      "y"
    """
    abs_paths = [os.path.abspath(p) for p in paths]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in abs_paths]
    if len(paths) <= 1:
        return stems
    parent_parts = [os.path.dirname(p).split(os.sep) for p in abs_paths]
    comps = [parts + [stem] for parts, stem in zip(parent_parts, stems)]

    # Drop trailing components shared by all paths — they don't disambiguate.
    while comps and len(comps[0]) > 1 and len({c[-1] for c in comps}) == 1:
        comps = [c[:-1] for c in comps]

    max_depth = max((len(c) for c in comps), default=1)
    for k in range(1, max_depth + 1):
        labels = [os.sep.join(c[-k:]) for c in comps]
        if len(set(labels)) == len(labels):
            return labels
    return [os.sep.join(c) for c in comps]


def load_combined(paths):
    """Read all CSVs, tag each with a unique `_source` label, concat.

    Every CSV must contain ``min_ca`` and expose the same set of columns.
    Pooling files with mismatched columns (e.g. one written before and one
    after the ``v_sphere_box_r*`` change) makes ``pd.concat`` fill NaNs, which
    later surface only as opaque curve-fit / "SVD did not converge" errors. We
    catch the mismatch here and raise a clear, actionable message instead.
    """
    labels = _unique_path_labels(paths)
    frames = []
    ref_cols = None
    ref_label = None
    for path, label in zip(paths, labels):
        df = pd.read_csv(path)
        if "min_ca" not in df.columns:
            raise SystemExit(f"Error: '{path}' has no 'min_ca' column; "
                             "is it a spatial_metrics CSV?")
        cols = set(df.columns)
        if ref_cols is None:
            ref_cols, ref_label = cols, label
        elif cols != ref_cols:
            missing = sorted(ref_cols - cols)
            extra = sorted(cols - ref_cols)
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if extra:
                detail.append(f"has unexpected {extra}")
            raise SystemExit(
                f"Error: column mismatch — '{label}' {' and '.join(detail)} "
                f"relative to '{ref_label}'.\n"
                "All input CSVs must come from the same evaluator version; "
                "re-run evaluate_synapse_distribution_spatial.py to regenerate "
                "the stale file(s).")
        df["_source"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True), labels


def _source_colors(df):
    sources = list(dict.fromkeys(df["_source"].tolist()))
    cmap = plt.get_cmap("tab10")
    return sources, {s: cmap(i % 10) for i, s in enumerate(sources)}


def plot_scatter_grid(df, predictors, out_path, color_sources=False):
    n = len(predictors)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes = axes.flatten()
    y = df["min_ca"].values
    if color_sources:
        sources, color_map = _source_colors(df)
    for ax, col in zip(axes, predictors):
        x = df[col].values
        if color_sources:
            for s in sources:
                m = (df["_source"] == s).values
                ax.scatter(x[m], y[m], alpha=0.35, s=14, color=color_map[s])
        else:
            ax.scatter(x, y, alpha=0.35, s=14)
        fit = fit_saturating(x, y)
        if fit is not None:
            xx = np.linspace(x.min(), x.max(), 128)
            ax.plot(xx, _sat_model(xx, *fit["popt"]), "r-", lw=2,
                    label=fr"$R^2 = {fit['r2']:.3f}$")
        ax.set_xlabel(col)
        ax.set_ylabel("min_ca")
        ax.set_title(f"min_ca vs {col}")
        if col.startswith("vfrac_local_"):
            ax.set_xlim(0.0, 1.0)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    if color_sources:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              color=color_map[s], label=s) for s in sources]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(sources), 4),
                   bbox_to_anchor=(0.5, -0.02), title="source")
    fig.suptitle(f"Univariate regressions  (N = {len(df)})", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def pca_analysis(df, predictors):
    """Return PCA loadings, explained variance, projections, and R^2 of min_ca."""
    X = df[predictors].to_numpy()
    finite_col = np.isfinite(X).all(axis=0)
    if not finite_col.all():
        bad = [p for p, ok in zip(predictors, finite_col) if not ok]
        raise SystemExit(
            f"Error: non-finite (NaN/inf) values in predictor column(s) {bad}; "
            "cannot run PCA. Check the input CSVs for missing entries or "
            "zero sphere-box volumes.")
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    # SVD-based PCA. Vt rows are component vectors; pcs = Xs @ Vt.T.
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    variance = (S ** 2) / (len(Xs) - 1)
    explained = variance / variance.sum()
    pcs = Xs @ Vt.T

    y = df["min_ca"].to_numpy()
    y_c = y - y.mean()
    ss_tot = (y_c ** 2).sum()

    per_pc_r2 = np.empty(len(predictors))
    for k in range(len(predictors)):
        res = linregress(pcs[:, k], y)
        per_pc_r2[k] = res.rvalue ** 2

    cumulative_r2 = np.empty(len(predictors))
    for k in range(1, len(predictors) + 1):
        # Centered design: fit min_ca = sum_j beta_j * PC_j (PCs are already
        # zero-mean from standardization).
        Xk = pcs[:, :k]
        beta, *_ = np.linalg.lstsq(Xk, y_c, rcond=None)
        cumulative_r2[k - 1] = 1.0 - ((y_c - Xk @ beta) ** 2).sum() / ss_tot

    return {
        "predictors": predictors,
        "explained": explained,
        "loadings": Vt,
        "pcs": pcs,
        "per_pc_r2": per_pc_r2,
        "cumulative_r2": cumulative_r2,
    }


def plot_pca(pca, df, out_path, color_sources=False):
    n_pred = len(pca["predictors"])
    pc_labels = [f"PC{k + 1}" for k in range(n_pred)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.bar(pc_labels, pca["explained"] * 100, color="C0",
           label="per-PC variance")
    ax.plot(pc_labels, np.cumsum(pca["explained"]) * 100, "ko-",
            label="cumulative")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    vmax = float(np.abs(pca["loadings"]).max())
    im = ax.imshow(pca["loadings"], aspect="auto", cmap="coolwarm",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_pred))
    ax.set_xticklabels(pca["predictors"], rotation=45, ha="right")
    ax.set_yticks(range(n_pred))
    ax.set_yticklabels(pc_labels)
    ax.set_title("PCA loadings (rows = PCs)")
    for i in range(n_pred):
        for j in range(n_pred):
            ax.text(j, i, f"{pca['loadings'][i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(pca["loadings"][i, j]) < 0.5 * vmax else "white")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    x = pca["pcs"][:, 0]
    y = df["min_ca"].to_numpy()
    if color_sources:
        sources, color_map = _source_colors(df)
        for s in sources:
            m = (df["_source"] == s).values
            ax.scatter(x[m], y[m], alpha=0.35, s=14, color=color_map[s])
    else:
        ax.scatter(x, y, alpha=0.35, s=14)
    fit = fit_saturating(x, y)
    if fit is not None:
        xx = np.linspace(x.min(), x.max(), 128)
        ax.plot(xx, _sat_model(xx, *fit["popt"]), "r-", lw=2,
                label=fr"$R^2 = {fit['r2']:.3f}$")
    ax.set_xlabel(f"PC1 ({pca['explained'][0] * 100:.1f}% of predictor variance)")
    ax.set_ylabel("min_ca")
    ax.set_title("min_ca vs PC1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(pc_labels, pca["per_pc_r2"], color="C2", alpha=0.6,
           label="univariate (single PC)")
    ax.plot(pc_labels, pca["cumulative_r2"], "ko-",
            label="cumulative (first k PCs)")
    ax.set_ylabel(r"$R^2$ with min_ca")
    ax.set_title("min_ca predictability from PCs")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if color_sources:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              color=color_map[s], label=s) for s in sources]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(sources), 4),
                   bbox_to_anchor=(0.5, -0.02), title="source")
    fig.suptitle(f"PCA on predictors (N = {len(df)})", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="+",
                   help="Path(s) to spatial_metrics.csv file(s); data from all is combined")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for plots (default: same dir as first csv)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Threshold for min_ca; rows with min_ca below this value are dropped")
    p.add_argument("--color-sources", action="store_true",
                   help="Color points by source CSV; otherwise all points are one color")
    p.add_argument("--show", action="store_true",
                   help="Show plots interactively after saving")
    return p.parse_args()


def main():
    args = parse_args()
    df, labels = load_combined(args.csv)
    df = normalize_v_local(df)
    n_raw = len(df)
    df = df[df["min_ca"] >= args.threshold].reset_index(drop=True)
    n_dropped = n_raw - len(df)
    predictors = predictor_columns(df)
    if not predictors:
        raise SystemExit(
            "Error: no predictor columns (expected d_* and/or v_local_r* "
            f"columns); got columns {list(df.columns)}.")
    src_str = ", ".join(args.csv) if len(args.csv) <= 3 else f"{len(args.csv)} CSVs"
    print(f"Loaded {n_raw} rows from {src_str}; "
          f"dropped {n_dropped} with min_ca < {args.threshold}, kept {len(df)}")
    print(f"Sources ({len(labels)}): {labels}")
    print(f"Predictors ({len(predictors)}): {predictors}")
    print(f"min_ca:  min={df['min_ca'].min():.4f}  max={df['min_ca'].max():.4f}  "
          f"mean={df['min_ca'].mean():.4f}  std={df['min_ca'].std():.4f}")
    print()

    out_dir = (args.out_dir
               or os.path.dirname(os.path.abspath(args.csv[0]))
               or ".")
    os.makedirs(out_dir, exist_ok=True)
    scatter_path = os.path.join(out_dir, "scatter_min_ca.png")
    pca_path = os.path.join(out_dir, "pca_min_ca.png")

    plot_scatter_grid(df, predictors, scatter_path,
                      color_sources=args.color_sources)
    print(f"Wrote {scatter_path}")

    print()
    print("Saturating-exponential fits  min_ca ~ a*(1 - exp(-b*(predictor - c))):")
    print(f"  {'predictor':>16s}  {'R^2':>7s}  {'a':>10s}  {'b':>10s}  {'c':>10s}")
    for col in predictors:
        fit = fit_saturating(df[col].to_numpy(), df["min_ca"].to_numpy())
        if fit is None:
            print(f"  {col:>16s}    (fit failed)")
            continue
        a, b, c = fit["popt"]
        print(f"  {col:>16s}  {fit['r2']:7.4f}  {a:10.4g}  {b:10.4g}  {c:10.4g}")

    pca = pca_analysis(df, predictors)
    print()
    print("PCA on standardized predictors:")
    print(f"  {'PC':>4s}  {'var %':>7s}  " + "  ".join(f"{c:>14s}" for c in predictors))
    for k in range(len(predictors)):
        print(f"  PC{k + 1:>2d}  {pca['explained'][k] * 100:7.2f}  " +
              "  ".join(f"{v:>14.3f}" for v in pca["loadings"][k]))

    print()
    print("min_ca regressed on PCs:")
    print(f"  {'PC':>4s}  {'univariate R^2':>16s}  {'cumulative R^2':>16s}")
    for k in range(len(predictors)):
        print(f"  PC{k + 1:>2d}  {pca['per_pc_r2'][k]:16.4f}  {pca['cumulative_r2'][k]:16.4f}")

    plot_pca(pca, df, pca_path, color_sources=args.color_sources)
    print(f"\nWrote {pca_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
