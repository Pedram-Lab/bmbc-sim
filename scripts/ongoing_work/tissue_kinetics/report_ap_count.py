"""Fewer action potentials: no-buffer depletion for 1-10 pulses, buffer effects for 1, 2 and 5
pulses (all synapses and solitary synapse), and mechanics with 2 pulses. Figures + HTML report.

    uv run scripts/ongoing_work/tissue_kinetics/report_ap_count.py ap_count.csv \\
        --mech results/mech-2p-analysis/readout.csv --mech-runs results/mech-2p_2026-10-24-154840 --out results/fewap-analysis/

ap_count.csv comes from evaluate_dose_response.py over the 1-10 pulse references and the
fewap_* / dose_* / single_* sweeps (ecs 0.1); the mechanics table from mech_calib_readout.py.
"""
import argparse
import functools
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis import compute_local_ca
from bmbcsim.simulation.result_io import find_run_dirs
from plot_dose_response import KD0, KF0, LABEL
from report_dose_response import _table, img, sig, table

METRICS = ["uptake_cum", "peak_deficit"]
COLOR = {"uniform": "k", "half": "tab:red", "on": "tab:red", "off": "tab:blue"}
SETTINGS = [("all synapses, 0.3 mM", "all", 0.3), ("all synapses, 3 mM", "all", 3.0),
            ("solitary, 1 mM", "solitary", 1.0)]
TARGETS = [28, 49, 3, 39]  # the singled-out synapses (solitary runs)


def references(df):
    """No-buffer runs, synapses with flux: median and IQR per pulse count (all synapses), the
    active synapse per target (solitary)."""
    ref = df[(df.scenario == "reference") & df.is_active]
    rows = []
    for (solitary, n), g in ref.groupby([ref.active != "all", "n_pulses"]):
        for m in METRICS:
            v = g[m].dropna()
            rows.append(dict(setting="solitary" if solitary else "all synapses", n_pulses=n, metric=m,
                             median=v.median(), q25=v.quantile(0.25), q75=v.quantile(0.75), n_syn=len(v)))
    return pd.DataFrame(rows)


def effects(df):
    """Per (setting, pulse count, placement): median effect on the synapses that carry flux and
    sit at the buffer (all buffered synapses of the half placement; the active synapse for the
    solitary runs; every synapse for uniform), also in % of the synapse's own reference value."""
    buf = df[(df.scenario != "reference") & (df.ecm_kf == KF0) & (df.ecm_kd == KD0) & df.is_active]
    ref = df[df.scenario == "reference"].set_index(["stimulus", "active", "synapse"])
    rows = []
    for label, kind, dose in SETTINGS:
        sub = buf[(buf.ecm_total == dose) & ((buf.active == "all") == (kind == "all"))]
        if kind == "all":
            sub = sub[(sub.scenario == "uniform") | ((sub.scenario == "half") & sub.buffered
                                                      & (sub.local_width == 1.0))]
        else:
            sub = sub[sub.local_width.isna() | (sub.local_width == 2.0)]
        for (n, scen), g in sub.groupby(["n_pulses", "scenario"]):
            row = dict(setting=label, n_pulses=n, placement=scen, n_syn=len(g))
            r = pd.MultiIndex.from_frame(g[["stimulus", "active", "synapse"]])
            for m in METRICS:
                pct = 100 * g[f"effect_{m}"].to_numpy() / ref[m].reindex(r).to_numpy()
                thr = g[f"thr_{m}"].iloc[0]
                e = g[f"effect_{m}"].median()
                row[f"{m}: effect"] = e
                row[f"{m}: min"], row[f"{m}: max"] = g[f"effect_{m}"].min(), g[f"effect_{m}"].max()
                row[f"{m}: % ref"] = np.nanmedian(pct)
                row[f"{m}: thr"] = thr
                row[f"{m}: sig"] = bool(sig(e, thr))
            rows.append(row)
    return pd.DataFrame(rows)


def fig_reference(ref, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, m in zip(axes, METRICS):
        for setting, color in (("all synapses", "k"), ("solitary", "tab:red")):
            g = ref[(ref.setting == setting) & (ref.metric == m)].sort_values("n_pulses")
            ax.plot(g.n_pulses, g["median"], "o-", color=color, label=f"{setting} (median, IQR)")
            ax.fill_between(g.n_pulses, g.q25, g.q75, color=color, alpha=0.15)
        ax.set_xlabel("action potentials")
        ax.set_ylabel(LABEL[m])
        ax.set_xticks(range(1, 11))
    axes[0].legend(fontsize=8)
    fig.suptitle("No buffer: depletion and uptake vs number of action potentials")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ap_reference.png"), dpi=130)
    plt.close(fig)


def fig_effects(eff, out):
    """Effect vs pulse count per setting (rows) and metric (columns); error bars = min/max over
    the synapses, grey band = reference IQR of that pulse count."""
    fig, axes = plt.subplots(len(SETTINGS), len(METRICS), figsize=(9, 2.8 * len(SETTINGS)),
                             squeeze=False, sharex=True)
    for i, (label, _, _) in enumerate(SETTINGS):
        sub = eff[eff.setting == label]
        for j, m in enumerate(METRICS):
            ax = axes[i, j]
            for scen, g in sub.groupby("placement"):
                g = g.sort_values("n_pulses")
                ax.errorbar(g.n_pulses, g[f"{m}: effect"],
                            yerr=[g[f"{m}: effect"] - g[f"{m}: min"], g[f"{m}: max"] - g[f"{m}: effect"]],
                            fmt="o-", color=COLOR[scen], capsize=3, label=scen)
            t = sub.drop_duplicates("n_pulses").sort_values("n_pulses")
            ax.fill_between(t.n_pulses, -t[f"{m}: thr"], t[f"{m}: thr"], color="0.85", zorder=0,
                            label="±IQR(reference)")
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_title(label, fontsize=10)
            ax.set_ylabel("Δ " + LABEL[m], fontsize=8)
            if i == len(SETTINGS) - 1:
                ax.set_xlabel("action potentials")
                ax.set_xticks([1, 2, 5])
        axes[i, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ap_effects.png"), dpi=130)
    plt.close(fig)


@functools.cache
def _ca(run):
    return compute_local_ca(run)


def _trace(ax, run, syn, **kw):
    t, ca = _ca(run)
    ax.plot(t, ca[:, syn], **{"lw": 1, **kw})


def _grid(n_rows, title):
    fig, axes = plt.subplots(n_rows, len(TARGETS), figsize=(3.2 * len(TARGETS), 2.3 * n_rows + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for ax, s in zip(axes[0], TARGETS):
        ax.set_title(f"synapse {s}", fontsize=10)
    fig.suptitle(title)
    return fig, axes


def _save(fig, axes, xlim, path):
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    axes[0, 0].set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_trace_reference(df, out):
    """Ca at each singled-out synapse without buffer, one line per pulse count."""
    ref = df[df.scenario == "reference"]
    fig, axes = _grid(2, "No buffer: Ca at the synapse (nearest ECS vertex)")
    for i, solitary in enumerate((False, True)):
        for j, s in enumerate(TARGETS):
            g = ref[ref.active == (str(s) if solitary else "all")]
            for n, run in g.groupby("n_pulses").run.first().items():
                _trace(axes[i, j], run, s, color=plt.cm.viridis((n - 1) / 9), label=f"{n} AP")
        axes[i, 0].set_ylabel(("solitary" if solitary else "all synapses") + "\nCa (mM)")
    axes[0, -1].legend(fontsize=6, ncol=2)
    _save(fig, axes, (80, 400), os.path.join(out, "trace_reference.png"))


def fig_trace_effects(df, out):
    """Ca at each singled-out synapse per buffer placement, for 1, 2 and 5 pulses."""
    d = df[(df.scenario == "reference") | ((df.ecm_kf == KF0) & (df.ecm_kd == KD0) & (df.scenario != "all"))]
    rows = [("all synapses, 3 mM", "all", 3.0, 1.0), ("solitary, 1 mM", None, 1.0, 2.0)]
    fig, axes = _grid(6, "Ca at the synapse per buffer placement")
    for i, ((label, act, dose, width), n) in enumerate(itertools.product(rows, (1, 2, 5))):
        for j, s in enumerate(TARGETS):
            g = d[(d.active == (act or str(s))) & (d.n_pulses == n) & (d.synapse == s)
                  & ((d.scenario == "reference")
                     | ((d.ecm_total == dose) & (d.local_width.isna() | (d.local_width == width))))]
            for scen, r in g.groupby("scenario").first().iterrows():
                lab = {"reference": "no buffer", "half": "half, " + ("buffered" if r.buffered else "not buffered")}
                _trace(axes[i, j], r.run, s, color=COLOR.get(scen, "0.7"), label=lab.get(scen, scen),
                       lw=2.5 if scen == "reference" else 1)
            axes[i, j].legend(fontsize=6)
        axes[i, 0].set_ylabel(f"{label}\n{n} AP\nCa (mM)", fontsize=8)
    _save(fig, axes, (80, 300), os.path.join(out, "trace_effects.png"))


def fig_trace_mechanics(runs_dir, out):
    """Ca at each singled-out synapse per condensation k (k 0 = mechanics off)."""
    panels = {}
    for r in find_run_dirs(runs_dir):
        cfg = yaml.safe_load(open(f"{r}/config.yaml"))
        mech = cfg["mechanics"]
        k = float(str(mech["ecm_ca_condensation"]).split()[0]) if mech["enabled"] else 0.0
        active = cfg["synapse"].get("active")
        row = ("solitary, on target" if active else "all synapses, "
               + ("uniform" if cfg["ecm"].get("local_width") is None else "at every synapse"))
        for s in active or TARGETS:
            panels.setdefault(row, {}).setdefault(s, []).append((k, str(r)))
    names = sorted(panels)
    fig, axes = _grid(len(names), "Mechanics, 2 action potentials, 1 mM: Ca at the synapse")
    for i, row in enumerate(names):
        for j, s in enumerate(TARGETS):
            for c, (k, run) in zip(("k", "tab:orange", "tab:red"), sorted(panels[row].get(s, []))):
                _trace(axes[i, j], run, s, color=c, label="mechanics off" if k == 0 else f"k {k:g} /mM")
            if s in panels[row]:
                axes[i, j].legend(fontsize=6)
            else:
                axes[i, j].text(0.5, 0.5, "not run", transform=axes[i, j].transAxes, ha="center", color="0.5")
        axes[i, 0].set_ylabel(row.replace(", ", "\n") + "\nCa (mM)", fontsize=8)
    _save(fig, axes, (40, 200), os.path.join(out, "trace_mechanics.png"))


def mechanics_table(mech, eff):
    """Mechanics-on runs vs their twin, next to the chemical effect of the same buffer at 2 pulses."""
    m = mech[mech.k > 0].copy()
    m["placement"] = m.regime
    cols = {"k": "k (1/mM)", "ecs_vol_peak": "ECS vol peak", "local_vol_peak": "local ECS vol peak",
            "J_min": "J min", "d_uptake_amol_%": "Δ uptake % (vs twin)",
            "d_peak_deficit_%": "Δ peak deficit % (vs twin)"}
    m = m[["placement", *cols]].rename(columns=cols)
    chem = eff[eff.n_pulses == 2][["setting", "placement", "uptake_cum: % ref", "peak_deficit: % ref"]]
    return m, chem.rename(columns={"uptake_cum: % ref": "Δ uptake % (vs no buffer)",
                                   "peak_deficit: % ref": "Δ peak deficit % (vs no buffer)"})


def conclusions(ref, eff, mech):
    r = ref[(ref.setting == "all synapses")].set_index(["metric", "n_pulses"])["median"]
    e = eff.set_index(["setting", "placement", "n_pulses"])
    crowd, sol = "all synapses, 3 mM", "solitary, 1 mM"
    pct = lambda s, p, m: "/".join(f"{e.loc[(s, p, n), f'{m}: % ref']:+.0f}" for n in (1, 2, 5))
    n_sig = int(e.loc[crowd][["uptake_cum: sig", "peak_deficit: sig"]].to_numpy().sum())
    on_u = (e.loc[(sol, "on")]["uptake_cum: effect"] / e.loc[(sol, "uniform")]["uptake_cum: effect"])
    mk = lambda reg, k, c: mech.loc[(mech.placement.str.startswith(reg)) & (mech["k (1/mM)"] == k), c]
    return f"""
<li><b>Without buffer, depletion saturates, uptake does not.</b> With all synapses firing the peak deficit is
{r[('peak_deficit', 1)]:.2f} / {r[('peak_deficit', 2)]:.2f} / {r[('peak_deficit', 5)]:.2f} mM after 1 / 2 / 5 action potentials and
{r[('peak_deficit', 10)]:.2f} mM after 10; uptake grows linearly (about {(r[('uptake_cum', 10)] - r[('uptake_cum', 2)]) / 8:.2f} amol per action potential
from the second on).</li>
<li><b>Fewer action potentials make the buffer matter more for depletion, not for uptake.</b> All synapses, 3 mM,
1/2/5 action potentials: peak deficit {pct(crowd, 'half', 'peak_deficit')} % (half, sigma 1 um) vs
{pct(crowd, 'uniform', 'peak_deficit')} % (uniform); uptake {pct(crowd, 'half', 'uptake_cum')} % vs
{pct(crowd, 'uniform', 'uptake_cum')} %. Local stays about 2x uniform; {n_sig} of the 12 median effects exceed the
reference IQR of their pulse count (the uptake IQR shrinks with the pulse count). At 0.3 mM all effects are &lt;3 %.</li>
<li><b>Solitary synapse, 1 mM:</b> on-target buffer gives uptake {pct(sol, 'on', 'uptake_cum')} % and peak deficit
{pct(sol, 'on', 'peak_deficit')} % for 1/2/5 action potentials ({on_u.min():.0f}-{on_u.max():.0f}x the uniform effect,
off-target about half of on-target); nothing exceeds the (weak, 4-synapse) threshold.</li>
<li><b>Mechanics, 2 action potentials.</b> All synapses: k 2 / 5 swell the ECS by
{100 * (mk('crowd', 2, 'ECS vol peak').iloc[0] - 1):.0f} / {100 * (mk('crowd', 5, 'ECS vol peak').iloc[0] - 1):.0f} % but lower uptake by
{-mk('crowd', 2, 'Δ uptake % (vs twin)').iloc[0]:.1f} / {-mk('crowd', 5, 'Δ uptake % (vs twin)').iloc[0]:.1f} % and do not reduce depletion (presumably the swelling
dilutes the ECS Ca faster than it adds volume near the synapses). Solitary on-target, k 0.5: local ECS +{100 * (mk('solitary', 0.5, 'local ECS vol peak').mean() - 1):.0f} %,
peak deficit {mk('solitary', 0.5, 'Δ peak deficit % (vs twin)').mean():+.0f} %, uptake {mk('solitary', 0.5, 'Δ uptake % (vs twin)').mean():+.1f} % on top of the
buffer's own {pct(sol, 'on', 'peak_deficit').split('/')[1]} % / {pct(sol, 'on', 'uptake_cum').split('/')[1]} %: shape change at this k is as large as the chemical
effect on depletion, but uptake still changes by only a few percent. Mesh: J min {mech['J min'].min():.3f}.</li>"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--mech", required=True)
    p.add_argument("--mech-runs", required=True, help="mech_2p sweep directory (time traces)")
    p.add_argument("--out", default=".")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    ref, eff = references(df), effects(df)
    fig_reference(ref, args.out)
    fig_effects(eff, args.out)
    fig_trace_reference(df, args.out)
    fig_trace_effects(df, args.out)
    fig_trace_mechanics(args.mech_runs, args.out)
    mech, chem = mechanics_table(pd.read_csv(args.mech), eff)
    ref_wide = ref.pivot_table(index=["setting", "n_pulses"], columns="metric",
                               values=["median", "q25", "q75"]).reset_index()
    ref_wide.columns = [" ".join(map(str, c)).strip() for c in ref_wide.columns]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fewer action potentials</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:1em;line-height:1.4}}
details{{margin:0.5em 0}} summary{{cursor:pointer;color:#555;font-size:0.85em}}
.t{{border-collapse:collapse;font-size:12px}} .t td,.t th{{padding:2px 8px;border-bottom:1px solid #ddd;text-align:right}}
.t th{{background:#f3f3f3}} h2{{margin-top:2em;border-bottom:2px solid #333}}</style></head><body>
<h1>Fewer action potentials, and mechanics with 2</h1>
<p><b>Setup.</b> As the dose-response study (ECS ratio 0.1 = 16.5 % ECS, one seed, K_d {KD0:g} mM,
k_on {KF0:g} /(mM s)), pulses 10 ms apart from 100 ms, 0.5 s end. Buffer placements: uniform,
half of the synapses at sigma 1 um (all synapses firing), on the active synapse or on a neighbour
at sigma 2 um (solitary). Effects are differences to the no-buffer run with the same pulse count;
the threshold is the reference IQR of that pulse count. Mechanics runs: 2 pulses at 50/60 ms, 200 ms
end, 1 mM, compared with the same run with mechanics off.</p>
<h2>Conclusions</h2><ul>{conclusions(ref, eff, mech)}</ul>
<h2>No-buffer depletion vs number of action potentials</h2>
{img(os.path.join(args.out, 'ap_reference.png'))}{table(ref_wide)}
<p>Ca at the singled-out synapses (nearest ECS vertex), all synapses firing (1-10 action potentials) and
solitary (1, 2, 5):</p>{img(os.path.join(args.out, 'trace_reference.png'))}
<h2>Buffer effects with 1, 2 and 5 action potentials</h2>
<p>Median over the synapses at the buffer (bars: min/max); grey: reference IQR of that pulse count.</p>
{img(os.path.join(args.out, 'ap_effects.png'))}{table(eff)}
<p>Ca at the singled-out synapses per placement (all synapses: 3 mM, half at sigma 1 um; solitary: 1 mM,
sigma 2 um):</p>{img(os.path.join(args.out, 'trace_effects.png'))}
<h2>Mechanics with 2 action potentials</h2>
<p>Mechanics on vs off at the same buffer (1 mM); "local ECS vol" is within 2 um of the active synapse
(whole ECS for all synapses). For comparison, the chemical effect of the buffer at 2 pulses
(vs no buffer, same synapse; the all-synapses chemistry was run at 0.3 and 3 mM):</p>
{_table(mech)}<p></p>{_table(chem, "{:.2f}")}
<p>Ca at the singled-out synapses (nearest ECS vertex of the undeformed mesh, so it includes the dilution by
swelling); solitary mechanics ran for synapses 28 and 49 only:</p>{img(os.path.join(args.out, 'trace_mechanics.png'))}
</body></html>"""
    path = os.path.join(args.out, "report.html")
    open(path, "w").write(html)
    print(f"report -> {path}")
    return ref, eff, mech, chem


if __name__ == "__main__":
    main()
