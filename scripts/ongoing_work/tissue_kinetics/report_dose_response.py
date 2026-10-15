"""Self-contained HTML report for the buffer dose-response study.

    uv run scripts/ongoing_work/tissue_kinetics/report_dose_response.py dose_response.csv \\
        --plots plots/ --targets <ref run>/targets.json --out report.html
"""
import argparse
import base64
import json
import os

import numpy as np
import pandas as pd

from plot_dose_response import KD0, KF0, STIM0, core, relative_effects, targets

VERDICT_METRICS = ["uptake_cum", "peak_deficit", "frac_at_floor"]

DEVIATIONS = [
    "Step 0 (numerical error floor, kinetic convergence) skipped: the threshold is the "
    "geometric spread (IQR) of each metric across the reference run's synapses only.",
    "Buffer shape: Gaussians with standard deviation sigma on the synapse centers, not a "
    "smooth shell of thickness L_b. sigma = 1 um spans the 1 um slab.",
    "Many synapses share the dose N (all, or half of the active synapses) instead of one "
    "target per run; the 4 evaluation synapses are chosen by local ECS volume fraction "
    "(geodesic radius 1 um), not ECS thickness.",
    "Dose grid 0.1-10 mM (not 0.01-30), one buffer species, one seed.",
    "No mesh refinement; dt = 0.5 ms, records every 2 ms.",
    "Sub-saturating stimulus = a single pulse (not a reduced single-channel current).",
    "Step 3 runs the uniform and half-buffered scenarios only.",
]


def img(path):
    if not os.path.exists(path):
        return f"<p><em>missing: {os.path.basename(path)}</em></p>"
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%">'


def table(df, fmt="{:.3g}"):
    return df.to_html(index=False, float_format=lambda v: fmt.format(v), border=0,
                      classes="t", na_rep="")


def sig(effect, thr):
    """|effect| > threshold; a zero/NaN threshold (metric constant across the reference,
    e.g. no synapse ever reaches the floor) cannot classify anything -> False."""
    return (np.abs(effect) > thr) & (thr > 0)


def effects_table(sub, tg):
    """Per target, dose, scenario: effect and significance for the verdict metrics."""
    rows = []
    for s, cls in tg:
        g = sub[(sub.synapse == s) & (sub.scenario != "reference")]
        for (scen, width, dose), gg in g.groupby(["scenario", g.local_width.fillna(0), "ecm_total"]):
            r = gg.iloc[0]
            row = dict(synapse=s, target=cls, scenario=scen, sigma=width or "", dose=dose,
                       buffered=bool(r.buffered))
            for m in VERDICT_METRICS:
                row[m] = r[f"effect_{m}"]
                row[f"{m} sig"] = "*" if sig(r[f"effect_{m}"], r[f"thr_{m}"]) else ""
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["synapse", "dose", "scenario", "sigma"])


def local_gain(df, scenarios=("all", "half")):
    """Per target, dose and local scenario: effect(local) - effect(uniform) on the SAME synapse.

    This is the contrast the claim is about; comparing buffered with unbuffered
    *synapses* would confound placement with synapse identity.
    """
    key = ["seed", "stimulus", "ecm_kf", "ecm_kd", "ecm_total", "synapse"]
    # The Step 3 sweeps repeat the core grid point (kf 1e5, Kd 1): keep one copy of each run.
    uni = df[df.scenario == "uniform"].drop_duplicates(key).set_index(key)
    loc = df[df.scenario.isin(scenarios) & df.target_class.notna()].drop_duplicates(
        key + ["scenario", "local_width"]).copy()
    idx = pd.MultiIndex.from_frame(loc[key])
    for m in VERDICT_METRICS:
        loc[f"gain_{m}"] = loc[f"effect_{m}"].to_numpy() - uni[f"effect_{m}"].reindex(idx).to_numpy()
    return loc


def placement(df):
    """Median relative effect (local / uniform, same synapse) on buffered vs unbuffered
    synapses per dose and sigma, and their ratio = how much more a synapse gets from
    buffer on itself than from the same amount on a neighbour."""
    rel = relative_effects(df)
    rows = []
    for (dose, width), g in rel.groupby(["ecm_total", "local_width"]):
        row = dict(dose=dose, sigma=width, n_buffered=int(g.buffered.sum()),
                   n_unbuffered=int((~g.buffered).sum()))
        for m in ("uptake_cum", "peak_deficit"):
            on, off = g[g.buffered][f"rel_{m}"].median(), g[~g.buffered][f"rel_{m}"].median()
            row[f"{m}: on-target"] = on
            row[f"{m}: off-target"] = off
            row[f"{m}: on/off"] = on / off
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(df, axis, tg):
    """Local gain (half - uniform, same synapse) on buffered vs unbuffered targets, and the
    uniform effect itself, against the threshold. Claim holds where the buffered gain
    exceeds the threshold while the uniform effect does not."""
    sub = df[df.scenario.isin(["uniform", "half"]) & (df.local_width.fillna(1.0) == 1.0)
             & df.synapse.isin([s for s, _ in tg])]
    if axis == "stimulus":
        sub = sub[sub.stimulus != STIM0]
    else:
        fixed = "ecm_kd" if axis == "ecm_kf" else "ecm_kf"
        sub = sub[(sub.stimulus == STIM0) & (sub[fixed] == (KD0 if axis == "ecm_kf" else KF0))]
    g = local_gain(sub, ("half",))
    rows = []
    for (val, dose), gg in g.groupby([axis, "ecm_total"]):
        uni = sub[(sub.scenario == "uniform") & (sub[axis] == val) & (sub.ecm_total == dose)]
        row = {axis: val, "dose": dose}
        for m in VERDICT_METRICS:
            thr = gg[f"thr_{m}"].iloc[0]
            gb, gu = gg[gg.buffered][f"gain_{m}"].mean(), gg[~gg.buffered][f"gain_{m}"].mean()
            u = uni[f"effect_{m}"].abs().max()
            row[f"{m}: gain buffered"] = gb
            row[f"{m}: gain unbuffered"] = gu
            row[f"{m}: uniform max"] = u
            if not thr > 0:
                row[f"{m}: verdict"] = "n/a (threshold 0)"
                continue
            row[f"{m}: verdict"] = ("gain>thr" if abs(gb) > thr else "gain<thr") + ", " + \
                                   ("uniform>thr" if u > thr else "uniform<thr")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--plots", default="plots")
    p.add_argument("--targets", required=True)
    p.add_argument("--out", default="report.html")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "active" not in df:
        df["active"] = "all"
    tj = json.load(open(args.targets))
    tg = targets(df)
    sub = core(df)
    ref = df[(df.scenario == "reference")]

    inventory = df.drop_duplicates("run").groupby(["stimulus", "scenario"]).size().rename("runs").reset_index()
    tgt = pd.DataFrame([dict(synapse=s, target=cls, cell=tj["cell"][str(s)],
                             vfrac_r1=tj["vfrac"][str(s)], d_lateral_um=tj["d_lateral"][str(s)],
                             in_buffered_half=s in tj["buffered_half"]) for s, cls in tg])
    thr = pd.DataFrame([dict(stimulus=stim, **{m: g[f"thr_{m}"].iloc[0] for m in
                                                 ["uptake_cum", "time_to_floor", "frac_at_floor",
                                                  "peak_deficit", "recovery_half"]})
                        for stim, g in ref.groupby("stimulus")])
    ref5 = ref[ref.stimulus == STIM0]
    baseline = pd.DataFrame([dict(metric=m, median=ref5[m].median(), q25=ref5[m].quantile(0.25),
                                  q75=ref5[m].quantile(0.75), n=ref5[m].notna().sum())
                             for m in ["uptake_cum", "time_to_floor", "frac_at_floor", "peak_deficit",
                                       "recovery_half"]])
    peaks = sub[sub.scenario != "reference"].drop_duplicates("run").groupby(
        ["scenario", sub.local_width.fillna(0), "ecm_total"]).ecm_peak.first().rename(
        "peak buffer conc. (mM)").reset_index()
    eff = effects_table(sub, tg)
    g = local_gain(sub)
    gain_tab = g[["synapse", "target_class", "scenario", "local_width", "ecm_total", "buffered"]].copy()
    for m in VERDICT_METRICS:
        gain_tab[f"gain {m}"] = g[f"gain_{m}"]
        gain_tab[f"{m} sig"] = np.where(sig(g[f"gain_{m}"], g[f"thr_{m}"]), "*", "")
    gain_tab = gain_tab.sort_values(["synapse", "ecm_total", "scenario", "local_width"])
    top = g[(g.ecm_total == g.ecm_total.max()) & g.buffered]
    uni_top = sub[(sub.scenario == "uniform") & (sub.ecm_total == sub.ecm_total.max())
                  & sub.target_class.notna()]
    thr_u, thr_d = sub.thr_uptake_cum.iloc[0], sub.thr_peak_deficit.iloc[0]
    ratio = (top.effect_uptake_cum / uni_top.set_index("synapse").effect_uptake_cum.reindex(top.synapse).to_numpy())
    place = placement(sub)
    p_top = place[place.dose == place.dose.max()]
    conclusions = f"""
<h2>Conclusions</h2>
<ul>
<li>At the largest dose ({g.ecm_total.max():g} mM, ECS mean) the uniform buffer changes the cumulative uptake of the
four evaluation synapses by {uni_top.effect_uptake_cum.min():.2f} to {uni_top.effect_uptake_cum.max():.2f} amol
(threshold {thr_u:.2f} amol): {int(sig(uni_top.effect_uptake_cum, thr_u).sum())} of 4 cross it. Peak-deficit changes are
{uni_top.effect_peak_deficit.min():.3f} to {uni_top.effect_peak_deficit.max():.3f} mM (threshold {thr_d:.2f} mM).</li>
<li>Putting the same amount on the synapse multiplies its uptake effect by {ratio.min():.2f} to {ratio.max():.2f}
(all placements and widths, buffered targets, largest dose); the local gain over uniform is
{top.gain_uptake_cum.min():.2f} to {top.gain_uptake_cum.max():.2f} amol, i.e. {int(sig(top.gain_uptake_cum, thr_u).sum())} of
{len(top)} buffered target cases exceed the threshold. Below 3 mM no placement produces a significant effect or gain.</li>
<li>The equivalent uniform dose is only about 1-2x the actual dose although the buffer concentration at the synapse is
3-27x higher: the synapse draws Ca from a region several um wide, so buffer concentrated within sigma &lt;= 1 um
contributes little beyond its share of the total. Most of the effect is global (the buffer props up the whole ECS),
and unbuffered synapses in the half-buffered runs still receive a large part of the uniform effect.</li>
<li>Placement, re-scored on all synapses with each synapse normalised by its own uniform effect: at
{place.dose.max():g} mM a synapse carrying buffer gets {p_top["uptake_cum: on-target"].min():.2f}-{p_top["uptake_cum: on-target"].max():.2f}x
the uniform uptake effect, a synapse without gets {p_top["uptake_cum: off-target"].min():.2f}-{p_top["uptake_cum: off-target"].max():.2f}x,
so buffer on the synapse itself is worth {p_top["uptake_cum: on/off"].min():.1f}-{p_top["uptake_cum: on/off"].max():.1f}x the same
amount on a neighbour. Placement does matter in relative terms; the absolute gain over uniform still stays below the
threshold because the whole effect is small.</li>
<li>Robustness: the ordering does not change with k_on (effects saturate above ~1e2 /(mM s), the buffer is inert
below ~1 /(mM s)), with K_d (largest effect at K_d near the resting Ca, inert at 0.01 mM), or with a single-pulse
stimulus. The buffered-minus-uniform gain never exceeds the threshold in Step 3; the uniform effect itself exceeds it
only at 10 mM.</li>
<li>Floor-based metrics are uninformative: no synapse reaches the 0.47 mM floor with this stimulus. Recovery t1/2 is
quantized by the 2 ms record interval, so its 2-4 ms IQR is resolution, not biology.</li>
<li><b>Verdict.</b> Within this model and design the claim is not supported: concentrating a matched amount of buffer at
a synapse does not make a significant difference where the uniform buffer makes none. Placement changes the effect
by a factor of about two at most; the total amount and the synapse's own geometry dominate.</li>
</ul>"""
    eq_path = os.path.join(args.plots, "equivalent_dose.csv")
    eq = pd.read_csv(eq_path) if os.path.exists(eq_path) else pd.DataFrame()

    conf = sub[(sub.scenario == "half") & (sub.ecm_total == sub.ecm_total.max()) & ~sub.buffered]
    conf_rows = []
    for width, g in conf.groupby("local_width"):
        for m in VERDICT_METRICS:
            s_ = g[sig(g[f"effect_{m}"], g[f"thr_{m}"])]
            conf_rows.append(dict(sigma=width, metric=m, n_unbuffered=len(g), n_significant=len(s_),
                                  max_distance_significant_um=s_.d_buffered.max() if len(s_) else np.nan,
                                  min_distance_unbuffered_um=g.d_buffered.min()))
    conf_tab = pd.DataFrame(conf_rows)

    H = []
    H.append(f"""<!doctype html><html><head><meta charset="utf-8"><title>Buffer dose-response</title>
<style>body{{font-family:sans-serif;max-width:1400px;margin:auto;padding:1em;line-height:1.4}}
.t{{border-collapse:collapse;font-size:12px}} .t td,.t th{{padding:2px 8px;border-bottom:1px solid #ddd;text-align:right}}
.t th{{background:#f3f3f3}} h2{{margin-top:2em;border-bottom:2px solid #333}} small{{color:#666}}</style></head><body>
<h1>Synapse-local vs uniform ECM Ca buffer at matched total amount</h1>
<p><b>Claim under test.</b> For the same total buffer amount, concentrating the buffer near a synapse
changes that synapse's Ca depletion significantly, while spreading it uniformly over the ECS does not.</p>
<p><b>Setup.</b> tissue_kinetics default geometry (20x20x1 um, ECS ratio 0.1), 5 pulses at 100 Hz
(floor 0.47 mM), immobile buffer Ca + ECM &lt;-&gt; ECM_Ca with K_d = {KD0:g} mM, k_on = {KF0:g} /(mM s),
initialized at equilibrium with 1.3 mM Ca; dt = 0.5 ms, records every 2 ms, one seed. The buffer dose N is
the ECS mean concentration, identical across placements: uniform, Gaussians (sigma) on every active synapse,
or on half of them (<code>ecm.local_target</code>). Effects are per-synapse differences to the no-buffer
reference; the significance threshold is the IQR of the metric across the reference run's synapses.</p>
{conclusions}
<h2>Runs</h2>{table(inventory)}
<h2>Evaluation synapses</h2>
<p>Picked on the reference run by local ECS volume fraction within a geodesic radius of 1 um
(lowest two = restricted, highest two = connected), at least 2 um from the lateral box edge, alone in
their cell, pairwise &ge; 5 um apart. The buffered half ({len(tj['buffered_half'])} of {tj['n_synapses']}
active synapses) contains exactly one of each.</p>{table(tgt)}
<h2>Reference baseline and thresholds</h2>
<p>Metric distributions across all synapses of the 5-pulse reference run; the IQR is the threshold.
A zero IQR (floor metrics when no synapse reaches the depletion floor) makes that metric unusable
for the verdict; it is reported but not flagged.</p>
{table(baseline)}<p>Thresholds per stimulus:</p>{table(thr)}
<h2>Step 1: dose-response</h2>
<p>Peak total buffer concentration in the ECS at t=0 (the dilution the local placement buys):</p>{table(peaks)}
{img(os.path.join(args.plots, 'dose_response.png'))}
<p>Effects of the verdict metrics per evaluation synapse (* = |effect| &gt; IQR):</p>{table(eff)}
<p>Local gain = effect with the local buffer minus effect with the uniform buffer, on the same
synapse at the same dose (* = |gain| &gt; IQR). Filled/True = this synapse carries buffer in that run:</p>
{table(gain_tab)}
<h2>Step 2: equivalent uniform dose and confinement</h2>
<p>N_eq is the uniform dose whose effect on the same synapse matches the local-buffer effect
(log-interpolated along the uniform dose-response; blank where outside the sampled range).
The comparison line is the buffer concentration actually present at the synapse divided by N.</p>
{img(os.path.join(args.plots, 'equivalent_dose.png'))}{table(eq) if len(eq) else ''}
<p>Spatial confinement, half-buffered scenario at the largest dose: effects on the synapses that carry
no buffer, vs their distance to the nearest buffered one.</p>
{img(os.path.join(args.plots, 'confinement.png'))}{table(conf_tab)}
<h2>Placement: on- vs off-target at the same amount</h2>
<p>Comparing buffered with unbuffered <i>synapses</i> directly confounds placement with synapse identity, so each
synapse's effect in the half-buffered run is divided by its own effect in the uniform run at the same dose
(relative effect; 1 = same as uniform, 0 = no effect). Medians over the buffered and the unbuffered synapses of
the core runs and their ratio:</p>
{img(os.path.join(args.plots, 'placement.png'))}{table(place)}
<h2>Step 3: robustness</h2>
<p>Verdict per parameter value and dose (sigma = 1 um): <i>gain</i> = effect in the half-buffered run minus
effect in the uniform run on the same synapse, averaged over the two buffered and the two unbuffered evaluation
synapses; <i>uniform max</i> = largest |effect| over the four in the uniform run. The claim holds where the
buffered gain &gt; thr and uniform &lt; thr.</p>
""")
    for axis, title, fig in (("ecm_kf", "On-rate k_on (1/(mM s)), K_d = 1 mM", "robustness_ecm_kf.png"),
                             ("ecm_kd", "Affinity K_d (mM), k_on = 1e5 /(mM s)", "robustness_ecm_kd.png"),
                             ("stimulus", "Sub-saturating stimulus (1 pulse)", "robustness_stimulus.png")):
        v = verdict(df, axis, tg)
        if v.empty:
            continue
        H.append(f"<h3>{title}</h3>{img(os.path.join(args.plots, fig))}{table(v)}")
    H.append("<h2>Deviations from the original plan</h2><ul>" +
             "".join(f"<li>{d}</li>" for d in DEVIATIONS) + "</ul>")
    H.append(f"<p><small>Source: {os.path.abspath(args.csv)}; reference run {tj['reference_run']}</small></p>"
             "</body></html>")
    open(args.out, "w").write("\n".join(H))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
