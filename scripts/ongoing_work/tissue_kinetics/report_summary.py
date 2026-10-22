"""Succinct self-contained HTML summary of the tissue_kinetics study so far: chemical ECM
buffering (dose-response study) vs ECM shape change (mechanics phase 1 and the crowd swelling
push), built from the existing CSVs and figures.

    uv run scripts/ongoing_work/tissue_kinetics/report_summary.py \\
        --dose results/dose-response-analysis-ecs19 --mech results/mech-analysis --out report_summary.html

``--dose`` holds dose_response.csv and plots/ (plot_dose_response.py, plot_single_synapse.py),
``--mech`` holds mechanics.csv (mech_calib_readout.py) and plots/ (plot_mechanics.py).
"""
import argparse
import os

import numpy as np
import pandas as pd

from report_dose_response import img, table

STIM = "5p_0.5pA"


def buffering_tables(df):
    """Crowd: median effect of the uniform buffer per dose vs the reference IQR.
    Solitary: effect of the on-target buffer (1 mM) at the active synapse per target."""
    df = df[df.stimulus == STIM]
    ref = df[df.scenario == "reference"]
    core = df[(df.ecm_kf == 1e5) & (df.ecm_kd == 1.0)]
    crowd_ref = ref[ref.active == "all"]
    rows = []
    for dose, g in core[(core.scenario == "uniform") & (core.active == "all")].groupby("ecm_total"):
        half = core[(core.scenario == "half") & (core.active == "all") & (core.ecm_total == dose) & core.buffered]
        rows.append({"dose (mM, ECS mean)": dose,
                     "uniform: median d uptake (amol)": g.effect_uptake_cum.median(),
                     "local on buffered synapses: median d uptake (amol)": half.effect_uptake_cum.median(),
                     "uniform: median d peak deficit (mM)": g.effect_peak_deficit.median(),
                     "local on buffered synapses: median d peak deficit (mM)": half.effect_peak_deficit.median()})
    crowd = pd.DataFrame(rows)
    crowd_thr = dict(uptake=crowd_ref.thr_uptake_cum.iloc[0], deficit=crowd_ref.thr_peak_deficit.iloc[0],
                     deficit_median=crowd_ref.peak_deficit.median())
    sol_ref = ref[(ref.active != "all") & ref.is_active].set_index("active")
    rows = []
    on = core[(core.scenario == "on") & (core.ecm_total == 1.0) & core.is_active]
    uni = core[(core.scenario == "uniform") & (core.ecm_total == 1.0) & (core.active != "all") & core.is_active]
    for (a, w), g in on.groupby(["active", "local_width"]):
        r, u = sol_ref.loc[a], uni[uni.active == a].iloc[0]
        rows.append({"active synapse": a, "sigma (um)": w, "reference peak deficit (mM)": r.peak_deficit,
                     "buffer at synapse (mM)": g.ecm_at_synapse.iloc[0],
                     "d deficit uniform (mM)": u.effect_peak_deficit,
                     "d deficit on-target (mM)": g.effect_peak_deficit.iloc[0],
                     "d deficit on-target (%)": 100 * g.effect_peak_deficit.iloc[0] / r.peak_deficit,
                     "d uptake on-target (%)": 100 * g.effect_uptake_cum.iloc[0] / r.uptake_cum,
                     "on / uniform (uptake)": g.effect_uptake_cum.iloc[0] / u.effect_uptake_cum})
    sol = pd.DataFrame(rows).sort_values(["sigma (um)", "active synapse"])
    sol_thr = dict(uptake=sol_ref.thr_uptake_cum.iloc[0], deficit=sol_ref.thr_peak_deficit.iloc[0])
    return crowd, crowd_thr, sol, sol_thr


def mechanics_table(m):
    m = m.copy()
    m["swelling (%)"] = 100 * (np.where(m.regime == "solitary", m.local_vol_peak, m.ecs_vol_peak) - 1)
    m["block volume (%)"] = 100 * (m.total_vol_peak - 1)
    m["status"] = np.where(m.t_end_ms >= m.end_ms - 1e-6, "finished",
                           "runaway, failed at " + m.t_end_ms.round().astype(int).astype(str) + " ms")
    m["springs"] = np.where(m.boundary_stiffness == 0, "free block", "default")
    cols = {"regime": "regime", "k": "k (1/mM)", "springs": "springs", "swelling (%)": "peak ECS swelling (%)",
            "block volume (%)": "block volume (%)", "J_min": "J_min", "J_max_ecs": "J_max",
            "peak_deficit": "peak deficit (mM)", "d_peak_deficit_%": "d deficit vs twin (%)",
            "s_per_step": "s / step", "status": "status"}
    return m[list(cols)].rename(columns=cols)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dose", required=True)
    p.add_argument("--mech", required=True)
    p.add_argument("--out", default="report_summary.html")
    args = p.parse_args()
    df = pd.read_csv(os.path.join(args.dose, "dose_response.csv"))
    m = pd.read_csv(os.path.join(args.mech, "mechanics.csv"))
    crowd, cthr, sol, sthr = buffering_tables(df)
    mech = mechanics_table(m)
    dp, mp = (lambda f: os.path.join(args.dose, "plots", f)), (lambda f: os.path.join(args.mech, "plots", f))
    k5 = m[(m.regime == "crowd") & (m.k == 5)].iloc[0]
    s02 = m[(m.regime == "solitary") & (m.k == 0.2)].iloc[0]
    k10 = m[(m.regime == "crowd") & (m.k == 10) & (m.boundary_stiffness == 1)].iloc[0]
    k0 = m[(m.regime == "crowd") & (m.k == 0)].iloc[0]
    n_runs = len(df.drop_duplicates("run")) + len(m)

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Buffering vs shape change</title>
<style>body{{font-family:sans-serif;max-width:1300px;margin:auto;padding:1em;line-height:1.45}}
h1{{font-size:1.5em}} h2{{font-size:1.2em;margin-top:1.6em}} .t{{border-collapse:collapse;font-size:0.85em}}
.t td,.t th{{padding:2px 8px;border-bottom:1px solid #ddd;text-align:right}} .t th{{background:#f3f3f3}}
.key{{background:#fff6e0;border-left:4px solid #e0a800;padding:0.6em 1em;margin:1em 0}}
.small{{font-size:0.85em;color:#444}}</style></head><body>
<h1>ECS Ca<sup>2+</sup> depletion: chemical buffering vs. ECM shape change</h1>
<p class="small">tissue_kinetics, 2026-10-13 to 2026-10-22, {n_runs} runs. 20&times;20&times;1 &micro;m slab, ECS fraction
0.19 nominal (0.25 measured; ECS 0.10 gave the same picture), 60 NMDAR synapses (0.5 pA, 5 pulses at 0, 10, 20,
30, 40 ms), Ca<sub>ECS</sub> 1.3 mM, tissue reservoir on all faces (pool refill &tau; &asymp; 25 ms). ECM buffer K<sub>d</sub>
1 mM, k<sub>on</sub> 10<sup>5</sup> /(mM s). Metrics: peak free-Ca deficit at the synapse and cumulative cell uptake.
"Significant" = larger than the inter-quartile range (IQR) of the metric across the unbuffered synapses.</p>

<div class="key"><b>Buffering alone does not significantly influence Ca depletion; shape change might.</b><br>
A chemical ECM buffer changes the peak deficit by at most ~10 % (&le; 0.06 mM, at 10 mM buffer), far below the
between-synapse IQR, in a crowd and at a solitary synapse, uniformly and placed on the synapse. ECM swelling
driven by Ca unbinding is the only mechanism found that can move depletion by more than the IQR: a few percent
relief at moderate swelling, and a dilution-driven <i>increase</i> of the deficit at large swelling that
outruns the tissue reservoir and runs away in the crowd. Which regime is real depends on one number the
simulations do not fix: the ECS volume change per mM of ECM-Ca unbound (k).</div>

<h2>1. Chemical buffering: small, and a pool effect in the crowd</h2>
<p>Crowd (all synapses firing). The buffer, uniform or concentrated on half of the synapses at the same total
amount, raises uptake above the IQR only at 10 mM and never moves the peak deficit out of the IQR.
Depletion in the crowd is a tissue-level pool effect (0.3-0.5 mM everywhere, synapse-specific part ~0.1 mM), so
placing buffer on a synapse gains at most ~2&times; over spreading it; the 4 evaluation synapses behave like all the
others.</p>
{img(dp('dose_response.png'))}
<p class="small">Reference: median peak deficit {cthr['deficit_median']:.2f} mM, IQR {cthr['deficit']:.3f} mM; uptake IQR {cthr['uptake']:.2f} amol.</p>
{table(crowd)}
<p>Solitary synapse (one synapse firing, the others silent). Here placement matters: buffer on the active synapse
gives 15-60&times; the uniform effect, i.e. roughly its local concentration factor. But the absolute effect stays
tiny: +3-13 % uptake and a peak-deficit change of 0.01-0.04 mM, below the IQR of {sthr['deficit']:.3f} mM
across the four solitary references.</p>
{img(dp('single_effect.png'))}
{table(sol)}

<h2>2. Shape change: the same few percent at moderate swelling, dominant beyond</h2>
<p>Model: ECM-Ca unbinding makes the ECS swell (stress-free volume J<sub>g</sub> = 1 + k &middot; &Delta;[ECM-Ca]
per reference volume, neo-Hookean ECS 0.5 kPa / cells 1 kPa, block embedded in compliant tissue), concentrations
are diluted by the volume gain, each run has a mechanics-off twin. k is a free parameter.</p>
{img(mp('mech_k_sweep.png'))}
<ul>
<li>Moderate swelling <b>relieves</b> depletion, linearly in k: the widened ECS supplies more Ca than the
dilution removes. Solitary synapse: {s02['d_peak_deficit_%']:.1f} % deficit at k 0.2 (7 % local swelling), the same as
the 1 mM on-target buffer ({100*sol['d deficit on-target (mM)'].median()/sol['reference peak deficit (mM)'].median():.1f} % median).
Crowd: {k5['d_peak_deficit_%']:.1f} % at k 5 ({100*(k5.ecs_vol_peak-1):.0f} % ECS swelling, block +{100*(k5.total_vol_peak-1):.0f} %).
Uptake changes &le; 2 %. Same order as buffering, still below the IQR.</li>
<li>Large swelling (k &ge; 10) <b>amplifies</b> depletion: dilution lowers free Ca, which unbinds more ECM-Ca,
which swells more. The reservoir refills the pool at ~0.02 mM/ms, the dilution removes ~0.13 mM/ms, so the loop
runs away. At k 10 the synaptic peak deficit reaches {k10.peak_deficit:.2f} mM after 8 ms (k 0: {k0.peak_deficit:.2f} mM after
44 ms), the ECS mean free Ca drops to 0.84 mM, then the swelling overshoots, retreats and the solver loses the
equilibrium; k 15 pushes the synaptic deficit to {m[(m.regime == "crowd") & (m.k == 15)].peak_deficit.iloc[0]:.2f} mM
(+35 %, beyond the IQR of {cthr['deficit']:.2f} mM) within 6 ms and then inverts the mesh. This is the first effect larger
than the IQR, but it is an instability of the shared pool, not a steady state. The stable ceiling in this slab is 25-30 % ECS swelling (k 5-7).</li>
</ul>
{img(mp('mech_crowd_timeseries.png'))}
{table(mech)}
<p class="small">J: element volume ratio (min over the block, max in the ECS). Crowd deficits are the median over
single-synapse cells; the k &ge; 10 rows are from the records before the solver failed.</p>

<h2>3. What decides it, and what is next</h2>
<ul>
<li>Both mechanisms give a few percent for parameters we consider realistic. Buffering cannot do more: its effect is
capped by the pool. Shape change can, but only through dilution of a shared pool at tens of percent ECS swelling,
which the model says is unstable and which requires k &asymp; 10 /mM (10 % ECS volume per 0.01 mM ECM-Ca unbound).</li>
<li>The decisive number is the physical k: ECS volume change per ECM-Ca unbound (ECM decondensation per Ca). It
should be estimated from the literature before more sweeps. If it is &le; 1 /mM, shape change is as inert as
buffering; if it is &ge; 5 /mM, the crowd regime becomes a swelling/dilution problem and the model needs a
mechanism that limits the loop (a faster reservoir or a volume-limited ECM).</li>
<li>Caveats: one seed, one geometry, swelling of the ECS only (no cell volume change; physiologically the ECS
<i>shrinks</i> during activity, which concentrates Ca), boundary springs matter for the crowd (free block swells
40 % more at k 10) but not for the stability.</li>
</ul>
<p class="small">Scripts: evaluate_dose_response / plot_dose_response / plot_single_synapse (section 1),
mech_calib_readout / plot_mechanics (section 2), report_summary (this page). Configs: dose_*_sweep,
single_*_sweep, mech_calib_sweep, mech_crowd_sweep.</p>
</body></html>"""
    open(args.out, "w").write(html)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
