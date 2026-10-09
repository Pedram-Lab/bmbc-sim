"""Coupled bmbcsim run (production chemistry + mechanics) on the two-cell mesh.

Reproduces the production collapse cheaply: uniform NMDAR-like Ca sink on membrane_1,
ECM buffer at Kd 1.3 mM, ECM_Ca drives ECS swelling with strength -k. Per step it logs
min/max J, ECS volume ratio and the ECM_Ca range in the ECS, so the failure mechanism
(dilution feedback vs static solver) can be read off the time series.

Usage:
    uv run python scripts/ongoing_work/two_cell/coupled_run.py --k 2.0 --tag baseline
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from astropy import units as u
from astropy import constants as const
import ngsolve as ngs

import bmbcsim
from bmbcsim.simulation import transport
from bmbcsim.simulation.fem_details import MechanicSolver

sys.path.insert(0, os.path.dirname(__file__))
from geometry import two_cell_mesh, BOX_MIN, BOX_MAX  # noqa: E402

CA_ECS, ECM_TOTAL, KD, KF = 1.3, 2.0, 1.3, 769.23          # mM, mM, mM, 1/(mM s)
PULSES_MS = [5, 10, 15, 20, 25, 30]
TAU1, TAU2 = 10.0, 3.0                                       # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=2.0, help="condensation 1/mM")
    ap.add_argument("--n-syn", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.25, help="ms")
    ap.add_argument("--end", type=float, default=40.0, help="ms")
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--ecs-ratio", type=float, default=0.1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="results/two_cell_coupled")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mesh = two_cell_mesh(ecs_ratio=args.ecs_ratio, n_refine=args.refine)
    result_dir = bmbcsim.timestamped_directory(args.out, args.tag)
    sim = bmbcsim.Simulation(mesh=mesh, result_directory=result_dir, mechanics=True)
    geo = sim.simulation_geometry
    ecs, cells = geo.compartments["ecs"], [geo.compartments[f"cell_{i}"] for i in range(2)]
    i_ecs = list(geo.compartments).index("ecs")

    ecs.add_elasticity(youngs_modulus=0.5 * u.kPa, poisson_ratio=0.3)
    for c in cells:
        c.add_elasticity(youngs_modulus=1.0 * u.kPa, poisson_ratio=0.4)

    ca, ecm, ecm_ca = (sim.add_species(n) for n in ("Ca", "ECM", "ECM_Ca"))
    K = KF / (KD * KF)  # 1/mM
    ecm_ca_eq = K * CA_ECS * ECM_TOTAL / (1 + K * CA_ECS)
    mM = u.mmol / u.L
    ecs.initialize_species(ca, CA_ECS * mM)
    ecs.initialize_species(ecm, (ECM_TOTAL - ecm_ca_eq) * mM)
    ecs.initialize_species(ecm_ca, ecm_ca_eq * mM)
    for c in cells:
        c.initialize_species(ca, 0.0 * mM)
    ecs.add_driving_species(ecm_ca, -args.k / mM, baseline=ecm_ca_eq * mM)

    ecs.add_diffusion(ca, 0.7 * u.um**2 / u.ms)
    for c in cells:
        c.add_diffusion(ca, 0.22 * u.um**2 / u.ms)
    ecs.add_reaction(reactants=[ca, ecm], products=[ecm_ca], k_f=KF / (mM * u.s), k_r=KF * KD / u.s)

    faraday = const.e.si * const.N_A
    q = args.n_syn * 35 * 0.5 * u.pA / (2 * faraday)

    def waveform(t):
        t_ms = t.to_value(u.ms)
        return sum(math.exp(-(t_ms - p) / TAU1) - math.exp(-(t_ms - p) / TAU2) for p in PULSES_MS if t_ms >= p)

    geo.membranes["membrane_1"].add_transport(
        ca, transport.ProportionalFlux(flux=q, saturation=CA_ECS * mM, depletion=0.47 * mM, temporal=waveform),
        ecs, cells[1])
    l_char = (BOX_MAX - BOX_MIN).max() / 2 * u.um
    perm = 0.7 * u.um**2 / u.ms / 1.6**2 / l_char
    for bnd in ["top", "bottom", "left", "right", "front", "back"]:
        b = geo.membranes[bnd]
        b.add_transport(ca, transport.Passive(perm * b.area, CA_ECS * mM), None, ecs)

    # --- per-step diagnostics, hooked after dilution ---
    vol0 = ngs.Integrate(ngs.CoefficientFunction(1), mesh, element_wise=True).NumPy().copy()
    mats = np.array([el.mat for el in mesh.Elements(ngs.VOL)])
    in_ecs = mats == "ecs"
    verts = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
    coords = np.array([v.point for v in mesh.vertices])
    centroids = coords[verts].mean(axis=1)
    from scipy.spatial import cKDTree
    mem_tree = cKDTree(coords[sorted({v.nr for el in mesh.Elements(ngs.BND) if el.mat.startswith("membrane")
                                      for v in el.vertices})])
    d_box = lambda p: min(np.abs(p - BOX_MIN).min(), np.abs(p - BOX_MAX).min())
    rows, t_last = [], [time.time()]
    log = open(os.path.join(args.out, f"steps_{args.tag}.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "J_min", "J_min_mat", "J_max", "ecs_vol_ratio", "ecmca_min", "ecmca_max",
                     "Jg_min", "Jg_max", "wall", "Jmin_x", "Jmin_y", "Jmin_z", "Jmin_d_mem", "Jmin_d_box",
                     "Jmin_Jg", "Jmin_c", "Jmin_Jprev", "patch_J_mean"])
    orig_adjust = MechanicSolver.adjust_concentrations

    def adjust(self, concentrations):
        orig_adjust(self, concentrations)
        self._mesh.UnsetDeformation()
        J = ngs.Integrate(ngs.Det(ngs.Id(3) + ngs.Grad(self.deformation)), mesh, element_wise=True).NumPy() / vol0
        c_cf = concentrations[ecm_ca].components[i_ecs]
        jprev_cf = self._volume_ratio.components[i_ecs]
        i = int(J.argmin())
        ew = lambda cf: float(ngs.Integrate(cf, mesh, element_wise=True).NumPy()[i]) / vol0[i]
        c_i, jp_i = ew(c_cf), ew(jprev_cf)
        self._mesh.SetDeformation(self.deformation)
        c = concentrations[ecm_ca].components[i_ecs].vec.FV().NumPy()
        patch = np.any(np.isin(verts, verts[i]), axis=1)
        now = time.time()
        row = [len(rows), J.min(), mats[i], J.max(), (J * vol0)[in_ecs].sum() / vol0[in_ecs].sum(),
               c.min(), c.max(), 1 - args.k * (c.max() - ecm_ca_eq), 1 - args.k * (c.min() - ecm_ca_eq),
               now - t_last[0], *centroids[i], mem_tree.query(centroids[i])[0], d_box(centroids[i]),
               1 - args.k * (c_i * jp_i - ecm_ca_eq), c_i, jp_i, (J * vol0)[patch].sum() / vol0[patch].sum()]
        t_last[0] = now
        rows.append(row)
        writer.writerow([f"{v:.4g}" if isinstance(v, float) else v for v in row])
        log.flush()

    MechanicSolver.adjust_concentrations = adjust

    cause = "ok"
    try:
        sim.run(end_time=args.end * u.ms, time_step=args.dt * u.ms, record_interval=20 * args.dt * u.ms,
                n_threads=args.threads)
    except RuntimeError as e:
        cause = f"FAIL step {len(rows)} (t={len(rows) * args.dt:.3g} ms): {str(e).splitlines()[0][:120]}"
    log.close()
    r = np.array([row[1:] for row in rows], dtype=object)
    print(f"{args.tag}: k={args.k} n_syn={args.n_syn} dt={args.dt} -> {cause}")
    print(f"  steps {len(rows)}, J_min overall {min(x[1] for x in rows):.3g}, ECS vol ratio max "
          f"{max(x[4] for x in rows):.3f}, ECM_Ca min {min(x[5] for x in rows):.3g} max {max(x[6] for x in rows):.3g}, "
          f"Jg range [{min(x[7] for x in rows):.3g}, {max(x[8] for x in rows):.3g}], "
          f"mean wall/step {np.mean([x[9] for x in rows]):.2f} s")


if __name__ == "__main__":
    main()
