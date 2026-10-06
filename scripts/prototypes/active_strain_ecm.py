"""Standalone NGSolve prototype of the Ca-dependent active-strain ECM model.

Static problem on the unit cube with a prescribed Ca field (or prescribed
swelling stretch theta). Multiplicative split F = F_e * theta*I, network
energy Psi_e in {neohooke, wlc}, total energy Psi = theta^3 Psi_e.

Usage:
    uv run python active_strain_ecm.py --experiments sanity push sweeps ca --out results/active_strain
"""
import argparse
import csv
import glob
import itertools
import math
import os
import time

import ngsolve
import numpy as np
from netgen.occ import Box, OCCGeometry, Pnt, X, Y, Z
from ngsolve import (BilinearForm, BitArray, CoefficientFunction, Det, Grad, GridFunction, Id,
                     InnerProduct, IntegrationRule, Mesh, Parameter, Projector, TaskManager,
                     Trace, VectorH1, VOL, TET, VTKOutput, Variation, ds, dx, exp, log, specialcf, sqrt, x, y, z)

# ---------------------------------------------------------------- parameters
DEFAULTS = dict(mu=1.0, nu=0.25, kT=1.0, Kd=0.7, Cref=1.3, L_over_lp0=30.0, lp0=1.0, lp1=1.7 / 3.2)


def lp_of_C(C, p):
    """Persistence length as function of Ca; the single replaceable function."""
    f = C / (C + p["Kd"])
    return p["lp0"] + (p["lp1"] - p["lp0"]) * f


def lam_of_nu(mu, nu):
    return 2 * mu * nu / (1 - 2 * nu)


def g_wlc(xx):
    """Bracket of w'(r): w'(r) = (kT/l_p) g(x)."""
    return 1 / (4 * (1 - xx) ** 2) - 1 / 4 + xx


def tanh(t):
    return 1 - 2 / (exp(2 * t) + 1)


# ---------------------------------------------------------------- fields
def shape_field(kind, w):
    """Normalised profile g(x) in [0, 1]."""
    if kind == "homogeneous":
        return CoefficientFunction(1.0)
    if kind == "linear":
        return x
    if kind == "blob":
        return exp(-((x - .5) ** 2 + (y - .5) ** 2 + (z - .5) ** 2) / (2 * 0.15 ** 2))
    if kind == "step":
        return 0.5 * (1 + tanh((x - 0.5) / w))
    raise ValueError(kind)


# ---------------------------------------------------------------- model
class Model:
    """Energy, diagnostics and solver state for one (mesh, order, bc, energy, theta_target)."""

    def __init__(self, mesh, order, bc, energy, theta_target, p, pin=False):
        self.mesh, self.order, self.energy, self.p = mesh, order, energy, p
        self.V = VectorH1(mesh, order=order, dirichlet={"bottom": "bottom", "all": ".*", "none": "", "spring": ""}[bc])
        self.u = GridFunction(self.V)
        self.free = BitArray(self.V.FreeDofs())
        if pin:  # remove 6 rigid modes: corner (0,0,0) fully, (1,0,0) in y,z, (0,1,0) in z
            for pt, comps in [((0, 0, 0), (0, 1, 2)), ((1, 0, 0), (1, 2)), ((0, 1, 0), (2,))]:
                v = min(mesh.vertices, key=lambda vv: sum((a - b) ** 2 for a, b in zip(vv.point, pt)))
                for c in comps:
                    self.free[self.V.Range(c).start + v.nr] = False
        self.proj = Projector(self.free, True)

        self.s = Parameter(0.0)
        self.theta = 1 + self.s * (theta_target - 1)
        mu, lam, kT, L = p["mu"], lam_of_nu(p["mu"], p["nu"]), p["kT"], p["L_over_lp0"] * p["lp0"]
        lp_ref = lp_of_C(p["Cref"], p)
        lp = self.theta ** 2 * lp_ref  # consistent with theta = sqrt(lp/lp_ref)

        def terms(uu):
            """Energy density and diagnostics as functions of a displacement (proxy or GridFunction)."""
            F = Id(3) + Grad(uu)
            Fe = F / self.theta
            J = Det(F)
            Je = J / self.theta ** 3
            trCe = Trace(Fe.trans * Fe)
            vol = lam / 4 * (Je * Je - 1 - 2 * log(Je))
            d = dict(J=J, Je=Je, FeI=sqrt(InnerProduct(Fe - Id(3), Fe - Id(3))), xcf=None, x0cf=None)
            if energy == "neohooke":
                Psi_e = mu / 2 * (trCe - 3 - 2 * log(Je)) + vol
            elif energy == "wlc":
                R0_ref = math.sqrt(2 * lp_ref * L)
                x_ref = R0_ref / L
                n = 3 * mu / (kT / lp_ref * g_wlc(x_ref) * R0_ref)  # calibrate n at C_ref
                R0 = sqrt(2 * lp * L)
                lch = sqrt(trCe / 3)
                xx = R0 * lch / L
                x0 = R0 / L
                w = lambda xi: kT * L / lp * (xi * xi / 2 + 1 / (4 * (1 - xi)) - xi / 4 - 1 / 4)
                R0wp = kT / lp * g_wlc(x0) * R0
                # subtract w(R0) so Psi_e = 0 in the stress-free state, as for neohooke
                Psi_e = n * (w(xx) - w(x0) - R0wp / 3 * log(Je)) + vol
                d.update(xcf=xx, x0cf=x0, lch_over_lmax_ref=lch * R0_ref / L)
            else:
                raise ValueError(energy)
            return self.theta ** 3 * Psi_e, d

        Psi, _ = terms(self.V.TrialFunction())
        _, diag = terms(self.u)
        self.J, self.Je, self.FeI, self.xcf, self.x0cf = (diag[k] for k in ["J", "Je", "FeI", "xcf", "x0cf"])
        self.lch_over_lmax_ref = diag.get("lch_over_lmax_ref")

        self.a = BilinearForm(self.V, symmetric=True)
        self.a += Variation(Psi.Compile() * dx)
        if bc == "spring":  # local compliant embedding on all faces, as in bmbcsim fem_details
            uu, n, lc = self.V.TrialFunction(), specialcf.normal(3), 1 / math.sqrt(3)
            E, un = 2 * mu * (1 + p["nu"]), InnerProduct(uu, n)
            self.a += Variation((E * un * un + mu * (InnerProduct(uu, uu) - un * un)) / (2 * lc) * ds)
        self.pts = mesh.MapToAllElements(IntegrationRule(TET, 2 * order), VOL)
        self.r = self.u.vec.CreateVector()
        self.du = self.u.vec.CreateVector()

    def ev(self, cf):
        return cf(self.pts).ravel()

    def admissible(self):
        """Rejection test at quadrature points. Returns None or the cause."""
        if self.ev(self.J).min() <= 0:
            return "J->0"
        if self.xcf is not None:
            if self.ev(self.x0cf).max() >= 1 - 1e-8:
                return "x0->1"
            if self.ev(self.xcf).max() >= 1 - 1e-8:
                return "x->1"
        return None

    def newton(self, maxit=30, rtol=1e-8, atol=1e-10):
        """Energy-backtracking Newton. Returns (converged, iterations, cause)."""
        a, u, r, du = self.a, self.u, self.r, self.du
        causes = []
        for it in range(maxit):
            a.Apply(u.vec, r)
            self.proj.Project(r)
            res = r.Norm()
            if it == 0:
                res0 = res
            if res <= atol or res <= rtol * res0:
                return True, it, None
            a.AssembleLinearization(u.vec)
            try:
                inv = a.mat.Inverse(self.free, inverse="pardiso")
            except Exception:
                return False, it, "singular"
            du.data = inv * r
            slope = InnerProduct(r, du)
            if not slope > 0:
                return False, it, "indefinite"
            E0 = a.Energy(u.vec)
            alpha = 1.0
            while alpha > 1e-4:
                u.vec.data -= alpha * du
                E = a.Energy(u.vec)
                cause = "nan" if not math.isfinite(E) else self.admissible()
                if cause is None and E <= E0 - 1e-4 * alpha * slope:
                    break
                u.vec.data += alpha * du
                if cause:
                    causes.append(cause)
                alpha /= 2
            else:
                return False, it, max(set(causes), key=causes.count) if causes else "stagnation"
        return False, maxit, "maxit"

    def continuation(self, ds0, s_max=1.0):
        """Ramp s in [0, s_max] adaptively. Returns dict of run statistics."""
        s, ds, steps, iters, cause = 0.0, ds0, 0, 0, None
        u_old = self.u.vec.CreateVector()
        t0 = time.time()
        while s < s_max - 1e-12 and ds >= 1e-4:
            s_try = min(s_max, s + ds)
            self.s.Set(s_try)
            u_old.data = self.u.vec
            ok, it, c = self.newton()
            iters += it
            if ok:
                s, steps = s_try, steps + 1
                if it <= 4:
                    ds *= 1.5
            else:
                self.u.vec.data = u_old
                ds /= 2
                cause = c
        self.s.Set(s)
        d = dict(success=s >= s_max - 1e-12, s=s, n_steps=steps, newton_iters=iters,
                 energy_value=self.a.Energy(self.u.vec), wall=time.time() - t0,
                 cause="" if s >= s_max - 1e-12 else cause, ndof=self.V.ndof)
        th = self.ev(self.theta)
        J, Je = self.ev(self.J), self.ev(self.Je)
        d.update(theta_min=th.min(), theta_max=th.max(), J_min=J.min(), J_max=J.max(),
                 Je_min=Je.min(), Je_max=Je.max(), max_FeI=self.ev(self.FeI).max())
        if self.xcf is not None:
            d.update(x_max=self.ev(self.xcf).max(), x0_max=self.ev(self.x0cf).max(),
                     lch_over_lmax=self.ev(self.lch_over_lmax_ref).max())
        return d

    def vtk(self, filename):
        coefs, names = [self.u, self.J, self.Je, self.theta], ["u", "J", "Je", "theta"]
        if self.xcf is not None:
            coefs.append(self.xcf), names.append("x")
        VTKOutput(self.mesh, coefs=coefs, names=names, filename=filename, subdivision=1).Do()


# ---------------------------------------------------------------- mesh
_meshes = {}


def unit_cube(maxh):
    if maxh not in _meshes:
        box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
        for ax, lo, hi in [(X, "left", "right"), (Y, "front", "back"), (Z, "bottom", "top")]:
            box.faces.Min(ax).name, box.faces.Max(ax).name = lo, hi
        _meshes[maxh] = Mesh(OCCGeometry(box).GenerateMesh(maxh=maxh))
    return _meshes[maxh]


# ---------------------------------------------------------------- experiments
A_EXPAND, A_CONTRACT = 9.0, -0.99  # theta caps: 10 and 0.01 (g in [0, 1])
COLUMNS = ["experiment", "energy", "bc", "field", "w", "mode", "A", "nu", "maxh", "order", "L_over_lp0",
           "success", "s", "theta_min", "theta_max", "n_steps", "newton_iters", "J_min", "J_max",
           "Je_min", "Je_max", "max_FeI", "x_max", "x0_max", "lch_over_lmax", "energy_value", "wall",
           "cause", "ndof"]


def run_case(experiment, energy, bc, field, w=0.05, mode="theta", A=1.0, nu=0.25, maxh=0.1, order=2,
             L_over_lp0=30.0, C_range=(0.0, 100.0), vtk=None):
    p = dict(DEFAULTS, nu=nu, L_over_lp0=L_over_lp0)
    g = shape_field(field, w)
    if mode == "theta":
        theta_target = 1 + A * g
    else:  # Ca mode: C = C0 + (C1 - C0) g, theta = sqrt(lp(C)/lp(Cref))
        C = C_range[0] + (C_range[1] - C_range[0]) * g
        theta_target = sqrt(lp_of_C(C, p) / lp_of_C(p["Cref"], p))
        A = float("nan")
    m = Model(unit_cube(maxh), order, bc, energy, theta_target, p)
    d = m.continuation(ds0=min(0.1, 0.05 / abs(A)) if mode == "theta" else 0.25)
    row = dict(experiment=experiment, energy=energy, bc=bc, field=field, w=w, mode=mode, A=A, nu=nu,
               maxh=maxh, order=order, L_over_lp0=L_over_lp0, **d)
    print(f"[{experiment}] {energy:8s} {bc:6s} {field:11s} A={A:+.2f} nu={nu} h={maxh} p={order} "
          f"L={L_over_lp0:g} -> s={d['s']:.4f} theta=[{d['theta_min']:.3f},{d['theta_max']:.3f}] "
          f"steps={d['n_steps']} it={d['newton_iters']} cause={d['cause'] or 'ok'} {d['wall']:.0f}s",
          flush=True)
    if vtk:
        m.vtk(vtk)
    return row


def experiment_push(vtk_dir, bcs=("bottom", "all")):
    for energy, bc, field, A in itertools.product(["neohooke", "wlc"], bcs,
                                                  ["homogeneous", "linear", "blob", "step"],
                                                  [A_EXPAND, A_CONTRACT]):
        tag = f"{energy}_{bc}_{field}_{'expand' if A > 0 else 'contract'}"
        yield run_case("push", energy, bc, field, A=A, vtk=os.path.join(vtk_dir, tag) if vtk_dir else None)


def experiment_ca(bcs=("bottom", "all")):
    for energy, bc, field in itertools.product(["neohooke", "wlc"], bcs,
                                               ["homogeneous", "linear", "blob", "step"]):
        yield run_case("ca", energy, bc, field, mode="ca")


SWEEPS = dict(nu=[0.0, 0.25, 0.4, 0.45, 0.49], w=[0.2, 0.05, 0.01], maxh=[0.2, 0.1, 0.05],
              order=[1, 2], L_over_lp0=[10.0, 30.0, 100.0, 1000.0])


def experiment_sweep(name):
    """One-at-a-time sweep from the base case (step field, all faces clamped)."""
    for val, energy, A in itertools.product(SWEEPS[name], ["neohooke", "wlc"], [A_EXPAND, A_CONTRACT]):
        if name == "L_over_lp0" and energy == "neohooke":
            continue
        yield run_case(f"sweep_{name}", energy, "all", "step", A=A, **{name: val})


# ---------------------------------------------------------------- sanity checks
def sanity():
    mesh, order, p = unit_cube(0.2), 2, dict(DEFAULTS)
    rel = lambda a, b: np.linalg.norm(a.vec.FV().NumPy() - b.vec.FV().NumPy()) / np.linalg.norm(b.vec.FV().NumPy())

    for energy in ["neohooke", "wlc"]:
        # theta == 1: u == 0, zero stress, Newton converges immediately
        m = Model(mesh, order, "bottom", energy, CoefficientFunction(1.0), p)
        m.s.Set(1.0)
        ok, it, _ = m.newton()
        assert ok and it <= 1 and m.u.vec.Norm() < 1e-10 and m.ev(m.FeI).max() < 1e-10, (energy, it)

        # homogeneous theta, pinned: u = (theta-1)(X - X0), X0 = origin
        th = 1.1
        m = Model(mesh, order, "none", energy, CoefficientFunction(th), p, pin=True)
        d = m.continuation(0.25)
        exact = GridFunction(m.V)
        exact.Set((th - 1) * CoefficientFunction((x, y, z)))
        assert d["success"] and rel(m.u, exact) < 1e-8 and m.ev(m.FeI).max() < 1e-8, (energy, d["cause"], rel(m.u, exact))

        # homogeneous theta, bottom clamped: Fe ~ I (stress-free) away from the clamp.
        # The cube is not slender, so the clamp's influence only decays to ~10 % at the top.
        m = Model(mesh, order, "bottom", energy, CoefficientFunction(th), p)
        d = m.continuation(0.25)
        f, zz = m.ev(m.FeI) / (th - 1), m.ev(z)
        rms = lambda sel: math.sqrt((f[sel] ** 2).mean())
        lo, hi = rms(zz < 0.25), rms(zz > 0.75)
        assert d["success"] and hi < 0.15 and hi < lo / 2, (energy, lo, hi)
        print(f"  {energy}: rms |Fe-I|/(theta-1): z<0.25 -> {lo:.3f}, z>0.75 -> {hi:.3f}")

        # scaling mu, lam by a constant leaves u unchanged
        tt = 1 + 0.5 * shape_field("step", 0.05)
        us = []
        for scale in [1.0, 10.0]:
            m = Model(mesh, order, "all", energy, tt, dict(p, mu=scale * p["mu"]))
            m.continuation(0.25)
            us.append(m.u)
        assert rel(us[0], us[1]) < 1e-7, (energy, rel(us[0], us[1]))

    # small strain: wlc -> neohooke as L/lp0 grows
    tt = 1 + 0.02 * shape_field("step", 0.05)
    ref = Model(mesh, order, "all", "neohooke", tt, p)
    ref.continuation(1.0)
    diffs = []
    for L in [30.0, 100.0, 1000.0]:
        m = Model(mesh, order, "all", "wlc", tt, dict(p, L_over_lp0=L))
        m.continuation(1.0)
        diffs.append(rel(m.u, ref.u))
    print(f"  small strain |u_wlc-u_nh|/|u_nh| for L/lp0=30,100,1000: {diffs}")
    assert diffs[0] < 0.05 and diffs == sorted(diffs, reverse=True) and diffs[-1] < 0.01, diffs

    # large strain, L/lp0 = 1000: wlc ~ neohooke over a push range
    for A in [2.0, -0.6]:
        tt = 1 + A * shape_field("step", 0.05)
        ref = Model(mesh, order, "all", "neohooke", tt, p)
        ref.continuation(0.05)
        diffs = []
        for L in [100.0, 1000.0]:
            m = Model(mesh, order, "all", "wlc", tt, dict(p, L_over_lp0=L))
            m.continuation(0.05)
            diffs.append(rel(m.u, ref.u))
        print(f"  large strain A={A:+.1f}: |u_wlc-u_nh|/|u_nh| for L/lp0=100,1000: {diffs}")
        assert diffs[1] < diffs[0] and diffs[1] < 0.1, diffs
    print("sanity checks passed")


# ---------------------------------------------------------------- summary
def summary(csv_paths):
    import pandas as pd
    df = pd.concat([pd.read_csv(f) for f in csv_paths], ignore_index=True)
    df["dir"] = np.where(df["A"] > 0, "expand", "contract")
    df["push"] = np.where(df["A"] > 0, df["theta_max"], df["theta_min"])
    pd.set_option("display.width", 200, "display.max_rows", 500)
    push = df[df.experiment == "push"]
    if len(push):
        print("\n=== Push test: attained theta (cap 10 / 0.01) and failure cause ===")
        print(push.pivot_table(index=["energy", "bc", "field"], columns="dir", values="push", aggfunc="first").round(3))
        print(push.pivot_table(index=["energy", "bc", "field"], columns="dir", values="cause", aggfunc="first").fillna("ok"))
        print("--- state at the last accepted step ---")
        print(push.set_index(["energy", "bc", "field", "dir"])[
            ["J_min", "Je_min", "Je_max", "max_FeI", "x_max", "x0_max", "n_steps", "newton_iters", "wall"]].round(3))
    for name in SWEEPS:
        sw = df[df.experiment == f"sweep_{name}"]
        if not len(sw):
            continue
        print(f"\n=== Sweep {name} (step field, all clamped): attained theta ===")
        print(sw.pivot_table(index=name, columns=["energy", "dir"], values="push", aggfunc="first").round(3))
        print(sw.pivot_table(index=name, columns=["energy", "dir"], values="cause", aggfunc="first").fillna("ok"))
        print(sw.pivot_table(index=name, columns=["energy", "dir"], values="J_min", aggfunc="first").round(3).add_prefix("J_min "))
    ca = df[df.experiment == "ca"]
    if len(ca):
        print("\n=== Ca mode (C from 0 to 100, theta in [0.874, 1.199]) ===")
        print(ca[["energy", "bc", "field", "success", "theta_min", "theta_max", "n_steps", "newton_iters", "cause"]])
    print("\n=== Failure causes overall ===")
    print(df.groupby(["energy", "cause"], dropna=False).size())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["sanity", "push", "ca", "sweeps"],
                    help="sanity push ca push_spring ca_spring sweeps sweep_nu sweep_w sweep_maxh sweep_order sweep_L_over_lp0 summary")
    ap.add_argument("--out", default="results/active_strain")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    exps = list(args.experiments)
    if "sweeps" in exps:
        exps = [e for e in exps if e != "sweeps"] + [f"sweep_{n}" for n in SWEEPS]

    ngsolve.SetNumThreads(args.threads)
    with TaskManager():
        for exp in exps:
            if exp == "sanity":
                sanity()
                continue
            if exp == "summary":
                continue
            if exp == "push":
                os.makedirs(os.path.join(args.out, "vtk"), exist_ok=True)  # VTKOutput does not create it
                rows = experiment_push(os.path.join(args.out, "vtk"))
            elif exp == "ca":
                rows = experiment_ca()
            elif exp == "push_spring":
                os.makedirs(os.path.join(args.out, "vtk"), exist_ok=True)
                rows = experiment_push(os.path.join(args.out, "vtk"), bcs=("spring",))
            elif exp == "ca_spring":
                rows = experiment_ca(bcs=("spring",))
            elif exp.startswith("sweep_"):
                rows = experiment_sweep(exp[len("sweep_"):])
            else:
                raise ValueError(exp)
            path = os.path.join(args.out, f"{exp}.csv")
            with open(path, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=COLUMNS)
                wr.writeheader()
                for row in rows:
                    wr.writerow({k: row.get(k, "") for k in COLUMNS})
                    fh.flush()
    if "summary" in exps or len(exps) > 1:
        summary(sorted(glob.glob(os.path.join(args.out, "*.csv"))))


if __name__ == "__main__":
    main()
