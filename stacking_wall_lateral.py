#!/usr/bin/env python3
"""
Non-axisymmetric (3D) tip-induced band bending across the stacking wall, with the lateral
solver of the bandbending repository (branch ``lateral-solver``, ``engine/lateral.py``).

The stack changes along x as s(x) = [1 - tanh(x/ell)]/2 (the fitted wall; (+4-) at x < 0,
(+5) at x > 0; x relative to the wall centre).  Every layer is a rigid-band charge sheet
whose density follows the zero-field layer DOS of the stack at s(x) (``WallTB`` of
stacking_wall_tibb.py for the intermediate stacks), with the fixed charge that makes the
(+5) stack neutral at the measured flat-band position (E_F = -E_fb below its neutrality
point; one energy reference for all stacks).  The tip is the engine's hyperboloid
(R = 20 nm, d = 0.7 nm, CPot = 0) at U_tip = V with V the bias of the measured wall peak;
the outer boundaries carry the tip-free columns of the local stack plus the engine's
axisymmetric tip field of the (+5) stack.

For every tip position the layer potentials U_l(x, y = 0) are stored with the tip-free
two-dimensional solution U0_l(x).  With --kp the k.p wall model is solved with
V_l(x) = U_l(x, 0) - mu0 (the lateral solution inside the box, the columns elsewhere; layer
order reversed for the k.p model, which numbers the layers from the bottom) and
the top-layer LDOS under the tip is read, as in stacking_wall_tibb_bump.py.

    python stacking_wall_lateral.py --tag 514                # potentials, many positions
    python stacking_wall_lateral.py --tag 514 --kp           # k.p spectra at --kp-positions
    python stacking_wall_lateral.py --plot

Writes examples/stacking_wall_lateral_dos.npz (DOS tables, shared),
examples/stacking_wall_lateral_TAG.json and, with --plot, examples/stacking_wall_lateral.png.
"""

import argparse
import json
import math
import os
import time

import numpy as np

import stacking_wall as sw
import stacking_wall_tibb as swt
from stacking_wall_tibb_bump import interp_in_s, load_series, peak

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
S_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
LAYER_Z = -0.335 * np.arange(6)                       # nm, top layer first
EPS_SPACER, EPS_OXIDE, OXIDE = 2.6, 3.9, 90.0         # as in the engine fits (SiO2, 90 nm)
TEMPERATURE, BROADENING = 9.0, 0.01
POSITIONS = [-20.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
KP_POSITIONS = [-10.0, -6.0, -3.0, 0.0, 4.0]


# ---------------------------------------------------------------------------
# DOS tables of the wall stacks
# ---------------------------------------------------------------------------

def dos_tables(path=os.path.join(EXAMPLES, "stacking_wall_lateral_dos.npz")):
    """(s, E, D): zero-field layer DOS of the stack at every s of S_GRID on the uniform
    PROJ_DE grid, D (n_s, nE, 6) in states/(eV cm^2), the stack neutral at E = 0.
    Cached in ``path``."""
    from bandbending.materials.multilayer import PROJ_DE, PROJ_FINE
    if os.path.exists(path):
        f = np.load(path)
        if np.allclose(f["s"], S_GRID):
            return f["s"], f["E"], f["D"]
    p = swt.library.TB_MCELLISTRIM2023
    D_all = []
    for s in S_GRID:
        t0 = time.time()
        m = swt.WallTB(p, s=s, temperature=0.0, broadening=1e-3, h_fine=1e-3, e_window=PROJ_FINE)
        emax = 3.0 * p.gamma0 + p.gamma1 * m.stacking.dimer_norm + 0.05
        n = int(math.ceil(emax / PROJ_DE))
        E = PROJ_DE * np.arange(-n, n + 1)
        _, D = m.ldos(None, E, resolve="layer")
        D_all.append(np.maximum(D, 0.0))
        print(f"DOS table s = {s:.2f}: {E.size} energies, {time.time() - t0:.0f} s", flush=True)
    s_arr, D_all = np.array(S_GRID), np.array(D_all)
    np.savez_compressed(path, s=s_arr, E=E, D=D_all)
    return s_arr, E, D_all


def sheet_tables(mu0):
    from bandbending.engine import lateral as lat
    s, E, D = dos_tables()
    return lat.SheetTables.from_dos(s, [(E, D[i]) for i in range(s.size)], float(mu0), None,
                                    temperature=TEMPERATURE, broadening=BROADENING)


# ---------------------------------------------------------------------------
# the solves
# ---------------------------------------------------------------------------

def tip_field(tables, V, radius, separation):
    """The engine's axisymmetric rigid solution of the (+5) stack at bias V (its far field is
    zero: the stack is neutral in this frame).  Returns (field(rho, z), Result)."""
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, Simulation, Tip, library
    from bandbending.engine import lateral as lat
    tip = Tip(radius=radius, separation=separation, contact_potential=0.0)
    ml = GrapheneMultilayer(swt.SEQ_RG, name="RG", coupling="rigid", doping=tuple(tables.doping), broadening=BROADENING)
    sim = Simulation(Sample(surface=ml, layers=(Layer(OXIDE, library.SIO2, name="SiO2"),), gate=Gate()), tip,
                     temperature=TEMPERATURE, grid="normal")
    res = sim.solve(V, 0.0, fields=True)
    # The engine's own densities differ from the tables by about 1e-4 of the layer charge;
    # the middle layers' small quantum capacitance turns that into a far field of a few meV.
    # Only the tip-induced part is used: the field minus the engine's far-field column.
    field = lat.axisymmetric_field(res)
    r_far = float(np.max(res.r))

    def tip_part(rho, zz):
        rho = np.asarray(rho, dtype=float)
        return field(rho, zz) - field(np.full(rho.shape, r_far), zz)
    return tip_part, res, tip


def solve_position(tables, s_of_x, x_tip, U_tip, tip, field, *, half_fine=15.0, h_lat=0.5, x_ext=100.0, y_ext=80.0,
                   z_top=40.0, h_gap=0.1):
    """One tip position: returns the lateral problem (solved) and the tip-free 2D problem on the
    same x grid."""
    from bandbending.engine import lateral as lat
    x = lat.lateral_axis([x_tip, 0.0], half_fine, h_lat, x_ext)
    y = lat.half_axis(half_fine, h_lat, y_ext)
    z = lat.vertical_axis(LAYER_Z, OXIDE, z_top, h_gap=h_gap)
    bc0 = lat.column_boundary(tables, S_GRID, z, LAYER_Z, EPS_SPACER, EPS_OXIDE, s_of_x)
    bc = lat.column_boundary(tables, S_GRID, z, LAYER_Z, EPS_SPACER, EPS_OXIDE, s_of_x, tip_part=field, x_tip=x_tip)
    ref = lat.LateralPoisson(x, np.array([0.0, 1.0]), z, LAYER_Z, EPS_SPACER, EPS_OXIDE, tables, s_of_x,
                             boundary=bc0, top_neumann=True, y_neumann=True)
    ref.solve()
    prob = lat.LateralPoisson(x, y, z, LAYER_Z, EPS_SPACER, EPS_OXIDE, tables, s_of_x, tip=tip, x_tip=x_tip,
                              U_tip=U_tip, boundary=bc)
    prob.solve()
    return prob, ref


def columns(tables):
    from bandbending.engine import lateral as lat
    z = lat.vertical_axis(LAYER_Z, OXIDE, 40.0, h_gap=0.1)
    k = [int(np.argmin(np.abs(z - zl))) for zl in LAYER_Z]
    return np.array([lat.column_solve(tables, s, z, LAYER_Z, EPS_SPACER, EPS_OXIDE)[k] for s in S_GRID])   # (n_s, 6)


def potential_profile(x_box, U_box, cols, ell, L, mu0):
    """x (A, periodic cell) -> (6, n) on-site energies for the k.p model: the columns of the
    local stack everywhere, replaced by the lateral solution inside the box.  The inputs are
    top layer first (as the engine); the output is in the k.p order, bottom layer first."""
    x_box = np.asarray(x_box, dtype=float)
    s_box = 0.5 * (1.0 - np.tanh(x_box / ell))
    col_box = np.array([np.interp(s_box, S_GRID, cols[:, l]) for l in range(6)])            # (6, nx)
    dU = np.asarray(U_box) - col_box

    def f(xa):
        xa = np.asarray(xa, dtype=float)
        s = sw.wall_profile(xa, 10.0 * ell, 10.0 * L)
        xn = (np.mod(xa + 5.0 * L, 10.0 * L) - 5.0 * L) / 10.0                            # nm, main wall at 0
        V = np.array([np.interp(s, S_GRID, cols[:, l]) + np.interp(xn, x_box, dU[l], left=0.0, right=0.0)
                      for l in range(6)])
        return (V - mu0)[::-1]
    return f


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--positions", type=float, nargs="+", default=POSITIONS,
                    help="tip positions relative to the wall centre (nm, positive on the (+5) side)")
    ap.add_argument("--kp", action="store_true", help="k.p spectra under the tip at --kp-positions")
    ap.add_argument("--kp-positions", type=float, nargs="+", default=KP_POSITIONS)
    ap.add_argument("--bias", type=float, default=None, help="tip bias (V); default: measured wall peak")
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--separation", type=float, default=0.7)
    ap.add_argument("--length", type=float, default=300.0, help="k.p periodic cell (nm)")
    ap.add_argument("--dky", type=float, default=0.002)
    ap.add_argument("--p-cut", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.006)
    ap.add_argument("--e-max", type=float, default=0.35)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.tag is not None:
        run(args)
    if args.plot:
        plot()


def run(args):
    out_path = os.path.join(EXAMPLES, f"stacking_wall_lateral_{args.tag}.json")
    state = json.load(open(out_path)) if os.path.exists(out_path) else {}
    rec, settings, meas, fit = load_series(args.tag)
    E_fb = settings["E_fb_measured"]
    mu0 = float(rec[0.0]["mu0"])
    V = args.bias if args.bias is not None else peak(meas["bias"], meas["wall"], E_fb)[0]
    ell = float(fit["ell"])
    s_of_x = lambda xx: 0.5 * (1.0 - np.tanh(np.asarray(xx, dtype=float) / ell))            # noqa: E731
    dU_local = interp_in_s(rec, lambda r: np.array([np.interp(V, r["bias"], r["U0"][:, l]) for l in range(6)]) - r["U_far"])
    t0 = time.time()
    tables = sheet_tables(mu0)
    cols = columns(tables)
    field, res, tip = tip_field(tables, V, args.radius, args.separation)
    apex_rg = [float(res.sheets[f"RG.L{l}"].U[0] - res.sheets[f"RG.L{l}"].U_far) for l in range(1, 7)]
    far_rg = [float(res.sheets[f"RG.L{l}"].U_far) for l in range(1, 7)]
    print(f"[{args.tag}] V = {V:+.3f} V, mu0 = {1e3 * mu0:+.1f} meV, ell = {ell:.1f} nm; tables, columns and the engine "
          f"field in {time.time() - t0:.0f} s; engine apex bending of (+5): {np.round(1e3 * np.array(apex_rg), 2)} meV "
          f"(its far field {np.round(1e3 * np.array(far_rg), 2)} meV)", flush=True)
    state["settings"] = dict(bias=V, radius=args.radius, separation=args.separation, E_fb=E_fb, mu0=mu0, x0=fit["x0"], ell=ell,
                             s_grid=list(S_GRID), doping=tables.doping.tolist(), layer_z=LAYER_Z.tolist(),
                             eps_spacer=EPS_SPACER, eps_oxide=EPS_OXIDE, oxide=OXIDE, temperature=TEMPERATURE,
                             broadening=BROADENING, apex_rg_engine=apex_rg, far_rg_engine=far_rg, length=args.length, dky=args.dky,
                             p_cut=args.p_cut, sigma=args.sigma)
    state["columns"] = cols.tolist()
    runs = state.setdefault("runs", {})
    for xt in (args.kp_positions if args.kp else args.positions):
        key = f"{xt:+.1f}"
        if key in runs:
            continue
        t0 = time.time()
        prob, ref = solve_position(tables, s_of_x, xt, V, tip, field)
        x = prob.x
        U, U0 = prob.layer_potentials(0), ref.layer_potentials(0)
        i0 = int(np.argmin(np.abs(x - xt)))
        runs[key] = dict(x_tip=xt, s=float(s_of_x(xt)), x=x.tolist(), U=U.tolist(), U0=U0.tolist(),
                         apex=U[:, i0].tolist(), apex0=U0[:, i0].tolist(), dU_apex=(U[:, i0] - U0[:, i0]).tolist(),
                         dU_local=dU_local(s_of_x(xt)).tolist(), seconds=time.time() - t0,
                         iterations=prob.info["iterations"], n_free=prob.info["n_free"], converged=prob.info["converged"],
                         shape=list(prob.shape))
        d, dl = runs[key]["dU_apex"], runs[key]["dU_local"]
        print(f"[{args.tag}] tip at {xt:+5.1f} nm (s = {s_of_x(xt):.2f}): top-layer bending {1e3 * d[0]:+6.2f} meV "
              f"(local approximation {1e3 * dl[0]:+6.2f}), layer 2 {1e3 * d[1]:+6.2f} ({1e3 * dl[1]:+6.2f}); "
              f"{prob.info['iterations']} it, {prob.n_free} nodes, {time.time() - t0:.0f} s", flush=True)
        json.dump(state, open(out_path, "w"))
    # the local approximation with the same solver: the uniform stack at s(x_tip) under the tip
    for key, r in runs.items():
        if "dU_uniform" in r:
            continue
        t0 = time.time()
        s_t = float(r["s"])
        prob, ref = solve_position(tables, lambda xx, s_t=s_t: np.full(np.shape(xx), s_t), r["x_tip"], V, tip, field)
        i0 = int(np.argmin(np.abs(prob.x - r["x_tip"])))
        U, U0 = prob.layer_potentials(0), ref.layer_potentials(0)
        r["dU_uniform"] = (U[:, i0] - U0[:, i0]).tolist()
        r["apex_uniform"] = U[:, i0].tolist()
        print(f"[{args.tag}] uniform stack at s = {s_t:.2f}: top-layer bending {1e3 * r['dU_uniform'][0]:+6.2f} meV "
              f"(lateral {1e3 * r['dU_apex'][0]:+6.2f}, engine sweeps {1e3 * r['dU_local'][0]:+6.2f}); {time.time() - t0:.0f} s",
              flush=True)
        json.dump(state, open(out_path, "w"))
    if args.kp:
        kp = state.setdefault("kp", {})
        energy = np.arange(-args.e_max, args.e_max + 1e-9, 0.0025)
        for xt in args.kp_positions:
            r = runs[f"{xt:+.1f}"]
            for variant in ("tip", "none"):
                key = f"{xt:+.1f}_{variant}"
                if key in kp:
                    continue
                t0 = time.time()
                lpp = potential_profile(r["x"], r["U"] if variant == "tip" else r["U0"], cols, ell, args.length, mu0)
                out = sw.ldos_wall(wall_length=10.0 * ell, length=10.0 * args.length, p_cut=args.p_cut, ky_max=args.p_cut,
                                   dky=args.dky, sigma=args.sigma, energy=energy, x_out=np.array([10.0 * xt]),
                                   layer_potential_profile=lpp)
                y = out["ldos"][-1][:, 0]
                pk, h = peak(energy, y, E_fb)
                kp[key] = dict(x_tip=xt, variant=variant, energy=energy.tolist(), ldos=y.tolist(), peak=pk, height=h,
                               seconds=time.time() - t0)
                print(f"[{args.tag}] k.p, tip at {xt:+5.1f} nm, {variant:4s}: peak {1e3 * pk:+6.1f} meV, {time.time() - t0:.0f} s",
                      flush=True)
                json.dump(state, open(out_path, "w"))


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [t for t in ("514", "940") if os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_lateral_{t}.json"))]
    fig, axs = plt.subplots(len(tags), 3, figsize=(15, 4.2 * len(tags)), squeeze=False)
    for row, t in enumerate(tags):
        st = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_lateral_{t}.json")))
        s, x0, E_fb = st["settings"], st["settings"]["x0"], st["settings"]["E_fb"]
        runs = sorted(st["runs"].values(), key=lambda r: r["x_tip"])
        # 1. layer potentials along the line
        ax = axs[row, 0]
        r0 = min(runs, key=lambda r: abs(r["x_tip"]))
        x, U0 = np.array(r0["x"]), np.array(r0["U0"])
        for l in range(6):
            ax.plot(x + x0, 1e3 * U0[l], color="#1f6fb2", lw=1.6 if l == 0 else 0.7, alpha=1.0 if l == 0 else 0.5,
                    label="tip-free, layer 1" if l == 0 else ("tip-free, layers 2-6" if l == 1 else None))
        shown = [r for r in runs if r["x_tip"] in KP_POSITIONS]
        cmap = plt.get_cmap("Oranges")
        for i, r in enumerate(shown):
            ax.plot(np.array(r["x"]) + x0, 1e3 * np.array(r["U"])[0], color=cmap(0.4 + 0.6 * i / max(len(shown) - 1, 1)), lw=1.2,
                    label=f"tip at {r['x_tip']:+.0f} nm" if i in (0, len(shown) - 1) else None)
        ax.axvline(x0, color="#605d57", lw=0.6, ls="--")
        ax.set(xlim=(x0 - 40, x0 + 40), xlabel="x (nm)", ylabel="layer potential U (meV)",
               title=f"{t}: layer potentials along the line, V = {s['bias']:+.3f} V")
        ax.legend(fontsize=7, frameon=False)
        # 2. apex bending against tip position: lateral vs local
        ax = axs[row, 1]
        xs = np.array([r["x_tip"] for r in runs]) + x0
        for l, c in ((0, "#d95926"), (1, "#8c6d31")):
            ax.plot(xs, [1e3 * r["dU_apex"][l] for r in runs], "o-", color=c, ms=3, label=f"layer {l + 1}, lateral solve")
            if all("dU_uniform" in r for r in runs):
                ax.plot(xs, [1e3 * r["dU_uniform"][l] for r in runs], "s--", color=c, ms=3, alpha=0.7,
                        label=f"layer {l + 1}, uniform stack at s(x_tip), same solver")
            ax.plot(xs, [1e3 * r["dU_local"][l] for r in runs], "^:", color=c, ms=3, alpha=0.5, label=f"layer {l + 1}, engine sweeps (local approximation)")
        ax.axvline(x0, color="#605d57", lw=0.6, ls="--")
        ax.set(xlabel="tip position (nm)", ylabel="tip-induced bending under the apex (meV)",
               title="U_l(x_tip) with the tip minus without")
        ax.legend(fontsize=7, frameon=False)
        # 3. flat-band peak under the tip from the k.p model
        ax = axs[row, 2]
        kp = st.get("kp", {})
        bump_path = os.path.join(EXAMPLES, f"stacking_wall_bump_{t}.json")
        bump = json.load(open(bump_path))["runs"] if os.path.exists(bump_path) else {}
        series = [("tip", "lateral solve, with the tip", "#d95926", "o-"), ("none", "lateral solve, tip-free", "#1f6fb2", "s--")]
        for variant, label, color, style in series:
            pts = sorted((r for r in kp.values() if r["variant"] == variant), key=lambda r: r["x_tip"])
            h_max = max((r["height"] for r in pts), default=0.0)
            alive = [r for r in pts if r["height"] > 0.15 * h_max and abs(r["peak"] - E_fb) < 0.06]
            if alive:
                ax.plot([r["x_tip"] + x0 for r in alive], [1e3 * r["peak"] for r in alive], style, color=color, ms=4, label=label)
        pts = sorted((r for r in bump.values() if r["variant"] == "bump"), key=lambda r: r["x"])
        h_max = max((r["height"] for r in pts), default=0.0)
        alive = [r for r in pts if r["height"] > 0.15 * h_max and abs(r["peak"] - E_fb) < 0.06]
        if alive:
            ax.plot([r["x"] + x0 for r in alive], [1e3 * r["peak"] for r in alive], "^:", color="#605d57", ms=4,
                    label="Gaussian bump (hybrid + bump)")
        ax.axhline(1e3 * E_fb, color="#605d57", lw=0.6, ls=":")
        ax.axvline(x0, color="#605d57", lw=0.6, ls="--")
        ax.set(xlabel="tip position (nm)", ylabel="flat-band peak (mV, k.p energy)", title="apparent flat-band energy under the tip")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(EXAMPLES, "stacking_wall_lateral.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
