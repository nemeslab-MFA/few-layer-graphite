#!/usr/bin/env python3
"""
The tip-induced part of the band bending in the wall model: one wall calculation per
tip position.

For a tip at x_tip the layer potentials along the line are the far-field Poisson
potentials of the hybrid (V_l(x) = U_far,l − μ0 of the stack at s(x)) plus the bump
ΔU_l exp(−(x − x_tip)²/2w²), with ΔU_l = U0,l(V) − U_far,l the tip-induced bending of
the stack at s(x_tip) at the bias V where the flat-band peak is measured at the wall,
and w = sqrt(2 R d) the size of the gated spot.  The k·p wall model is solved with these
potentials and the top-layer LDOS is read at x_tip.  Results are cached per position.

    python stacking_wall_tibb_bump.py --tag 514 --positions -10 -6 -3 0 4 [--sm4 FILE]

Writes examples/stacking_wall_bump_TAG.json (spectra per position) and, with --plot,
examples/stacking_wall_bump.png for every tag that has results.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

import stacking_wall as sw

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")


def load_series(tag):
    main = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")))
    best = main["scan"][f"{main['best_cpot']:+.3f}"]
    recs = {0.0: best, 1.0: main["runs"]["sf"]}
    for f in [os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")] + sorted(
            glob.glob(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}_s*.json"))):
        for k, r in json.load(open(f)).get("runs", {}).items():
            if k.startswith("wall_s") and r.get("s") is not None:
                recs[float(r["s"])] = r
    out = {}
    for s, r in sorted(recs.items()):
        sweep = r["sweep"]
        out[s] = dict(bias=np.array(sweep["bias"]), U0=np.array(sweep["U0"]),
                      U_far=np.array(sweep["U_far"]).mean(axis=0), mu0=sweep["mu0"])
    fit = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_{tag}.json")))["fit"]
    return out, main["settings"], {k: np.array(v) for k, v in main["measured"].items()}, fit


def interp_in_s(rec, values):
    """values(s_record) -> (6,) ; returns f(s) interpolating linearly between the stacks."""
    s_vals = np.array(sorted(rec))
    table = np.array([values(rec[s]) for s in s_vals])            # (n_s, 6)
    return lambda s: np.array([np.interp(np.asarray(s, dtype=float), s_vals, table[:, l]) for l in range(6)])


def peak(E, y, centre, half=0.09):
    sel = (E >= centre - half) & (E <= centre + half)
    return float(E[sel][np.argmax(y[sel])]), float(y[sel].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--positions", type=float, nargs="+", default=[-10.0, -6.0, -3.0, 0.0, 4.0],
                    help="tip positions relative to the wall centre (nm, positive on the (+5) side)")
    ap.add_argument("--bias", type=float, default=None, help="bias of the bump amplitude (V); default: measured wall peak")
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--separation", type=float, default=0.7)
    ap.add_argument("--length", type=float, default=300.0, help="periodic cell (nm)")
    ap.add_argument("--dky", type=float, default=0.002)
    ap.add_argument("--p-cut", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.006)
    ap.add_argument("--e-max", type=float, default=0.35)
    ap.add_argument("--no-bump", action="store_true", help="also compute the reference without the bump at each position")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out_path = os.path.join(EXAMPLES, f"stacking_wall_bump_{args.tag}.json")
    state = json.load(open(out_path)) if os.path.exists(out_path) else {}
    rec, settings, meas, fit = load_series(args.tag)
    E_fb = settings["E_fb_measured"]
    V_bump = args.bias if args.bias is not None else peak(meas["bias"], meas["wall"], E_fb)[0]
    V_of_s = interp_in_s(rec, lambda r: r["U_far"] - r["mu0"])
    dU_of_s = interp_in_s(rec, lambda r: np.array([np.interp(V_bump, r["bias"], r["U0"][:, l]) for l in range(6)]) - r["U_far"])
    L, ell = 10.0 * args.length, 10.0 * fit["ell"]
    w = np.sqrt(2.0 * 10.0 * args.radius * 10.0 * args.separation)       # A
    energy = np.arange(-args.e_max, args.e_max + 1e-9, 0.0025)
    state["settings"] = dict(bias=V_bump, radius=args.radius, separation=args.separation, spot=w / 10.0,
                             length=args.length, dky=args.dky, p_cut=args.p_cut, sigma=args.sigma, E_fb=E_fb,
                             x0=fit["x0"], ell=fit["ell"])
    print(f"[{args.tag}] bump amplitude at V = {V_bump:+.3f} V, spot w = {w / 10:.1f} nm, cell {args.length:.0f} nm")
    runs = state.setdefault("runs", {})
    for xt in args.positions:
        for variant in (["bump"] + (["none"] if args.no_bump else [])):
            key = f"{xt:+.1f}_{variant}"
            if key in runs:
                continue
            x_tip = 10.0 * xt
            s_t = float(sw.wall_profile(x_tip, ell, L))
            bump = dU_of_s(s_t) if variant == "bump" else np.zeros(6)
            profile = lambda x, ell=ell: sw.wall_profile(x, ell, L)                                    # noqa: E731
            lpp = lambda x, bump=bump, x_tip=x_tip: (V_of_s(profile(x))                                 # noqa: E731
                                                     + bump[:, None] * np.exp(-0.5 * ((np.asarray(x, dtype=float) - x_tip) / w) ** 2)[None, :])
            t0 = time.time()
            res = sw.ldos_wall(wall_length=ell, length=L, p_cut=args.p_cut, ky_max=args.p_cut, dky=args.dky,
                               sigma=args.sigma, energy=energy, x_out=np.array([x_tip]), layer_potential_profile=lpp)
            y = res["ldos"][-1][:, 0]
            pk, h = peak(energy, y, E_fb)
            runs[key] = dict(x=xt, variant=variant, s=s_t, bump=bump.tolist(), energy=energy.tolist(), ldos=y.tolist(),
                             peak=pk, height=h, seconds=time.time() - t0)
            print(f"[{args.tag}] tip at {xt:+5.1f} nm (s = {s_t:.2f}), {variant:4s}: top-layer bump {1e3 * bump[0]:+6.1f} meV, "
                  f"peak {1e3 * pk:+6.1f} meV, {time.time() - t0:.0f} s", flush=True)
            json.dump(state, open(out_path, "w"))
    if args.plot:
        plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [t for t in ("514", "940") if os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_bump_{t}.json"))]
    fig, axs = plt.subplots(len(tags), 2, figsize=(12, 4 * len(tags)), squeeze=False)
    for row, t in enumerate(tags):
        st = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_bump_{t}.json")))
        s = st["settings"]
        x0, E_fb = s["x0"], s["E_fb"]
        # hybrid without the bump, from the cache, aligned as in the report
        hyb = sw.load_ldos(os.path.join(EXAMPLES, f"stacking_wall_hybrid_{t}.npz"))
        xh, Eh, top = hyb["x"] / 10.0 + x0, hyb["energy"], hyb["ldos"][-1]
        far = xh > xh.max() - 30
        E_far = peak(Eh, top[:, far].mean(axis=1), 0.0, 0.12)[0]
        shift = E_fb - E_far
        runs = sorted((r for r in st["runs"].values() if r["variant"] == "bump"), key=lambda r: r["x"])
        ax = axs[row, 0]
        step = 0.0
        for r in runs:
            E, y = np.array(r["energy"]) + shift, np.array(r["ldos"])
            j = int(np.argmin(np.abs(xh - (x0 + r["x"]))))
            y0 = top[:, j]
            scale = 1.0 / top[:, far].mean(axis=1).max()
            ax.plot(Eh + shift, y0 * scale + step, color="#1f6fb2", lw=1.0)
            ax.plot(E, y * scale + step, color="#d95926", lw=1.2)
            ax.text(0.33, step + 0.05, f"tip at {r['x']:+.0f} nm (s = {r['s']:.2f}), bump {1e3 * r['bump'][0]:+.0f} meV",
                    fontsize=7, ha="right")
            step += 0.6
        ax.set(xlim=(-0.3, 0.35), xlabel="E − E_F (eV)", ylabel="top-layer LDOS under the tip (offset)",
               title=f"{t}: hybrid without (blue) and with (orange) the tip-induced bump, V = {s['bias']:+.3f} V")
        ax = axs[row, 1]
        h_max = max(r["height"] for r in runs)
        alive = [r for r in runs if r["height"] > 0.15 * h_max and abs(r["peak"] - E_fb) < 0.06]   # the peak is still there
        xs = [x0 + r["x"] for r in alive]
        ax.plot(xs, [1e3 * (r["peak"] + shift) for r in alive], "o-", color="#d95926", label="hybrid + bump")
        refs = [st["runs"].get(f"{r['x']:+.1f}_none") for r in alive]
        if all(refs):
            ax.plot(xs, [1e3 * (q["peak"] + shift) for q in refs], "s--", color="#1f6fb2", label="hybrid (same numerics)")
        else:
            ax.plot(xs, [1e3 * (peak(Eh + shift, top[:, int(np.argmin(np.abs(xh - x)))], E_fb)[0]) for x in xs], "s--",
                    color="#1f6fb2", label="hybrid")
        ax.axhline(1e3 * E_fb, color="#605d57", lw=0.6, ls=":")
        ax.axvline(x0, color="#605d57", lw=0.6, ls="--")
        ax.set(xlabel="tip position (nm)", ylabel="flat-band peak (mV)", title="apparent flat-band energy under the tip")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(EXAMPLES, "stacking_wall_bump.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
