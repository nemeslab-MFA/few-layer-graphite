#!/usr/bin/env python3
"""
Charge redistribution against tip-induced band bending at the (+4-)|(+5) wall, and the effect of a
wider flat band.

For every stack of the local approximation (stacking_wall_tibb.py: (+5), the wall stacks at s and
(+4-)) the simulated top-layer spectrum is computed twice from the same self-consistent sweep: with the
tip (the layer potentials under the apex, U0(V), as in the report) and without it (the far-field
potentials U_far of the same stack at every bias, so that only the charge redistribution of the stack
itself moves the bands).  The flat-band peak of each, measured from that of the (+5) stack, gives the
bend with and without the tip; their difference is the tip-induced part.

The flat band of the gamma0/gamma1 stacks is narrower than the measured one.  ``calibrate`` finds the
Gaussian width sigma (the engine's ``broadening``: std of the occupation and of the LDOS) at which the
simulated (+5) peak is as wide as the measured one, by re-broadening the spectrum of the existing
sweep; ``wide`` then solves the (+5) stack and the wall stack at s = 0.75 again with that sigma in the
occupation and the LDOS (contact potential 0, as fitted), so that the screening sees the wider band.

    python stacking_wall_separation.py calibrate
    python stacking_wall_separation.py separate                 # sigma = 10 meV, the existing series
    python stacking_wall_separation.py wide --tag 940 --sigma 0.013
    python stacking_wall_separation.py plot

Writes examples/stacking_wall_separation.json (and _wide_TAG.json) and, with plot,
examples/stacking_wall_separation.png.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
sys.path.insert(0, HERE)
TAGS = ("514", "940")
OUT = os.path.join(EXAMPLES, "stacking_wall_separation.json")


def load_stacks(tag):
    """s -> (sweep dict, run record) of every stack of the local approximation, s = 0 the fitted (+5) stack."""
    main = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")))
    recs = {0.0: main["scan"][f"{main['best_cpot']:+.3f}"], 1.0: main["runs"]["sf"]}
    for f in [os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")] + sorted(
            glob.glob(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}_s*.json"))):
        for k, r in json.load(open(f)).get("runs", {}).items():
            if k.startswith("wall_s") and r.get("s") is not None:
                recs[float(r["s"])] = r
    return main, dict(sorted(recs.items()))


def tb_of(s, broadening, temperature, n_phi=6, wall_kw=None):
    """The tight-binding object of the stack at s (s = 0 (+5), s = 1 (+4-), otherwise the wall)."""
    import stacking_wall_tibb as swt
    from bandbending import Simulation, Tip
    tip = Tip(radius=20.0, separation=0.7, contact_potential=0.0)
    if s == 0.0 or s == 1.0:
        seq, name = (swt.SEQ_RG, "RG") if s == 0.0 else (swt.SEQ_SF, "SF")
        sample = swt.make_sample(seq, name, fermi_level=0.0, broadening=broadening)
        sim = Simulation(sample, tip, temperature=temperature, grid="draft")
    else:
        sample = swt.make_sample(swt.SEQ_RG, swt.WALL_NAME, fermi_level=0.0, broadening=broadening)
        sim = swt.WallSimulation(sample, tip, temperature, wall_s=s, n_phi=n_phi, grid="draft",
                                 wall_kw=wall_kw or {"h_fine": 0.012, "h_coarse": 0.3})
    return sim._multilayer_tb(sample.multilayers[0].spec)


def spectra(tb, sweep, fine, *, broadening, settings, tip=True):
    """Top-layer dI/dV on ``fine`` from a stored sweep: with the tip (U0) or tip-free (U_far)."""
    from bandbending.results.spectroscopy import interpolate_potentials, tunneling_spectrum
    V = np.asarray(sweep["bias"])
    U = np.asarray(sweep["U0"] if tip else sweep["U_far"])
    ok = np.all(np.isfinite(U), axis=1)
    if tip:
        Ui, dUdV = interpolate_potentials(V[ok], U[ok], fine)
    else:
        Uf = np.nanmean(U[ok], axis=0)                       # far field: the same at every bias
        Ui, dUdV = np.repeat(Uf[None, :], fine.size, axis=0), np.zeros((fine.size, Uf.size))
    sp = tunneling_spectrum(tb, sweep["mu0"], fine, Ui, dUdV, broadening=broadening,
                            modulation=settings["modulation"], barrier=settings["barrier"],
                            separation=settings["separation"], temperature=settings["temperature"])
    return np.asarray(sp.didv)


def peak(E, y, centre, half=0.09):
    E, y = np.asarray(E), np.asarray(y)
    sel = (E >= centre - half) & (E <= centre + half)
    i = int(np.argmax(y[sel]))
    return float(E[sel][i]), float(y[sel][i])


def centroid(E, y, pk, half=0.03):
    E, y = np.asarray(E), np.asarray(y)
    sel = (E >= pk - half) & (E <= pk + half)
    yb = y[sel] - y[sel].min()
    return float(np.sum(E[sel] * yb) / np.sum(yb))


def fwhm(E, y, centre, half=0.07):
    E, y = np.asarray(E), np.asarray(y)
    sel = (E >= centre - half) & (E <= centre + half)
    v, w = E[sel], y[sel]
    i = int(np.argmax(w))
    base = min(w[:i + 1].min(), w[i:].min())
    h = base + 0.5 * (w[i] - base)
    lo, hi = i, i
    while lo > 0 and w[lo] > h:
        lo -= 1
    while hi < len(w) - 1 and w[hi] > h:
        hi += 1
    # linear interpolation of the half-height crossings
    xl = v[lo] + (h - w[lo]) * (v[lo + 1] - v[lo]) / (w[lo + 1] - w[lo]) if w[lo + 1] != w[lo] else v[lo]
    xr = v[hi - 1] + (h - w[hi - 1]) * (v[hi] - v[hi - 1]) / (w[hi] - w[hi - 1]) if w[hi] != w[hi - 1] else v[hi]
    return float(xr - xl)


def calibrate(args):
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cal = out.setdefault("calibration", {})
    for tag in TAGS:
        main, recs = load_stacks(tag)
        s = main["settings"]
        fine = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 4)
        E_fb = s["E_fb_measured"]
        meas = fwhm(main["measured"]["bias"], main["measured"]["rg"], E_fb)
        tb = tb_of(0.0, s["broadening"], s["temperature"])
        rows = []
        for sig in args.sigmas:
            y = spectra(tb, recs[0.0]["sweep"], fine, broadening=sig, settings=s)
            rows.append(dict(sigma=sig, fwhm=fwhm(fine, y, E_fb)))
            print(f"[{tag}] sigma {1e3 * sig:5.1f} meV: simulated (+5) FWHM {1e3 * rows[-1]['fwhm']:5.1f} mV "
                  f"(measured {1e3 * meas:.1f} mV)", flush=True)
        f = np.array([r["fwhm"] for r in rows])
        sg = np.array([r["sigma"] for r in rows])
        match = float(np.interp(meas, f, sg)) if f.min() <= meas <= f.max() else None
        cal[tag] = dict(measured_fwhm=meas, rows=rows, sigma_match=match)
        print(f"[{tag}] sigma for the measured width: {('%.1f meV' % (1e3 * match)) if match else 'out of range'}")
    json.dump(out, open(OUT, "w"))


def separate_records(tag, recs, settings, broadening, fine, stored=None):
    """Per stack: flat-band peak and centroid with and without the tip (eV).  ``stored``: s -> (bias, didv) of
    with-tip spectra already computed with the same settings (the with-tip spectrum needs three LDOS per bias
    point, the tip-free one a single LDOS)."""
    E_fb = settings["E_fb_measured"]
    rows = []
    for s_val, r in recs.items():
        tb = tb_of(s_val, broadening, settings["temperature"])
        if stored and s_val in stored:
            y_tip = np.interp(fine, *stored[s_val])
        else:
            y_tip = spectra(tb, r["sweep"], fine, broadening=broadening, settings=settings, tip=True)
        y_free = spectra(tb, r["sweep"], fine, broadening=broadening, settings=settings, tip=False)
        row = dict(s=s_val)
        for lab, y in (("tip", y_tip), ("free", y_free)):
            pk, h = peak(fine, y, E_fb)
            row[lab] = dict(peak=pk, height=h, centroid=centroid(fine, y, pk))
        rows.append(row)
        print(f"[{tag}] s = {s_val:.2f}: peak with tip {1e3 * row['tip']['peak']:+6.1f} mV, tip-free "
              f"{1e3 * row['free']['peak']:+6.1f} mV (centroids {1e3 * row['tip']['centroid']:+6.1f}, "
              f"{1e3 * row['free']['centroid']:+6.1f})", flush=True)
    return rows


def separate(args):
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    sep = out.setdefault("separation", {})
    for tag in TAGS:
        main, recs = load_stacks(tag)
        s = main["settings"]
        fine = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 4)
        fb = np.asarray(main["fine_bias"])
        stored = {sv: (fb, np.asarray(r["didv"])) for sv, r in recs.items()}
        sep[tag] = dict(broadening=s["broadening"], E_fb=s["E_fb_measured"], s_wall=s["s"],
                        measured_wall=float(peak(main["measured"]["bias"], main["measured"]["wall"], s["E_fb_measured"])[0]),
                        rows=separate_records(tag, recs, s, s["broadening"], fine, stored=stored))
        json.dump(out, open(OUT, "w"))


def wide(args):
    """(+5) and the wall stack at s = 0.75 with a wider Gaussian, contact potential 0."""
    import stacking_wall_tibb as swt
    from bandbending import Tip
    tag, sig = args.tag, args.sigma
    main, _ = load_stacks(tag)
    s = dict(main["settings"])
    s["broadening"] = sig
    E_fb = s["E_fb_measured"]
    path = os.path.join(EXAMPLES, f"stacking_wall_separation_wide_{tag}.json")
    state = json.load(open(path)) if os.path.exists(path) else {}
    state["settings"] = dict(s, sigma=sig, cpot=0.0, bias_step=args.bias_step, bias_max=args.bias_max, s_values=args.s)
    ckdir = os.path.join(HERE, "stacking_wall_epsfit", "wideband")
    os.makedirs(ckdir, exist_ok=True)
    biases = np.round(np.arange(-args.bias_max, args.bias_max + 1e-9, args.bias_step), 4) + 0.0
    fine = np.round(np.arange(-args.bias_max, args.bias_max + 1e-9, 0.0025), 4)
    tip = Tip(radius=s["radius"], separation=s["separation"], contact_potential=0.0)
    runs = state.setdefault("runs", {})
    t0 = time.time()
    if "0.00" not in runs:
        rg = swt.make_sample(swt.SEQ_RG, "RG", fermi_level=-E_fb, broadening=sig)
        m, dt, fails = swt.run_sweep(rg, tip, biases, temperature=s["temperature"], grid=s["grid"],
                                     checkpoint=os.path.join(ckdir, f"{tag}_sig{1e3 * sig:.0f}_rg.ckpt.json"))
        runs["0.00"] = dict(s=0.0, seconds=dt, failures=len(fails), sweep=swt.sweep_arrays(m), doping=np.asarray(m.doping).tolist())
        json.dump(state, open(path, "w"))
        print(f"[{tag}] sigma {1e3 * sig:.1f} meV: (+5) sweep {dt:.0f} s, {len(fails)} failures", flush=True)
    doping = runs["0.00"]["doping"]
    for sv in args.s:
        key = f"{sv:.2f}"
        if key in runs:
            continue
        sample = swt.make_sample(swt.SEQ_RG, swt.WALL_NAME, doping=doping, broadening=sig)
        m, dt, fails = swt.run_sweep(sample, tip, biases, temperature=s["temperature"], wall_s=sv, grid=s["wall_grid"],
                                     n_phi=s["n_phi"], wall_kw={"h_fine": s["wall_hfine"], "h_coarse": s["wall_hcoarse"]},
                                     checkpoint=os.path.join(ckdir, f"{tag}_sig{1e3 * sig:.0f}_s{sv:.2f}.ckpt.json"))
        runs[key] = dict(s=sv, seconds=dt, failures=len(fails), sweep=swt.sweep_arrays(m))
        json.dump(state, open(path, "w"))
        print(f"[{tag}] sigma {1e3 * sig:.1f} meV: wall s = {sv:.2f} sweep {dt:.0f} s, {len(fails)} failures", flush=True)
    recs = {float(k): {"sweep": v["sweep"]} for k, v in sorted(runs.items(), key=lambda kv: float(kv[0]))}
    near = np.round(np.arange(E_fb - 0.13, E_fb + 0.13 + 1e-9, 0.0025), 4)     # the flat band and the wall peak
    state["rows"] = separate_records(tag, recs, s, sig, near)
    state["fwhm_rg"] = None
    tb = tb_of(0.0, sig, s["temperature"])
    y = spectra(tb, runs["0.00"]["sweep"], near, broadening=sig, settings=s)
    state["fwhm_rg"] = fwhm(near, y, E_fb)
    state["seconds"] = time.time() - t0
    json.dump(state, open(path, "w"))
    print(f"[{tag}] wrote {path}; (+5) FWHM {1e3 * state['fwhm_rg']:.1f} mV")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    out = json.load(open(OUT))
    sep = out["separation"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, tag in zip(axs, TAGS):
        d = sep[tag]
        rows = d["rows"]
        h0 = {lab: rows[0][lab]["height"] for lab in ("tip", "free")}
        for lab, col, name in (("tip", ORANGE, "with the tip"), ("free", BLUE, "without the tip (charge redistribution)")):
            pts = [(r["s"], r[lab]["centroid"] - rows[0][lab]["centroid"]) for r in rows if r[lab]["height"] > 0.15 * h0[lab]]
            ax.plot([p[0] for p in pts], [1e3 * p[1] for p in pts], "o-", color=col, lw=2, ms=5, mec="white", mew=1.0,
                    label=name)
        wpath = os.path.join(EXAMPLES, f"stacking_wall_separation_wide_{tag}.json")
        if os.path.exists(wpath):
            w = json.load(open(wpath))
            if w.get("rows"):
                wr = w["rows"]
                for lab, mk, col in (("tip", "D", ORANGE), ("free", "s", BLUE)):
                    xs = [r["s"] for r in wr[1:]]
                    ax.plot(xs, [1e3 * (r[lab]["centroid"] - wr[0][lab]["centroid"]) for r in wr[1:]], mk, ms=8,
                            mfc="none", mec=col, mew=1.6,
                            label=f"{'with' if lab == 'tip' else 'without'} the tip, σ = {1e3 * w['settings']['sigma']:.0f} meV")
        meas = d["measured_wall"] - d["E_fb"]
        ax.plot([d["s_wall"]], [1e3 * meas], "*", ms=13, color=INK, label=f"measured at the wall spectrum ({1e3 * meas:+.0f} mV)")
        ax.axhline(0.0, color=GRID, lw=0.8)
        ax.set_xlabel("s, (+⁴−) weight of the stack")
        ax.set_ylabel("flat-band peak − (+⁵) peak (mV)")
        ax.set_title(f"line {tag}: bend in the local approximation, σ = {1e3 * d['broadening']:.0f} meV (lines)",
                     fontsize=9.5, loc="left", color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left" if meas < 0 else "upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path = os.path.join(EXAMPLES, "stacking_wall_separation.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "separate", "wide", "plot"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.010, 0.012, 0.014, 0.016, 0.018, 0.020])
    ap.add_argument("--s", type=float, nargs="+", default=[0.75])
    ap.add_argument("--bias-step", type=float, default=0.025)
    ap.add_argument("--bias-max", type=float, default=0.2)
    args = ap.parse_args()
    {"calibrate": calibrate, "separate": separate, "wide": wide, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    main()
