#!/usr/bin/env python3
"""
Line spectrum 940 with the polarized stack, in the local approximation.

The best polarized stack of the (+5) refit (stacking_wall_epsfit/, examples/stacking_wall_epsfit_940.json):
interlayer permittivity eps, starting doping Delta E_F and the fixed SiO2 charge n_sub in the top
2 nm of the 90 nm oxide, with the tip that leaves the (+5) bands flat at zero bias.  The wall
stack (stacking_wall_tibb.WallTB) is solved at several (+4-) weights s with the same fixed
charge per layer (that of the doped (+5) stack), the same n_sub and the same tip; s = 0 is the
(+5) stack, s = 1 the (+4-) stack.  The top-layer dI/dV of every stack is placed along the
fitted profile s(x) of line 940 next to the measurement.

Parameter sets (--set): eps15, the best polarized stack of the eps x Delta E_F scan (default); eps26_peaks and
eps26_fb, the eps = 2.6 refit (stacking_wall_epsfit/fixeps.py, examples/stacking_wall_eps26_940.json): its best
cell by the peak positions and its best cell with the flat band within 10 mV.

    python stacking_wall_polar_series.py run WORKER [--set S]           # 4 workers share a queue (lock files)
    python stacking_wall_polar_series.py queue WORKER --sets S1 S2      # wall stacks split in bias halves, both sets
    python stacking_wall_polar_series.py collect --sm4 PATH [--set S]   # merge, maps, figure

Writes examples/stacking_wall_polar[_SET]_940.json, examples/stacking_wall_polar[_SET]_map_940.png/.npz and
examples/stacking_wall_polar[_SET]_spectra_940.png (no suffix for eps15).
"""

import argparse
import glob
import json
import os
import time

import numpy as np

import stacking_wall_tibb as swt
from stacking_wall_tibb_map import simulated_map

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
TAG = "940"
PSETS = {
    "eps15": dict(work="series", suffix="", s=[0.15, 0.30, 0.45, 0.60, 0.0, 1.0, 0.75, 0.90],
                  source=(f"stacking_wall_epsfit_{TAG}.json", ("polarized", "best")),
                  label="polarized stack, \u03b5 = 15"),
    "eps26_peaks": dict(work="series_eps26_peaks", suffix="_eps26_peaks",
                        s=[0.0, 1.0, 0.50, 0.33, 0.67, 0.41, 0.59, 0.25, 0.75, 0.19, 0.14, 0.08, 0.88],
                        source=(f"stacking_wall_eps26_{TAG}.json", ("best",)),
                        label="\u03b5 = 2.6, best peak fit"),
    # s every ~1 nm within 3 nm of the wall centre of line 940 (l = 5.6 nm), wider beyond; the centre first
    "eps26_fb": dict(work="series_eps26_fb", suffix="_eps26_fb",
                     s=[0.0, 1.0, 0.50, 0.33, 0.67, 0.41, 0.59, 0.25, 0.75, 0.19, 0.14, 0.08, 0.88],
                     source=(f"stacking_wall_eps26_{TAG}.json", ("flat_band_in_place",)),
                     label="\u03b5 = 2.6, best flat-band fit"),
}
PSET = "eps15"
WORK = os.path.join(HERE, "stacking_wall_epsfit", PSETS[PSET]["work"])
S_ORDER = PSETS[PSET]["s"]                                         # slow wall stacks first


def use(name):
    """Switch the module to parameter set ``name``."""
    global PSET, WORK, S_ORDER
    PSET, WORK, S_ORDER = name, os.path.join(HERE, "stacking_wall_epsfit", PSETS[name]["work"]), PSETS[name]["s"]
    os.makedirs(WORK, exist_ok=True)


def out_name(kind, ext):
    return os.path.join(EXAMPLES, f"stacking_wall_polar{PSETS[PSET]['suffix']}{kind}_{TAG}.{ext}")
# 50 mV steps, 25 mV within 75 mV of zero (the spectra agree with a full 25 mV sweep to 0.2 % of the peak and
# give the same feature positions; checked on the (+5) stack)
BIASES = np.unique(np.round(np.concatenate([np.arange(-0.45, 0.45 + 1e-9, 0.05), [-0.075, -0.025, 0.025, 0.075]]), 4))
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
SPEC = dict(modulation=0.00375, barrier=4.6, separation=0.7, broadening=0.01, temperature=9.0)
TRAP, OXIDE = 2.0, 90.0
WALL_KW = {"h_fine": 0.012, "h_coarse": 0.3}


def best_params():
    fname, path = PSETS[PSET]["source"]
    src = json.load(open(os.path.join(EXAMPLES, fname)))
    s = src
    for k in path:
        s = s[k]
    p = dict(eps=s.get("eps", src.get("eps")), dEF=s["dEF"], nsub=s["nsub"], cpot=s.get("cpot"))
    if p["cpot"] is None:                      # the tip that leaves the (+5) bands flat at zero bias: CPot = U_1 far
        from bandbending import Simulation
        r = Simulation(sample(swt.SEQ_RG, "RG", p["eps"], p["nsub"], fermi_level=p["dEF"]), None,
                       temperature=9.0).solve(0.0, 0.0)
        p["cpot"] = float(np.asarray(r.multilayers["RG"].U)[0][0])
    return p


def sample(stacking, name, eps, nsub, *, doping=None, fermi_level=None):
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, library
    kw = {"doping": tuple(float(d) for d in doping)} if doping is not None else {"fermi_level": float(fermi_level)}
    ml = GrapheneMultilayer(stacking, name=name, broadening=0.01, e_window=0.6, eps_spacer=eps, **kw)
    return Sample(surface=ml, layers=(Layer(TRAP, library.SIO2, name="traps", fixed_charge=nsub / (TRAP * 1e-7)),
                                      Layer(OXIDE - TRAP, library.SIO2, name="SiO2")), gate=Gate())


def layer_doping(p):
    """Fixed charge per layer of the (+5) stack doped to Delta E_F (the field-free doped stack)."""
    from bandbending import Simulation
    r = Simulation(sample(swt.SEQ_RG, "RG", p["eps"], p["nsub"], fermi_level=p["dEF"]), None, temperature=9.0).solve(0.0, 0.0)
    return np.asarray(r.multilayers["RG"].doping)


def key(s):
    return f"s{s:.2f}"


def run_one(s, p, doping):
    from bandbending import Tip
    out = os.path.join(WORK, key(s) + ".json")
    if s == 0.0:
        name, stacking, wall_s = "RG", swt.SEQ_RG, None
    elif s == 1.0:
        name, stacking, wall_s = "SF", swt.SEQ_SF, None
    else:
        name, stacking, wall_s = swt.WALL_NAME, swt.SEQ_RG, s
    tip = Tip(radius=20.0, separation=0.7, contact_potential=p["cpot"])
    t0 = time.time()
    m, dt, fails = swt.run_sweep(sample(stacking, name, p["eps"], p["nsub"], doping=doping), tip, BIASES,
                                 temperature=9.0, wall_s=wall_s, grid="draft", n_phi=6, wall_kw=WALL_KW,
                                 checkpoint=os.path.join(WORK, key(s) + ".ckpt.json"))
    sp = m.spectrum(0, bias=FINE, **SPEC)
    sw = swt.sweep_arrays(m)
    res = dict(s=s, stacking=stacking, seconds=time.time() - t0, failures=len(fails), didv=sp.didv.tolist(),
               didv_frozen=sp.didv_frozen.tolist(), sweep=sw, params=p)
    with open(out, "w") as fh:
        json.dump(res, fh)
    V, U0, Uf = np.asarray(sw["bias"]), np.asarray(sw["U0"]), np.asarray(sw["U_far"])
    print(f"s = {s:.2f}: {time.time() - t0:.0f} s, {len(fails)} failures; far-field U - E_F (meV) "
          f"{np.round(1e3 * (Uf[0] - sw['mu0']), 1)}, top-layer bending at 0 V {1e3 * np.interp(0, V, U0[:, 0] - Uf[:, 0]):+.2f} meV",
          flush=True)


def run_half(s, sign):
    """One half of the sweep of stack s (bias <= 0 or >= 0) in its own checkpoint; the process that
    completes the second half merges both into the stack's checkpoint and computes the spectrum."""
    os.makedirs(WORK, exist_ok=True)
    p = best_params()
    doping = layer_doping(p)
    from bandbending import Tip
    name, stacking, wall_s = (swt.WALL_NAME, swt.SEQ_RG, s) if 0.0 < s < 1.0 else (("RG", swt.SEQ_RG, None) if s == 0.0 else ("SF", swt.SEQ_SF, None))
    from bandbending import Simulation
    smp = sample(stacking, name, p["eps"], p["nsub"], doping=doping)
    tip = Tip(radius=20.0, separation=0.7, contact_potential=p["cpot"])
    sim = Simulation(smp, tip, temperature=9.0, grid="draft") if wall_s is None else \
        swt.WallSimulation(smp, tip, 9.0, wall_s=wall_s, n_phi=6, grid="draft", wall_kw=WALL_KW)
    half = BIASES[BIASES <= 0] if sign == "neg" else BIASES[BIASES >= 0]
    swt.CheckpointedSweep(sim, name, half, os.path.join(WORK, f"{key(s)}_{sign}.ckpt.json")).run()
    print(f"{key(s)} {sign} half done", flush=True)
    parts = [os.path.join(WORK, f"{key(s)}_{h}.ckpt.json") for h in ("neg", "pos")]
    pts = {}
    for f, h in zip(parts, ("neg", "pos")):
        if not os.path.exists(f):
            return
        d = json.load(open(f))
        need = BIASES[BIASES <= 0] if h == "neg" else BIASES[BIASES >= 0]
        if any(swt.CheckpointedSweep.key(float(v)) not in d for v in need):
            return
        pts.update(d)
    try:
        fd = os.open(os.path.join(WORK, key(s) + ".merge.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    with open(os.path.join(WORK, key(s) + ".ckpt.json"), "w") as fh:
        json.dump(pts, fh)
    run_one(s, p, doping)


def run(worker):
    os.makedirs(WORK, exist_ok=True)
    p = best_params()
    doping = layer_doping(p)
    print(f"worker {worker}: eps {p['eps']}, dEF {1e3 * p['dEF']:.0f} meV, n_sub {p['nsub']:.4e}, CPot {1e3 * p['cpot']:.1f} meV; "
          f"doping per layer (1e12 cm^-2) {np.round(doping / 1e12, 3)}", flush=True)
    for s in S_ORDER:
        if os.path.exists(os.path.join(WORK, key(s) + ".json")):
            continue
        try:
            fd = os.open(os.path.join(WORK, key(s) + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        run_one(s, p, doping)
    print(f"worker {worker} done", flush=True)


def queue(worker, sets):
    """Work through the stacks of several parameter sets: every wall stack as two bias halves
    (merged by whichever half finishes last), the end stacks whole; one lock file per job."""
    jobs = []
    for n in sets:                                   # in the order of the set's s list
        for s in PSETS[n]["s"]:
            jobs += [(n, s, "neg"), (n, s, "pos")] if 0.0 < s < 1.0 else [(n, s, None)]
    for n, s, half in jobs:
        use(n)
        if os.path.exists(os.path.join(WORK, key(s) + ".json")):
            continue
        lock = os.path.join(WORK, key(s) + (f"_{half}" if half else "") + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        print(f"worker {worker}: {n} s = {s:.2f} {half or 'whole'}", flush=True)
        if half:
            run_half(s, half)
        else:
            p = best_params()
            run_one(s, p, layer_doping(p))
    print(f"worker {worker} done", flush=True)


# ---------------------------------------------------------------------------
# collect: merge, map, figure
# ---------------------------------------------------------------------------

def fb_peak(E, y, centre, half=0.09):
    sel = (E >= centre - half) & (E <= centre + half) & np.isfinite(y)
    return float(E[sel][np.argmax(y[sel])]), float(y[sel].max())


def collect(sm4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import stacking_wall as sw

    runs = {}
    for f in glob.glob(os.path.join(WORK, "s*.json")):
        if f.endswith(".ckpt.json"):
            continue
        r = json.load(open(f))
        runs[float(r["s"])] = r
    s_vals = np.array(sorted(runs))
    main = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{TAG}.json")))
    meas = main["measured"]
    Vm, ym = np.asarray(meas["bias"]), np.asarray(meas["rg"], dtype=float)
    E_fb = main["settings"]["E_fb_measured"]
    a, res = swt.shape_distance(Vm, ym, FINE, np.asarray(runs[0.0]["didv"]))        # one global scale, (+5)
    D = np.array([runs[s]["didv"] for s in s_vals]) * a * 1e3
    fit = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_{TAG}.json")))["fit"]
    exp = sw.load_line_spectrum(sm4)
    dist, bias_e, exp_map = exp["dist"], exp["bias"], exp["didv"]
    sim, s_x = simulated_map(s_vals, D, dist, fit)
    old = np.load(os.path.join(EXAMPLES, f"stacking_wall_tibb_map_{TAG}.npz"))
    np.savez_compressed(out_name("_map", "npz"), dist=dist, bias=FINE,
                        didv=sim.astype(np.float32), s=s_x, s_values=s_vals)

    # flat-band peak along the line: measured, polarized model, earlier uniformly doped model
    def track(E, Z, centre, h_min):
        pk, h = [], []
        for j in range(Z.shape[1]):
            e, v = fb_peak(E, Z[:, j], centre)
            pk.append(e)
            h.append(v)
        pk, h = np.array(pk), np.array(h)
        far = dist > dist.max() - 15
        alive = h > h_min * np.nanmean(h[far])
        return np.where(alive, pk, np.nan)

    pk_meas = track(bias_e, exp_map, E_fb, 0.5)
    pk_pol = track(FINE, sim, E_fb, 0.3)
    pk_old = track(np.asarray(old["bias"]), np.asarray(old["didv"]), E_fb, 0.3)
    peaks_s = {f"{s:.2f}": fb_peak(FINE, np.asarray(runs[s]["didv"]), E_fb)[0] for s in s_vals}

    lo, hi = -0.45, 0.45
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for col, (E, Z, title) in enumerate(((bias_e, exp_map * 1e3, f"experiment, line spectrum {TAG}"),
                                         (FINE, sim, f"model: {PSETS[PSET]['label']}, local approximation"))):
        sel = (E >= lo) & (E <= hi)
        v = np.nanpercentile(Z[sel], [1, 99.5])
        im = axs[0, col].imshow(Z[sel], aspect="auto", origin="lower", cmap="plasma",
                                extent=[dist.min(), dist.max(), lo, hi], vmin=v[0], vmax=v[1])
        fig.colorbar(im, ax=axs[0, col], pad=0.02).set_label("dI/dV (a.u.)")
        axs[0, col].set_title(title, fontsize=10, loc="left")
        g = np.gradient(Z[sel], E[sel], axis=0)
        vg = np.nanpercentile(np.abs(g), 98)
        im = axs[1, col].imshow(g, aspect="auto", origin="lower", cmap="coolwarm",
                                extent=[dist.min(), dist.max(), lo, hi], vmin=-vg, vmax=vg)
        fig.colorbar(im, ax=axs[1, col], pad=0.02).set_label("d(dI/dV)/dV (a.u.)")
        axs[1, col].set_title("bias derivative", fontsize=10, loc="left")
    for ax in axs.ravel():
        ax.axvline(fit["x0"], color="w", ls="--", lw=0.8)
        ax.set_xlabel("distance (nm)")
        ax.set_ylabel("sample bias (V)")
    fig.suptitle(f"Line spectrum {TAG}: eps = {runs[0.0]['params']['eps']:g}, Delta E_F = {1e3 * runs[0.0]['params']['dEF']:.0f} meV, "
                 f"n_sub = {runs[0.0]['params']['nsub'] / 1e13:.3f}e13 cm^-2, same fixed charge per layer and tip for every stack; "
                 f"s(x) from the fitted profile (x0 = {fit['x0']:.1f} nm, l = {fit['ell']:.1f} nm), s = "
                 f"{', '.join(f'{s:g}' for s in s_vals)}", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_name("_map", "png"), dpi=160)
    plt.close(fig)

    # spectra at the s values and the flat-band peak along the line
    INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axs[0]
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(s_vals):
        ax.plot(FINE, D[i] / D[0].max() + 0.45 * i, color=cmap(i / max(len(s_vals) - 1, 1)), lw=1.3, label=f"s = {s:g}")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("sample bias (V)")
    ax.set_ylabel("top-layer dI/dV under the tip (offset)")
    ax.set_title(f"{PSETS[PSET]['label']}: each s, same tip", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax = axs[1]
    ax.plot(dist, 1e3 * pk_meas, color=INK, lw=1.6, label="measured")
    ax.plot(dist, 1e3 * pk_pol, color="#2a78d6", lw=1.6, label=PSETS[PSET]["label"])
    ax.plot(dist, 1e3 * pk_old, color="#eb6834", lw=1.2, ls=(0, (3, 2)), label="uniformly doped stack, \u03b5 = 2.6")
    for other, color in (("eps15", "#1baf7a"), ("eps26_fb", "#6250d6"), ("eps26_peaks", "#e87ba4")):
        if other == PSET:
            continue
        f_other = os.path.join(EXAMPLES, f"stacking_wall_polar{PSETS[other]['suffix']}_{TAG}.json")
        if os.path.exists(f_other):
            tr = json.load(open(f_other))["track"]
            ax.plot(np.asarray(tr["dist"]), 1e3 * np.asarray(tr["polarized"], dtype=float), color=color, lw=1.1,
                    ls=(0, (1, 1.5)), label=PSETS[other]["label"])
    ax.axvline(fit["x0"], color=MUTED, lw=0.8, ls="--")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlabel("distance (nm)")
    ax.set_ylabel("flat-band peak (mV)")
    ax.set_title("flat-band peak along the line (while visible)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_name("_spectra", "png"), dpi=160)
    plt.close(fig)

    # lowest edge above E_F (steepest rise between +20 and +260 mV) along the line
    from scipy.signal import savgol_filter

    def edge_track(E, Z, smooth):
        sel = (E > 0.02) & (E < 0.26)
        out_e = []
        for j in range(Z.shape[1]):
            y = np.nanmean(Z[:, max(j - 2, 0):j + 3], axis=1) if smooth else Z[:, j]
            if smooth:
                y = savgol_filter(y, 9, 3)
            d = np.gradient(y, E)
            out_e.append(float(E[sel][np.argmax(d[sel])]))
        return np.array(out_e)

    edge_meas = edge_track(bias_e, exp_map, True)
    edge_pol = edge_track(FINE, sim, False)
    edge_old = edge_track(np.asarray(old["bias"]), np.asarray(old["didv"]), False)
    y0 = np.asarray(runs[0.0]["didv"])
    win = (FINE > E_fb - 0.09) & (FINE < E_fb + 0.09)
    bg = (FINE > -0.2) & (FINE < -0.12)
    heights = {f"{s:.2f}": float((np.asarray(runs[s]["didv"])[win].max() - np.median(np.asarray(runs[s]["didv"])[bg]))
                                 / y0[win].max()) for s in s_vals}
    # the flat-band peak of every stack without the tip (same spectroscopy, layer potentials of the column)
    from bandbending import Simulation
    from bandbending.results.spectroscopy import tunneling_spectrum
    p_set = best_params()
    dop = layer_doping(p_set)
    tipfree = {}
    for s in s_vals:
        if s == 0.0:
            nm, sim0 = "RG", Simulation(sample(swt.SEQ_RG, "RG", p_set["eps"], p_set["nsub"], doping=dop), None, temperature=9.0)
        elif s == 1.0:
            nm, sim0 = "SF", Simulation(sample(swt.SEQ_SF, "SF", p_set["eps"], p_set["nsub"], doping=dop), None, temperature=9.0)
        else:
            nm = swt.WALL_NAME
            sim0 = swt.WallSimulation(sample(swt.SEQ_RG, nm, p_set["eps"], p_set["nsub"], doping=dop), None, 9.0, wall_s=s,
                                      n_phi=6, wall_kw=WALL_KW)
        q = sim0.solve(0.0, 0.0).multilayers[nm]
        Uc = np.asarray(q.U)[0]
        yc = tunneling_spectrum(q.tb, q.mu0, FINE, np.tile(Uc, (FINE.size, 1)), np.zeros((FINE.size, Uc.size)), **SPEC).didv
        tipfree[f"{s:.2f}"] = fb_peak(FINE, yc, E_fb)[0]
    far = dist > dist.max() - 15
    out = dict(tipfree_peaks=tipfree,params=runs[0.0]["params"], s_values=s_vals.tolist(), scale=a, residual_rg=res, fine_bias=FINE.tolist(),
               peaks_s=peaks_s, runs={key(s): {k: runs[s][k] for k in ("s", "stacking", "seconds", "failures", "didv", "sweep")}
                                      for s in s_vals},
               track=dict(dist=dist.tolist(), measured=pk_meas.tolist(), polarized=pk_pol.tolist(), uniform=pk_old.tolist()),
               edge=dict(measured=edge_meas.tolist(), polarized=edge_pol.tolist(), uniform=edge_old.tolist()),
               heights=heights,
               fit=fit, E_fb=E_fb)
    # the bend: deepest flat-band peak while visible, minus the far (+5) value
    for k, pk in (("measured", pk_meas), ("polarized", pk_pol), ("uniform", pk_old)):
        ok = np.isfinite(pk)
        out[f"far_{k}"] = float(np.nanmean(pk[far]))
        out[f"extreme_{k}"] = float(pk[ok][np.argmax(np.abs(pk[ok] - np.nanmean(pk[far])))]) if ok.any() else None
        out[f"extreme_dist_{k}"] = float(dist[ok][np.argmax(np.abs(pk[ok] - np.nanmean(pk[far])))]) if ok.any() else None
    json.dump(out, open(out_name("", "json"), "w"))
    print(f"wrote maps; (+5) residual {res:.3f}; flat-band peak per s (mV): "
          f"{ {k: round(1e3 * v, 1) for k, v in peaks_s.items()} }")
    for k in ("measured", "polarized", "uniform"):
        print(f"  {k}: far {1e3 * out[f'far_{k}']:+.1f} mV, extreme {1e3 * out[f'extreme_{k}']:+.1f} mV at {out[f'extreme_dist_{k}']:.1f} nm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "half", "queue", "collect"])
    ap.add_argument("--set", default="eps15", choices=sorted(PSETS))
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("worker", nargs="?", type=int, default=0)
    ap.add_argument("--s", type=float, default=None)
    ap.add_argument("--sign", choices=["neg", "pos"], default=None)
    ap.add_argument("--sm4", default=None)
    args = ap.parse_args()
    use(args.set)
    if args.cmd == "run":
        run(args.worker)
    elif args.cmd == "queue":
        queue(args.worker, args.sets or [args.set])
    elif args.cmd == "half":
        run_half(args.s, args.sign)
    else:
        collect(args.sm4)
