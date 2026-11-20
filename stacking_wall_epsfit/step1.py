"""Step 1 of the rebuilt charge model: the (+5) spectra of lines 514 and 940.

Model: six neutral gamma0/gamma1 graphene layers (no per-layer doping), interlayer permittivity eps (free, the
same for both lines), a fixed charge n_sub in the top 2 nm of the 90 nm SiO2 (free, one per line), the Si gate at
the sample potential, the tip fixed (R = 20 nm, d = 0.7 nm, contact potential 0).  The Gaussian width of the
occupation and the LDOS reproduces the measured flat-band width (sigma = 13.7 meV for 514, 12.5 meV for 940).

Every cell is solved twice: without the tip (one self-consistent solve; the spectrum of the far-field potentials)
and with the tip (a self-consistent bias sweep; the spectrum with the band-bending term).  Both are compared with
the measured (+5) spectrum:
  * feature positions: lower side-band peak, flat-band peak, the two upper side-band edges (and, for 940, the
    onset at +69 mV);
  * flat-band asymmetry: the spectrum relative to the flat-band peak 40, 60 and 80 mV from the peak on the side
    of E_F and on the far side;
  * shape: least-squares scale and relative residual over -0.45 .. 0.45 V;
  * the bottom flat band: the far-field potential of layer 6 (where the bottom flat band sits) and the top-layer
    LDOS there.

    python step1.py run        # 4 processes; cells in step1/
    python step1.py summary    # table and best cells; writes ../examples/stacking_wall_step1.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "step1")
TRAP = 2.0
SIGMA = {"514": 0.0137, "940": 0.0125}
TIP = dict(radius=20.0, separation=0.7, contact_potential=0.0)
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
BIASES = np.round(np.concatenate([np.arange(-0.45, -0.199, 0.05), np.arange(-0.15, 0.151, 0.025),
                                  np.arange(0.2, 0.451, 0.05)]), 4)
GRID = {
    "940": {2.6: [1.0, 2.0, 3.0], 4.0: [1.0, 1.5, 2.0, 2.5], 6.0: [0.6, 1.0, 1.4, 1.8], 8.0: [0.6, 0.9, 1.2, 1.5],
            10.0: [0.5, 0.8, 1.0, 1.3], 15.0: [0.5, 0.7, 0.9, 1.1]},
    "514": {2.6: [-1.0, -2.0, -3.0], 4.0: [-1.0, -1.3, -1.6, -2.0], 6.0: [-0.6, -0.9, -1.2, -1.5],
            8.0: [-0.5, -0.75, -1.0, -1.25], 10.0: [-0.4, -0.6, -0.8, -1.0], 15.0: [-0.4, -0.55, -0.7, -0.85]},
}
TAILS = (0.04, 0.06, 0.08)


def key(tag, eps, n13, cpot=0.0):
    return f"{tag}_eps{eps:04.1f}_n{n13:+.2f}" + (f"_c{cpot:+.3f}" if cpot else "")


def measured(tag):
    m = json.load(open(os.path.join(REPO, "examples", f"stacking_wall_tibb_{tag}.json")))
    return np.asarray(m["measured"]["bias"]), np.asarray(m["measured"]["rg"], float), m["settings"]["E_fb_measured"]


def asymmetry(V, y, E_fb):
    """Spectrum over the flat-band peak at TAILS from the peak: (towards E_F ..., away from E_F ...)."""
    V, y = np.asarray(V), np.asarray(y)
    sel = np.abs(V - E_fb) < 0.04
    i0 = np.where(sel)[0][np.argmax(y[sel])]
    pk, v0 = y[i0], V[i0]
    s = np.sign(v0) if v0 != 0 else np.sign(E_fb)
    inner = [float(np.interp(v0 - s * d, V, y) / pk) for d in TAILS]
    outer = [float(np.interp(v0 + s * d, V, y) / pk) for d in TAILS]
    return np.array(inner + outer)


def feats(tag, V, y, smooth=0):
    sys.path.insert(0, HERE)
    from features import features, features5
    return features5(V, y, smooth) if tag == "940" else features(V, y, smooth)


def evaluate(job):
    tag, eps, n13 = job[:3]
    cpot = job[3] if len(job) > 3 else 0.0
    path = os.path.join(OUT, key(tag, eps, n13, cpot) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, Simulation, Tip, library
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    t0 = time.time()
    sig = SIGMA[tag]
    nsub = n13 * 1e13
    spec = dict(modulation=0.00375, barrier=4.6, separation=TIP["separation"], broadening=sig, temperature=9.0)
    ml = GrapheneMultilayer(swt.SEQ_RG, name="RG", broadening=sig, e_window=0.8, doping=0.0, eps_spacer=eps)
    sample = Sample(surface=ml, layers=(Layer(TRAP, library.SIO2, name="traps", fixed_charge=nsub / (TRAP * 1e-7)),
                                        Layer(90.0 - TRAP, library.SIO2, name="SiO2")), gate=Gate())
    # without the tip
    r = Simulation(sample, None, temperature=9.0).solve(0.0, 0.0)
    p = r.multilayers["RG"]
    U = np.asarray(p.U)[0]
    free = tunneling_spectrum(p.tb, p.mu0, FINE, np.tile(U, (FINE.size, 1)), np.zeros((FINE.size, U.size)), **spec).didv
    E = np.arange(-0.6, 0.6 + 1e-9, 0.001)
    _, D = p.tb.ldos(U, E + p.mu0, broadening=sig, resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == E.size else D.T
    bot = float((U - p.mu0)[5])
    j = int(np.argmin(np.abs(E - bot)))
    # with the tip
    sim = Simulation(sample, Tip(**dict(TIP, contact_potential=cpot)), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "RG", BIASES, os.path.join(OUT, key(tag, eps, n13, cpot) + ".ckpt.json")).run()
    tipped = m.spectrum(0, bias=FINE, **spec).didv
    Vm, ym, E_fb = measured(tag)
    win = (Vm >= -0.45) & (Vm <= 0.45)
    fm = feats(tag, Vm[win], ym[win], smooth=9)
    am = asymmetry(Vm[win], ym[win], E_fb)
    out = dict(tag=tag, eps=eps, nsub=nsub, cpot=cpot, sigma=sig, U_minus_mu0=(U - p.mu0).tolist(), mu0=float(p.mu0),
               carriers=float(np.sum(p.n)), carriers_layers=np.asarray(p.n).ravel().tolist(), bottom_fb=bot,
               bottom_in_top=float(D[j, 0] / np.median(D[(np.abs(E - bot) < 0.1), 0])),
               measured_features=fm.tolist(), measured_asym=am.tolist(), failures=len(m.failures),
               sweep=swt.sweep_arrays(m), seconds=None)
    for lab, y in (("free", np.asarray(free)), ("tip", np.asarray(tipped))):
        f = feats(tag, FINE, y)
        a = asymmetry(FINE, y, E_fb)
        sc, res = swt.shape_distance(Vm, ym, FINE, y)
        ok = np.isfinite(f)
        out[lab] = dict(didv=y.tolist(), features=f.tolist(), asym=a.tolist(), scale=sc, residual=res,
                        pos_rms=float(np.sqrt(np.mean((f[ok] - fm[ok]) ** 2))) if ok.any() else 1.0,
                        n_features=int(ok.sum()), fb_err=float(f[1] - fm[1]),
                        asym_rms=float(np.sqrt(np.mean((a - am) ** 2))))
        # one number to rank cells: 10 mV of feature rms counts as much as 0.05 of tail height rms
        out[lab]["score"] = out[lab]["pos_rms"] / 0.010 + out[lab]["asym_rms"] / 0.05 + 2.0 * (len(f) - ok.sum())
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    print(f"{key(tag, eps, n13, cpot)}: bottom FB {1e3 * bot:+.0f} meV; with tip FB {1e3 * out['tip']['features'][1]:+.1f} "
          f"(meas {1e3 * fm[1]:+.1f}), feature rms {1e3 * out['tip']['pos_rms']:.1f} mV, tails {out['tip']['asym_rms']:.3f}, "
          f"res {out['tip']['residual']:.2f}; no tip FB {1e3 * out['free']['features'][1]:+.1f}, rms {1e3 * out['free']['pos_rms']:.1f}; "
          f"{out['seconds']:.0f} s", flush=True)
    return out


CPOT_CELLS = {"940": [(4.0, 1.5), (4.0, 2.0), (6.0, 1.8)], "514": [(15.0, -0.4), (10.0, -0.4), (8.0, -0.5), (2.6, -2.0)]}
CPOTS = (-0.15, -0.075, 0.075, 0.15)


def run(cpot_scan=False):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(t, e, n) for t in ("940", "514") for e, ns in GRID[t].items() for n in ns]
    if cpot_scan:
        jobs = [(t, e, n, c) for t, cells in CPOT_CELLS.items() for e, n in cells for c in CPOTS]
    with mp.get_context("spawn").Pool(4) as pool:
        for _ in pool.imap_unordered(evaluate, jobs):
            pass


def summary():
    rows = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json") and ".ckpt" not in f]
    out = {}
    for tag in ("514", "940"):
        rs = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["tip"]["score"])
        print(f"== {tag}: measured features {np.round(1e3 * np.asarray(rs[0]['measured_features']))} mV, tails "
              f"{np.round(rs[0]['measured_asym'], 2)}")
        for r in rs:
            print(f"  eps {r['eps']:4.1f} n_sub {r['nsub'] / 1e13:+5.2f}e13 CPot {r.get('cpot', 0.0):+.3f} | tip: score {r['tip']['score']:5.1f} feats "
                  f"{np.round(1e3 * np.asarray(r['tip']['features']))} rms {1e3 * r['tip']['pos_rms']:5.1f} tails "
                  f"{r['tip']['asym_rms']:.3f} res {r['tip']['residual']:.2f} | no tip: score {r['free']['score']:5.1f} "
                  f"FB {1e3 * r['free']['features'][1]:+5.1f} rms {1e3 * r['free']['pos_rms']:5.1f} | bottom FB "
                  f"{1e3 * r['bottom_fb']:+5.0f} meV")
        out[tag] = [{k: v for k, v in r.items() if k not in ("sweep",)} for r in rs]
    # eps shared by both lines: best sum of the best score per line at each eps
    eps_list = sorted({r["eps"] for r in rows})
    joint = []
    for e in eps_list:
        best = {t: min((r for r in rows if r["tag"] == t and r["eps"] == e), key=lambda r: r["tip"]["score"]) for t in ("514", "940")}
        joint.append(dict(eps=e, score=sum(b["tip"]["score"] for b in best.values()),
                          cells={t: (b["nsub"], b["tip"]["score"]) for t, b in best.items()}))
        print(f"eps {e:4.1f}: joint score {joint[-1]['score']:5.1f}  (514 n_sub {best['514']['nsub'] / 1e13:+.2f}e13, "
              f"940 n_sub {best['940']['nsub'] / 1e13:+.2f}e13)")
    json.dump(dict(cells=out, joint=joint), open(os.path.join(REPO, "examples", "stacking_wall_step1.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    if sys.argv[1] == "cells":                   # python step1.py cells 940:2.6:2.0:-0.15 ...
        jobs = [tuple([a.split(":")[0]] + [float(x) for x in a.split(":")[1:]]) for a in sys.argv[2:]]
        os.makedirs(OUT, exist_ok=True)
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(evaluate, jobs))
    else:
        {"run": run, "cpot": lambda: run(True), "summary": summary}[sys.argv[1]]()
