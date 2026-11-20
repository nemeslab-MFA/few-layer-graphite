"""Step 1 with the global offset: the (+5) spectra of lines 514 and 940, polarized-stack charge model.

Model: six gamma0/gamma1 graphene layers with the starting doping Delta E_F (GrapheneMultilayer.fermi_level: each
layer carries the fixed charge that neutralises its own carriers at E_F = Delta E_F, so the stack alone is flat),
interlayer permittivity eps, a fixed charge n_sub in the top 2 nm of the 90 nm SiO2, the Si gate at the sample
potential, the tip fixed (R = 20 nm, d = 0.7 nm) at zero band bending at zero bias (CPot = U_1 far from the tip:
tip work function = sample work function).  sigma = 13.7 meV (514) and 12.5 meV (940) as in step1.py.

For every (eps, Delta E_F) n_sub is set by bisection so that the tip-free top-layer flat band sits at the measured
flat-band peak; the cell is then solved without the tip (spectrum of the far-field potentials) and with it (a
self-consistent bias sweep).  Targets (2026-11-20, rescore.py): the positions of the lower side-band peak and of
the two upper side-band edges, the flat-band peak, and the presence (sign) of the flat-band asymmetry; the size of
the shoulder and the +69 mV step are not fitted.

    python step1g.py run                 # 4 processes; cells in step1g/
    python step1g.py cells 940:10:0.13   # single cells (tag:eps:Delta E_F in eV)
    python step1g.py summary             # table; writes ../examples/stacking_wall_step1g.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "step1g")
sys.path.insert(0, HERE)
import rescore  # noqa: E402
import step1  # noqa: E402

EPS = (2.6, 4.0, 6.0, 10.0, 15.0)
DEF = {"940": (0.04, 0.07, 0.10, 0.13, 0.16, 0.20), "514": (-0.04, -0.07, -0.10, -0.13, -0.16, -0.20)}
FB_MEASURED = {"940": -0.0325, "514": 0.0225}      # replaced by the measured feature at run time


def key(tag, eps, dEF):
    return f"{tag}_eps{eps:04.1f}_dEF{1e3 * dEF:+06.1f}"


def build(tag, eps, dEF, nsub):
    sys.path.insert(0, REPO)
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, library
    import stacking_wall_tibb as swt
    ml = GrapheneMultilayer(swt.SEQ_RG, name="RG", broadening=step1.SIGMA[tag], e_window=0.8, fermi_level=dEF,
                            eps_spacer=eps)
    return Sample(surface=ml, layers=(Layer(step1.TRAP, library.SIO2, name="traps", fixed_charge=nsub / (step1.TRAP * 1e-7)),
                                      Layer(90.0 - step1.TRAP, library.SIO2, name="SiO2")), gate=Gate())


E = np.arange(-0.6, 0.6 + 1e-9, 0.0005)


def tipfree(tag, eps, dEF, nsub):
    from bandbending import Simulation
    r = Simulation(build(tag, eps, dEF, nsub), None, temperature=9.0).solve(0.0, 0.0)
    p = r.multilayers["RG"]
    U = np.asarray(p.U)[0]
    _, D = p.tb.ldos(U, E + p.mu0, broadening=step1.SIGMA[tag], resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == E.size else D.T
    sel = (E > -0.12) & (E < 0.08)
    fb = float(E[sel][np.argmax(D[sel, 0])])
    return dict(U=U, mu0=float(p.mu0), carriers=float(np.sum(p.n)), doping=np.asarray(p.doping), fb=fb, D=D, p=p)


def place(tag, eps, dEF, target):
    """n_sub (cm^-2) putting the tip-free top-layer flat band at target: bisection (the peak falls as n_sub rises)."""
    doping = tipfree(tag, eps, dEF, 0.0)["doping"]
    n0 = -float(np.sum(doping))
    # n0 leaves the stack neutral; the flat band crosses the gap strip there.  Beyond about 2e13 from it the peak
    # window picks up the side bands, so the bracket stays inside.
    for w in (0.5e13, 1.0e13, 1.5e13, 2.0e13):
        lo, hi = n0 - w, n0 + w
        flo, fhi = tipfree(tag, eps, dEF, lo)["fb"], tipfree(tag, eps, dEF, hi)["fb"]
        if fhi <= target <= flo:
            break
    else:
        return None
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if tipfree(tag, eps, dEF, mid)["fb"] > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 2e9:
            break
    return 0.5 * (lo + hi)


def evaluate(job):
    tag, eps, dEF = job
    path = os.path.join(OUT, key(tag, eps, dEF) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, REPO)
    from bandbending import Simulation, Tip
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vm, ym = rescore.measured(tag)
    fm = rescore.features(Vm, ym, smooth=9)
    nsub = place(tag, eps, dEF, float(fm[1]))
    if nsub is None:
        out = dict(tag=tag, eps=eps, dEF=dEF, placed=False)
        json.dump(out, open(path, "w"))
        print(f"{key(tag, eps, dEF)}: flat band cannot be placed", flush=True)
        return out
    t_place = time.time() - t0
    sig = step1.SIGMA[tag]
    spec = dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=sig, temperature=9.0)
    tf = tipfree(tag, eps, dEF, nsub)
    U, p = tf["U"], tf["p"]
    free = np.asarray(tunneling_spectrum(p.tb, p.mu0, step1.FINE, np.tile(U, (step1.FINE.size, 1)),
                                         np.zeros((step1.FINE.size, U.size)), **spec).didv)
    cpot = float(U[0])                                     # zero band bending at zero bias
    sim = Simulation(build(tag, eps, dEF, nsub), Tip(**dict(step1.TIP, contact_potential=cpot)), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "RG", step1.BIASES, os.path.join(OUT, key(tag, eps, dEF) + ".ckpt.json")).run()
    tipped = np.asarray(m.spectrum(0, bias=step1.FINE, **spec).didv)
    D = tf["D"]
    Ul = U - tf["mu0"]
    out = dict(tag=tag, eps=eps, dEF=dEF, placed=True, nsub=nsub, cpot=cpot, sigma=sig, U_minus_mu0=Ul.tolist(),
               mu0=tf["mu0"], carriers=tf["carriers"], doping=tf["doping"].tolist(), bottom_fb=float(Ul[5]),
               ldos_E=E[::4].tolist(), ldos_top=D[::4, 0].tolist(), ldos_bottom=D[::4, 5].tolist(),
               measured_features=fm.tolist(), measured_asym=rescore.asym(Vm, ym, fm[1]), failures=len(m.failures),
               sweep=swt.sweep_arrays(m))
    for lab, y in (("free", free), ("tip", tipped)):
        sc, res = swt.shape_distance(Vm, ym, step1.FINE, y)
        out[lab] = dict(didv=y.tolist(), scale=sc, residual=res, **rescore.score(tag, y, fm))
    out["seconds"] = time.time() - t0
    out["seconds_place"] = t_place
    json.dump(out, open(path, "w"))
    t, f = out["tip"], out["free"]
    print(f"{key(tag, eps, dEF)}: n_sub {nsub / 1e13:+.3f}e13, carriers {tf['carriers'] / 1e12:+.2f}e12, U-E_F "
          f"{np.round(1e3 * Ul).astype(int).tolist()} meV | tip: feats {np.round(1e3 * np.asarray(t['features'])).tolist()} "
          f"edges {1e3 * t['edge_rms']:.1f} FB {1e3 * t['fb_err']:+.1f} asym {t['asym']:+.2f} | free: feats "
          f"{np.round(1e3 * np.asarray(f['features'])).tolist()} edges {1e3 * f['edge_rms']:.1f} asym {f['asym']:+.2f}; "
          f"{t_place:.0f} + {out['seconds'] - t_place:.0f} s", flush=True)
    return out


def summary():
    rows = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json") and ".ckpt" not in f]
    rows = [r for r in rows if r.get("placed")]
    out = {}
    for tag in ("940", "514"):
        rs = sorted([r for r in rows if r["tag"] == tag], key=lambda r: (not r["tip"]["asym_ok"], r["tip"]["edge_rms"]))
        print(f"== {tag}: measured {np.round(1e3 * np.asarray(rs[0]['measured_features'])).tolist()} mV, "
              f"asymmetry {rs[0]['measured_asym']:+.2f}" if rs else f"== {tag}: none")
        for r in rs:
            t, f = r["tip"], r["free"]
            print(f"  eps {r['eps']:4.1f} dEF {1e3 * r['dEF']:+6.1f} n_sub {r['nsub'] / 1e13:+.3f}e13 | tip: edges {1e3 * t['edge_rms']:5.1f} "
                  f"FB {1e3 * t['fb_err']:+5.1f} asym {t['asym']:+.2f} feats {np.round(1e3 * np.asarray(t['features'])).astype(int).tolist()} | "
                  f"free: edges {1e3 * f['edge_rms']:5.1f} asym {f['asym']:+.2f} feats {np.round(1e3 * np.asarray(f['features'])).astype(int).tolist()} "
                  f"| U-E_F {np.round(1e3 * np.asarray(r['U_minus_mu0'])).astype(int).tolist()}")
        out[tag] = [{k: v for k, v in r.items() if k not in ("sweep",)} for r in rs]
    json.dump(out, open(os.path.join(REPO, "examples", "stacking_wall_step1g.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] in ("run", "cells"):
        if sys.argv[1] == "run":
            jobs = [(t, e, d) for e in (10.0, 4.0, 2.6, 15.0, 6.0) for t in ("940", "514") for d in DEF[t]]
        else:
            jobs = [(a.split(":")[0], float(a.split(":")[1]), float(a.split(":")[2])) for a in sys.argv[2:]]
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(evaluate, jobs))
    elif sys.argv[1] == "time":
        t0 = time.time()
        r = tipfree("940", 10.0, 0.13, -1.4e13)
        print(f"one tip-free solve {time.time() - t0:.1f} s, FB {1e3 * r['fb']:+.1f} meV")
    else:
        summary()
