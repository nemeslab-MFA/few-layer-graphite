"""The 940 (+5) flat band placed by the tip alone: neutral stack, contact potential swept.

Hypothesis (user, 2026-11-23): the shift of the top-layer flat band to -32 mV is set mostly by the tip (it changes
slightly when the tip is pulsed), and since the bias is small there, by the tip - sample work-function difference.
Model: (+5) stack CBACBA, full SWMcC engine tight binding, interlayer eps 2.6, charge neutral (doping 0, no global
offset), no SiO2 charge; tip R 20 nm, d 0.7 nm; sigma 12.5 meV, T 9 K.  CPot = W_tip - W_RG is the only free
parameter, fitted to the flat-band position under the tip and nothing else.

    python cpot940.py place           # bisection: CPot putting the apex flat band at the measured bias (single solves)
    python cpot940.py run CPOT ...    # full self-consistent bias sweeps (4 processes): cpot940/cpot*.json
    python cpot940.py summary         # table; ../examples/stacking_wall_cpot940.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "cpot940")
sys.path.insert(0, HERE)
sys.path.insert(1, REPO)
import rescore  # noqa: E402
import step1  # noqa: E402
import swmcc_cells as S  # noqa: E402

TAG, EPS, MODEL = "940", 2.6, "full"
SIG = step1.SIGMA[TAG]
SPEC = dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=SIG, temperature=9.0)
E = np.arange(-0.30, 0.10 + 1e-9, 0.0005)


def sample():
    return S.build(TAG, EPS, 0.0, 0.0, MODEL, "RG")                # doping 0, n_sub 0


def tip(cpot):
    from bandbending import Tip
    return Tip(**dict(step1.TIP, contact_potential=float(cpot)))


def apex_fb(cpot, bias):
    """Flat-band energy (eV from E_F) of the top-layer LDOS under the apex, solved at the given bias."""
    from bandbending import Simulation
    r = Simulation(sample(), tip(cpot), temperature=9.0, grid="draft").solve(float(bias), 0.0)
    p = r.multilayers["ML"]
    U = np.asarray(p.U)[0]
    _, D = p.tb.ldos(U, E + p.mu0, broadening=SIG, resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == E.size else D.T
    return float(E[np.argmax(D[:, 0])]), U - p.mu0


def place():
    """CPot at which the apex flat band sits at the measured peak bias (resonance condition E_FB(V) = eV)."""
    Vr, yr = rescore.measured(TAG)
    vfb = float(rescore.features(Vr, yr, smooth=9)[1])
    log = []

    def f(c):
        fb, U = apex_fb(c, vfb)
        log.append(dict(cpot=c, fb=fb, U_minus_EF=U.tolist()))
        print(f"CPot {c:+.4f} eV: apex flat band {1e3 * fb:+6.1f} meV at V = {1e3 * vfb:+.1f} mV, apex U-E_F "
              f"{np.round(1e3 * U).astype(int).tolist()}", flush=True)
        return fb - vfb

    lo, hi = -0.8, 0.0                                              # f(lo) < 0 < f(hi) expected
    flo, fhi = f(lo), f(hi)
    if not flo < 0 < fhi:
        raise SystemExit(f"bracket failed: f({lo}) = {flo}, f({hi}) = {fhi}")
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.004:
            break
    cpot = 0.5 * (lo + hi)
    json.dump(dict(cpot=cpot, v_fb=vfb, log=log), open(os.path.join(OUT, "place.json"), "w"))
    print(f"placed: CPot {cpot:+.4f} eV", flush=True)
    return cpot


def key(cpot):
    return f"cpot{1e3 * cpot:+07.1f}"


def run(cpot):
    path = os.path.join(OUT, key(cpot) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vr, yr = rescore.measured(TAG)
    fm = rescore.features(Vr, yr, smooth=9)
    tf = S.tipfree(TAG, EPS, 0.0, 0.0, MODEL, "RG")
    U, p = tf["U"], tf["p"]
    free = np.asarray(tunneling_spectrum(p.tb, p.mu0, step1.FINE, np.tile(U, (step1.FINE.size, 1)),
                                         np.zeros((step1.FINE.size, U.size)), **SPEC).didv)
    sim = Simulation(sample(), tip(cpot), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", step1.BIASES, os.path.join(OUT, key(cpot) + ".ckpt.json")).run()
    tipped = np.asarray(m.spectrum(0, bias=step1.FINE, **SPEC).didv)
    out = dict(tag=TAG, eps=EPS, model=MODEL, cpot=cpot, U_free_minus_EF=(U - tf["mu0"]).tolist(), mu0=tf["mu0"],
               failures=len(m.failures), sweep=swt.sweep_arrays(m), measured_features=fm.tolist(),
               measured_asym=rescore.asym(Vr, yr, fm[1]))
    for lab, y in (("free", free), ("tip", tipped)):
        sc, res = swt.shape_distance(Vr, yr, step1.FINE, y)
        out[lab] = dict(didv=y.tolist(), scale=sc, residual=res, **rescore.score(TAG, y, fm))
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    t = out["tip"]
    print(f"CPot {cpot:+.4f}: tip feats {S._mv(t['features'])} (meas {S._mv(fm)}) FB err {1e3 * t['fb_err']:+.1f} "
          f"asym {t['asym']:+.2f} (meas {out['measured_asym']:+.2f}) edges {1e3 * t['edge_rms']:.1f} res {t['residual']:.2f}; "
          f"free feats {S._mv(out['free']['features'])}; {out['seconds']:.0f} s", flush=True)
    return out


def rows():
    return sorted((json.load(open(os.path.join(OUT, f))) for f in os.listdir(OUT)
                   if f.startswith("cpot") and f.endswith(".json") and ".ckpt" not in f), key=lambda r: r["cpot"])


def summary():
    keep = []
    for r in rows():
        t, f = r["tip"], r["free"]
        print(f"CPot {r['cpot']:+.4f} | tip feats {S._mv(t['features'])} FB err {1e3 * t['fb_err']:+5.1f} asym {t['asym']:+.2f} "
              f"edges {1e3 * t['edge_rms']:5.1f} res {t['residual']:.2f} | free feats {S._mv(f['features'])} "
              f"| failures {r['failures']}")
        keep.append({k: v for k, v in r.items() if k != "sweep"} | dict(V=r["sweep"]["bias"], U0=r["sweep"]["U0"]))
    pl = os.path.join(OUT, "place.json")
    json.dump(dict(place=json.load(open(pl)) if os.path.exists(pl) else None, runs=keep),
              open(os.path.join(REPO, "examples", "stacking_wall_cpot940.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "place":
        place()
    elif cmd == "run":
        cps = [float(x) for x in sys.argv[2:]]
        with mp.get_context("spawn").Pool(min(4, len(cps))) as pool:
            list(pool.imap_unordered(run, cps))
    else:
        summary()
