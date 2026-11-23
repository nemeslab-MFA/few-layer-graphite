"""Substrate-charge sweep on the CPot-only 940 (+5) fit: tip R 20 nm, d 0.7 nm, CPot -0.367 eV, neutral stack (no
Fermi-level offset), eps 2.6, full SWMcC; fixed sheet charge n_sub (e/cm^2, sign of the charge) in the 2 nm SiO2 trap
layer under the stack, 88 nm SiO2 to the grounded gate.  n_sub = 0 is the fit itself (its checkpoint is reused).

    python cpot940_nsub.py run       # full self-consistent bias sweeps (4 processes): cpot940/nsub*.json
    python cpot940_nsub.py summary   # table; ../examples/stacking_wall_cpot940_nsub.json
"""
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402
import cpot940_geom_fine as GF  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

CPOT = -0.367
NSUB = (-2e13, -1.5e13, -1e13, -0.5e13, 0.0, 0.5e13, 1e13, 1.5e13, 2e13)   # +-3e13 puts L6 at the 0.5 eV TB window


def peak(V, y):
    try:
        return GF.peak(V, y, lo=-0.20, hi=0.15)
    except (IndexError, ValueError):
        return None


def key(n):
    return f"nsub{n / 1e13:+05.2f}e13"


def run(n):
    path = os.path.join(C.OUT, key(n) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vr, yr = rescore.measured(C.TAG)
    fm = rescore.features(Vr, yr, smooth=9)
    ck = os.path.join(C.OUT, key(n) + ".ckpt.json")
    if n == 0.0 and not os.path.exists(ck):
        shutil.copy(os.path.join(C.OUT, C.key(CPOT) + ".ckpt.json"), ck)
    tf = C.S.tipfree(C.TAG, C.EPS, 0.0, n, C.MODEL, "RG")
    U, p = tf["U"], tf["p"]
    F = step1.FINE
    free = np.asarray(tunneling_spectrum(p.tb, p.mu0, F, np.tile(U, (F.size, 1)), np.zeros((F.size, U.size)),
                                         **C.SPEC).didv)
    sim = Simulation(C.S.build(C.TAG, C.EPS, 0.0, n, C.MODEL, "RG"), C.tip(CPOT), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", step1.BIASES, ck).run()
    tipped = np.asarray(m.spectrum(0, bias=F, **C.SPEC).didv)
    out = dict(nsub=n, cpot=CPOT, failures=len(m.failures), sweep=swt.sweep_arrays(m), mu0_free=tf["mu0"],
               U_free_minus_EF=(U - tf["mu0"]).tolist(), carriers_free=tf["carriers"], n_free=tf["n"].tolist(),
               measured_features=fm.tolist(), measured_peak=peak(Vr, yr))
    for lab, y in (("free", free), ("tip", tipped)):
        sc, res = swt.shape_distance(Vr, yr, F, y)
        out[lab] = dict(didv=y.tolist(), scale=sc, residual=res, peak=peak(F, y),
                        **rescore.score(C.TAG, y, fm))
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    line(out)
    return out


def line(r):
    t, f = r["tip"], r["free"]
    pk = lambda q: (f"FB {1e3 * q['v']:+6.1f} mV, HWHM ratio {q['ratio']:.2f}" if q else "FB ?")  # noqa: E731
    print(f"n_sub {r['nsub'] / 1e13:+5.2f}e13 | tip: {pk(t['peak'])}, "
          f"feats {C.S._mv(t['features'])}, edges {1e3 * t['edge_rms']:5.1f}, res {t['residual']:.2f} | no tip: "
          f"{pk(f['peak'])}, feats {C.S._mv(f['features'])}, stack carriers {r['carriers_free']:+.2e} | "
          f"failures {r['failures']}" + (f"; {r['seconds']:.0f} s" if "seconds" in r else ""), flush=True)


def rows():
    return [json.load(open(os.path.join(C.OUT, key(n) + ".json"))) for n in NSUB
            if os.path.exists(os.path.join(C.OUT, key(n) + ".json"))]


def summary():
    rs = rows()
    print(f"measured features {C.S._mv(rs[0]['measured_features'])}")
    for r in rs:
        line(r)
    json.dump([{k: v for k, v in r.items() if k != "sweep"} | dict(V=r["sweep"]["bias"], U0=r["sweep"]["U0"],
                                                                     mu0=r["sweep"]["mu0"]) for r in rs],
              open(os.path.join(REPO, "examples", "stacking_wall_cpot940_nsub.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(C.OUT, exist_ok=True)
    if sys.argv[1] == "run":
        jobs = sorted(NSUB, key=lambda n: (n != 0.0, abs(n)))
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(run, jobs))
    else:
        summary()
