"""Tip geometry on the CPot-only fit of the 940 (+5) flat band (cpot940.py): neutral stack, eps 2.6, full SWMcC,
CPot fixed at the fitted -0.367 eV; tip-sample distance d = 0.4-1.0 nm at R = 20 nm and apex radius R = 5-80 nm at
d = 0.7 nm.  The WKB barrier width of the simulated spectrum follows d.

    python cpot940_geom.py run       # full self-consistent bias sweeps (4 processes): cpot940/geom_*.json
    python cpot940_geom.py summary   # table; ../examples/stacking_wall_cpot940_geom.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

CPOT = -0.367
D_SWEEP = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)                    # at R = 20 nm
R_SWEEP = (5.0, 10.0, 20.0, 40.0, 80.0)                          # at d = 0.7 nm
JOBS = sorted({(d, 20.0) for d in D_SWEEP} | {(0.7, r) for r in R_SWEEP}, key=lambda j: (j[0] != 0.7, j))


def key(d, R):
    return f"geom_d{d:.2f}_R{R:05.1f}"


def peak(V, y, lo=-0.15, hi=0.06):
    """Peak bias, height above the local floor, half widths at half maximum on the outer (negative) and inner side."""
    w = (V >= lo) & (V <= hi)
    v, z = V[w], y[w]
    k = int(np.argmax(z))
    floor = min(z[:k].min(), z[k:].min())
    half = floor + 0.5 * (z[k] - floor)
    left = v[:k][np.where(z[:k] < half)[0][-1]] if np.any(z[:k] < half) else np.nan
    right = v[k:][np.where(z[k:] < half)[0][0]] if np.any(z[k:] < half) else np.nan
    return dict(v=float(v[k]), height=float(z[k]), floor=float(floor), hwhm_out=float(v[k] - left),
                hwhm_in=float(right - v[k]))


def run(job):
    d, R = job
    path = os.path.join(C.OUT, key(d, R) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation, Tip
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vr, yr = rescore.measured(C.TAG)
    fm = rescore.features(Vr, yr, smooth=9)
    spec = dict(C.SPEC, separation=d)
    sim = Simulation(C.sample(), Tip(radius=R, separation=d, contact_potential=CPOT), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", step1.BIASES, os.path.join(C.OUT, key(d, R) + ".ckpt.json")).run()
    y = np.asarray(m.spectrum(0, bias=step1.FINE, **spec).didv)
    sc, res = swt.shape_distance(Vr, yr, step1.FINE, y)
    out = dict(d=d, R=R, cpot=CPOT, failures=len(m.failures), sweep=swt.sweep_arrays(m),
               tip=dict(didv=y.tolist(), scale=sc, residual=res, peak=peak(step1.FINE, y), **rescore.score(C.TAG, y, fm)),
               measured_features=fm.tolist(), measured_peak=peak(Vr, yr), seconds=time.time() - t0)
    json.dump(out, open(path, "w"))
    line(out)
    return out


def line(r):
    t, p = r["tip"], r["tip"]["peak"]
    print(f"d {r['d']:.2f} nm R {r['R']:5.1f} nm | FB {1e3 * p['v']:+6.1f} mV, HWHM out {1e3 * p['hwhm_out']:5.1f} in "
          f"{1e3 * p['hwhm_in']:5.1f} (ratio {p['hwhm_out'] / p['hwhm_in']:.2f}) asym {t['asym']:+.2f} | feats "
          f"{C.S._mv(t['features'])} edges {1e3 * t['edge_rms']:5.1f} res {t['residual']:.2f} | failures {r['failures']}"
          + (f"; {r['seconds']:.0f} s" if "seconds" in r else ""), flush=True)


def rows():
    return [json.load(open(os.path.join(C.OUT, key(*j) + ".json"))) for j in JOBS
            if os.path.exists(os.path.join(C.OUT, key(*j) + ".json"))]


def summary():
    rs = rows()
    mp_ = rs[0]["measured_peak"]
    print(f"measured: FB {1e3 * mp_['v']:+.1f} mV, HWHM out {1e3 * mp_['hwhm_out']:.1f} in {1e3 * mp_['hwhm_in']:.1f} "
          f"(ratio {mp_['hwhm_out'] / mp_['hwhm_in']:.2f}), features {C.S._mv(rs[0]['measured_features'])}")
    for r in sorted(rs, key=lambda r: (r["R"], r["d"])):
        line(r)
    json.dump([{k: v for k, v in r.items() if k != "sweep"} | dict(V=r["sweep"]["bias"], U0=r["sweep"]["U0"],
                                                                     mu0=r["sweep"]["mu0"]) for r in rs],
              open(os.path.join(REPO, "examples", "stacking_wall_cpot940_geom.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(C.OUT, exist_ok=True)
    if sys.argv[1] == "run":
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(run, JOBS))
    else:
        summary()
