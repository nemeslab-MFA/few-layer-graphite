"""Two-parameter fit of the 940 (+5) peak positions: tip contact potential CPot and SiO2 trap charge n_sub, everything
else fixed (neutral stack, eps 2.6, full SWMcC, tip R 20 nm, d 0.7 nm, sigma 12.5 meV, T 9 K).  Objective: rms error
of the four feature positions (lower bulk edge, flat band, upper edges 1 and 2) against the measurement; peak shapes
are ignored.

Stage 1 (valley): for each n_sub, the CPot that puts the apex flat band at the measured bias (single solves, Illinois
regula falsi).  Stage 2: full bias sweeps on a 16-point grid at the valley points -> features -> rms.  Stage 3: local
points chosen from stage 2 (``point CPOT:NSUB``).

    python cpot940_fit2.py valley [N ...]    # stages 1 and 2 (4 processes); N in 1e13 cm^-2, default NSUB
    python cpot940_fit2.py point C:N [...]   # extra full sweeps
    python cpot940_fit2.py summary           # table; ../examples/stacking_wall_cpot940_fit2.json
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

OUT = os.path.join(C.OUT, "fit2")
NSUB = (0.25e13, 0.0, -0.125e13, -0.25e13, -0.375e13, -0.5e13, -0.75e13)
BIAS16 = (-0.45, -0.35, -0.30, -0.25, -0.20, -0.10, -0.06, -0.03, 0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
LO, HI = (-1.2, 0.0) if C.TAG == "940" else (-0.6, 1.2)          # CPot bracket (eV)


def sample(n):
    return C.S.build(C.TAG, C.EPS, 0.0, n, C.MODEL, "RG")


def apex_fb(cpot, n, bias):
    from bandbending import Simulation
    r = Simulation(sample(n), C.tip(cpot), temperature=9.0, grid="draft").solve(float(bias), 0.0)
    p = r.multilayers["ML"]
    U = np.asarray(p.U)[0]
    _, D = p.tb.ldos(U, C.E + p.mu0, broadening=C.SIG, resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == C.E.size else D.T
    return float(C.E[np.argmax(D[:, 0])]), U - p.mu0


def nkey(n):
    return f"n{n / 1e13:+06.3f}"


def place(n):
    path = os.path.join(OUT, "place_" + nkey(n) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    Vr, yr = rescore.measured(C.TAG)
    vfb = float(rescore.features(Vr, yr, smooth=9)[1])
    log = []

    def f(c):
        fb, U = apex_fb(c, n, vfb)
        log.append(dict(cpot=c, fb=fb, U_minus_EF=U.tolist()))
        print(f"  {nkey(n)}: CPot {c:+.4f} -> apex FB {1e3 * fb:+6.1f} meV (target {1e3 * vfb:+.1f})", flush=True)
        return fb - vfb

    a, b = LO, HI
    fa, fb_ = f(a), f(b)
    if not fa < 0 < fb_:
        out = dict(nsub=n, cpot=None, v_fb=vfb, log=log, note=f"no bracket: f({a}) = {fa}, f({b}) = {fb_}")
        json.dump(out, open(path, "w"))
        return out
    side = 0
    for _ in range(20):                                           # Illinois regula falsi
        c = (a * fb_ - b * fa) / (fb_ - fa)
        fc = f(c)
        if abs(fc) <= 0.0006 or b - a < 0.003:
            break
        if fc < 0:
            a, fa = c, fc
            if side == -1:
                fb_ *= 0.5
            side = -1
        else:
            b, fb_ = c, fc
            if side == 1:
                fa *= 0.5
            side = 1
    best = min(log, key=lambda x: abs(x["fb"] - vfb))
    out = dict(nsub=n, cpot=best["cpot"], v_fb=vfb, log=log)
    json.dump(out, open(path, "w"))
    print(f"placed {nkey(n)}: CPot {best['cpot']:+.4f} eV (apex FB {1e3 * best['fb']:+.1f} meV)", flush=True)
    return out


def pkey(c, n):
    return f"sweep_c{1e3 * c:+07.1f}_{nkey(n)}"


def sweep(cn):
    c, n = cn
    path = os.path.join(OUT, pkey(c, n) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vr, yr = rescore.measured(C.TAG)
    fm = rescore.features(Vr, yr, smooth=9)
    sim = Simulation(sample(n), C.tip(c), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", BIAS16, os.path.join(OUT, pkey(c, n) + ".ckpt.json")).run()
    y = np.asarray(m.spectrum(0, bias=step1.FINE, **C.SPEC).didv)
    sc, res = swt.shape_distance(Vr, yr, step1.FINE, y)
    s = rescore.score(C.TAG, y, fm)
    err = np.asarray(s["err"], dtype=float)
    out = dict(cpot=c, nsub=n, failures=len(m.failures), sweep=swt.sweep_arrays(m), measured_features=fm.tolist(),
               didv=y.tolist(), scale=sc, residual=res, rms4=float(np.sqrt(np.nanmean(err ** 2))),
               seconds=time.time() - t0, **s)                    # s carries 'missing' (features not found)
    json.dump(out, open(path, "w"))
    line(out)
    return out


def line(r):
    print(f"CPot {r['cpot']:+.4f} eV, n_sub {r['nsub'] / 1e13:+.3f}e13 | feats {C.S._mv(r['features'])} "
          f"(meas {C.S._mv(r['measured_features'])}) | err {C.S._mv(r['err'])} | rms4 {1e3 * r['rms4']:5.1f} mV, "
          f"edges {1e3 * r['edge_rms']:5.1f}, missing {r['missing']} | failures {r['failures']}"
          + (f"; {r['seconds']:.0f} s" if "seconds" in r else ""), flush=True)


def rows():
    return [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT))
            if f.startswith("sweep_") and f.endswith(".json") and not f.endswith(".ckpt.json")]


def summary():
    rs = sorted(rows(), key=lambda r: (r["nsub"], r["cpot"]))
    for r in rs:
        line(r)
    if rs:
        b = min(rs, key=lambda r: r["rms4"] + 1.0 * r["missing"])
        print("best:", end=" ")
        line(b)
    pl = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.startswith("place_")]
    json.dump(dict(valley=[{k: v for k, v in p.items() if k != "log"} for p in pl],
                   sweeps=[{k: v for k, v in r.items() if k not in ("sweep", "didv")} for r in rs]),
              open(os.path.join(REPO, "examples", f"stacking_wall_cpot{C.TAG}_fit2.json"), "w"))


def valley(nsub=NSUB):
    with mp.get_context("spawn").Pool(4) as pool:
        placed = pool.map(place, nsub)
        jobs = [(p["cpot"], p["nsub"]) for p in placed if p["cpot"] is not None]
        for p in placed:
            if p["cpot"] is None:
                print(f"{nkey(p['nsub'])}: {p['note']}", flush=True)
        list(pool.imap_unordered(sweep, jobs))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "valley":
        valley(tuple(float(a) * 1e13 for a in sys.argv[2:]) or NSUB)
    elif sys.argv[1] == "point":
        pts = [tuple(float(x) for x in a.split(":")) for a in sys.argv[2:]]
        with mp.get_context("spawn").Pool(min(4, len(pts))) as pool:
            list(pool.imap_unordered(sweep, [(c, n * 1e13) for c, n in pts]))
    summary()
