"""Fixed oxide charge spread uniformly over the whole 90 nm SiO2 instead of the 2 nm trap layer under the stack
(user, 2026-11-23), at the two-parameter best fit of the 940 (+5) peak positions (CPot -0.615 eV, n_sub -0.5e13 cm^-2 in
the trap layer).  Same mesh as before (2 nm + 88 nm SiO2 layers), now carrying the same volume density.

    python cpot940_bulk.py induced      # tip-free: induced stack charge vs total oxide charge; equivalent density
    python cpot940_bulk.py run          # full sweeps at the start point (n_tot -0.5e13, CPot -0.615) and at the
                                        # equivalent density; CPot placement (flat band at -32.5 mV) there
    python cpot940_bulk.py summary      # table; ../examples/stacking_wall_cpot940_bulk.json
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
import cpot940_fit2 as F2  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

OUT = os.path.join(C.OUT, "bulk")
BEST_C, BEST_N = -0.615, -0.5e13
D_OX = 90.0                                                      # nm


def build_bulk(n_tot):
    """Neutral (+5) stack on 90 nm SiO2 with a uniform fixed charge n_tot (e/cm^2 in total) over the whole oxide."""
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, library
    import stacking_wall_tibb as swt
    ml = GrapheneMultilayer(swt.SEQ_RG, name="ML", broadening=step1.SIGMA[C.TAG], e_window=C.S.E_WINDOW,
                            eps_spacer=C.EPS, params=C.S.params(C.MODEL), doping=0.0)
    rho = n_tot / (D_OX * 1e-7)                                  # e/cm^3
    return Sample(surface=ml, layers=(Layer(step1.TRAP, library.SIO2, name="ox_top", fixed_charge=rho),
                                      Layer(D_OX - step1.TRAP, library.SIO2, name="ox", fixed_charge=rho)), gate=Gate())


def carriers(sample):
    from bandbending import Simulation
    p = Simulation(sample, None, temperature=9.0).solve(0.0, 0.0).multilayers["ML"]
    return float(np.sum(p.n)), (np.asarray(p.U)[0] - p.mu0)


def induced():
    path = os.path.join(OUT, "induced.json")
    if os.path.exists(path):
        return json.load(open(path))
    c_trap, U_trap = carriers(C.S.build(C.TAG, C.EPS, 0.0, BEST_N, C.MODEL, "RG"))
    c_start, _ = carriers(build_bulk(BEST_N))
    lo, hi = 1.5 * BEST_N, 2.5 * BEST_N                          # n_tot bracket (both negative)
    f = lambda n: carriers(build_bulk(n))[0] - c_trap            # noqa: E731
    flo, fhi = f(lo), f(hi)
    log = [(lo, flo), (hi, fhi)]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        log.append((mid, fm))
        if abs(fm) < 1e8:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    c_eq, U_eq = carriers(build_bulk(mid))
    out = dict(trap=dict(n=BEST_N, carriers=c_trap, U=U_trap.tolist()), start=dict(n=BEST_N, carriers=c_start),
               equivalent=dict(n=mid, carriers=c_eq, U=U_eq.tolist()), log=log)
    json.dump(out, open(path, "w"))
    print(f"trap layer n_sub {BEST_N:.3e}: stack carriers {c_trap:+.4e} cm^-2 (fraction {c_trap / BEST_N:.3f}); "
          f"uniform n_tot {BEST_N:.3e}: {c_start:+.4e} ({c_start / BEST_N:.3f}); equivalent uniform n_tot {mid:.4e} "
          f"({mid / (D_OX * 1e-7):.3e} cm^-3): {c_eq:+.4e}; U-E_F trap {np.round(1e3 * U_trap, 2)}, "
          f"uniform {np.round(1e3 * U_eq, 2)} meV", flush=True)
    return out


def pkey(c, n):
    return f"bulk_c{1e3 * c:+07.1f}_n{n / 1e13:+07.4f}"


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
    sim = Simulation(build_bulk(n), C.tip(c), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", F2.BIAS16, os.path.join(OUT, pkey(c, n) + ".ckpt.json")).run()
    y = np.asarray(m.spectrum(0, bias=step1.FINE, **C.SPEC).didv)
    sc, res = swt.shape_distance(Vr, yr, step1.FINE, y)
    s = rescore.score(C.TAG, y, fm)
    err = np.asarray(s["err"], dtype=float)
    out = dict(cpot=c, nsub=n, failures=len(m.failures), sweep=swt.sweep_arrays(m), measured_features=fm.tolist(),
               didv=y.tolist(), scale=sc, residual=res, rms4=float(np.sqrt(np.nanmean(err ** 2))),
               seconds=time.time() - t0, **s)
    json.dump(out, open(path, "w"))
    F2.line(out)
    return out


def place(n):
    path = os.path.join(OUT, f"place_n{n / 1e13:+07.4f}.json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation
    Vr, yr = rescore.measured(C.TAG)
    vfb = float(rescore.features(Vr, yr, smooth=9)[1])
    log = []

    def f(c):
        r = Simulation(build_bulk(n), C.tip(c), temperature=9.0, grid="draft").solve(vfb, 0.0)
        p = r.multilayers["ML"]
        U = np.asarray(p.U)[0]
        _, D = p.tb.ldos(U, C.E + p.mu0, broadening=C.SIG, resolve="layer")
        D = np.asarray(D)
        D = D if D.shape[0] == C.E.size else D.T
        fb = float(C.E[np.argmax(D[:, 0])])
        log.append(dict(cpot=c, fb=fb))
        print(f"  bulk n {n / 1e13:+.4f}: CPot {c:+.4f} -> apex FB {1e3 * fb:+.1f} meV", flush=True)
        return fb - vfb

    a, b = BEST_C - 0.08, BEST_C + 0.08
    fa, fb_ = f(a), f(b)
    side = 0
    for _ in range(12):
        if not fa < 0 < fb_:
            break
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
    out = dict(nsub=n, cpot=best["cpot"], fb=best["fb"], v_fb=vfb, log=log)
    json.dump(out, open(path, "w"))
    print(f"placed bulk n {n / 1e13:+.4f}: CPot {best['cpot']:+.4f} eV (apex FB {1e3 * best['fb']:+.1f} meV)", flush=True)
    return out


def task(t):
    return place(t[1]) if t[0] == "place" else sweep(t[1])


def rows():
    return [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT))
            if f.startswith("bulk_") and f.endswith(".json") and not f.endswith(".ckpt.json")]


def summary():
    ind = induced()
    for r in rows():
        F2.line(r)
    pl = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.startswith("place_")]
    for p in pl:
        print(f"placement: bulk n {p['nsub'] / 1e13:+.4f}e13 -> CPot {p['cpot']:+.4f} eV")
    json.dump(dict(induced=ind, placements=pl, sweeps=[{k: v for k, v in r.items() if k not in ("sweep", "didv")}
                                                       for r in rows()]),
              open(os.path.join(REPO, "examples", "stacking_wall_cpot940_bulk.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "induced":
        induced()
    elif sys.argv[1] == "run":
        n_eq = induced()["equivalent"]["n"]
        jobs = [("sweep", (BEST_C, BEST_N)), ("sweep", (BEST_C, n_eq)), ("place", n_eq)]
        with mp.get_context("spawn").Pool(3) as pool:
            list(pool.imap_unordered(task, jobs))
        summary()
    else:
        summary()
