"""Line 940, (+5) side, eps = 2.6: the polarized stack with the tip work function as a third fit
parameter, next to the starting doping Delta E_F and the SiO2 charge n_sub (through the hole density
it leaves in the stack, as in fixeps.py).

The tip enters as the offset delta of its contact potential from the value that leaves the bands flat
at zero bias, CPot = U_1(far) + delta: delta > 0 is a tip work function above the sample's, and the bias
of zero band bending is -delta.  Compass search from the flat-band fit of fixeps.py (Delta E_F = 60 meV,
1e11 holes as targeted there, delta = 0): at every step the six points +-step along Delta E_F, log10(holes) and delta are
solved in parallel; the search moves to the best of them if it improves the feature rms, and halves the
steps otherwise.  Objective and validity as in analyze_fixeps.py (five features; the lower side-band peak
below -200 mV and at least half as prominent as the measured one).

    python fit_cpot.py search           # 4 processes; log in fitcpot/log.jsonl, cells in fitcpot/
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fitcpot")
START = (60.0, 11.0, 0.0)                       # Delta E_F (meV), log10(holes / cm^-2), delta (meV): the fixeps cell
STEP0 = (10.0, 0.25, 10.0)
STEP_MIN = (2.5, 0.06, 2.5)
BOUNDS = ((20.0, 160.0), (9.5, 12.5), (-150.0, 150.0))
NPROC = 4


def key(p):
    return f"dEF{p[0]:06.2f}_lh{p[1]:05.3f}_d{p[2]:+07.2f}"


def evaluate(p):
    """Feature rms (eV) of one parameter point, with the cell written to fitcpot/KEY.json."""
    sys.path.insert(0, HERE)
    import fixeps
    from analyze_fixeps_lib import low_prominence, PROM_M
    from features import features5
    polar, swt = fixeps.polar, fixeps.swt
    path = os.path.join(OUT, key(p) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    dEF, holes, delta = p[0] / 1e3, 10.0 ** p[1], p[2] / 1e3
    D, xn = fixeps.neutral_point(dEF)
    nsub = -D + xn - holes
    U_far, mu0, carriers, _, didv_free = polar.tipfree(fixeps.EPS, dEF, nsub)
    cpot = float(U_far[0]) + delta
    tip = polar.Tip(radius=20.0, separation=0.7, contact_potential=cpot)
    sim = polar.Simulation(polar.sample(fixeps.EPS, dEF, nsub), tip, temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "RG", fixeps.BIASES, os.path.join(OUT, key(p) + ".ckpt.json")).run()
    sp = m.spectrum(0, bias=polar.FINE, **polar.SPEC)
    f = features5(polar.FINE, sp.didv)
    prom = low_prominence(polar.FINE, sp.didv, f[0]) if np.isfinite(f[0]) else float("nan")
    ok = bool(np.all(np.isfinite(f)) and np.all(np.diff(f) > 0) and f[1] > -0.115 and f[0] < -0.2 and prom >= 0.5 * PROM_M)
    rms = float(np.sqrt(np.mean((f - polar.FM5) ** 2))) if ok else 1.0
    a, res = swt.shape_distance(polar.VM, polar.YM, polar.FINE, sp.didv)
    sw = swt.sweep_arrays(m)
    V, U0, Uf = np.asarray(sw["bias"]), np.asarray(sw["U0"]), np.asarray(sw["U_far"])
    out = dict(p=list(p), dEF=dEF, holes=holes, delta=delta, carriers=carriers, nsub=nsub, cpot=cpot, U_far=U_far.tolist(),
               mu0=mu0, features=f.tolist(), prominence=prom, valid=ok, rms=rms, residual=res, scale=a,
               didv=sp.didv.tolist(), didv_free=didv_free.tolist(), sweep=sw, failures=len(m.failures),
               bend_at_zero=float(np.interp(0.0, V, U0[:, 0] - Uf[:, 0])), seconds=time.time() - t0)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def clip(p):
    return tuple(float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(p, BOUNDS))


def search():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "log.jsonl"), "a")
    ctx = mp.get_context("spawn")
    with ctx.Pool(NPROC) as pool:
        x = START
        fx = pool.apply(evaluate, (x,))
        step = list(STEP0)
        print(f"start {key(x)}: rms {1e3 * fx['rms']:.1f} mV, features {np.round(1e3 * np.asarray(fx['features']))}", flush=True)
        log.write(json.dumps(dict(event="start", p=list(x), rms=fx["rms"])) + "\n")
        log.flush()
        for it in range(20):
            trial = []
            for k in range(3):
                for sgn in (+1, -1):
                    q = list(x)
                    q[k] += sgn * step[k]
                    q = clip(q)
                    if q != x and q not in trial:
                        trial.append(q)
            res = pool.map(evaluate, trial)
            for q, r in zip(trial, res):
                log.write(json.dumps(dict(event="eval", it=it, p=list(q), rms=r["rms"], valid=r["valid"],
                                          features=r["features"], residual=r["residual"])) + "\n")
            log.flush()
            ib = int(np.argmin([r["rms"] for r in res]))
            if res[ib]["rms"] < fx["rms"] - 1e-4:
                x, fx = trial[ib], res[ib]
                print(f"it {it}: move to {key(x)}: rms {1e3 * fx['rms']:.1f} mV, features {np.round(1e3 * np.asarray(fx['features']))}, "
                      f"steps {step}", flush=True)
            else:
                step = [s / 2 for s in step]
                print(f"it {it}: no improvement around {key(x)} (best neighbour {1e3 * res[ib]['rms']:.1f} mV); steps -> {step}",
                      flush=True)
                if all(s < sm for s, sm in zip(step, STEP_MIN)):
                    break
            log.write(json.dumps(dict(event="state", it=it, p=list(x), rms=fx["rms"], step=step)) + "\n")
            log.flush()
    print(f"best {key(x)}: rms {1e3 * fx['rms']:.1f} mV", flush=True)
    log.write(json.dumps(dict(event="done", p=list(x), rms=fx["rms"])) + "\n")
    log.close()


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    if sys.argv[1] == "search":
        search()
