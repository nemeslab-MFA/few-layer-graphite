"""Line 940, (+5) side: the polarized stack at a fixed interlayer permittivity (eps = 2.6, the
sigma-band value), with the starting doping Delta E_F and the SiO2 charge n_sub as the two fit
parameters, under the tip.

n_sub is set through the carrier density of the stack it leaves: for every Delta E_F the neutral
point x_n of the net fixed charge x = n_sub + sum(doping) is found by bisection on the tip-free
carriers, and n_sub = -sum(doping) + x_n + c with c the target carrier density (negative: holes).
Inside the neutral strip E_F moves through the gap at a few meV per 1e9 cm^-2; outside it the stack
carries c.  The tip leaves the bands flat at zero bias (CPot = U_1 far from the tip).  Five features
as in polar.py; 23 biases per sweep (25 mV steps within 75 mV of zero, 50 mV beyond).

    python fixeps.py WORKER NWORKERS          # the grid
    python fixeps.py one DEF C                # one cell (Delta E_F in eV, C in cm^-2)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import polar                                                         # noqa: E402
from features import features5                                      # noqa: E402

swt = polar.swt
EPS = 2.6
OUT = os.path.join(HERE, "fixeps")
DEF = [0.040, 0.060, 0.080, 0.100, 0.130]
CARRIERS = [-60e10, -25e10, -10e10, -4e10, -1.5e10]
BIASES = np.unique(np.round(np.concatenate([np.arange(-0.45, 0.45 + 1e-9, 0.05), [-0.075, -0.025, 0.025, 0.075]]), 4))


def cells():
    return [(d, c) for d in DEF for c in CARRIERS]


def cells2():
    """Second round: more holes, and the dopings between the best rows of the first."""
    return ([(d, c) for d in (0.040, 0.050, 0.060, 0.070) for c in (-100e10, -150e10)]
            + [(d, c) for d in (0.050, 0.070) for c in (-25e10, -60e10)])


def key(dEF, c):
    return f"dEF{1e3 * dEF:05.1f}_c{c / 1e10:+06.1f}"


def neutral_point(dEF):
    """(sum of the layer doping, net fixed charge x_n at which the tip-free stack is neutral)."""
    _, _, _, doping, _ = polar.tipfree(EPS, dEF, 0.0)
    D = float(doping.sum())
    lo, hi = -0.05 * D, 0.02 * D                                     # carriers(lo) < 0 < carriers(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        car = polar.tipfree(EPS, dEF, -D + mid)[2]
        if car < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e8:
            break
    return D, 0.5 * (lo + hi)


def run_cell(dEF, c):
    path = os.path.join(OUT, key(dEF, c) + ".json")
    if os.path.exists(path):
        return
    t0 = time.time()
    D, xn = neutral_point(dEF)
    nsub = -D + xn + c
    U_far, mu0, carriers, _, didv_free = polar.tipfree(EPS, dEF, nsub)
    cpot = float(U_far[0])
    tip = polar.Tip(radius=20.0, separation=0.7, contact_potential=cpot)
    sim = polar.Simulation(polar.sample(EPS, dEF, nsub), tip, temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "RG", BIASES, os.path.join(OUT, key(dEF, c) + ".ckpt.json")).run()
    sp = m.spectrum(0, bias=polar.FINE, **polar.SPEC)
    f5 = features5(polar.FINE, sp.didv)
    ff = features5(polar.FINE, didv_free)
    a, res = swt.shape_distance(polar.VM, polar.YM, polar.FINE, sp.didv)
    sw = swt.sweep_arrays(m)
    V, U0, Uf = np.asarray(sw["bias"]), np.asarray(sw["U0"]), np.asarray(sw["U_far"])
    out = dict(eps=EPS, dEF=dEF, carriers_target=c, carriers=carriers, nsub=nsub, doping_sum=D, x_neutral=xn, cpot=cpot,
               U_far=U_far.tolist(), mu0=mu0, features=f5.tolist(), rms=polar.rms(f5, polar.FM5),
               n_features=int(np.isfinite(f5).sum()), features_free=ff.tolist(), rms_free=polar.rms(ff, polar.FM5),
               didv=sp.didv.tolist(), didv_free=didv_free.tolist(), scale=a, residual=res, sweep=sw,
               measured_features=polar.FM5.tolist(), failures=len(m.failures), seconds=time.time() - t0,
               bend_at_zero=float(np.interp(0.0, V, U0[:, 0] - Uf[:, 0])))
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"dEF {1e3 * dEF:5.1f} meV, holes {-c / 1e10:5.1f}e10: n_sub {nsub / 1e13:+.4f}e13 (carriers {carriers / 1e10:+.2f}e10), "
          f"U1-E_F {1e3 * (U_far[0] - mu0):+.1f}, U6-U1 {1e3 * (U_far[-1] - U_far[0]):+.1f} meV; tip-free rms {1e3 * out['rms_free']:.1f}; "
          f"under tip {np.round(1e3 * f5)} rms {1e3 * out['rms']:.1f} mV; {out['seconds']:.0f} s, {len(m.failures)} failures", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "one":
        run_cell(float(sys.argv[2]), float(sys.argv[3]))
    else:
        w, n = int(sys.argv[1]), int(sys.argv[2])
        todo = cells2() if len(sys.argv) > 3 and sys.argv[3] == "round2" else cells()
        for i, (d, c) in enumerate(todo):
            if i % n == w:
                run_cell(d, c)
        print(f"worker {w} done", flush=True)
