"""Line 940, (+5) side: the polarized stack.  For each interlayer permittivity eps and starting
doping Delta E_F (GrapheneMultilayer.fermi_level), a fixed charge n_sub in the top 2 nm of the
90 nm SiO2 acts on the doped stack; n_sub is the value whose tip-free top-layer spectrum fits the
five measured features best.  Then one self-consistent tip sweep with the tip at zero band
bending at zero bias (CPot = U_1 far from the tip).

    python polar.py WORKER NWORKERS
    python polar.py one EPS DEF          # a single cell, for testing
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stacking_wall_tibb as swt                                    # noqa: E402
from features import features, features5                            # noqa: E402
from bandbending import Gate, GrapheneMultilayer, Layer, Sample, Simulation, Tip, library   # noqa: E402
from bandbending.results.spectroscopy import tunneling_spectrum     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "polar")
SRC = os.path.join(REPO, "examples", "stacking_wall_tibb_940.json")
EPS = [1.0, 1.8, 2.6, 4.0, 6.0, 10.0, 15.0, 25.0]
DEF = [0.040, 0.070, 0.100, 0.130, 0.160, 0.200]
BIASES = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.025), 4)
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
SPEC = dict(modulation=0.00375, barrier=4.6, separation=0.7, broadening=0.01, temperature=9.0)
TRAP = 2.0                                                        # nm of SiO2 holding n_sub


def cells():
    return [(e, d) for d in DEF for e in EPS]


def key(eps, dEF):
    return f"eps{eps:05.1f}_dEF{1e3 * dEF:05.1f}"


st = json.load(open(SRC))
VM, YM = np.asarray(st["measured"]["bias"]), np.asarray(st["measured"]["rg"], dtype=float)
FM5 = features5(VM, YM, smooth=9)
FM4 = features(VM, YM, smooth=9)


def sample(eps, dEF, nsub):
    ml = GrapheneMultilayer(swt.SEQ_RG, name="RG", broadening=0.01, e_window=0.6, fermi_level=dEF, eps_spacer=eps)
    return Sample(surface=ml, layers=(Layer(TRAP, library.SIO2, name="traps", fixed_charge=nsub / (TRAP * 1e-7)),
                                      Layer(90.0 - TRAP, library.SIO2, name="SiO2")), gate=Gate())


def tipfree(eps, dEF, nsub):
    """Tip-free column: (layer potentials U_far (eV), mu0, carriers (cm^-2), tip-free dI/dV on FINE)."""
    r = Simulation(sample(eps, dEF, nsub), None, temperature=9.0).solve(0.0, 0.0)
    p = r.multilayers["RG"]
    U = np.asarray(p.U)[0]
    sp = tunneling_spectrum(p.tb, p.mu0, FINE, np.tile(U, (FINE.size, 1)), np.zeros((FINE.size, U.size)), **SPEC)
    return U, float(p.mu0), float(np.sum(p.n)), np.asarray(p.doping), sp.didv


def rms(f, fm):
    return float(np.sqrt(np.nanmean((f - fm) ** 2))) if np.isfinite(f).sum() >= 3 else 1.0


def place(eps, dEF):
    """n_sub of the best tip-free fit: a coarse scan across the neutral strip, then Brent."""
    _, _, _, doping, _ = tipfree(eps, dEF, 0.0)
    n0 = -1.016 * doping.sum()
    cache = {}

    def cost(ns):
        ns = float(ns)
        if ns not in cache:
            U, mu0, car, _, didv = tipfree(eps, dEF, ns)
            cache[ns] = (rms(features5(FINE, didv), FM5), U, mu0, car, didv)
        return cache[ns][0]

    grid = n0 + np.linspace(-4e10, 4e10, 17)
    vals = [cost(g) for g in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": 2e8})
    best = min(cache, key=lambda k: cache[k][0])
    return best, n0, cache[best]


def run_cell(eps, dEF):
    path = os.path.join(OUT, key(eps, dEF) + ".json")
    if os.path.exists(path):
        return
    t0 = time.time()
    nsub, n0, (r_free, U_far, mu0, carriers, didv_free) = place(eps, dEF)
    t_place = time.time() - t0
    cpot = float(U_far[0])                                         # zero band bending at zero bias
    tip = Tip(radius=20.0, separation=0.7, contact_potential=cpot)
    sim = Simulation(sample(eps, dEF, nsub), tip, temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "RG", BIASES, os.path.join(OUT, key(eps, dEF) + ".ckpt.json")).run()
    sp = m.spectrum(0, bias=FINE, **SPEC)
    f5 = features5(FINE, sp.didv)
    a, res = swt.shape_distance(VM, YM, FINE, sp.didv)
    sw = swt.sweep_arrays(m)
    V, U0, Uf = np.asarray(sw["bias"]), np.asarray(sw["U0"]), np.asarray(sw["U_far"])
    out = dict(eps=eps, dEF=dEF, nsub=nsub, nsub_neutral_guess=n0, carriers=carriers, cpot=cpot, U_far=U_far.tolist(),
               mu0=mu0, seconds=time.time() - t0, seconds_place=t_place, failures=len(m.failures),
               features_free=features5(FINE, didv_free).tolist(), rms_free=r_free, didv_free=didv_free.tolist(),
               features=f5.tolist(), rms=rms(f5, FM5), rms4=rms(np.delete(f5, 2), FM4), n_features=int(np.isfinite(f5).sum()),
               measured_features=FM5.tolist(), scale=a, residual=res, didv=sp.didv.tolist(), sweep=sw,
               bend_at_zero=float(np.interp(0.0, V, U0[:, 0] - Uf[:, 0])))
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"eps {eps:5.1f} dEF {1e3 * dEF:5.1f} meV: n_sub {nsub / 1e13:+.4f}e13 (carriers {carriers / 1e10:+.2f}e10), "
          f"U1-E_F {1e3 * (U_far[0] - mu0):+.1f}, U6-U1 {1e3 * (U_far[-1] - U_far[0]):+.1f} meV; tip-free {np.round(1e3 * np.asarray(out['features_free']))} "
          f"rms {1e3 * r_free:.1f}; under tip {np.round(1e3 * f5)} rms {1e3 * out['rms']:.1f} mV; bend at 0 V {1e3 * out['bend_at_zero']:+.2f} meV; "
          f"{t_place:.0f} + {out['seconds'] - t_place:.0f} s, {len(m.failures)} failures", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "one":
        run_cell(float(sys.argv[2]), float(sys.argv[3]))
    else:
        w, n = int(sys.argv[1]), int(sys.argv[2])
        for i, (e, d) in enumerate(cells()):
            if i % n == w:
                run_cell(e, d)
        print(f"worker {w} done", flush=True)
