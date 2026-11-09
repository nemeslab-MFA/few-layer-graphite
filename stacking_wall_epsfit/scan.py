"""Line 940, (+5) side: self-consistent tip sweeps over interlayer permittivity x starting doping.

Model as in stacking_wall_tibb.py for the (+5) stack (CBACBA on 90 nm SiO2 over a grounded
gate, tip R = 20 nm, d = 0.7 nm, CPot = 0, 9 K, 10 meV broadening), with eps_spacer and
fermi_level (Delta E_F, the starting doping) free.  One JSON per cell in OUT; checkpointed
per bias point.

    python scan.py WORKER NWORKERS        # run this worker's share of the grid
    python scan.py collect                # print the table of the finished cells
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stacking_wall_tibb as swt                                    # noqa: E402
from features import features                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cells")
SRC = os.path.join(REPO, "examples", "stacking_wall_tibb_940.json")
EPS = [1.0, 1.8, 2.6, 4.0, 6.0, 10.0, 15.0, 25.0]
DEF = [0.010, 0.020, 0.030, 0.040, 0.055, 0.070]
BIASES = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.025), 4)
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
SPEC = dict(modulation=0.00375, barrier=4.6, separation=0.7, broadening=0.01, temperature=9.0)
GRID = "draft"


def cells():
    return [(e, d) for d in DEF for e in EPS]


def key(eps, dEF):
    return f"eps{eps:05.1f}_dEF{1e3 * dEF:05.1f}"


def measured():
    st = json.load(open(SRC))
    return np.asarray(st["measured"]["bias"]), np.asarray(st["measured"]["rg"], dtype=float)


def run_cell(eps, dEF):
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, Simulation, Tip, library
    path = os.path.join(OUT, key(eps, dEF) + ".json")
    if os.path.exists(path):
        return
    tip = Tip(radius=20.0, separation=0.7, contact_potential=0.0)
    ml = GrapheneMultilayer(swt.SEQ_RG, name="RG", broadening=0.01, e_window=0.6, fermi_level=dEF, eps_spacer=eps)
    sample = Sample(surface=ml, layers=(Layer(90.0, library.SIO2, name="SiO2"),), gate=Gate())
    sim = Simulation(sample, tip, temperature=9.0, grid=GRID)
    t0 = time.time()
    m = swt.CheckpointedSweep(sim, "RG", BIASES, os.path.join(OUT, key(eps, dEF) + ".ckpt.json")).run()
    sp = m.spectrum(0, bias=FINE, **SPEC)
    Vm, ym = measured()
    fm = features(Vm, ym, smooth=9)
    fs = features(FINE, sp.didv)
    a, res = swt.shape_distance(Vm, ym, FINE, sp.didv)
    out = dict(eps=eps, dEF=dEF, seconds=time.time() - t0, failures=len(m.failures), features=fs.tolist(),
               measured_features=fm.tolist(), rms=float(np.sqrt(np.nanmean((fs - fm) ** 2))),
               n_features=int(np.isfinite(fs).sum()), scale=a, residual=res, didv=sp.didv.tolist(),
               didv_frozen=sp.didv_frozen.tolist(), sweep=swt.sweep_arrays(m))
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"eps {eps:5.1f}  dEF {1e3 * dEF:5.1f} meV: features {np.round(1e3 * fs)} mV, rms {1e3 * out['rms']:.1f} mV, "
          f"shape residual {res:.3f}, {out['seconds']:.0f} s, {len(m.failures)} failures", flush=True)


def collect():
    rows = []
    for e, d in cells():
        p = os.path.join(OUT, key(e, d) + ".json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "collect":
        rows = sorted(collect(), key=lambda r: r["rms"])
        Vm, ym = measured()
        print("measured", np.round(1e3 * features(Vm, ym, smooth=9)))
        for r in rows:
            print(f"eps {r['eps']:5.1f} dEF {1e3 * r['dEF']:5.1f}: rms {1e3 * r['rms']:5.1f}  {np.round(1e3 * np.array(r['features']))}"
                  f"  residual {r['residual']:.3f}")
        print(len(rows), "of", len(cells()), "cells")
    else:
        w, n = int(sys.argv[1]), int(sys.argv[2])
        for i, (e, d) in enumerate(cells()):
            if i % n == w:
                run_cell(e, d)
        print(f"worker {w} done", flush=True)
