"""Convergence of the flat-band peak shape in the sweep grid: cpot940_geom.py cases re-solved with 5 mV bias steps
from -120 to +20 mV added to the 25 mV sweep (the apex potentials between solved biases are PCHIP-interpolated, and
the peak asymmetry is set by how the lever arm changes across the peak).  Starts from a copy of the 25 mV checkpoint.

    python cpot940_geom_dense.py D:R [D:R ...]     # cpot940/dense_*.json; table
"""
import json
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402
import cpot940_geom as G  # noqa: E402
import cpot940_geom_fine as GF  # noqa: E402
import step1  # noqa: E402

DENSE = np.unique(np.round(np.concatenate([np.asarray(step1.BIASES), np.arange(-0.120, 0.020 + 1e-9, 0.005)]), 6))


def run(job):
    from bandbending import Simulation, Tip
    import stacking_wall_tibb as swt
    d, R = job
    path = os.path.join(C.OUT, "dense_" + G.key(d, R)[5:] + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    ck = os.path.join(C.OUT, "dense_" + G.key(d, R)[5:] + ".ckpt.json")
    if not os.path.exists(ck):
        shutil.copy(os.path.join(C.OUT, G.key(d, R) + ".ckpt.json"), ck)
    sim = Simulation(C.sample(), Tip(radius=R, separation=d, contact_potential=G.CPOT), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", DENSE, ck).run()
    y = np.asarray(m.spectrum(0, bias=GF.VF, **dict(C.SPEC, separation=d)).didv)
    out = dict(d=d, R=R, failures=len(m.failures), didv=y.tolist(), peak=GF.peak(GF.VF, y), sweep=swt.sweep_arrays(m))
    json.dump(out, open(path, "w"))
    return out


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    jobs = [tuple(float(x) for x in a.split(":")) for a in sys.argv[1:]]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
    for r in res:
        p = r["peak"]
        print(f"dense d {r['d']:.2f} R {r['R']:5.1f}: FB {1e3 * p['v']:+6.1f} mV, HWHM out {1e3 * p['hwhm_out']:5.2f} "
              f"in {1e3 * p['hwhm_in']:5.2f} ratio {p['ratio']:.3f}; failures {r['failures']}")
