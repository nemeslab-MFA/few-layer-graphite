"""Flat-band peak shape of the cpot940_geom.py sweeps on a 0.25 mV bias grid (the 2.5 mV grid of the stored spectra
quantises the half widths in 2.5 mV steps, which is the size of the asymmetry being measured).  No new solves: the
apex potentials come from the checkpoints and are PCHIP-interpolated between the 25 mV sweep points as before.

    python cpot940_geom_fine.py        # cpot940/geom_fine.json and a table
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402
import cpot940_geom as G  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

VF = np.round(np.arange(-0.150, 0.080 + 1e-9, 0.00025), 6)


def peak(V, y, lo=-0.15, hi=0.06):
    """Peak (parabolic vertex), height above the common floor (lower of the two side minima in the window) and the
    half widths at half maximum on the outer (negative) and inner side, crossings linearly interpolated."""
    w = (V >= lo) & (V <= hi)
    v, z = V[w], y[w]
    k = int(np.argmax(z))
    a, b, c = z[k - 1], z[k], z[k + 1]
    dk = 0.5 * (a - c) / (a - 2 * b + c)
    vk, zk = v[k] + dk * (v[1] - v[0]), b - 0.25 * (a - c) * dk
    floor = min(z[:k].min(), z[k:].min())
    half = floor + 0.5 * (zk - floor)
    i = np.where(z[:k] < half)[0][-1]
    left = v[i] + (half - z[i]) / (z[i + 1] - z[i]) * (v[i + 1] - v[i])
    j = k + np.where(z[k:] < half)[0][0]
    right = v[j - 1] + (half - z[j - 1]) / (z[j] - z[j - 1]) * (v[j] - v[j - 1])
    return dict(v=float(vk), height=float(zk), floor=float(floor), hwhm_out=float(vk - left), hwhm_in=float(right - vk),
                ratio=float((vk - left) / (right - vk)))


def tip_job(job):
    from bandbending import Simulation, Tip
    import stacking_wall_tibb as swt
    d, R = job
    sim = Simulation(C.sample(), Tip(radius=R, separation=d, contact_potential=G.CPOT), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", step1.BIASES, os.path.join(C.OUT, G.key(d, R) + ".ckpt.json")).run()
    assert not m.failures and all(m._valid)
    y = np.asarray(m.spectrum(0, bias=VF, **dict(C.SPEC, separation=d)).didv)
    return dict(d=d, R=R, didv=y.tolist(), peak=peak(VF, y))


def free_job(d):
    from bandbending.results.spectroscopy import tunneling_spectrum
    tf = C.S.tipfree(C.TAG, C.EPS, 0.0, 0.0, C.MODEL, "RG")
    U, p = tf["U"], tf["p"]
    y = np.asarray(tunneling_spectrum(p.tb, p.mu0, VF, np.tile(U, (VF.size, 1)), np.zeros((VF.size, U.size)),
                                      **dict(C.SPEC, separation=d)).didv)
    return dict(d=d, didv=y.tolist(), peak=peak(VF, y))


def run(task):
    return tip_job(task[1]) if task[0] == "tip" else free_job(task[1])


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    tasks = [("tip", j) for j in G.JOBS] + [("free", 0.7)]
    with mp.get_context("spawn").Pool(4) as pool:
        res = list(pool.map(run, tasks))
    Vr, yr = rescore.measured(C.TAG)
    pm = peak(Vr, yr)
    out = dict(V=VF.tolist(), tip=res[:-1], free=res[-1], measured=pm,
               measured_step=float(np.median(np.diff(Vr))))
    json.dump(out, open(os.path.join(C.OUT, "geom_fine.json"), "w"))
    f = lambda p: (f"FB {1e3 * p['v']:+6.1f} mV, HWHM out {1e3 * p['hwhm_out']:5.2f} in {1e3 * p['hwhm_in']:5.2f} "
                   f"ratio {p['ratio']:.3f}")
    print(f"measured (step {1e3 * out['measured_step']:.2f} mV) {f(pm)}")
    print(f"no tip                {f(res[-1]['peak'])}")
    for r in sorted(res[:-1], key=lambda r: (r["R"], r["d"])):
        print(f"d {r['d']:.2f} R {r['R']:5.1f}  {f(r['peak'])}")
