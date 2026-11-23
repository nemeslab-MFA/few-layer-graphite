"""(+4-) spectra of lines 940 and 514 predicted with the best-fit parameters of their (+5) spectra (two-parameter fit
of CPot and n_sub, cpot940_fit2.py): same tip, same substrate charge, same everything else (neutral stack, eps 2.6, full
SWMcC, R 20 nm, d 0.7 nm, T 9 K, sigma of the line); only the stacking changes to (+4-), ABACBA.  Scored like the
earlier (+4-) runs: the minimum near E_F and the first edge above it (step2.minimum, step2.first_edge).

    python cpot_sf.py run        # full sweeps (2 processes), cpot_sf/<tag>.json
    python cpot_sf.py summary    # table; ../examples/stacking_wall_cpot_sf.json
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
sys.path.insert(1, REPO)
import step1  # noqa: E402
import step2  # noqa: E402
import swmcc_cells as S  # noqa: E402

OUT = os.path.join(HERE, "cpot_sf")
EPS, MODEL = 2.6, "full"
BEST = {"940": dict(cpot=-0.615, nsub=-0.5e13), "514": dict(cpot=0.370, nsub=0.375e13)}


def spec(tag):
    return dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=step1.SIGMA[tag],
                temperature=9.0)


def run(tag):
    path = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation, Tip
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    t0 = time.time()
    p = BEST[tag]
    F = step1.FINE
    Vm, ym, _ = step2.measured_sf(tag)
    win = (Vm >= -0.45) & (Vm <= 0.45)
    Vm, ym = Vm[win], ym[win]
    mm = step2.minimum(Vm, ym)
    out = dict(tag=tag, cpot=p["cpot"], nsub=p["nsub"], measured_min=mm, measured_edge=step2.first_edge(Vm, ym, mm))
    tf = S.tipfree(tag, EPS, 0.0, p["nsub"], MODEL, "SF")
    U, pp = tf["U"], tf["p"]
    free = np.asarray(tunneling_spectrum(pp.tb, pp.mu0, F, np.tile(U, (F.size, 1)), np.zeros((F.size, U.size)),
                                         **spec(tag)).didv)
    sim = Simulation(S.build(tag, EPS, 0.0, p["nsub"], MODEL, "SF"), Tip(**dict(step1.TIP, contact_potential=p["cpot"])),
                     temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", step1.BIASES, os.path.join(OUT, f"{tag}.ckpt.json")).run()
    tipped = np.asarray(m.spectrum(0, bias=F, **spec(tag)).didv)
    out.update(failures=len(m.failures), sweep=swt.sweep_arrays(m), U_free_minus_EF=(U - tf["mu0"]).tolist(),
               mu0_free=tf["mu0"])
    for lab, y in (("free", free), ("tip", tipped)):
        mn = step2.minimum(F, y)
        sc, res = swt.shape_distance(Vm, ym, F, y)
        out[lab] = dict(didv=y.tolist(), minimum=mn, edge=step2.first_edge(F, y, mn), scale=sc, residual=res)
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    line(out)
    return out


def line(r):
    t, f = r["tip"], r["free"]
    print(f"{r['tag']} (+4-) at CPot {r['cpot']:+.3f} eV, n_sub {r['nsub'] / 1e13:+.3f}e13 | minimum tip {1e3 * t['minimum']:+.0f} "
          f"(no tip {1e3 * f['minimum']:+.0f}, measured {1e3 * r['measured_min']:+.0f}) mV | first edge tip "
          f"{1e3 * t['edge']:+.0f} (no tip {1e3 * f['edge']:+.0f}, measured {1e3 * r['measured_edge']:+.0f}) mV | "
          f"residual {t['residual']:.2f} | failures {r['failures']}" + (f"; {r['seconds']:.0f} s" if "seconds" in r else ""),
          flush=True)


def summary():
    rs = [json.load(open(os.path.join(OUT, f"{t}.json"))) for t in BEST if os.path.exists(os.path.join(OUT, f"{t}.json"))]
    for r in rs:
        line(r)
    json.dump([{k: v for k, v in r.items() if k != "sweep"} for r in rs],
              open(os.path.join(REPO, "examples", "stacking_wall_cpot_sf.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "run":
        with mp.get_context("spawn").Pool(2) as pool:
            list(pool.imap_unordered(run, list(BEST)))
    summary()
