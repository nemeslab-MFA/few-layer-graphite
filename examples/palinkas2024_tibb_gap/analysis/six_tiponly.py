"""6L (+5) tip-only sweeps: best-fit CPot of each line with the SiO2 charge removed (n_sub = 0)."""
import json, multiprocessing as mp, os, sys, time
A = os.environ.get("KANDRAI_ANALYSIS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "bandbending", "examples", "kandrai2026_rg_tip_gating", "analysis"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "six")
JOBS = [("940", -0.615, 0.0), ("514", 0.370, 0.0)]

def run(job):
    tag, c, n = job
    os.environ["CPOT_TAG"] = tag
    sys.path.insert(0, A)
    import numpy as np
    import cpot940_fit2 as F, step1, stacking_wall_tibb as swt
    from bandbending import Simulation
    t0 = time.time()
    path = os.path.join(OUT, f"tiponly_{tag}.json")
    sim = Simulation(F.sample(n), F.C.tip(c), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", F.BIAS16, path.replace(".json", ".ckpt.json")).run()
    y = np.asarray(m.spectrum(0, bias=step1.FINE, **F.C.SPEC).didv)
    json.dump(dict(tag=tag, cpot=c, nsub=n, sweep=swt.sweep_arrays(m), bias=step1.FINE.tolist(), didv=y.tolist(),
                   failures=len(m.failures), seconds=time.time() - t0), open(path, "w"))
    print(tag, "done", time.time() - t0, flush=True)

if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with mp.get_context("spawn").Pool(2) as pool:
        list(pool.imap_unordered(run, JOBS))
