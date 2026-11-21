"""Interlayer eps sweep for line 514 with the full SWMcC engine tight binding (protocol of swmcc_cells.py).

    python swmcc_eps.py scan [EPS:DEF ...]  # tip-free (+5) and (+4-), default EPS x DEF: swmcc/scan_*.json
    python swmcc_eps.py tip EPS:DEF ... # self-consistent tip sweeps (both stacks) via swmcc_cells.evaluate
    python swmcc_eps.py summary         # tables; ../examples/stacking_wall_swmcc_eps514.json

eps -> infinity removes the interlayer Hartree term: every layer at one potential, charge set by the quantum
capacitance alone (rigid bands).  Large eps is therefore a proxy for anything that weakens the interlayer Hartree
cost of the sheet model (exchange, finite thickness of the pi charge, sigma screening beyond 2.6).
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
import rescore  # noqa: E402
import step1  # noqa: E402
import step2  # noqa: E402
import swmcc_cells as S  # noqa: E402

TAG, MODEL = "514", "full"
EPS = (4.0, 6.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 1000.0)
DEF = (0.0, -0.04, -0.07, -0.10)


def spectrum(tf):
    from bandbending.results.spectroscopy import tunneling_spectrum
    spec = dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=step1.SIGMA[TAG],
                temperature=9.0)
    U, p = tf["U"], tf["p"]
    return np.asarray(tunneling_spectrum(p.tb, p.mu0, step1.FINE, np.tile(U, (step1.FINE.size, 1)),
                                         np.zeros((step1.FINE.size, U.size)), **spec).didv)


def scan(cell):
    eps, dEF = cell
    path = os.path.join(S.OUT, "scan_" + S.key(TAG, eps, dEF, MODEL) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    Vr, yr = rescore.measured(TAG)
    fm = rescore.features(Vr, yr, smooth=9)
    nsub = S.place(TAG, eps, dEF, MODEL, float(fm[1]))
    out = dict(tag=TAG, eps=eps, dEF=dEF, model=MODEL, placed=nsub is not None)
    if nsub is not None:
        rg = S.tipfree(TAG, eps, dEF, nsub, MODEL, "RG")
        sf = S.tipfree(TAG, eps, dEF, nsub, MODEL, "SF")
        yrg, ysf = spectrum(rg), spectrum(sf)
        Vm, ym, _ = step2.measured_sf(TAG)
        win = (Vm >= -0.45) & (Vm <= 0.45)
        mn = step2.minimum(step1.FINE, ysf)
        out.update(nsub=nsub, U_RG=(rg["U"] - rg["mu0"]).tolist(), U_SF=(sf["U"] - sf["mu0"]).tolist(),
                   n_RG=rg["n"].tolist(), RG=dict(didv=yrg.tolist(), **rescore.score(TAG, yrg, fm)),
                   SF=dict(didv=ysf.tolist(), minimum=mn, edge=step2.first_edge(step1.FINE, ysf, mn),
                           residual=step2_residual(Vm, ym, ysf)),
                   measured_features=fm.tolist(), measured_min=step2.minimum(Vm[win], ym[win]))
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    line(out)
    return out


def step2_residual(Vm, ym, y):
    import stacking_wall_tibb as swt
    return swt.shape_distance(Vm, ym, step1.FINE, y)[1]


def line(r):
    head = f"eps {r['eps']:6.1f} dEF {1e3 * r['dEF']:+5.0f}"
    if not r["placed"]:
        print(head + " | not placed", flush=True)
        return
    g, s = r["RG"], r["SF"]
    print(head + f" n_sub {r['nsub'] / 1e13:+.3f}e13 | RG feats {S._mv(g['features'])} edges {1e3 * g['edge_rms']:5.1f} "
          f"asym {g['asym']:+.2f} | SF min {1e3 * s['minimum']:+4.0f} (meas {1e3 * r['measured_min']:+.0f}) res {s['residual']:.2f} "
          f"| U-E_F {np.round(1e3 * np.asarray(r['U_RG'])).astype(int).tolist()}", flush=True)


def summary():
    rows = [json.load(open(os.path.join(S.OUT, f))) for f in sorted(os.listdir(S.OUT))
            if f.startswith("scan_" + TAG) and f.endswith(f"_{MODEL}.json")]
    print("== tip-free scan (514, full SWMcC); measured features", S._mv(rows[0]["measured_features"]), "asym +0.23")
    for r in sorted(rows, key=lambda r: (r["dEF"], r["eps"])):
        line(r)
    tips = []
    for f in sorted(os.listdir(S.OUT)):
        if f.startswith(TAG) and f.endswith(f"_{MODEL}_RG.json") and ".ckpt" not in f:
            rg = json.load(open(os.path.join(S.OUT, f)))
            p = os.path.join(S.OUT, f.replace("_RG.json", "_SF.json"))
            sf = json.load(open(p)) if os.path.exists(p) else None
            if rg.get("placed"):
                tips.append((rg, sf))
    print("== with the tip")
    for rg, sf in sorted(tips, key=lambda x: (x[0]["dEF"], x[0]["eps"])):
        t = rg["tip"]
        s = (f"SF min tip {1e3 * sf['tip']['minimum']:+4.0f} free {1e3 * sf['free']['minimum']:+4.0f} res {sf['tip']['residual']:.2f}"
             if sf else "SF -")
        print(f"eps {rg['eps']:6.1f} dEF {1e3 * rg['dEF']:+5.0f} n_sub {rg['nsub'] / 1e13:+.3f}e13 | tip feats {S._mv(t['features'])} "
              f"edges {1e3 * t['edge_rms']:5.1f} FB {1e3 * t['fb_err']:+5.1f} asym {t['asym']:+.2f} | free edges "
              f"{1e3 * rg['free']['edge_rms']:5.1f} | {s}")
    json.dump(dict(scan=rows, tip=[{k: v for k, v in r.items() if k != "sweep"} for pair in tips for r in pair if r]),
              open(os.path.join(REPO, "examples", "stacking_wall_swmcc_eps514.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(S.OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "scan":
        cells = ([tuple(float(x) for x in a.split(":")) for a in sys.argv[2:]] if len(sys.argv) > 2
                 else [(e, d) for d in DEF for e in EPS])
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(scan, cells))
    elif cmd == "tip":
        pairs = [tuple(float(x) for x in a.split(":")) for a in sys.argv[2:]]
        jobs = [(TAG, e, d, MODEL, s) for (e, d) in pairs for s in S.STACKS]
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(S.evaluate, jobs))
    else:
        summary()
