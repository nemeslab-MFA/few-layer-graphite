"""Step 1 at the physical interlayer eps = 2.6 with the full SWMcC engine tight binding, both lines.

Protocol of swmcc_cells.py: global offset Delta E_F (0 = clean), SiO2 charge n_sub placed by bisection so that the
tip-free top-layer flat band of the (+5) stack sits at the measured peak, sigma per line, tip R = 20 nm, d = 0.7 nm at
zero band bending at zero bias over (+5) (CPot = U_1 far), the same tip and charges for (+4-).  The eps sweep of line
514 (swmcc_eps.py) showed that the (+5) spectrum fixes the fixed charge over eps, so at eps 2.6 the offset is about a
third of the eps 10 one: scan it in 10 meV steps.

    python step1_eps26.py scan      # tip-free (+5) and (+4-) over DEF[tag]: swmcc/scan_*.json
    python step1_eps26.py tip [N]   # tip sweeps, both stacks, on the N (default 2) best tip-free cells per line
    python step1_eps26.py summary   # tables; ../examples/stacking_wall_step1_eps26.json
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

EPS, MODEL = 2.6, "full"
DEF = {"514": (0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06),
       "940": (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)}


def spectrum(tag, tf):
    from bandbending.results.spectroscopy import tunneling_spectrum
    spec = dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=step1.SIGMA[tag],
                temperature=9.0)
    U, p = tf["U"], tf["p"]
    return np.asarray(tunneling_spectrum(p.tb, p.mu0, step1.FINE, np.tile(U, (step1.FINE.size, 1)),
                                         np.zeros((step1.FINE.size, U.size)), **spec).didv)


def scan_path(tag, dEF):
    return os.path.join(S.OUT, "scan_" + S.key(tag, EPS, dEF, MODEL) + ".json")


def scan(cell):
    tag, dEF = cell
    path = scan_path(tag, dEF)
    if os.path.exists(path):
        return json.load(open(path))
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vr, yr = rescore.measured(tag)
    fm = rescore.features(Vr, yr, smooth=9)
    nsub = S.place(tag, EPS, dEF, MODEL, float(fm[1]))
    out = dict(tag=tag, eps=EPS, dEF=dEF, model=MODEL, placed=nsub is not None)
    if nsub is not None:
        rg = S.tipfree(tag, EPS, dEF, nsub, MODEL, "RG")
        sf = S.tipfree(tag, EPS, dEF, nsub, MODEL, "SF")
        yrg, ysf = spectrum(tag, rg), spectrum(tag, sf)
        Vm, ym, _ = step2.measured_sf(tag)
        win = (Vm >= -0.45) & (Vm <= 0.45)
        mn = step2.minimum(step1.FINE, ysf)
        out.update(nsub=nsub, U_RG=(rg["U"] - rg["mu0"]).tolist(), U_SF=(sf["U"] - sf["mu0"]).tolist(),
                   n_RG=rg["n"].tolist(), RG=dict(didv=yrg.tolist(), **rescore.score(tag, yrg, fm)),
                   SF=dict(didv=ysf.tolist(), minimum=mn, edge=step2.first_edge(step1.FINE, ysf, mn),
                           residual=swt.shape_distance(Vm, ym, step1.FINE, ysf)[1]),
                   measured_features=fm.tolist(), measured_min=step2.minimum(Vm[win], ym[win]),
                   measured_edge=step2.first_edge(Vm[win], ym[win], step2.minimum(Vm[win], ym[win])))
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    line(out)
    return out


def line(r):
    head = f"{r['tag']} dEF {1e3 * r['dEF']:+5.0f}"
    if not r["placed"]:
        print(head + " | not placed", flush=True)
        return
    g, s = r["RG"], r["SF"]
    print(head + f" n_sub {r['nsub'] / 1e13:+.3f}e13 | RG feats {S._mv(g['features'])} (meas {S._mv(r['measured_features'])}) "
          f"edges {1e3 * g['edge_rms']:5.1f} asym {g['asym']:+.2f} | SF min {1e3 * s['minimum']:+4.0f} (meas "
          f"{1e3 * r['measured_min']:+.0f}) res {s['residual']:.2f} | U-E_F {np.round(1e3 * np.asarray(r['U_RG'])).astype(int).tolist()}",
          flush=True)


def scans(tag):
    out = []
    for d in DEF[tag]:
        p = scan_path(tag, d)
        if os.path.exists(p):
            r = json.load(open(p))
            if r["placed"]:
                out.append(r)
    return out


def best(tag, n):
    """The n placed cells with the smallest tip-free edge error and all three bulk features found."""
    rows = [r for r in scans(tag) if r["RG"]["missing"] == 0]
    return [r["dEF"] for r in sorted(rows, key=lambda r: r["RG"]["edge_rms"])[:n]]


def tip_result(tag, dEF):
    out = []
    for st in S.STACKS:
        p = os.path.join(S.OUT, S.key(tag, EPS, dEF, MODEL, st) + ".json")
        out.append(json.load(open(p)) if os.path.exists(p) else None)
    return out


def summary():
    keep = {}
    for tag in ("514", "940"):
        rows = scans(tag)
        print(f"== {tag}: tip-free, eps {EPS}, full SWMcC")
        for r in rows:
            line(r)
        print(f"== {tag}: with the tip")
        tips = []
        for d in sorted({r["dEF"] for r in rows} | {DEF[tag][0]}):
            rg, sf = tip_result(tag, d)
            if rg and rg.get("placed"):
                t = rg["tip"]
                s = (f"SF min tip {1e3 * sf['tip']['minimum']:+4.0f} free {1e3 * sf['free']['minimum']:+4.0f} (meas "
                     f"{1e3 * sf['measured_min']:+.0f}), edge tip {1e3 * sf['tip']['edge']:+4.0f} (meas {1e3 * sf['measured_edge']:+.0f}), "
                     f"res {sf['tip']['residual']:.2f}" if sf else "SF -")
                print(f"{tag} dEF {1e3 * d:+5.0f} n_sub {rg['nsub'] / 1e13:+.3f}e13 | tip feats {S._mv(t['features'])} edges "
                      f"{1e3 * t['edge_rms']:5.1f} FB {1e3 * t['fb_err']:+5.1f} asym {t['asym']:+.2f} res {t['residual']:.2f} | "
                      f"free edges {1e3 * rg['free']['edge_rms']:5.1f} | {s}")
                tips += [{k: v for k, v in x.items() if k != "sweep"} for x in (rg, sf) if x]
        keep[tag] = dict(scan=rows, tip=tips)
    json.dump(keep, open(os.path.join(REPO, "examples", "stacking_wall_step1_eps26.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(S.OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "scan":
        cells = [(t, d) for t in ("514", "940") for d in DEF[t]]
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(scan, cells))
    elif cmd == "tip":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        cells = [(t, d) for t in ("514", "940") for d in best(t, n)]
        print("tip sweeps on", cells, flush=True)
        jobs = [(t, EPS, d, MODEL, s) for (t, d) in cells for s in S.STACKS]
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(S.evaluate, jobs))
    else:
        summary()
