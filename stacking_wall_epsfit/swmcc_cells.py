"""The selected charge-model cells of both lines with the full SWMcC tight binding in the engine, against gamma0/gamma1.

Engine tight binding: ``library.TB_MCELLISTRIM2023`` (gamma0, gamma1) or ``library.TB_SWMCC_MCELLISTRIM2023``
(gamma0..gamma5, Delta' of McEllistrim et al. 2023, the parameters of the k.p wall model; bandbending plan P5).
Charge model and protocol as in step1g.py: starting doping Delta E_F (global offset; 0 for the clean model), interlayer
eps, SiO2 charge n_sub placed by bisection so that the tip-free top-layer flat band of the (+5) stack sits at the
measured peak (with each tight binding separately), sigma per line, the tip (R = 20 nm, d = 0.7 nm) at zero band
bending at zero bias over the (+5) stack (CPot = U_1 far), the same tip and charges for the (+4-) stack.  Every job
is one stack of one cell with one tight binding: the tip-free spectrum and the self-consistent bias sweep.

    python swmcc_cells.py run        # 4 processes; jobs in swmcc/
    python swmcc_cells.py summary    # table; ../examples/stacking_wall_swmcc.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "swmcc")
sys.path.insert(0, HERE)
sys.path.insert(1, REPO)
import rescore  # noqa: E402
import step1  # noqa: E402
import step2  # noqa: E402

CELLS = [("940", 10.0, 0.10), ("940", 10.0, 0.13), ("940", 2.6, 0.07), ("940", 4.0, 0.0),
         ("514", 10.0, -0.07), ("514", 10.0, -0.04), ("514", 2.6, -0.07), ("514", 4.0, 0.0)]
MODELS = ("full", "g01")
STACKS = ("RG", "SF")
E_WINDOW = 0.5


def key(tag, eps, dEF, model, stack=None):
    return f"{tag}_eps{eps:04.1f}_dEF{1e3 * dEF:+06.1f}_{model}" + (f"_{stack}" if stack else "")


def params(model):
    sys.path.insert(0, REPO)
    from bandbending import library
    return library.TB_SWMCC_MCELLISTRIM2023 if model == "full" else library.TB_MCELLISTRIM2023


def build(tag, eps, dEF, nsub, model, stack):
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, library
    import stacking_wall_tibb as swt
    seq = swt.SEQ_RG if stack == "RG" else swt.SEQ_SF
    kw = dict(fermi_level=dEF) if dEF != 0.0 else dict(doping=0.0)
    ml = GrapheneMultilayer(seq, name="ML", broadening=step1.SIGMA[tag], e_window=E_WINDOW, eps_spacer=eps,
                            params=params(model), **kw)
    return Sample(surface=ml, layers=(Layer(step1.TRAP, library.SIO2, name="traps", fixed_charge=nsub / (step1.TRAP * 1e-7)),
                                      Layer(90.0 - step1.TRAP, library.SIO2, name="SiO2")), gate=Gate())


E = np.arange(-0.6, 0.6 + 1e-9, 0.0005)


def tipfree(tag, eps, dEF, nsub, model, stack="RG"):
    from bandbending import Simulation
    r = Simulation(build(tag, eps, dEF, nsub, model, stack), None, temperature=9.0).solve(0.0, 0.0)
    p = r.multilayers["ML"]
    U = np.asarray(p.U)[0]
    _, D = p.tb.ldos(U, E + p.mu0, broadening=step1.SIGMA[tag], resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == E.size else D.T
    sel = (E > -0.12) & (E < 0.08)
    return dict(U=U, mu0=float(p.mu0), carriers=float(np.sum(p.n)), n=np.asarray(p.n).ravel(), doping=np.asarray(p.doping),
                fb=float(E[sel][np.argmax(D[sel, 0])]), D=D, p=p)


def place(tag, eps, dEF, model, target):
    """n_sub putting the tip-free (+5) top-layer flat band at target (cached in swmcc/place_*.json)."""
    path = os.path.join(OUT, "place_" + key(tag, eps, dEF, model) + ".json")
    if os.path.exists(path):
        return json.load(open(path))["nsub"]
    n0 = -float(np.sum(tipfree(tag, eps, dEF, 0.0, model)["doping"]))
    for w in (0.5e13, 1.0e13, 1.5e13, 2.0e13, 3.0e13):
        lo, hi = n0 - w, n0 + w
        if tipfree(tag, eps, dEF, hi, model)["fb"] <= target <= tipfree(tag, eps, dEF, lo, model)["fb"]:
            break
    else:
        return None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if tipfree(tag, eps, dEF, mid, model)["fb"] > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 2e9:
            break
    nsub = 0.5 * (lo + hi)
    json.dump(dict(nsub=nsub, target=target), open(path, "w"))
    return nsub


def evaluate(job):
    tag, eps, dEF, model, stack = job
    path = os.path.join(OUT, key(tag, eps, dEF, model, stack) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, REPO)
    from bandbending import Simulation, Tip
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    t0 = time.time()
    Vr, yr = rescore.measured(tag)
    fm = rescore.features(Vr, yr, smooth=9)
    nsub = place(tag, eps, dEF, model, float(fm[1]))
    if nsub is None:
        out = dict(tag=tag, eps=eps, dEF=dEF, model=model, stack=stack, placed=False)
        json.dump(out, open(path, "w"))
        return out
    rg = tipfree(tag, eps, dEF, nsub, model, "RG")
    cpot = float(rg["U"][0])                                  # zero band bending at zero bias over (+5)
    tf = rg if stack == "RG" else tipfree(tag, eps, dEF, nsub, model, stack)
    sig = step1.SIGMA[tag]
    spec = dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=sig, temperature=9.0)
    U, p = tf["U"], tf["p"]
    free = np.asarray(tunneling_spectrum(p.tb, p.mu0, step1.FINE, np.tile(U, (step1.FINE.size, 1)),
                                         np.zeros((step1.FINE.size, U.size)), **spec).didv)
    t1 = time.time()
    sim = Simulation(build(tag, eps, dEF, nsub, model, stack), Tip(**dict(step1.TIP, contact_potential=cpot)),
                     temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "ML", step1.BIASES, os.path.join(OUT, key(tag, eps, dEF, model, stack) + ".ckpt.json")).run()
    tipped = np.asarray(m.spectrum(0, bias=step1.FINE, **spec).didv)
    Ul = U - tf["mu0"]
    D = tf["D"]
    out = dict(tag=tag, eps=eps, dEF=dEF, model=model, stack=stack, placed=True, nsub=nsub, cpot=cpot, sigma=sig,
               U_minus_mu0=Ul.tolist(), mu0=tf["mu0"], carriers=tf["carriers"], n_layers=tf["n"].tolist(),
               doping=tf["doping"].tolist(), e_shift=float(p.tb.e_shift), ldos_E=E[::4].tolist(),
               ldos=D[::4].tolist(), failures=len(m.failures), sweep=swt.sweep_arrays(m), seconds_sweep=time.time() - t1)
    if stack == "RG":
        for lab, y in (("free", free), ("tip", tipped)):
            sc, res = swt.shape_distance(Vr, yr, step1.FINE, y)
            out[lab] = dict(didv=y.tolist(), scale=sc, residual=res, **rescore.score(tag, y, fm))
        out["measured_features"] = fm.tolist()
        out["measured_asym"] = rescore.asym(Vr, yr, fm[1])
    else:
        Vm, ym, _ = step2.measured_sf(tag)
        win = (Vm >= -0.45) & (Vm <= 0.45)
        mm = step2.minimum(Vm[win], ym[win])
        out["measured_min"], out["measured_edge"] = mm, step2.first_edge(Vm[win], ym[win], mm)
        for lab, y in (("free", free), ("tip", tipped)):
            mn = step2.minimum(step1.FINE, y)
            sc, res = swt.shape_distance(Vm, ym, step1.FINE, y)
            out[lab] = dict(didv=y.tolist(), minimum=mn, edge=step2.first_edge(step1.FINE, y, mn), scale=sc, residual=res)
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    t, f = out["tip"], out["free"]
    extra = (f"feats tip {np.round(1e3 * np.asarray(t['features'])).astype(int).tolist()} edges {1e3 * t['edge_rms']:.1f} "
             f"FB {1e3 * t['fb_err']:+.1f} asym {t['asym']:+.2f} | free edges {1e3 * f['edge_rms']:.1f} asym {f['asym']:+.2f}"
             if stack == "RG" else
             f"min tip {1e3 * t['minimum']:+.0f} free {1e3 * f['minimum']:+.0f} (meas {1e3 * out['measured_min']:+.0f}), "
             f"edge tip {1e3 * t['edge']:+.0f} (meas {1e3 * out['measured_edge']:+.0f}), res {t['residual']:.2f}")
    print(f"{key(tag, eps, dEF, model, stack)}: n_sub {nsub / 1e13:+.3f}e13, U-E_F {np.round(1e3 * Ul).astype(int).tolist()} | "
          f"{extra}; {out['seconds']:.0f} s", flush=True)
    return out


def summary():
    rows = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT))
            if f.endswith(".json") and not f.startswith("place_") and ".ckpt" not in f]
    rows = [r for r in rows if r.get("placed")]
    for tag in ("940", "514"):
        print(f"== {tag}")
        for r in sorted([r for r in rows if r["tag"] == tag], key=lambda r: (r["eps"], r["dEF"], r["stack"], r["model"])):
            t, f = r["tip"], r["free"]
            head = f"  eps {r['eps']:4.1f} dEF {1e3 * r['dEF']:+6.1f} {r['model']:4s} {r['stack']} n_sub {r['nsub'] / 1e13:+.3f}"
            if r["stack"] == "RG":
                print(head + f" | tip feats {np.round(1e3 * np.asarray(t['features'])).astype(int).tolist()} edges {1e3 * t['edge_rms']:5.1f} "
                      f"FB {1e3 * t['fb_err']:+5.1f} asym {t['asym']:+.2f} res {t['residual']:.2f} | free feats "
                      f"{np.round(1e3 * np.asarray(f['features'])).astype(int).tolist()} edges {1e3 * f['edge_rms']:5.1f} asym {f['asym']:+.2f} "
                      f"| U-E_F {np.round(1e3 * np.asarray(r['U_minus_mu0'])).astype(int).tolist()}")
            else:
                print(head + f" | min tip {1e3 * t['minimum']:+4.0f} free {1e3 * f['minimum']:+4.0f} (meas {1e3 * r['measured_min']:+.0f}); "
                      f"edge tip {1e3 * t['edge']:+4.0f} free {1e3 * f['edge']:+4.0f} (meas {1e3 * r['measured_edge']:+.0f}); res tip "
                      f"{t['residual']:.2f} free {f['residual']:.2f} | U-E_F {np.round(1e3 * np.asarray(r['U_minus_mu0'])).astype(int).tolist()}")
    json.dump([{k: v for k, v in r.items() if k != "sweep"} for r in rows],
              open(os.path.join(REPO, "examples", "stacking_wall_swmcc.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "run":
        jobs = [(t, e, d, m, s) for m in MODELS for (t, e, d) in CELLS for s in STACKS]
        with mp.get_context("spawn").Pool(4) as pool:
            list(pool.imap_unordered(evaluate, jobs))
    else:
        summary()
