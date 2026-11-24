"""S7 gaps of the six-layer (+5) spectra of lines 940 and 514: measured, SWMcC TB (neutral, no tip), SiO2 charge only
(tip-free), tip only (best-fit CPot, n_sub = 0; and the CPot that alone places the flat band), tip + SiO2 charge
(best fit of kandrai2026_rg_tip_gating)."""
import json, os, sys
import numpy as np
W = os.path.dirname(os.path.abspath(__file__))
A = os.environ.get("KANDRAI_ANALYSIS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "bandbending", "examples", "kandrai2026_rg_tip_gating", "analysis"))
sys.path.insert(0, W)
import s7
WIN = ((-0.32, -0.14), (0.14, 0.33))
BEST = {"940": (-0.615, -0.5e13, "sweep_c-0615.0_n-0.500.json", "sweep_c-0365.7_n+0.000.json"),
        "514": (0.370, 0.375e13, "sweep_c+0370.0_n+0.375.json", "sweep_c+0358.1_n+0.000.json")}


def main(tag):
    os.environ["CPOT_TAG"] = tag
    sys.path.insert(0, A)
    import cpot940 as C, step1, rescore
    from bandbending.results.spectroscopy import tunneling_spectrum
    FINE = step1.FINE
    cpot, nsub, fbest, fcp = BEST[tag]
    res = {}
    Vm, ym = rescore.measured(tag)
    m = json.load(open(os.path.join(os.path.dirname(A), "data", f"stacking_wall_tibb_{tag}.json")))
    res["measured"] = dict(bias=Vm.tolist(), didv=ym.tolist())

    def free(n, zeroU=False):
        tf = C.S.tipfree(tag, C.EPS, 0.0, n, C.MODEL, "RG")
        U, p = tf["U"], tf["p"]
        if zeroU:
            U = np.full_like(U, p.mu0)       # bands at the neutrality frame: U - mu0 = 0 (pure TB)
        y = np.asarray(tunneling_spectrum(p.tb, p.mu0, FINE, np.tile(U, (FINE.size, 1)), np.zeros((FINE.size, U.size)),
                                          **C.SPEC).didv)
        return dict(bias=FINE.tolist(), didv=y.tolist(), U_minus_mu0=(U - p.mu0).tolist())
    res["tb_U0"] = free(0.0, zeroU=True)
    res["tb"] = free(0.0)
    res["sub"] = free(nsub)
    r = json.load(open(os.path.join(A, f"cpot{tag}", "fit2", fbest)))
    res["tipsub"] = dict(bias=FINE.tolist(), didv=r["didv"], sweep=r["sweep"])
    r = json.load(open(os.path.join(A, f"cpot{tag}", "fit2", fcp)))
    res["tip_placed"] = dict(bias=FINE.tolist(), didv=r["didv"], sweep=r["sweep"], cpot=r["cpot"])
    p = os.path.join(W, "six", f"tiponly_{tag}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        res["tip"] = dict(bias=FINE.tolist(), didv=r["didv"], sweep=r["sweep"], cpot=r["cpot"])
    import features as FE
    for k, v in res.items():
        V, y = np.array(v["bias"]), np.array(v["didv"])
        f = FE.features(V, y, smooth=9 if k == "measured" else 0, low_max=-0.15)
        wh, we = (f[0] - 0.05, f[0] + 0.09), (f[2] - 0.07, f[2] + 0.06)     # around the -1 peak and the +1 rise
        h, e = s7.fit_edge(V, y, wh), s7.fit_edge(V, y, we)
        g = e - h
        v.update(x0_h=h, x0_e=e, gap=g, win_h=wh, win_e=we, features=f.tolist())
        print(f"{tag} {k:10s}: x0_h {1e3 * h:+7.1f}  x0_e {1e3 * e:+7.1f}  gap {1e3 * g:6.1f} mV | -1 peak {1e3*f[0]:+.1f}, +1 rise {1e3*f[2]:+.1f}, FB {1e3*f[1]:+.1f}", flush=True)
    res["meta"] = dict(cpot=cpot, nsub=nsub, win=WIN)
    json.dump(res, open(os.path.join(W, "six", f"gap_{tag}.json"), "w"))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    main(sys.argv[1])
