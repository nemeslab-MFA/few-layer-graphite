"""Fit (CPot, n_sub) of each 3L / 4L map from the grid sweeps, then the gap decomposition at the optimum.

    python thin_fit.py fit        # interpolated optimum per map (no new solves); thin/fit.json
    python thin_fit.py final      # verification sweeps at the optima + tip-only (n_sub = 0) sweeps (4 processes)
    python thin_fit.py decomp     # TB / substrate-only / tip-only / tip+substrate S7 gaps; thin/decomp.json
"""
import glob, json, multiprocessing as mp, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rgfit as R


def grid_rows(N):
    """The grid sweeps of grid.py only (the verification sweeps of ``final`` are left out, so that ``fit`` returns
    the optimum the verification sweeps were run at)."""
    from grid import CP, NS
    rows = []
    for f in glob.glob(os.path.join(R.OUT, f"sweep_N{N}_*.json")):
        if f.endswith(".ckpt.json") or "_d" in os.path.basename(f):
            continue
        r = json.load(open(f))
        if any(abs(r["cpot"] - c) < 1e-9 for c in CP) and any(abs(r["nsub"] - n) < 1e3 for n in NS):
            rows.append(r)
    return rows


def surface(N, name):
    from scipy.interpolate import RBFInterpolator
    rows = [r for r in grid_rows(N) if name in r["spectra"] and np.all(np.isfinite(r["spectra"][name]["features"]))]
    X = np.array([[r["cpot"], r["nsub"] / 1e13] for r in rows])
    F = np.array([r["spectra"][name]["features"] for r in rows])
    return X, F, RBFInterpolator(X, F, kernel="thin_plate_spline", smoothing=0.0)


def fit():
    out = {}
    for name, m in R.MAPS.items():
        N = m["N"]
        fm = R.measured_features(name)
        X, F, rbf = surface(N, name)
        c = np.linspace(X[:, 0].min(), X[:, 0].max(), 241)
        n = np.linspace(X[:, 1].min(), X[:, 1].max(), 201)
        C, Nn = np.meshgrid(c, n, indexing="ij")
        P = rbf(np.column_stack([C.ravel(), Nn.ravel()])).reshape(C.shape + (3,))
        err = 1e3 * (P - fm)
        rms = np.sqrt(np.mean(err ** 2, axis=-1))
        i = np.unravel_index(np.argmin(rms), rms.shape)
        gap = 1e3 * (P[..., 2] - P[..., 1])
        good = rms < rms[i] + 5.0                          # the valley: within 5 mV of the best rms
        out[name] = dict(N=N, measured=fm.tolist(), cpot=float(C[i]), nsub=float(Nn[i]) * 1e13, rms=float(rms[i]),
                         pred=P[i].tolist(), gap_pred=float(gap[i]), n_grid=len(X),
                         valley_cpot=[float(C[good].min()), float(C[good].max())],
                         valley_nsub=[float(Nn[good].min()) * 1e13, float(Nn[good].max()) * 1e13],
                         valley_gap=[float(gap[good].min()), float(gap[good].max())])
        print(f"{name}: meas FB {1e3*fm[0]:+.0f} h {1e3*fm[1]:+.0f} e {1e3*fm[2]:+.0f} gap {1e3*(fm[2]-fm[1]):.0f} | best CPot "
              f"{C[i]:+.3f} eV n_sub {Nn[i]:+.3f}e13 rms {rms[i]:.1f} mV pred {np.round(1e3*P[i]).astype(int).tolist()} gap "
              f"{gap[i]:.0f} | valley (rms<+5 mV): CPot {C[good].min():+.2f}..{C[good].max():+.2f}, n_sub "
              f"{Nn[good].min():+.2f}..{Nn[good].max():+.2f}e13, gap {gap[good].min():.0f}..{gap[good].max():.0f}", flush=True)
    json.dump(out, open(os.path.join(R.OUT, "fit.json"), "w"), indent=1)
    return out


D_SENS = (0.5, 0.9)                                     # tip-sample distance sensitivity (nm)


def rnd(c, n):
    return round(c / 0.005) * 0.005, round(n / 0.025e13) * 0.025e13


def final():
    fits = json.load(open(os.path.join(R.OUT, "fit.json")))
    jobs = set()
    for name, f in fits.items():
        c, n = rnd(f["cpot"], f["nsub"])
        jobs.add((f["N"], c, n))
        jobs.add((f["N"], c, 0.0))
        for d in D_SENS:
            jobs.add((f["N"], c, n, d))
    with mp.get_context("spawn").Pool(4) as pool:
        list(pool.imap_unordered(R.sweep, sorted(jobs)))


def decomp():
    from s7 import gap as s7gap
    fits = json.load(open(os.path.join(R.OUT, "fit.json")))
    out = {}
    for name, f in fits.items():
        N, mod = f["N"], R.MAPS[name]["mod"]
        c, n = rnd(f["cpot"], f["nsub"])
        B = R.grid_for(mod)
        res = {}
        Vm, ym = R.measured(name)
        res["measured"] = dict(bias=Vm.tolist(), didv=ym.tolist())
        y, _ = R.free_spectrum(N, 0.0, mod, zeroU=True)
        res["tb"] = dict(bias=B.tolist(), didv=y.tolist())
        y, U = R.free_spectrum(N, n, mod)
        res["sub"] = dict(bias=B.tolist(), didv=y.tolist(), U_minus_mu0=U.tolist())
        for lab, nn in (("tip", 0.0), ("tipsub", n)):
            r = json.load(open(os.path.join(R.OUT, "sweep_" + R.skey(N, c, nn) + ".json")))
            res[lab] = dict(bias=r["spectra"][name]["bias"], didv=r["spectra"][name]["didv"], sweep_bias=r["bias"],
                            U0=r["U0"], U_far=r["U_far"], mu0=r["mu0"])
        for k, v in res.items():
            V, y = np.asarray(v["bias"]), np.asarray(v["didv"])
            h, e, g = s7gap(V, y, *R.WIN[N])
            v.update(x0_h=h, x0_e=e, gap=g, fb=R.flat_band(V, y), win_h=R.WIN[N][0], win_e=R.WIN[N][1])
            print(f"{name} {k:8s}: FB {1e3*v['fb']:+6.1f} x0_h {1e3*h:+7.1f} x0_e {1e3*e:+7.1f} gap {1e3*g:6.1f} mV", flush=True)
        sens = {}
        for d in D_SENS:
            r = json.load(open(os.path.join(R.OUT, "sweep_" + R.skey(N, c, n, d) + ".json")))
            V, y = np.asarray(r["spectra"][name]["bias"]), np.asarray(r["spectra"][name]["didv"])
            h, e, g = s7gap(V, y, *R.WIN[N])
            sens[f"{d:.2f}"] = dict(x0_h=h, x0_e=e, gap=g, fb=R.flat_band(V, y))
            print(f"{name} tipsub d={d:.2f} nm: FB {1e3*R.flat_band(V, y):+6.1f} x0_h {1e3*h:+7.1f} x0_e {1e3*e:+7.1f} gap {1e3*g:6.1f} mV", flush=True)
        res["meta"] = dict(N=N, cpot=c, nsub=n, d_sens=sens, rms_fit=f["rms"], mod=mod, valley_gap=f["valley_gap"],
                           valley_cpot=f["valley_cpot"], valley_nsub=f["valley_nsub"])
        out[name] = res
        json.dump(res, open(os.path.join(R.OUT, f"gap_{name}.json"), "w"))
    return out


if __name__ == "__main__":
    R.threads1()
    {"fit": fit, "final": final, "decomp": decomp}[sys.argv[1]]()
