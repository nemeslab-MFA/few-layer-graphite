"""Self-consistent Hartree + full-SWMcC tight-binding model of the 3- and 4-layer rhombohedral graphite STM spectra
of Palinkas et al., Carbon 230, 119608 (2024) (Zenodo 10931524), with the same model as the six-layer fits of
kandrai2026_rg_tip_gating: neutral stack (fermi_level 0: the zero-field SWMcC layer charges are screened), interlayer
eps_z 2.6, 2 nm SiO2 trap layer with the fixed charge n_sub on 88 nm SiO2, grounded Si gate (90 nm SiO2 wafers,
Methods 3.1.1), tip R 20 nm, d 0.7 nm, contact potential CPot; T 9 K; WKB barrier 4.6 eV over d.

Free parameters per map: CPot and n_sub.  Targets: the flat-band peak bias and the S7 band-edge positions x0(-1),
x0(+1) of the map-averaged spectrum.
"""
import json
import os
import sys
import time

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(W, "thin")
sys.path.insert(0, W)
import s7  # noqa: E402

TRAP, SIO2 = 2.0, 88.0
TIP = dict(radius=20.0, separation=0.7)
SIGMA = 0.010
E_WINDOW = 0.9
FINE = np.round(np.arange(-0.65, 0.75 + 1e-9, 0.00125), 5)
BIASES = np.round(np.array([-0.65, -0.55, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.10, -0.05, -0.025, 0.0, 0.025,
                            0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65, 0.75]), 4)
WIN = {3: ((-0.6, -0.15), (0.2, 0.7)), 4: ((-0.6, -0.2), (0.2, 0.7))}         # S7 windows of the Zenodo notebook
MAPS = {"3L_647": dict(N=3, mod=0.0040), "3L_280": dict(N=3, mod=0.0025),
        "4L_209": dict(N=4, mod=0.0055), "4L_660": dict(N=4, mod=0.0050)}
E_FB = np.arange(-0.15, 0.15 + 1e-9, 0.0005)


def spec_kw(mod, d=None):
    return dict(modulation=mod, barrier=4.6, separation=TIP["separation"] if d is None else d, broadening=SIGMA,
                temperature=9.0, energy_step=0.0005)


def grid_for(mod):
    """Bias grid of the simulated spectrum: 2.5 mV, or 1.25 mV for modulations below 3.6 mV rms."""
    step = 0.0025 if mod >= 0.0036 else 0.00125
    return np.round(np.arange(-0.65, 0.75 + 1e-9, step), 5)


def seq(N):
    return ("CBA" * 4)[:N]


def build(N, nsub):
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, library
    ml = GrapheneMultilayer(seq(N), name="ML", broadening=SIGMA, e_window=E_WINDOW, eps_spacer=2.6,
                            params=library.TB_SWMCC_MCELLISTRIM2023, fermi_level=0.0)
    return Sample(surface=ml, layers=(Layer(TRAP, library.SIO2, name="traps", fixed_charge=nsub / (TRAP * 1e-7)),
                                      Layer(SIO2, library.SIO2, name="SiO2")), gate=Gate())


def tip(cpot, d=None):
    from bandbending import Tip
    kw = dict(TIP) if d is None else dict(TIP, separation=float(d))
    return Tip(**kw, contact_potential=float(cpot))


def tipfree(N, nsub):
    from bandbending import Simulation
    r = Simulation(build(N, nsub), None, temperature=9.0).solve(0.0, 0.0)
    p = r.multilayers["ML"]
    return np.asarray(p.U)[0], p


def free_spectrum(N, nsub, mod, zeroU=False, bias=None):
    bias = grid_for(mod) if bias is None else bias
    from bandbending.results.spectroscopy import tunneling_spectrum
    U, p = tipfree(N, nsub)
    if zeroU:
        U = np.full_like(U, p.mu0)
    y = np.asarray(tunneling_spectrum(p.tb, p.mu0, bias, np.tile(U, (bias.size, 1)), np.zeros((bias.size, U.size)),
                                      **spec_kw(mod)).didv)
    return y, U - p.mu0


class Sweep:
    """Apex and far-field layer potentials of a bias sweep (N layers), warm-started from 0 V outwards and
    checkpointed after every point (the CheckpointedSweep of kandrai2026_rg_tip_gating for any N)."""

    def __init__(self, sim, biases, path=None):
        self.sim, self.path = sim, path
        self.bias = np.asarray(biases, dtype=np.float64)
        self.points = json.load(open(path)) if path and os.path.exists(path) else {}
        self.tb, self.failures = None, []

    @staticmethod
    def key(v):
        return f"{v:+.4f}"

    def run(self):
        prev = {}
        for i in np.argsort(np.abs(self.bias), kind="stable"):
            vb = float(self.bias[i])
            k = self.key(vb)
            if k in self.points:
                continue
            init = prev.get(np.sign(vb)) or prev.get(0.0)
            try:
                r = self.sim.solve(vb, 0.0, init=init, fields=False)
            except Exception as err:  # noqa: BLE001
                print(f"    bias {vb:+.3f} V failed: {err}", flush=True)
                self.failures.append(vb)
                self.points[k] = None
                continue
            p = r.multilayers["ML"]
            self.points[k] = {"U0": np.asarray(p.U[0]).tolist(), "U_far": np.asarray(p.U_far).tolist(),
                              "mu0": float(p.mu0), "iterations": int(r.diagnostics.get("iterations", -1))}
            self.tb = p.tb
            prev[np.sign(vb)] = r
            if self.path:
                json.dump(self.points, open(self.path, "w"))
        if self.tb is None:
            self.tb = self.sim._multilayer_tb(self.sim._sample.multilayers[0].spec)
        return self

    def arrays(self):
        ok = [self.points.get(self.key(v)) is not None for v in self.bias]
        b = self.bias[ok]
        U0 = np.array([self.points[self.key(v)]["U0"] for v in b])
        Uf = np.array([self.points[self.key(v)]["U_far"] for v in b])
        return b, U0, Uf, next(pt["mu0"] for pt in self.points.values() if pt)

    def spectrum(self, bias, **kw):
        from bandbending.results.spectroscopy import interpolate_potentials, tunneling_spectrum
        b, U0, _, mu0 = self.arrays()
        U, dUdV = interpolate_potentials(b, U0, np.asarray(bias, float))
        return np.asarray(tunneling_spectrum(self.tb, mu0, np.asarray(bias, float), U, dUdV, **kw).didv)


def measured(name):
    d = json.load(open(os.path.join(W, "..", "data", "stm_3L4L_avg.json")))[name]
    return np.asarray(d["bias"]), np.asarray(d["didv"])


def flat_band(V, y, lim=(-0.1, 0.1)):
    s = (V > lim[0]) & (V < lim[1])
    return float(V[s][np.argmax(y[s])])


def features(N, V, y):
    """(flat-band peak, x0(-1), x0(+1)) of a spectrum; S7 windows of the Zenodo notebook."""
    h, e, _ = s7.gap(V, y, *WIN[N])
    return np.array([flat_band(V, y), h, e])


def measured_features(name):
    V, y = measured(name)
    return features(MAPS[name]["N"], V, y)


def apex_fb(N, cpot, nsub, bias):
    """Flat-band energy (eV from E_F) of the top-layer LDOS under the apex, solved at the given bias."""
    from bandbending import Simulation
    r = Simulation(build(N, nsub), tip(cpot), temperature=9.0, grid="draft").solve(float(bias), 0.0)
    p = r.multilayers["ML"]
    U = np.asarray(p.U)[0]
    _, D = p.tb.ldos(U, E_FB + p.mu0, broadening=SIGMA, resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == E_FB.size else D.T
    return float(E_FB[np.argmax(D[:, 0])]), U - p.mu0


def skey(N, cpot, nsub, d=None):
    k = f"N{N}_c{1e3 * cpot:+07.1f}_n{nsub / 1e13:+06.3f}"
    return k if d is None or abs(d - TIP["separation"]) < 1e-9 else k + f"_d{d:.2f}"


def sweep(job):
    """Full bias sweep at (N, cpot, nsub); spectra for every map modulation of that N are produced on demand."""
    N, cpot, nsub = job[:3]
    d = job[3] if len(job) > 3 else None
    path = os.path.join(OUT, "sweep_" + skey(N, cpot, nsub, d) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    from bandbending import Simulation
    t0 = time.time()
    sim = Simulation(build(N, nsub), tip(cpot, d), temperature=9.0, grid="draft")
    m = Sweep(sim, BIASES, path.replace(".json", ".ckpt.json")).run()
    b, U0, Uf, mu0 = m.arrays()
    out = dict(N=N, cpot=cpot, nsub=nsub, d=d if d is not None else TIP["separation"], bias=b.tolist(), U0=U0.tolist(), U_far=Uf.tolist(), mu0=mu0,
               failures=len(m.failures), seconds=time.time() - t0, spectra={})
    for name, mp_ in MAPS.items():
        if mp_["N"] == N:
            B = grid_for(mp_["mod"])
            y = m.spectrum(B, **spec_kw(mp_["mod"], d))
            out["spectra"][name] = dict(bias=B.tolist(), didv=y.tolist(), features=features(N, B, y).tolist())
    json.dump(out, open(path, "w"))
    f = out["spectra"]
    print(f"{skey(N, cpot, nsub, d)}: " + "; ".join(f"{k} FB {1e3 * v['features'][0]:+.0f} h {1e3 * v['features'][1]:+.0f} "
                                                f"e {1e3 * v['features'][2]:+.0f} gap {1e3 * (v['features'][2] - v['features'][1]):.0f}"
                                                for k, v in f.items()) + f" ({out['seconds']:.0f} s)", flush=True)
    return out


def threads1():
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
