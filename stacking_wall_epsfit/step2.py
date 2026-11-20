"""Step 2 of the rebuilt charge model: the (+4-) spectra of lines 514 and 940 with the parameters of step 1.

The (+4-) stack (ABACBA, top layer in the Bernal position) with exactly the charge model and tip fitted to the
(+5) side in step 1 (shared eps = 2.6; 940: n_sub = +2e13 cm^-2, contact potential -0.15 eV; 514: n_sub = -2e13,
+0.075 eV), solved without the tip (one self-consistent solve) and with it (a bias sweep), against the measured
(+4-) spectrum far from the wall.  Recorded: the layer potentials, the minimum of the spectrum near zero bias (the
Dirac point of the (+4-) surface cone), the lowest edge above it, the shape residual.

    python step2.py run      # both lines in parallel; cells in step2/
    python step2.py plot     # ../examples/stacking_wall_step2.png
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "step2")
sys.path.insert(0, HERE)
import step1  # noqa: E402

PARAMS = {"940": dict(eps=2.6, n13=2.0, cpot=-0.15), "514": dict(eps=2.6, n13=-2.0, cpot=0.075)}


def measured_sf(tag):
    m = json.load(open(os.path.join(REPO, "examples", f"stacking_wall_tibb_{tag}.json")))
    return np.asarray(m["measured"]["bias"]), np.asarray(m["measured"]["sf"], float), m


def minimum(V, y, lo=-0.15, hi=0.15):
    V, y = np.asarray(V), np.asarray(y)
    sel = (V > lo) & (V < hi)
    return float(V[sel][np.argmin(y[sel])])


def first_edge(V, y, v0, hi=0.25):
    """Steepest rise between the minimum + 20 mV and hi (the lowest edge above the minimum)."""
    V, y = np.asarray(V), np.asarray(y)
    d = np.gradient(y / np.nanmax(y), V)
    sel = (V > v0 + 0.02) & (V < hi)
    return float(V[sel][np.argmax(d[sel])]) if sel.any() else float("nan")


def evaluate(tag):
    path = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, REPO)
    from bandbending import Gate, GrapheneMultilayer, Layer, Sample, Simulation, Tip, library
    from bandbending.results.spectroscopy import tunneling_spectrum
    import stacking_wall_tibb as swt
    p = PARAMS[tag]
    t0 = time.time()
    sig = step1.SIGMA[tag]
    nsub = p["n13"] * 1e13
    spec = dict(modulation=0.00375, barrier=4.6, separation=step1.TIP["separation"], broadening=sig, temperature=9.0)
    ml = GrapheneMultilayer(swt.SEQ_SF, name="SF", broadening=sig, e_window=0.8, doping=0.0, eps_spacer=p["eps"])
    sample = Sample(surface=ml, layers=(Layer(step1.TRAP, library.SIO2, name="traps", fixed_charge=nsub / (step1.TRAP * 1e-7)),
                                        Layer(90.0 - step1.TRAP, library.SIO2, name="SiO2")), gate=Gate())
    r = Simulation(sample, None, temperature=9.0).solve(0.0, 0.0)
    q = r.multilayers["SF"]
    U = np.asarray(q.U)[0]
    free = np.asarray(tunneling_spectrum(q.tb, q.mu0, step1.FINE, np.tile(U, (step1.FINE.size, 1)),
                                         np.zeros((step1.FINE.size, U.size)), **spec).didv)
    E = np.arange(-0.6, 0.6 + 1e-9, 0.001)
    _, D = q.tb.ldos(U, E + q.mu0, broadening=sig, resolve="layer")
    D = np.asarray(D)
    D = D if D.shape[0] == E.size else D.T
    sim = Simulation(sample, Tip(**dict(step1.TIP, contact_potential=p["cpot"])), temperature=9.0, grid="draft")
    m = swt.CheckpointedSweep(sim, "SF", step1.BIASES, os.path.join(OUT, f"{tag}.ckpt.json")).run()
    tipped = np.asarray(m.spectrum(0, bias=step1.FINE, **spec).didv)
    Vm, ym, _ = measured_sf(tag)
    win = (Vm >= -0.45) & (Vm <= 0.45)
    mm = minimum(Vm[win], ym[win])
    out = dict(tag=tag, **p, sigma=sig, U_minus_mu0=(U - q.mu0).tolist(), mu0=float(q.mu0), carriers=float(np.sum(q.n)),
               carriers_layers=np.asarray(q.n).ravel().tolist(), ldos_E=E.tolist(), ldos_top=D[:, 0].tolist(),
               ldos_l2=D[:, 1].tolist(), measured_min=mm, measured_edge=first_edge(Vm[win], ym[win], mm),
               sweep=swt.sweep_arrays(m), failures=len(m.failures))
    for lab, y in (("free", free), ("tip", tipped)):
        mn = minimum(step1.FINE, y)
        sc, res = swt.shape_distance(Vm, ym, step1.FINE, y)
        out[lab] = dict(didv=y.tolist(), minimum=mn, edge=first_edge(step1.FINE, y, mn), scale=sc, residual=res)
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"))
    print(f"[{tag}] (+4-): U - E_F {np.round(1e3 * (U - q.mu0), 1)} meV; minimum measured {1e3 * mm:+.1f}, "
          f"with tip {1e3 * out['tip']['minimum']:+.1f}, without {1e3 * out['free']['minimum']:+.1f} mV; first edge measured "
          f"{1e3 * out['measured_edge']:+.0f}, tip {1e3 * out['tip']['edge']:+.0f}, free {1e3 * out['free']['edge']:+.0f} mV; "
          f"residual tip {out['tip']['residual']:.2f}, free {out['free']['residual']:.2f}; {out['seconds']:.0f} s", flush=True)
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.6), gridspec_kw={"width_ratios": [1.6, 1.0]})
    for row, tag in enumerate(("940", "514")):
        r = json.load(open(os.path.join(OUT, f"{tag}.json")))
        Vm, ym, m = measured_sf(tag)
        w = (Vm >= -0.45) & (Vm <= 0.45)
        ax = axs[row, 0]
        ax.plot(Vm[w], ym[w], color=INK, lw=1.8, label="measured (+⁴−), far from the wall")
        ax.plot(step1.FINE, np.asarray(r["tip"]["didv"]) * r["tip"]["scale"], color=BLUE, lw=1.5,
                label=f"model with the tip (CPot {r['cpot']:+.3f} eV): residual {r['tip']['residual']:.2f}")
        ax.plot(step1.FINE, np.asarray(r["free"]["didv"]) * r["free"]["scale"], color=ORANGE, lw=1.2, ls="--",
                label=f"model without the tip: residual {r['free']['residual']:.2f}")
        old = m["runs"].get("sf")
        if old:
            fb = np.asarray(m["fine_bias"])
            sc = step1  # noqa: F841
            y = np.asarray(old["didv"])
            a = float(np.dot(np.interp(Vm[w], fb, y), ym[w]) / np.dot(np.interp(Vm[w], fb, y), np.interp(Vm[w], fb, y)))
            ax.plot(fb, a * y, color=MUTED, lw=1.0, ls=":", label="per-layer doping model (earlier), with the tip")
        for v, c in ((r["measured_min"], INK), (r["tip"]["minimum"], BLUE), (r["free"]["minimum"], ORANGE)):
            ax.axvline(v, color=c, lw=0.8, ls=":")
        ax.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (measured units)")
        ax.set_title(f"line {tag}, (+⁴−) side: ε = {r['eps']:g}, n_sub = {r['n13']:+.1f}×10¹³ cm⁻² (step-1 parameters); "
                     f"minimum measured {1e3 * r['measured_min']:+.0f}, model {1e3 * r['tip']['minimum']:+.0f} (tip) / "
                     f"{1e3 * r['free']['minimum']:+.0f} mV (no tip)", fontsize=8.8, loc="left", color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper center")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax = axs[row, 1]
        E = np.asarray(r["ldos_E"])
        ax.semilogy(E, r["ldos_top"], color=BLUE, lw=1.5, label="layer 1 (top)")
        ax.semilogy(E, r["ldos_l2"], color=ORANGE, lw=1.5, label="layer 2 (buried flat band)")
        ax.axvline(0.0, color=GRID, lw=0.8)
        U = np.round(1e3 * np.asarray(r["U_minus_mu0"])).astype(int).tolist()
        ax.set(xlim=(-0.3, 0.3), xlabel="E − E_F (eV)", ylabel="layer LDOS, no tip (log)")
        ax.set_title(f"layer potentials U − E_F (top → bottom) {U} meV", fontsize=8.8, loc="left", color=INK)
        ax.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(REPO, "examples", "stacking_wall_step2.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    if sys.argv[1] == "run":
        os.makedirs(OUT, exist_ok=True)
        with mp.get_context("spawn").Pool(2) as pool:
            list(pool.imap_unordered(evaluate, ["940", "514"]))
    else:
        plot()
