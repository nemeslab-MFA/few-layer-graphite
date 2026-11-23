"""Full-SWMcC band structure of the six-layer (+5) stack of the CPot-only 940 fit, without the tip and with the layer
potentials under the apex (CPot -0.367 eV, R 20 nm, d 0.7 nm) at the flat-band bias.  ../examples/stacking_wall_cpot940_bands.png

Bands along a line through K (towards Gamma on the left, towards M on the right; gamma3 warping makes the two
directions differ), coloured by the weight on the top layer (L1, the layer the tip probes).  Energies relative to the
sample Fermi level: E - E_F = eig(H(k) + diag(U_l)) - mu0."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE = "#2a78d6"
CMAP = LinearSegmentedColormap.from_list("l1", ["#cfcec8", "#86b6ef", "#2a78d6", "#0d366b"])   # weight 0 -> 1
QMAX = 0.8                                                       # 1/nm from K


def kline(a, n=1601):
    """k along Gamma <- K -> M, q in 1/nm (q < 0 towards Gamma, q > 0 towards M)."""
    K = np.array([4.0 * math.pi / (3.0 * a), 0.0])
    M = np.array([math.pi / a, math.pi / (math.sqrt(3.0) * a)])
    dG, dM = -K / np.linalg.norm(K), (M - K) / np.linalg.norm(M - K)
    q = np.linspace(-QMAX, QMAX, n)
    k = K[None] + np.where(q[:, None] < 0, -q[:, None] * dG[None], q[:, None] * dM[None])
    return q, k


def bands(tb, U, mu0, k):
    from bandbending.materials.multilayer import structure_factor_swmcc
    u = structure_factor_swmcc(k[:, 0], k[:, 1], tb.params.a)
    E, P = np.linalg.eigh(tb._h_full(u, np.asarray(U)))
    w = (np.abs(P.reshape(P.shape[0], tb.N, 2, 2 * tb.N)) ** 2).sum(axis=2)
    return E - mu0, w[:, 0, :]                                  # (nk, 2N), top-layer weight (nk, 2N)


def draw(ax, q, E, w, lw=1.8):
    for b in range(E.shape[1]):
        pts = np.column_stack([q, 1e3 * E[:, b]])
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=CMAP, norm=plt.Normalize(0, 1), lw=lw, capstyle="round")
        lc.set_array(0.5 * (w[:-1, b] + w[1:, b]))
        ax.add_collection(lc)
    return lc


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    tf = C.S.tipfree(C.TAG, C.EPS, 0.0, 0.0, C.MODEL, "RG")
    tb, mu0f, Uf = tf["p"].tb, tf["mu0"], tf["U"]
    r = json.load(open(os.path.join(C.OUT, "cpot-0367.0.json")))
    sw = r["sweep"]
    fine = json.load(open(os.path.join(C.OUT, "geom_fine.json")))
    v_fb = next(t for t in fine["tip"] if t["d"] == 0.7 and t["R"] == 20.0)["peak"]["v"]
    P = PchipInterpolator(np.asarray(sw["bias"]), np.asarray(sw["U0"]), axis=0)
    cases = {"no tip": (Uf, mu0f), f"tip, V = {1e3 * v_fb:+.0f} mV": (P(v_fb), sw["mu0"])}
    extra = {f"tip, V = {v:+.0f} mV": (P(1e-3 * v), sw["mu0"]) for v in (-270, 355)}

    q, k = kline(tb.params.a)
    res = {lab: bands(tb, U, m, k) for lab, (U, m) in cases.items()}

    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 5, width_ratios=(1, 1, 0.035, 0.1, 0.8), height_ratios=(1, 0.9), hspace=0.3, wspace=0.3)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, (lab, (E, w)) in zip(axs, res.items()):
        lc = draw(ax, q, E, w)
        ax.axhline(0, color=INK, lw=0.9, ls="--")
        ax.axvline(0, color=GRID, lw=0.9)
        ax.set(xlim=(-QMAX, QMAX), ylim=(-450, 450), xlabel="k − K (1/nm):  ← towards Γ   |   towards M →",
               ylabel="E − E_F (meV)")
        ax.set_title("no tip" if lab == "no tip" else f"{lab}: potentials under the apex (CPot −0.367 eV)",
                     fontsize=9, loc="left")
        ax.text(0.78, 6, "E_F", fontsize=7.5, color=INK2, ha="right", va="bottom")
        if lab != "no tip":
            ax.set_ylabel("")
        style(ax)
    cb = fig.colorbar(lc, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("weight on the top layer L1")
    cb.outline.set_visible(False)

    # layer potentials
    c = fig.add_subplot(gs[0, 4])
    L = np.arange(1, 7)
    c.axhline(0, color=INK, lw=0.9, ls="--")
    c.plot(L, 1e3 * (np.asarray(Uf) - mu0f), "o--", color=MUTED, lw=1.6, ms=5, label="no tip")
    c.plot(L, 1e3 * (P(v_fb) - sw["mu0"]), "o-", color=BLUE, lw=1.8, ms=5, label=f"tip, V = {1e3 * v_fb:+.0f} mV")
    for (lab, (U, m)), col in zip(extra.items(), ("#0d366b", "#86b6ef")):
        c.plot(L, 1e3 * (U - m), "o-", color=col, lw=1.2, ms=3.5, label=lab)
    c.set(xticks=L, xlabel="layer (1 = top, faces the tip)", ylabel="U_l − E_F (meV)",
          title="layer potentials under the apex")
    c.title.set_fontsize(9)
    c.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(c)

    # zoom overlay near E_F
    z = fig.add_subplot(gs[1, :2])
    (E0, w0), (E1, w1) = res.values()
    for b in range(E0.shape[1]):
        z.plot(q, 1e3 * E0[:, b], color=MUTED, lw=1.3, ls="--", label="no tip" if b == 0 else None)
        z.plot(q, 1e3 * E1[:, b], color=BLUE, lw=1.6, label=f"tip, V = {1e3 * v_fb:+.0f} mV" if b == 0 else None)
    z.axhline(0, color=INK, lw=0.9)
    z.axvline(0, color=GRID, lw=0.9)
    z.set(xlim=(-0.45, 0.45), ylim=(-140, 100), xlabel="k − K (1/nm):  ← towards Γ   |   towards M →",
          ylabel="E − E_F (meV)", title="the two surface flat bands, overlaid")
    i0 = int(np.argmin(np.abs(q)))
    N = E1.shape[1] // 2
    lo, hi = (N - 1, N) if E1[i0, N - 1] < E1[i0, N] else (N, N - 1)
    z.annotate(f"top-surface band: {100 * w1[i0, lo]:.0f} % on L1 at K", (0.0, 1e3 * E1[i0, lo]), textcoords="offset points",
               xytext=(40, -22), fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    z.annotate(f"bottom-surface band: {100 * w1[i0, hi]:.0f} % on L1", (0.12, 1e3 * E1[np.argmin(np.abs(q - 0.12)), hi]),
               textcoords="offset points", xytext=(30, 26), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    z.annotate(f"no tip: the two surface bands hybridise,\n{100 * w0[i0, N - 1]:.0f} / {100 * w0[i0, N]:.0f} % on L1 at K",
               (-0.05, 1e3 * E0[i0, N - 1]), textcoords="offset points", xytext=(-150, 40), fontsize=8, color=INK2,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    z.title.set_fontsize(9)
    z.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    style(z)

    # top-layer LDOS
    d = fig.add_subplot(gs[1, 4])
    Eg = np.arange(-0.45, 0.45 + 1e-9, 5e-4)
    for (lab, (U, m)), col, ls in zip(cases.items(), (MUTED, BLUE), ("--", "-")):
        _, D = tb.ldos(np.asarray(U), Eg + m, broadening=C.SIG, resolve="layer")
        D = np.asarray(D)
        D = D if D.shape[0] == Eg.size else D.T
        d.plot(D[:, 0] / 1e13, 1e3 * Eg, color=col, ls=ls, lw=1.6, label=lab)
    d.axhline(0, color=INK, lw=0.9)
    d.set(ylim=(-450, 450), xlabel="top-layer LDOS (10¹³ states / eV cm²)", ylabel="E − E_F (meV)",
          title=f"top-layer LDOS, σ {1e3 * C.SIG:.1f} meV")
    d.title.set_fontsize(9)
    d.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(d)

    fig.suptitle("Six-layer rhombohedral graphite (+⁵), full SWMcC, neutral stack, ε 2.6: bands without the tip and "
                 "with the self-consistent potentials under the apex at the fitted contact potential", fontsize=10,
                 x=0.01, ha="left")
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940_bands.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for lab, (E, w) in res.items():
        N = E.shape[1] // 2
        i0 = int(np.argmin(np.abs(q)))
        print(f"{lab}: E at K (meV) {np.round(1e3 * E[i0], 1)}; top-layer weight at K {np.round(w[i0], 2)}")
    for lab, (U, m) in {**cases, **extra}.items():
        print(f"{lab}: U_l - E_F (meV) {np.round(1e3 * (np.asarray(U) - m), 1)}")
