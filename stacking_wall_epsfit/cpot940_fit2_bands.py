"""Band structure at the two-parameter best fit of the 940 (+5) peak positions (CPot -0.615 eV, n_sub -0.5e13 cm^-2,
neutral stack, eps 2.6, tip R 20 nm, d 0.7 nm), without the tip and with the layer potentials under the apex.

    ../examples/stacking_wall_cpot940_fit2_bands.png   engine tight binding (full SWMcC), coloured by top-layer weight
    ../examples/stacking_wall_cpot940_fit2_kp.png      the paper's SWMcC k.p model (multilayer_graphene, McEllistrim
                                                       et al. 2023) with the same layer on-site energies, coloured by
                                                       the layer the state sits on; engine bands dotted for comparison
"""
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
sys.path.insert(1, REPO)
import cpot940 as C  # noqa: E402
import cpot940_bands as B  # noqa: E402
import cpot940_fit2 as F2  # noqa: E402
import multilayer_graphene as mlg  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, RED = "#2a78d6", "#d23a39"
BEST_C, BEST_N = -0.615, -0.5e13
V_FB, V_BOT = -0.0324, 0.112
LAYER = LinearSegmentedColormap.from_list("layer", ["#1c5cab", "#86b6ef", "#cfcec8", "#eb9a99", "#b8302f"])


def cases():
    r = json.load(open(os.path.join(F2.OUT, F2.pkey(BEST_C, BEST_N) + ".json")))
    sw = r["sweep"]
    P = PchipInterpolator(np.asarray(sw["bias"]), np.asarray(sw["U0"]), axis=0)
    tf = C.S.tipfree(C.TAG, C.EPS, 0.0, BEST_N, C.MODEL, "RG")
    return tf["p"].tb, {"no tip": (np.asarray(tf["U"]), tf["mu0"]),
                        f"tip, V = {1e3 * V_FB:+.0f} mV": (P(V_FB), sw["mu0"]),
                        f"tip, V = {1e3 * V_BOT:+.0f} mV": (P(V_BOT), sw["mu0"])}, r


def kp_bands(tb, U, mu0, q, dirs):
    """Paper's SWMcC k.p model (multilayer_graphene, layers bottom-first) with on-site U (engine order, top-first)."""
    p = mlg.get_parameters()
    a_cc = p["a"] / math.sqrt(3.0)
    pv = 0.1 * np.where(q[:, None] < 0, -q[:, None] * dirs[0][None], q[:, None] * dirs[1][None])   # 1/A
    u = -1.5 * a_cc * (pv[:, 0] - 1j * pv[:, 1])
    H = mlg._assemble(u, "ABCABC", p) + np.diag(np.repeat(np.asarray(U)[::-1], 2))
    E, V = np.linalg.eigh(H)
    w = (np.abs(V.reshape(V.shape[0], 6, 2, 12)) ** 2).sum(axis=2)            # (nk, layer bottom-first, band)
    centroid = 6.0 - np.einsum("klb,l->kb", w, np.arange(6))                 # 1 = top ... 6 = bottom
    return E + tb.e_shift - mu0, centroid


def tb_full(tb, U, mu0, k):
    from bandbending.materials.multilayer import structure_factor_swmcc
    u = structure_factor_swmcc(k[:, 0], k[:, 1], tb.params.a)
    E, V = np.linalg.eigh(tb._h_full(u, np.asarray(U)))
    w = (np.abs(V.reshape(V.shape[0], 6, 2, 12)) ** 2).sum(axis=2)            # top-first
    return E - mu0, w


def draw(ax, q, E, c, cmap, norm, lw=1.8):
    for b in range(E.shape[1]):
        pts = np.column_stack([q, 1e3 * E[:, b]])
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, lw=lw, capstyle="round")
        lc.set_array(0.5 * (c[:-1, b] + c[1:, b]))
        ax.add_collection(lc)
    return lc


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    tb, cs, r = cases()
    labs = list(cs)
    q, k = B.kline(tb.params.a)
    K = np.array([4.0 * math.pi / (3.0 * tb.params.a), 0.0])
    M = np.array([math.pi / tb.params.a, math.pi / (math.sqrt(3.0) * tb.params.a)])
    dirs = (-K / np.linalg.norm(K), (M - K) / np.linalg.norm(M - K))
    res = {lab: tb_full(tb, *cs[lab], k) for lab in labs}
    i0 = int(np.argmin(np.abs(q)))
    xl = "k − K (1/nm):  ← towards Γ   |   towards M →"

    # ---- figure 1: engine bands, top-layer weight
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 5, width_ratios=(1, 1, 0.035, 0.1, 0.8), height_ratios=(1, 0.9), hspace=0.3, wspace=0.3)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, lab in zip(axs, labs[:2]):
        E, w = res[lab]
        lc = draw(ax, q, E, w[:, 0, :], B.CMAP, plt.Normalize(0, 1))
        ax.axhline(0, color=INK, lw=0.9, ls="--")
        ax.axvline(0, color=GRID, lw=0.9)
        ax.set(xlim=(-B.QMAX, B.QMAX), ylim=(-450, 450), xlabel=xl, ylabel="E − E_F (meV)" if ax is axs[0] else "")
        ax.set_title("no tip" if lab == "no tip" else f"{lab}: potentials under the apex", fontsize=9, loc="left")
        B.style(ax)
    cb = fig.colorbar(lc, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("weight on the top layer L1")
    cb.outline.set_visible(False)

    c = fig.add_subplot(gs[0, 4])
    L = np.arange(1, 7)
    c.axhline(0, color=INK, lw=0.9, ls="--")
    for lab, col, ls in zip(labs, (MUTED, BLUE, "#0d366b"), ("--", "-", "-")):
        U, m = cs[lab]
        c.plot(L, 1e3 * (np.asarray(U) - m), "o" + ls, color=col, lw=1.6, ms=4.5, label=lab)
    c.set(xticks=L, xlabel="layer (1 = top, faces the tip)", ylabel="U_l − E_F (meV)", title="layer potentials")
    c.title.set_fontsize(9)
    c.legend(fontsize=7.5, frameon=False, loc="upper left")
    B.style(c)

    z = fig.add_subplot(gs[1, :2])
    (E0, w0), (E1, w1) = res[labs[0]], res[labs[1]]
    N = 6
    for b in range(E0.shape[1]):
        z.plot(q, 1e3 * E0[:, b], color=MUTED, lw=1.3, ls="--", label="no tip" if b == 0 else None)
        z.plot(q, 1e3 * E1[:, b], color=BLUE, lw=1.6, label=labs[1] if b == 0 else None)
    z.axhline(0, color=INK, lw=0.9)
    z.axvline(0, color=GRID, lw=0.9)
    top = N - 1 if w1[i0, 0, N - 1] > w1[i0, 0, N] else N
    bot = N if top == N - 1 else N - 1
    z.annotate(f"top-surface band: {100 * w1[i0, 0, top]:.0f} % on L1 at K, {1e3 * E1[i0, top]:+.0f} meV",
               (0.0, 1e3 * E1[i0, top]), textcoords="offset points", xytext=(40, -24), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    z.annotate(f"bottom-surface band: {100 * w1[i0, 5, bot]:.0f} % on L6 at K, {1e3 * E1[i0, bot]:+.0f} meV",
               (0.0, 1e3 * E1[i0, bot]), textcoords="offset points", xytext=(40, 22), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    z.set(xlim=(-0.45, 0.45), ylim=(-140, 170), xlabel=xl, ylabel="E − E_F (meV)",
          title="the two surface flat bands, overlaid")
    z.title.set_fontsize(9)
    z.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    B.style(z)

    d = fig.add_subplot(gs[1, 4])
    Eg = np.arange(-0.45, 0.45 + 1e-9, 5e-4)
    for lab, col, ls in zip(labs, (MUTED, BLUE, "#0d366b"), ("--", "-", "-")):
        U, m = cs[lab]
        _, D = tb.ldos(np.asarray(U), Eg + m, broadening=C.SIG, resolve="layer")
        D = np.asarray(D)
        D = D if D.shape[0] == Eg.size else D.T
        d.plot(D[:, 0] / 1e13, 1e3 * Eg, color=col, ls=ls, lw=1.5, label=lab)
    d.axhline(0, color=INK, lw=0.9)
    d.set(ylim=(-450, 450), xlabel="top-layer LDOS (10¹³ states / eV cm²)", ylabel="E − E_F (meV)",
          title=f"top-layer LDOS, σ {1e3 * C.SIG:.1f} meV")
    d.title.set_fontsize(9)
    d.legend(fontsize=7.5, frameon=False, loc="upper right")
    B.style(d)
    fig.suptitle(f"Best fit of the 940 (+⁵) peak positions: CPot {BEST_C:+.3f} eV, n_sub {BEST_N / 1e13:+.2f} × 10¹³ cm⁻²; "
                 "full SWMcC, neutral stack, ε 2.6, R 20 nm, d 0.7 nm", fontsize=10, x=0.01, ha="left")
    out1 = os.path.join(REPO, "examples", "stacking_wall_cpot940_fit2_bands.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: the paper's SWMcC k.p model with the same on-site energies
    kp = {lab: kp_bands(tb, *cs[lab], q, dirs) for lab in labs[:2]}
    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5), gridspec_kw=dict(hspace=0.28, wspace=0.18))
    norm = plt.Normalize(1, 6)
    for j, lab in enumerate(labs[:2]):
        Ek, cen = kp[lab]
        Et = res[lab][0]
        for i, (ylim, xlim) in enumerate((((-450, 450), (-B.QMAX, B.QMAX)), ((-140, 170), (-0.45, 0.45)))):
            ax = axs[i, j]
            lc = draw(ax, q, Ek, cen, LAYER, norm, lw=2.0 if i else 1.8)
            if i:
                for b in range(Et.shape[1]):
                    ax.plot(q, 1e3 * Et[:, b], color=INK, lw=0.7, ls=":", label="engine tight binding" if b == 0 else None)
                ax.legend(fontsize=7.5, frameon=False, loc="lower left")
            ax.axhline(0, color=INK, lw=0.9, ls="--")
            ax.axvline(0, color=GRID, lw=0.9)
            ax.set(xlim=xlim, ylim=ylim, xlabel=xl, ylabel="E − E_F (meV)" if j == 0 else "")
            ax.set_title(("no tip" if lab == "no tip" else f"{lab}, potentials under the apex")
                         + ("" if i == 0 else ", near E_F"), fontsize=9, loc="left")
            B.style(ax)
    cb = fig.colorbar(lc, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("layer the state sits on, ⟨l⟩ (1 = top, 6 = bottom)")
    cb.outline.set_visible(False)
    fig.suptitle("SWMcC k·p model of McEllistrim et al. (2023) (multilayer_graphene.py, as in the paper) with the layer "
                 f"on-site energies of the best fit (CPot {BEST_C:+.3f} eV, n_sub {BEST_N / 1e13:+.2f} × 10¹³ cm⁻²)",
                 fontsize=10, x=0.01, ha="left")
    out2 = os.path.join(REPO, "examples", "stacking_wall_cpot940_fit2_kp.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print("wrote", out1, out2)

    # numbers
    for lab in labs:
        E, w = res[lab]
        U, m = cs[lab]
        print(f"{lab}: U-E_F {np.round(1e3 * (np.asarray(U) - m), 1)}; pair at K {np.round(1e3 * E[i0, N - 1:N + 1], 1)} "
              f"meV, L1 {np.round(w[i0, 0, N - 1:N + 1], 2)}, L6 {np.round(w[i0, 5, N - 1:N + 1], 2)}")
    for lab in labs[:2]:
        Ek, _ = kp[lab]
        Et = res[lab][0]
        sel = np.abs(q) <= 0.3
        print(f"{lab}: k.p vs tight binding, max |dE| within 0.3 / 0.8 1/nm: "
              f"{1e3 * np.abs(Ek[sel] - Et[sel]).max():.1f} / {1e3 * np.abs(Ek - Et).max():.1f} meV")
