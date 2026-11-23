"""Why the tip moves the occupied-side features of the 940 (+5) spectrum and not the empty-side ones, in the CPot-only
fit (neutral stack, eps 2.6, full SWMcC, R 20 nm, d 0.7 nm).  ../examples/stacking_wall_cpot940_tibb_sketch.png

Top: energy diagrams (tip | vacuum | six layers) at the biases of three spectral features, with the self-consistent
layer potentials of the sweep.  Bottom: top-layer shift and local lever arm against bias for CPot -0.367 eV and 0."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, METAL = "#2a78d6", "#eb6834", "#dcdbd6"
cpot_fit = lambda col: col == BLUE  # noqa: E731
VAC = 0.80                     # sample vacuum level drawn at +0.80 eV (true value W_RG ~ 4.6 eV above E_F)
DZ = 0.335                     # layer spacing (nm)


def load(cp):
    r = json.load(open(os.path.join(C.OUT, f"cpot{1e3 * cp:+07.1f}.json")))
    sw = r["sweep"]
    V = np.asarray(sw["bias"])
    dU = np.asarray(sw["U0"]) - (np.asarray(r["U_free_minus_EF"]) + r["mu0"])
    return r, V, dU, PchipInterpolator(V, dU, axis=0)


def tibb0(V, d1):
    k = np.where(np.diff(np.sign(d1)) != 0)[0][0]
    return V[k] - d1[k] * (V[k + 1] - V[k]) / (d1[k + 1] - d1[k])


def diagram(ax, vb, dUl, levels, probe, cpot, v0, title, label_lines=False):
    z = DZ * np.arange(6)
    xt, xb = -C.step1.TIP["separation"], z[-1] + 0.45
    ax.fill_between([xt - 0.9, xt], -0.62, vb, color=METAL, lw=0)
    ax.plot([xt - 0.9, xt], [vb, vb], color=INK, lw=1.4)
    ax.text(xt - 0.86, vb + 0.025, "E_F tip", fontsize=7.5, color=INK2)
    ax.text(xt - 0.45, -0.57, "tip", fontsize=8, color=INK2, ha="center")
    ax.plot([-0.12, xb], [0, 0], color=INK, lw=1.0, ls="--")
    ax.text(xb + 0.03, 0, "E_F", fontsize=7.5, color=INK2, va="center")
    tv = vb + VAC + cpot                                                  # tip vacuum level
    zs = np.concatenate([[xt - 0.9, xt], z, [xb]])
    ev = np.concatenate([[tv, tv], VAC + dUl, [VAC + dUl[-1]]])
    ax.plot(zs, ev, color=INK2, lw=1.3)
    ax.text(xb + 0.03, VAC, "E_vac\n(drawn\n3.8 eV low)", fontsize=6.8, color=INK2, va="center")
    for name, e in levels.items():
        lab = name == probe
        col, lw = (BLUE, 2.2) if lab else ((INK, 1.6) if name == "flat band" else (MUTED, 1.1))
        xs = np.concatenate([z, [xb]]) if name != "flat band" else np.array([z[0] - 0.12, z[0] + 0.12])
        ys = e + (np.concatenate([dUl, [dUl[-1]]]) if name != "flat band" else np.full(2, dUl[0]))
        ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="round")
        if name != "flat band":
            ax.plot(xs, e + np.zeros_like(xs), color=GRID, lw=0.8, zorder=0)
    for zl in z:
        ax.plot([zl, zl], [-0.62, -0.50], color=MUTED, lw=0.8)
    ax.text(z[0], -0.47, "L1", fontsize=6.8, color=INK2, ha="center")
    ax.text(z[-1], -0.47, "L6", fontsize=6.8, color=INK2, ha="center")
    if vb > 0:
        ax.annotate("", xy=(0.0, vb), xytext=(xt, vb), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    else:
        ax.annotate("", xy=(xt, vb), xytext=(0.0, vb), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    ax.set_title(f"{title}\nV = {1e3 * vb:+.0f} mV,  V − V₀ = {vb - v0:+.2f} V,  ΔU₁ = {1e3 * dUl[0]:+.0f} meV",
                 fontsize=9, loc="left", color=INK)
    if label_lines:
        ax.text(xb + 0.03, levels["lower bulk edge"] + dUl[-1], "bulk edge", fontsize=6.8, color=INK2, va="center")
        ax.text(xb + 0.03, levels["upper bulk edge 1"] + dUl[-1], "bulk edge", fontsize=6.8, color=INK2, va="center")
        ax.text(xb + 0.03, levels["upper bulk edge 2"] + dUl[-1], "bulk edge", fontsize=6.8, color=INK2, va="center")
        ax.text(z[0] + 0.16, levels["flat band"] + dUl[0], "flat band (L1)", fontsize=6.8, color=INK2, va="center")
    ax.set(xlim=(xt - 0.95, xb + 0.55), ylim=(-0.62, 1.02), xticks=[], ylabel="electron energy − E_F,sample (eV)")
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    rf, V, dUf, Pf = load(-0.367)
    r0, V0s, dU0, P0 = load(0.0)
    v0f, v00 = tibb0(V, dUf[:, 0]), tibb0(V0s, dU0[:, 0])
    ff, ft, f0 = (np.asarray(x) for x in (rf["free"]["features"], rf["tip"]["features"], r0["tip"]["features"]))
    ff[1], ft[1] = -0.0038, -0.0324                                       # flat band from the 0.25 mV re-evaluation
    names = ("lower bulk edge", "flat band", "upper bulk edge 1", "upper bulk edge 2")
    levels = dict(zip(names, ff))

    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 6, height_ratios=(1.15, 1), hspace=0.32, wspace=0.9)
    for k, (i, title) in enumerate(((0, "lower bulk edge: strong field"), (1, "flat band: strong field"),
                                    (3, "upper bulk edge: field ~ 0 (TIBB = 0 at V₀)"))):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        diagram(ax, float(ft[i]), Pf(float(ft[i])), levels, names[i], -0.367, v0f, title, label_lines=(k == 0))

    a = fig.add_subplot(gs[1, :3])
    b = fig.add_subplot(gs[1, 3:])
    x = np.linspace(-0.45, 0.45, 361)
    for P, v0, col, lab in ((Pf, v0f, BLUE, "CPot −0.367 eV (fit)"), (P0, v00, ORANGE, "CPot 0")):
        a.plot(x, 1e3 * P(x)[:, 0], color=col, lw=2.0, label=lab)
        a.plot([v0], [0], "v", color=col, ms=8, mec="white", mew=1.0)
        a.annotate(f"TIBB = 0\nat {v0:+.2f} V", (v0, 0), textcoords="offset points",
                   xytext=(0, -32) if cpot_fit(col) else (-14, 10), ha="center" if cpot_fit(col) else "right",
                   fontsize=7.5, color=INK2)
        b.plot(x, P.derivative()(x)[:, 0], color=col, lw=2.0, label=lab)
    xs = x[x - 0.367 >= -0.45]
    a.plot(xs, 1e3 * P0(xs - 0.367)[:, 0], color=INK, lw=1.0, ls=(0, (3, 3)), label="CPot 0 curve shifted by +0.367 V")
    for f_t, f_f, col in ((ft, ff, BLUE), (f0, np.asarray(rf["free"]["features"]), ORANGE)):
        a.plot(f_t, 1e3 * (f_t - f_f), "o", mfc="white", mec=col, mew=1.6, ms=8, zorder=5)
    a.plot([], [], "o", mfc="white", mec=INK2, mew=1.6, ms=8, label="shift of a spectral feature (tip − no tip)")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(xlim=(-0.45, 0.45), xlabel="sample bias V (V)", ylabel="tip-induced shift (meV)",
          title="top-layer potential under the apex, ΔU₁ = U₁(tip) − U₁(no tip); circles: features")
    a.legend(fontsize=7.5, frameon=False, loc="lower right")
    b.set(xlim=(-0.45, 0.45), xlabel="sample bias V (V)", ylabel="dU₁/d(eV)",
          title="local lever arm of the top layer")
    for xy in ((-0.30, 0.218), (0.42, 0.122)):
        b.annotate("flat band pushed away from E_F:\nno top-layer states at E_F,\nthe field penetrates", xy,
                   xytext=(0.10, 0.185), textcoords="data", fontsize=7.5, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    b.text(-0.44, 0.214, "CPot −0.367 eV (fit)", fontsize=7.5, color=INK, va="top")
    b.text(-0.40, 0.162, "CPot 0", fontsize=7.5, color=INK, va="center")
    b.annotate("flat band at E_F screens", (0.30, 0.058), textcoords="offset points", xytext=(-40, 40),
               fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for ax in (a, b):
        ax.set_title(ax.get_title(), loc="left", fontsize=9)
        ax.set_title("", loc="center")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940_tibb_sketch.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, "V0 fit", v0f, "V0 cpot0", v00)
    print("feature shifts fit (mV)", 1e3 * (ft - ff), "cpot0", 1e3 * (f0 - np.asarray(rf["free"]["features"])))
    print("dU1 at features fit (meV)", 1e3 * Pf(ft)[:, 0])
