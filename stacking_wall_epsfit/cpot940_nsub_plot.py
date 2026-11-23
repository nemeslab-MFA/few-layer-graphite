"""Figure of cpot940_nsub.py: ../examples/stacking_wall_cpot940_nsub.png (top-layer dI/dV under the tip against the
substrate charge, CPot -0.367 eV, R 20 nm, d 0.7 nm, neutral stack, eps 2.6, full SWMcC)."""
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
import cpot940_nsub as N  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
NEG = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]                # n_sub -2 .. -0.5 (dark = large)
POS = ["#f5b8b7", "#eb6a69", "#d23a39", "#9e2b2a"]                # n_sub +0.5 .. +2
OFFSET = 0.004
V_FB = -0.0324


def colour(n):
    k = round(abs(n) / 0.5e13)
    if k == 0:
        return MUTED
    return NEG[4 - k] if n < 0 else POS[k - 1]


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    rs = sorted(N.rows(), key=lambda r: r["nsub"])
    Vr, yr = rescore.measured("940")
    fm = 1e3 * np.asarray(rs[0]["measured_features"])
    F = step1.FINE
    w = (Vr >= -0.45) & (Vr <= 0.45)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.9), height_ratios=(1.25, 1), hspace=0.32, wspace=0.28)
    a = fig.add_subplot(gs[0, :2])
    for k, r in enumerate(rs):
        y = np.asarray(r["tip"]["didv"]) * r["tip"]["scale"]
        a.plot(F, y + k * OFFSET, color=colour(r["nsub"]), lw=1.4)
        a.text(0.455, y[-1] + k * OFFSET, f"{r['nsub'] / 1e13:+.1f}", fontsize=7.5, va="center", color=INK)
    a.plot(Vr[w], yr[w] + len(rs) * OFFSET, color=INK, lw=1.9)
    a.text(0.455, yr[w][-1] + len(rs) * OFFSET, "measured", fontsize=7.5, va="center", color=INK)
    for f in fm:
        a.axvline(1e-3 * f, color=GRID, lw=0.9, zorder=0)
    a.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel=f"dI/dV (model scaled), offset {OFFSET}")
    a.set_title("top-layer dI/dV under the tip; labels: n_sub (10¹³ e/cm², charge in the trap layer; + induces "
                "electrons in the stack); grey lines: measured features", fontsize=8.5, loc="left")
    style(a)

    # layer potentials under the apex at the flat-band bias
    e = fig.add_subplot(gs[0, 2])
    L = np.arange(1, 7)
    e.axhline(0, color=INK, lw=0.9, ls="--")
    for r in rs:
        sw = r["sweep"]
        P = PchipInterpolator(np.asarray(sw["bias"]), np.asarray(sw["U0"]), axis=0)
        e.plot(L, 1e3 * (P(V_FB) - sw["mu0"]), "o-", color=colour(r["nsub"]), lw=1.4, ms=3.5)
    e.set(xticks=L, xlabel="layer (1 = top, faces the tip)", ylabel="U_l − E_F under the apex (meV)",
          title=f"layer potentials with the tip, V = {1e3 * V_FB:+.0f} mV")
    e.title.set_fontsize(8.5)
    style(e)

    # flat-band region, overlaid
    b = fig.add_subplot(gs[1, 0])
    wz = (Vr >= -0.16) & (Vr <= 0.10)
    for r in rs:
        y = np.asarray(r["tip"]["didv"]) * r["tip"]["scale"]
        b.plot(1e3 * F, y, color=colour(r["nsub"]), lw=1.3)
    b.plot(1e3 * Vr[wz], yr[wz], color=INK, lw=1.9, label="measured")
    b.set(xlim=(-160, 100), xlabel="sample bias (mV)", ylabel="dI/dV (model scaled)", title="flat-band region, overlaid")
    b.title.set_fontsize(8.5)
    b.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(b)

    x = np.array([r["nsub"] / 1e13 for r in rs])
    pv = lambda r, lab: 1e3 * r[lab]["peak"]["v"] if r[lab]["peak"] else np.nan   # noqa: E731
    c = fig.add_subplot(gs[1, 1])
    c.plot(x, [pv(r, "tip") for r in rs], "o-", color=INK2, lw=1.6, ms=4.5, label="with the tip (CPot −0.367 eV)")
    c.plot(x, [pv(r, "free") for r in rs], "o--", color=MUTED, lw=1.3, ms=4, mfc="white", label="no tip")
    c.axhline(fm[1], color=INK, lw=1.0, ls=":")
    c.annotate(f"measured {fm[1]:+.0f} mV", (x[0], fm[1]), textcoords="offset points", xytext=(2, 4), fontsize=7.5,
               color=INK2)
    c.set(xlabel="n_sub (10¹³ e/cm²)", ylabel="flat-band peak (mV)", title="flat-band position")
    c.title.set_fontsize(8.5)
    c.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(c)

    d = fig.add_subplot(gs[1, 2])
    names = ("lower bulk edge", "flat band", "upper edge 1", "upper edge 2")
    for i, mk in ((0, "v"), (2, "^"), (3, "s")):
        d.plot(x, [1e3 * r["tip"]["features"][i] - fm[i] if r["tip"]["features"][i] is not None else np.nan
                   for r in rs], marker=mk, color=INK2, lw=1.4, ms=5, label=names[i])
    d.axhline(0, color=INK, lw=1.0, ls=":")
    d.set(xlabel="n_sub (10¹³ e/cm²)", ylabel="model − measured (mV)", title="bulk edges with the tip, error")
    d.title.set_fontsize(8.5)
    d.legend(fontsize=7.5, frameon=False)
    style(d)

    fig.suptitle("Line 940 (+⁵): substrate-charge sweep at the fitted tip (CPot −0.367 eV, R 20 nm, d 0.7 nm); neutral "
                 "stack, ε 2.6, full SWMcC, σ 12.5 meV, T 9 K", fontsize=10, x=0.01, ha="left")
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940_nsub.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
