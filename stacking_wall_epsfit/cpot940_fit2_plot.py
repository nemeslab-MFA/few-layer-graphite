"""Figure of cpot940_fit2.py: ../examples/stacking_wall_cpot940_fit2.png (two-parameter fit of CPot and n_sub to the
940 (+5) peak positions)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402
import cpot940_fit2 as F2  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NAMES = ("lower bulk edge", "flat band", "upper edge 1", "upper edge 2")
MARK = ("v", "o", "^", "s")
OFFSET = 0.004


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def valley_rows():
    pl = {round(p["nsub"] / 1e13, 4): p["cpot"] for p in
          (json.load(open(os.path.join(F2.OUT, f))) for f in os.listdir(F2.OUT) if f.startswith("place_"))}
    rs = {(round(r["cpot"], 4), round(r["nsub"] / 1e13, 4)): r for r in F2.rows()}
    out = [rs[(round(c, 4), n)] for n, c in pl.items() if c is not None and (round(c, 4), n) in rs]
    return sorted(out, key=lambda r: -r["nsub"]), rs


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    val, allr = valley_rows()
    best = min(allr.values(), key=lambda r: r["rms4"])
    Vr, yr = rescore.measured(C.TAG)
    fm = 1e3 * np.asarray(best["measured_features"])
    FINE = step1.FINE
    ramp = ["#c5ddf7", "#9cc3f0", "#6fa6e8", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b", "#081f40"]

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.35, 1, 1), height_ratios=(1.25, 1), hspace=0.32, wspace=0.28)
    a = fig.add_subplot(gs[:, 0])
    w = (Vr >= -0.45) & (Vr <= 0.45)
    for k, r in enumerate(val):
        y = np.asarray(r["didv"]) * r["scale"]
        col = ramp[min(k, len(ramp) - 1)]
        a.plot(FINE, y + k * OFFSET, color=col, lw=2.0 if r is best else 1.3)
        a.text(0.455, y[-1] + k * OFFSET, f"{r['nsub'] / 1e13:+.3g}  ({r['cpot']:+.3f} eV)", fontsize=7.2,
               va="center", color=INK)
    a.plot(Vr[w], yr[w] + len(val) * OFFSET, color=INK, lw=1.9)
    a.text(0.455, yr[w][-1] + len(val) * OFFSET, "measured", fontsize=7.5, va="center", color=INK)
    for f in fm:
        a.axvline(1e-3 * f, color=GRID, lw=0.9, zorder=0)
    a.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel=f"dI/dV (model scaled), offset {OFFSET}")
    a.set_title("along the flat-band valley; labels: n_sub (10¹³ e/cm²)  (CPot)", fontsize=8.5, loc="left")
    a.set_xlim(-0.45, 0.62)
    style(a)

    x = np.array([r["nsub"] / 1e13 for r in val])
    b = fig.add_subplot(gs[0, 1])
    b.plot(x, [r["cpot"] for r in val], "o-", color=INK2, lw=1.6, ms=4.5)
    b.set(xlabel="n_sub (10¹³ e/cm²)", ylabel=f"CPot for the flat band at {fm[1]:+.1f} mV (eV)".replace("-", "−"),
          title="flat-band valley")
    b.title.set_fontsize(8.5)
    style(b)

    c = fig.add_subplot(gs[0, 2])
    for i in range(4):
        c.plot(x, [1e3 * r["err"][i] for r in val], marker=MARK[i], color=INK2 if i != 1 else MUTED, lw=1.3, ms=5,
               label=NAMES[i])
    c.plot(x, [1e3 * r["rms4"] for r in val], color=BLUE, lw=2.0, label="rms of the four")
    c.axhline(0, color=INK, lw=0.9, ls=":")
    c.set(xlabel="n_sub (10¹³ e/cm²)", ylabel="model − measured (mV)", ylim=(-45, 80),
          title="peak-position errors along the valley")
    c.title.set_fontsize(8.5)
    c.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(c)

    d = fig.add_subplot(gs[1, 1])
    off = sorted([r for (cc, n), r in allr.items() if n == round(best["nsub"] / 1e13, 4)], key=lambda r: r["cpot"])
    for i in (0, 1, 3):
        d.plot([r["cpot"] for r in off], [1e3 * r["err"][i] for r in off], marker=MARK[i],
               color=INK2 if i != 1 else MUTED, lw=1.3, ms=5, label=NAMES[i])
    d.plot([r["cpot"] for r in off], [1e3 * r["rms4"] for r in off], color=BLUE, lw=2.0, label="rms of the four")
    d.axhline(0, color=INK, lw=0.9, ls=":")
    d.set(xlabel="CPot (eV)", ylabel="model − measured (mV)",
          title=f"CPot across the valley at n_sub = {best['nsub'] / 1e13:+.3g} × 10¹³")
    d.title.set_fontsize(8.5)
    d.legend(fontsize=7.5, frameon=False)
    style(d)

    e = fig.add_subplot(gs[1, 2])
    zero = allr[min((k for k in allr if k[1] == 0.0), key=lambda k: abs(k[0] - [r["cpot"] for r in val if r["nsub"] == 0][0]))]
    e.plot(Vr[w], yr[w], color=INK, lw=1.9, label="measured")
    e.plot(FINE, np.asarray(zero["didv"]) * zero["scale"], color=ORANGE, lw=1.3,
           label=f"CPot only: {zero['cpot']:+.3f} eV, n_sub 0  (rms {1e3 * zero['rms4']:.0f} mV)")
    e.plot(FINE, np.asarray(best["didv"]) * best["scale"], color=BLUE, lw=1.6,
           label=f"best: {best['cpot']:+.3f} eV, {best['nsub'] / 1e13:+.3g} × 10¹³  (rms {1e3 * best['rms4']:.1f} mV)")
    for f in fm:
        e.axvline(1e-3 * f, color=GRID, lw=0.9, zorder=0)
    e.set(xlim=(-0.45, 0.45), ylim=(0, 0.0215), xlabel="sample bias (V)", ylabel="dI/dV (measured units; model scaled)",
          title="best fit against the CPot-only fit")
    e.title.set_fontsize(8.5)
    e.legend(fontsize=7, frameon=False, loc="upper center")
    style(e)

    fig.suptitle(f"Line {C.TAG} (+⁵): CPot and substrate charge fitted to the four peak positions (shapes ignored); neutral "
                 "stack, ε 2.6, full SWMcC, R 20 nm, d 0.7 nm", fontsize=10, x=0.01, ha="left")
    out = os.path.join(REPO, "examples", f"stacking_wall_cpot{C.TAG}_fit2.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out, "best", best["cpot"], best["nsub"], best["rms4"])
