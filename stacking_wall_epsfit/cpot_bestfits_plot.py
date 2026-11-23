"""Best two-parameter fits (CPot, n_sub) of the (+5) spectra of lines 940 and 514, side by side, with the fitted
parameters and the tip work function they imply for W_RG = 4.6 eV.  ../examples/stacking_wall_bestfits.png"""
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
import features as FE  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE = "#2a78d6"
W_RG = 4.6
TRAP_FRACTION = 0.985                     # share of the 2 nm trap charge induced in the stack (cpot940_bulk.py)
LINES = {"940": dict(dir="cpot940", cpot=-0.615, nsub=-0.5e13, low_max=None),
         "514": dict(dir="cpot514", cpot=0.370, nsub=0.375e13, low_max=-0.15)}
NAMES = ("lower bulk edge", "flat band", "upper edge 1", "upper edge 2")


def load(tag):
    p = LINES[tag]
    key = f"sweep_c{1e3 * p['cpot']:+07.1f}_n{p['nsub'] / 1e13:+06.3f}"
    r = json.load(open(os.path.join(HERE, p["dir"], "fit2", key + ".json")))
    y = np.asarray(r["didv"])
    f = FE.features(step1.FINE, y, low_max=p["low_max"])
    fm = np.asarray(r["measured_features"])
    return r, y, f, fm


def mvs(a):
    return ", ".join(f"{1e3 * x:+.0f}".replace("-", "−") for x in a)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(16, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(3.0, 1.15), hspace=0.18, wspace=0.16)
    out_rows = {}
    for col, tag in enumerate(("940", "514")):
        ax = fig.add_subplot(gs[0, col])
        tx = fig.add_subplot(gs[1, col])
        tx.axis("off")
        p = LINES[tag]
        r, y, f, fm = load(tag)
        Vr, yr = rescore.measured(tag)
        w = (Vr >= -0.45) & (Vr <= 0.45)
        F = step1.FINE
        ax.plot(Vr[w], yr[w], color=INK, lw=1.9, label="measured")
        ax.plot(F, y * r["scale"], color=BLUE, lw=1.7, label="best fit (model scaled)")
        top = max(yr[w].max(), (y * r["scale"]).max())
        for i, (vm, vf) in enumerate(zip(fm, f)):
            ax.axvline(vm, color=GRID, lw=1.0, zorder=0)
            ax.plot([vf, vf], [1.02 * top, 1.07 * top], color=BLUE, lw=2.0, solid_capstyle="butt")
            ax.plot([vm, vm], [1.08 * top, 1.13 * top], color=INK, lw=2.0, solid_capstyle="butt")
        err = 1e3 * (f - fm)
        rms = float(np.sqrt(np.mean(err ** 2)))
        carriers = abs(p["nsub"]) * TRAP_FRACTION
        kind = "holes" if p["nsub"] < 0 else "electrons"
        txt = (f"CPot = W_tip − W_RG = {p['cpot']:+.3f} eV\n"
               f"W_tip ≈ {W_RG + p['cpot']:.2f} eV  (W_RG = {W_RG} eV assumed)\n"
               f"n_sub = {p['nsub'] / 1e13:+.3g} × 10¹³ cm⁻² in the 2 nm trap layer\n"
               f"   → {carriers / 1e12:.1f} × 10¹² {kind} in the stack\n"
               f"fixed: neutral stack, ε 2.6, full SWMcC, d 0.7 nm, R 20 nm,\n"
               f"   σ {1e3 * step1.SIGMA[tag]:.1f} meV, T 9 K\n"
               f"peaks (mV): model {mvs(f)}\n"
               f"            measured {mvs(fm)}\n"
               f"rms of the four positions {rms:.1f} mV").replace("-", "−").replace("−>", "→")
        tx.text(0.0, 1.0, txt, transform=tx.transAxes, va="top", ha="left", fontsize=8.6, family="DejaVu Sans Mono",
                color=INK)
        ax.text(0.455, 1.105 * top, "measured", fontsize=7, color=INK, va="center")
        ax.text(0.455, 1.045 * top, "model", fontsize=7, color=BLUE, va="center")
        ax.set(xlim=(-0.45, 0.45), ylim=(0, 1.16 * top), xlabel="sample bias (V)",
               ylabel="dI/dV (measured units; model scaled)")
        ax.set_title(f"Line {tag} (+⁵)", fontsize=10.5, loc="left", color=INK)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right" if tag == "940" else "center right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        out_rows[tag] = dict(cpot=p["cpot"], w_tip=W_RG + p["cpot"], nsub=p["nsub"], carriers=carriers,
                             features=f.tolist(), measured=fm.tolist(), rms=rms)
    out = os.path.join(REPO, "examples", "stacking_wall_bestfits.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump(out_rows, open(os.path.join(REPO, "examples", "stacking_wall_bestfits.json"), "w"), indent=1)
    print("wrote", out)
    for k, v in out_rows.items():
        print(k, v)
