"""940 (+5): measured, fitted CPot, CPot 0 and no tip, vertically offset. ../examples/stacking_wall_cpot940_offset.png"""
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
import rescore  # noqa: E402
import step1  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
OFFSET = 0.005

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    runs = {round(r["cpot"], 3): r for r in C.rows()}
    best = min(runs, key=lambda c: abs(c - json.load(open(os.path.join(C.OUT, "place.json")))["cpot"]))
    Vr, yr = rescore.measured(C.TAG)
    w = (Vr >= -0.45) & (Vr <= 0.45)
    F = step1.FINE
    traces = [                                                          # bottom to top
        ("no tip", F, np.asarray(runs[best]["free"]["didv"]) * runs[best]["free"]["scale"], MUTED, "--"),
        ("CPot 0", F, np.asarray(runs[0.0]["tip"]["didv"]) * runs[0.0]["tip"]["scale"], "#eb6834", "-"),
        (f"CPot {best:+.3f} eV (fit)", F, np.asarray(runs[best]["tip"]["didv"]) * runs[best]["tip"]["scale"], "#2a78d6", "-"),
        ("measured", Vr[w], yr[w], INK, "-"),
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    for f in runs[best]["measured_features"]:
        ax.axvline(f, color=GRID, lw=0.9, zorder=0)
    for k, (lab, x, y, col, ls) in enumerate(traces):
        ax.plot(x, y + k * OFFSET, color=col, ls=ls, lw=1.9 if lab == "measured" else 1.5)
        ax.text(0.455, y[-1] + k * OFFSET, lab, color=INK, fontsize=9, va="center", ha="left")
    ax.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel=f"dI/dV (measured units; model scaled), offset {OFFSET}")
    ax.set_title("Line 940 (+⁵): neutral stack, ε 2.6, full SWMcC, tip R 20 nm, d 0.7 nm; grey lines: measured features",
                 fontsize=9, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(right=0.82)
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940_offset.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
