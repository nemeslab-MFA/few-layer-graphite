"""Layer on-site potentials U_l - E_F of the eps 2.6 full-SWMcC step-1 cells (step1_eps26.py), without the tip and under
the tip apex at selected sample biases.  Writes ../examples/stacking_wall_eps26_layer_U.png."""
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
import swmcc_cells as S  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CELLS = [("514", -0.03), ("940", 0.08), ("940", 0.10)]
BIAS = [(-0.25, "#1c5cab"), (-0.05, "#3987e5"), (-0.025, "#86b6ef"),            # diverging: blue = negative bias
        (0.025, "#f19a99"), (0.25, "#e34948"), (0.35, "#9e2b2a")]               # red = positive bias


def load(tag, dEF, stack):
    return json.load(open(os.path.join(S.OUT, S.key(tag, 2.6, dEF, "full", stack) + ".json")))


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True)
    L = np.arange(1, 7)
    for j, (tag, dEF) in enumerate(CELLS):
        for i, stack in enumerate(("RG", "SF")):
            ax = axs[i, j]
            r = load(tag, dEF, stack)
            sw = r["sweep"]
            V, U0, Uf, mu0 = np.asarray(sw["bias"]), np.asarray(sw["U0"]), np.asarray(sw["U_far"]), sw["mu0"]
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.plot(L, 1e3 * (Uf[0] - mu0), "o--", color=INK, lw=1.6, ms=5, label="no tip", zorder=5)
            for vb, col in BIAS:
                k = int(np.argmin(np.abs(V - vb)))
                ax.plot(L, 1e3 * (U0[k] - mu0), "o-", color=col, lw=1.5, ms=4, label=f"tip, {1e3 * V[k]:+.0f} mV")
            name = "(+⁵)" if stack == "RG" else "(+⁴−)"
            ax.set_title(f"line {tag} {name}: ε 2.6, ΔE_F {1e3 * dEF:+.0f} meV, n_sub {r['nsub'] / 1e13:+.2f}×10¹³ cm⁻²",
                         fontsize=8.5, loc="left", color=INK)
            ax.set(xticks=L)
            if i == 1:
                ax.set_xlabel("layer (1 = top, faces the tip)")
            if j == 0:
                ax.set_ylabel("U_l − E_F (meV)")
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axs[0, 0].legend(fontsize=7.2, frameon=False, loc="lower left")
    fig.suptitle("On-site potential of each layer relative to E_F (U > 0 moves the bands up): far from the tip, and "
                 "under the apex (R 20 nm, d 0.7 nm, CPot = U_1 far) at the sample bias shown", fontsize=9, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(REPO, "examples", "stacking_wall_eps26_layer_U.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
