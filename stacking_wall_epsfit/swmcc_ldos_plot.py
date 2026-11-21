"""Zero-field top-layer LDOS of the (+5) and (+4-) stacks in the engine tight binding: gamma0/gamma1 against the full
SWMcC set (energies from the neutrality point of each zero-field stack, sigma = 12.5 meV, 9 K), and the layer charges
of the neutral zero-field stacks.  Writes ../examples/stacking_wall_swmcc_ldos.png."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from bandbending.materials import library, multilayer as M  # noqa: E402
import stacking_wall_tibb as swt  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
E = np.arange(-0.5, 0.5 + 1e-9, 0.0005)

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    for j, (seq, lab) in enumerate(((swt.SEQ_RG, "(+⁵)"), (swt.SEQ_SF, "(+⁴−)"))):
        ax = axs[j]
        for p, col, name in ((library.TB_MCELLISTRIM2023, BLUE, "γ0/γ1"), (library.TB_SWMCC_MCELLISTRIM2023, ORANGE, "full SWMcC")):
            tb = M.MultilayerTB(seq, p, temperature=9.0, broadening=0.0125, e_window=0.5)
            _, D = tb.ldos(np.zeros(6), E, broadening=0.0125)
            near = np.abs(E) < 0.1
            pk = E[near][np.argmax(D[near, 0])]
            ax.plot(E, D[:, 0] / 1e13, color=col, lw=1.4, label=f"{name}" + (f": peak {1e3 * pk:+.1f} meV" if j == 0 else ""))
            if j == 0 or True:
                n = tb.densities(np.zeros(6), 0.0)
                axs[2].plot(np.arange(1, 7) + (0.08 if p is library.TB_SWMCC_MCELLISTRIM2023 else -0.08) + 0.0 * j,
                            n / 1e12, "o-" if j == 0 else "s--", color=col, ms=4, lw=1.0,
                            label=f"{name} {lab}" if p is library.TB_SWMCC_MCELLISTRIM2023 else None)
        ax.axvline(0.0, color=GRID, lw=0.8)
        ax.set(xlim=(-0.45, 0.45), xlabel="E − E_CNP (eV)", ylabel="top-layer LDOS (10¹³ eV⁻¹ cm⁻²)")
        ax.set_title(f"{lab} zero-field stack, no charges, no tip", fontsize=9.5, loc="left", color=INK)
        ax.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax = axs[2]
    ax.axhline(0.0, color=GRID, lw=0.8)
    ax.set(xlabel="layer (1 = top)", ylabel="excess electrons (10¹² cm⁻²)", xticks=range(1, 7))
    ax.set_title("layer charges of the neutral zero-field stack\n(full SWMcC; γ0/γ1 is zero)", fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(REPO, "examples", "stacking_wall_swmcc_ldos.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
