"""Step 1, shared eps = 2.6: measured (+5) spectra against the best cell of each line, with and without the tip."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step1
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
cells = {"940": "940_eps02.6_n+2.00_c-0.150", "514": "514_eps02.6_n-2.00_c+0.075"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(2, 1, figsize=(11, 8.4))
for ax, (tag, k) in zip(axs, cells.items()):
    r = json.load(open(os.path.join(step1.OUT, k + ".json")))
    Vm, ym, E_fb = step1.measured(tag)
    w = (Vm >= -0.45) & (Vm <= 0.45)
    ax.plot(Vm[w], ym[w], color=INK, lw=1.8, label="measured (+⁵)")
    ax.plot(step1.FINE, np.asarray(r["tip"]["didv"]) * r["tip"]["scale"], color=BLUE, lw=1.5,
            label=f"with the tip (CPot {r['cpot']:+.3f} eV): feature rms {1e3 * r['tip']['pos_rms']:.0f} mV, residual {r['tip']['residual']:.2f}")
    ax.plot(step1.FINE, np.asarray(r["free"]["didv"]) * r["free"]["scale"], color=ORANGE, lw=1.2, ls="--",
            label=f"without the tip: feature rms {1e3 * r['free']['pos_rms']:.0f} mV, residual {r['free']['residual']:.2f}")
    for f in r["measured_features"]:
        ax.axvline(f, color=MUTED, lw=0.8, ls=":")
    for f in r["tip"]["features"]:
        if np.isfinite(f):
            ax.axvline(f, color=BLUE, lw=0.6, ls=":", alpha=0.7)
    U = np.round(1e3 * np.asarray(r["U_minus_mu0"]))
    ax.set_title(f"line {tag}: ε = 2.6, n_sub = {r['nsub'] / 1e13:+.1f}×10¹³ cm⁻², neutral layers; layer potentials U − E_F "
                 f"(top → bottom) {U.astype(int).tolist()} meV; bottom flat band {1e3 * r['bottom_fb']:+.0f} meV",
                 fontsize=8.5, loc="left", color=INK)
    ax.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (measured units)")
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "stacking_wall_step1_eps26.png")
fig.savefig(out, dpi=160)
print("wrote", out)
