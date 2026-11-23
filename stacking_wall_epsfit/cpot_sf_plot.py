"""Figure of cpot_sf.py: ../examples/stacking_wall_cpot_sf.png.  (+4-) spectra of lines 940 and 514 predicted with the
best-fit parameters of their (+5) spectra, against the measured (+4-) spectra."""
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
import cpot_sf as SF  # noqa: E402
import step1  # noqa: E402
import step2  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE = "#2a78d6"
W_RG = 4.6


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 0.85, 0.42), hspace=0.32, wspace=0.16)
    F = step1.FINE
    for col, tag in enumerate(("940", "514")):
        r = json.load(open(os.path.join(SF.OUT, f"{tag}.json")))
        Vm, ym, _ = step2.measured_sf(tag)
        w = (Vm >= -0.45) & (Vm <= 0.45)
        t, f = r["tip"], r["free"]
        yt, yf = np.asarray(t["didv"]) * t["scale"], np.asarray(f["didv"]) * f["scale"]
        for row, (lim, title) in enumerate((((-0.45, 0.45), "full range"), ((-0.15, 0.25), "near E_F"))):
            ax = fig.add_subplot(gs[row, col])
            sel = (Vm >= lim[0]) & (Vm <= lim[1])
            ax.plot(Vm[sel], ym[sel], color=INK, lw=1.9, label="measured (+4−)")
            ax.plot(F, yt, color=BLUE, lw=1.7, label="model with the tip")
            ax.plot(F, yf, color=MUTED, lw=1.3, ls="--", label="model, no tip")
            if row == 1:
                for v, col_, lab in ((r["measured_min"], INK, "measured"), (t["minimum"], BLUE, "model")):
                    ax.axvline(v, color=col_, lw=1.0, ls=":")
                for v, col_ in ((r["measured_edge"], INK), (t["edge"], BLUE)):
                    ax.axvline(v, color=col_, lw=1.0, ls="-.")
                ax.text(0.01, 0.97, "dotted: minimum; dash-dot: first edge (black measured, blue model)",
                        transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
            ax.set(xlim=lim, xlabel="sample bias (V)", ylabel="dI/dV (measured units; model scaled)")
            ax.set_title(f"Line {tag} (+4−), {title}", fontsize=10, loc="left")
            if row == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper center")
            style(ax)
        tx = fig.add_subplot(gs[2, col])
        tx.axis("off")
        kind = "holes" if r["nsub"] < 0 else "electrons"
        txt = (f"parameters from the (+5) best fit: CPot {r['cpot']:+.3f} eV (W_tip ≈ {W_RG + r['cpot']:.2f} eV for W_RG "
               f"{W_RG} eV),\n   n_sub {r['nsub'] / 1e13:+.3g} × 10¹³ cm⁻² ({kind}); neutral stack, ε 2.6, full SWMcC, "
               f"d 0.7 nm, R 20 nm\n"
               f"minimum:    model {1e3 * t['minimum']:+.0f} mV (no tip {1e3 * f['minimum']:+.0f}), "
               f"measured {1e3 * r['measured_min']:+.0f} mV\n"
               f"first edge: model {1e3 * t['edge']:+.0f} mV (no tip {1e3 * f['edge']:+.0f}), "
               f"measured {1e3 * r['measured_edge']:+.0f} mV").replace("-", "−")
        tx.text(0.0, 1.0, txt, transform=tx.transAxes, va="top", ha="left", fontsize=8.6, family="DejaVu Sans Mono",
                color=INK)
    out = os.path.join(REPO, "examples", "stacking_wall_cpot_sf.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
