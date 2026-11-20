"""Figure of step 1: per line, the measured (+5) spectrum against the best cells with and without the tip, and the
score, flat-band error and lower side-band position over (eps, n_sub).  Writes ../examples/stacking_wall_step1.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)


def load():
    d = os.path.join(HERE, "step1")
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json") and ".ckpt" not in f]


if __name__ == "__main__":
    import sys
    import step1
    rows = load()
    with_cpot = len(sys.argv) > 1 and sys.argv[1] == "cpot"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.8), gridspec_kw={"width_ratios": [1.7, 1.0, 1.0]})
    for row, tag in enumerate(("940", "514")):
        rs = [r for r in rows if r["tag"] == tag and not r.get("cpot")]
        if not rs:
            continue
        Vm, ym, E_fb = step1.measured(tag)
        win = (Vm >= -0.45) & (Vm <= 0.45)
        ax = axs[row, 0]
        ax.plot(Vm[win], ym[win], color=INK, lw=1.8, label="measured (+⁵)")
        pool = [r for r in rows if r["tag"] == tag] if with_cpot else rs
        best = sorted(pool, key=lambda r: r["tip"]["score"])[:2]
        for i, r in enumerate(best):
            col = [BLUE, ORANGE][i]
            for lab, ls in (("tip", "-"), ("free", "--")):
                y = np.asarray(r[lab]["didv"]) * r[lab]["scale"]
                ax.plot(FINE, y, color=col, ls=ls, lw=1.3 if lab == "tip" else 1.0,
                        label=(f"ε = {r['eps']:g}, n_sub = {r['nsub'] / 1e13:+.2f}×10¹³, CPot {r.get('cpot', 0.0):+.3f} eV: "
                               f"{'with the tip' if lab == 'tip' else 'without'} (score {r[lab]['score']:.1f})"))
            ax.axvline(r["bottom_fb"], color=col, lw=0.8, ls=":")
        for f in rs[0]["measured_features"]:
            ax.axvline(f, color=GRID, lw=0.8, zorder=0)
        ax.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (measured units)")
        ax.set_title(f"line {tag}: best cells (dotted: bottom flat band, grey: measured features)", fontsize=10, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        eps = sorted({r["eps"] for r in rs})
        for k, (col_ax, what, lab) in enumerate(((axs[row, 1], "score", "score with the tip (lower is better)"),
                                                (axs[row, 2], "fb", "flat-band peak with the tip − measured (mV)"))):
            for r in rs:
                v = r["tip"]["score"] if what == "score" else 1e3 * r["tip"]["fb_err"]
                sc = col_ax.scatter(r["eps"], r["nsub"] / 1e13, c=[v], cmap="viridis_r" if what == "score" else "RdBu_r",
                                    vmin=5 if what == "score" else -30, vmax=15 if what == "score" else 30, s=160,
                                    edgecolors="white")
                col_ax.annotate(f"{v:.0f}", (r["eps"], r["nsub"] / 1e13), fontsize=6.5, ha="center", va="center",
                                color="white" if (what == "score" and v > 9) else INK)
            col_ax.set(xlabel="interlayer ε", ylabel="n_sub (10¹³ cm⁻²)", xscale="log")
            col_ax.set_xticks(eps, [f"{e:g}" for e in eps])
            col_ax.set_title(lab, fontsize=9.5, loc="left", color=INK)
            for sp in ("top", "right"):
                col_ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(REPO, "examples", "stacking_wall_step1_cpot.png" if with_cpot else "stacking_wall_step1.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)
