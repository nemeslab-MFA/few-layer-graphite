"""Figures of swmcc_cells.py: per line, every cell (rows) on the (+5) and (+4-) sides (columns), measured against the
gamma0/gamma1 and the full SWMcC engine tight binding, with the tip (solid) and without (dashed).
Writes ../examples/stacking_wall_swmcc_940.png and _514.png."""
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
import rescore  # noqa: E402
import step1  # noqa: E402
import step2  # noqa: E402
import swmcc_cells as S  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(tag, eps, dEF, model, stack):
    p = os.path.join(S.OUT, S.key(tag, eps, dEF, model, stack) + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    for tag in ("940", "514"):
        cells = [c for c in S.CELLS if c[0] == tag]
        fig, axs = plt.subplots(len(cells), 2, figsize=(15, 3.3 * len(cells)), squeeze=False)
        Vr, yr = rescore.measured(tag)
        Vs, ys, _ = step2.measured_sf(tag)
        ws = (Vs >= -0.45) & (Vs <= 0.45)
        for i, (_, eps, dEF) in enumerate(cells):
            for j, stack in enumerate(S.STACKS):
                ax = axs[i, j]
                V, y = (Vr, yr) if stack == "RG" else (Vs[ws], ys[ws])
                ax.plot(V, y, color=INK, lw=1.6, label="measured " + ("(+⁵)" if stack == "RG" else "(+⁴−)"))
                notes = []
                for model, col in (("g01", BLUE), ("full", ORANGE)):
                    r = load(tag, eps, dEF, model, stack)
                    if not r or not r.get("placed"):
                        continue
                    lab = "γ0/γ1" if model == "g01" else "full SWMcC"
                    for kind, ls, lw in (("tip", "-", 1.3), ("free", "--", 0.9)):
                        d = r[kind]
                        ax.plot(step1.FINE, np.asarray(d["didv"]) * d["scale"], color=col, ls=ls, lw=lw,
                                label=f"{lab}, {'with the tip' if kind == 'tip' else 'no tip'}")
                    t = r["tip"]
                    if stack == "RG":
                        notes.append(f"{lab}: n_sub {r['nsub'] / 1e13:+.2f}e13, edges {1e3 * t['edge_rms']:.0f} mV, "
                                     f"FB {1e3 * t['features'][1]:+.0f}, asym {t['asym']:+.2f}")
                    else:
                        notes.append(f"{lab}: minimum {1e3 * t['minimum']:+.0f} (tip) / {1e3 * r['free']['minimum']:+.0f} mV "
                                     f"(meas {1e3 * r['measured_min']:+.0f}), res {t['residual']:.2f}")
                if stack == "RG":
                    for f in rescore.features(Vr, yr, smooth=9):
                        ax.axvline(f, color=GRID, lw=0.8, zorder=0)
                kind = "clean" if dEF == 0 else f"ΔE_F {1e3 * dEF:+.0f} meV"
                ax.set_title(f"line {tag} {'(+⁵)' if stack == 'RG' else '(+⁴−)'}: ε {eps:g}, {kind}\n" + "\n".join(notes),
                             fontsize=7.8, loc="left", color=INK)
                ax.set(xlim=(-0.45, 0.45))
                if i == len(cells) - 1:
                    ax.set_xlabel("sample bias (V)")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out = os.path.join(REPO, "examples", f"stacking_wall_swmcc_{tag}.png")
        fig.savefig(out, dpi=130)
        print("wrote", out)
