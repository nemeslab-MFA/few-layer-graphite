"""Figure of step1_eps26.py (full SWMcC, global offset): ../examples/stacking_wall_step1_eps26_full.png (rows: lines 514 and 940)."""
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
import step1_eps26 as P  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.4), gridspec_kw=dict(width_ratios=(1, 1, 1.5, 1.5)))
    for row, tag in enumerate(("514", "940")):
        a, c, e, f = axs[row]
        rows = P.scans(tag)
        d = np.array([1e3 * r["dEF"] for r in rows])
        a.plot(d, [1e3 * r["RG"]["edge_rms"] for r in rows], color=INK2, lw=1.6, marker="o", ms=3.5, label="no tip")
        c.plot(d, [1e3 * r["SF"]["minimum"] for r in rows], color=INK2, lw=1.6, marker="o", ms=3.5, label="no tip")
        tipped = [(x, *P.tip_result(tag, x)) for x in sorted({r["dEF"] for r in rows})]
        tipped = [(x, rg, sf) for x, rg, sf in tipped if rg and rg.get("placed")]
        a.plot([1e3 * x for x, _, _ in tipped], [1e3 * rg["tip"]["edge_rms"] for _, rg, _ in tipped], "D", ms=7,
               color=INK, mec="#fcfcfb", mew=1.2, label="with the tip")
        ts = [(x, sf) for x, _, sf in tipped if sf]
        c.plot([1e3 * x for x, _ in ts], [1e3 * sf["tip"]["minimum"] for _, sf in ts], "D", ms=7, color=INK,
               mec="#fcfcfb", mew=1.2, label="with the tip")
        mm = 1e3 * rows[0]["measured_min"]
        c.axhline(mm, color=INK, lw=1.0, ls=":")
        c.annotate(f"measured {mm:+.0f} mV", (d.min(), mm), textcoords="offset points", xytext=(2, 4), color=INK2, fontsize=7.5)
        a.set(xlabel="ΔE_F (meV)", ylabel="rms error of the three bulk edges (mV)", ylim=(0, None),
              title=f"line {tag} (+⁵): bulk edges, ε 2.6")
        c.set(xlabel="ΔE_F (meV)", ylabel="(+⁴−) minimum (mV)", title=f"line {tag} (+⁴−): minimum")
        a.legend(fontsize=7, frameon=False)
        c.legend(fontsize=7, frameon=False)

        Vr, yr = rescore.measured(tag)
        Vs, ys, _ = step2.measured_sf(tag)
        ws = (Vs >= -0.45) & (Vs <= 0.45)
        e.plot(Vr, yr, color=INK, lw=1.8, label="measured")
        f.plot(Vs[ws], ys[ws], color=INK, lw=1.8, label="measured")
        ranked = sorted(tipped, key=lambda t: t[1]["tip"]["edge_rms"])[:2]
        for k, (x, rg, sf) in enumerate(ranked):
            lab = f"ΔE_F {1e3 * x:+.0f} meV"
            for kind, ls, lw in (("tip", "-", 1.4), ("free", "--", 0.9)):
                e.plot(step1.FINE, np.asarray(rg[kind]["didv"]) * rg[kind]["scale"], color=CAT[k], ls=ls, lw=lw,
                       label=f"{lab}, {'with the tip' if kind == 'tip' else 'no tip'}")
                if sf:
                    f.plot(step1.FINE, np.asarray(sf[kind]["didv"]) * sf[kind]["scale"], color=CAT[k], ls=ls, lw=lw,
                           label=f"{lab}, {'with the tip' if kind == 'tip' else 'no tip'}")
        for v in rows[0]["measured_features"]:
            e.axvline(v, color=GRID, lw=0.8, zorder=0)
        notes = [f"ΔE_F {1e3 * x:+.0f}: n_sub {rg['nsub'] / 1e13:+.2f}e13, edges {1e3 * rg['tip']['edge_rms']:.0f} mV, "
                 f"FB {1e3 * rg['tip']['features'][1]:+.0f} mV, asym {rg['tip']['asym']:+.2f}" for x, rg, _ in ranked]
        e.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (scaled)")
        e.set_title(f"line {tag} (+⁵), best two cells\n" + "\n".join(notes), fontsize=8, loc="left")
        f.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)")
        f.set_title(f"line {tag} (+⁴−), same cells", fontsize=8.5, loc="left")
        e.legend(fontsize=6.8, frameon=False, loc="upper left")
        for ax in (a, c, e, f):
            style(ax)
    fig.tight_layout()
    out = os.path.join(REPO, "examples", "stacking_wall_step1_eps26_full.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
