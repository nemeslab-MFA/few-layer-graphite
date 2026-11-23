"""Figure of cpot940_geom.py: ../examples/stacking_wall_cpot940_geom.png (rows: distance sweep, radius sweep).
Peak positions and half widths from the 0.25 mV re-evaluation (cpot940_geom_fine.py); open circles: sweeps re-solved
with 5 mV bias steps across the flat band (cpot940_geom_dense.py)."""
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
import cpot940_geom as G  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]   # ordinal blue, light = small
OFFSET = 0.004


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    rs = G.rows()
    fine = json.load(open(os.path.join(C.OUT, "geom_fine.json")))
    VF = np.asarray(fine["V"])
    fp = {(t["d"], t["R"]): t for t in fine["tip"]}
    dense = {}
    for r in rs:
        q = os.path.join(C.OUT, "dense_" + G.key(r["d"], r["R"])[5:] + ".json")
        if os.path.exists(q):
            dense[(r["d"], r["R"])] = json.load(open(q))["peak"]
    Vr, yr = rescore.measured("940")
    mp_ = fine["measured"]
    notip = fine["free"]["peak"]["ratio"]
    F = step1.FINE
    fig, axs = plt.subplots(2, 4, figsize=(18, 9.5), gridspec_kw=dict(width_ratios=(1.5, 1.1, 0.8, 0.8)))
    for row, (name, sel, par, unit) in enumerate((("distance d", lambda r: r["R"] == 20.0, "d", "nm"),
                                                  ("apex radius R", lambda r: r["d"] == 0.7, "R", "nm"))):
        sub = sorted([r for r in rs if sel(r)], key=lambda r: r[par])
        cols = RAMP[-len(sub):] if len(sub) <= len(RAMP) else RAMP
        full, zoom, pos, rat = axs[row]
        fixed = "R 20 nm" if par == "d" else "d 0.7 nm"
        w = (Vr >= -0.45) & (Vr <= 0.45)
        for k, r in enumerate(sub):
            y = np.asarray(r["tip"]["didv"]) * r["tip"]["scale"]
            full.plot(F, y + k * OFFSET, color=cols[k], lw=1.3)
            full.text(0.455, y[-1] + k * OFFSET, f"{par} {r[par]:g}", fontsize=7.5, va="center", color=INK)
            t = fp[(r["d"], r["R"])]
            p = t["peak"]
            zoom.plot(1e3 * VF, (np.asarray(t["didv"]) - p["floor"]) / (p["height"] - p["floor"]),
                      color=cols[k], lw=1.3, label=f"{par} {r[par]:g} {unit}")
        full.plot(Vr[w], yr[w] + len(sub) * OFFSET, color=INK, lw=1.8)
        full.text(0.455, yr[w][-1] + len(sub) * OFFSET, "measured", fontsize=7.5, va="center", color=INK)
        full.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel=f"dI/dV (model scaled), offset {OFFSET}")
        full.set_title(f"{name} sweep ({fixed}, CPot −0.367 eV, neutral stack, ε 2.6)", fontsize=8.5, loc="left")
        wz = (Vr >= -0.15) & (Vr <= 0.08)
        zoom.plot(1e3 * Vr[wz], (yr[wz] - mp_["floor"]) / (mp_["height"] - mp_["floor"]), color=INK, lw=1.8,
                  label="measured")
        zoom.set(xlim=(-150, 80), ylim=(-0.15, 1.1), xlabel="sample bias (mV)", ylabel="(dI/dV − floor) / peak height")
        zoom.set_title("flat band, normalised", fontsize=8.5, loc="left")
        zoom.legend(fontsize=7, frameon=False, loc="upper right")

        x = [r[par] for r in sub]
        pk = [fp[(r["d"], r["R"])]["peak"] for r in sub]
        pos.plot(x, [1e3 * p["v"] for p in pk], "o-", color=INK2, lw=1.5, ms=4)
        pos.axhline(1e3 * mp_["v"], color=INK, lw=1.0, ls=":")
        pos.set(xlabel=f"{par} ({unit})", ylabel="flat-band peak (mV)", title="flat-band position")
        rat.plot(x, [p["ratio"] for p in pk], "o-", color=INK2, lw=1.5, ms=4)
        for r in sub:
            q = dense.get((r["d"], r["R"]))
            if q:
                pos.plot(r[par], 1e3 * q["v"], "o", mfc="none", mec=INK, ms=7, mew=1.2)
                rat.plot(r[par], q["ratio"], "o", mfc="none", mec=INK, ms=7, mew=1.2)
        rat.axhline(mp_["ratio"], color=INK, lw=1.0, ls=":")
        rat.axhline(notip, color=MUTED, lw=1.0, ls="--")
        rat.annotate(f"measured {mp_['ratio']:.2f}", (x[0], mp_["ratio"]), textcoords="offset points", xytext=(2, 3),
                     fontsize=7, color=INK2)
        rat.annotate(f"no tip {notip:.2f}", (x[0], notip), textcoords="offset points", xytext=(2, 3), fontsize=7,
                     color=INK2)
        rat.set(xlabel=f"{par} ({unit})", ylabel="HWHM outer / inner", title="flat-band asymmetry")
        if par == "R":
            pos.set_xscale("log")
            rat.set_xscale("log")
        for ax in (pos, rat):
            ax.title.set_fontsize(8.5)
        for ax in (full, zoom, pos, rat):
            style(ax)
    fig.tight_layout()
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940_geom.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
