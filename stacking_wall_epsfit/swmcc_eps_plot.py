"""Figure of swmcc_eps.py (line 514, full SWMcC): ../examples/stacking_wall_swmcc_eps514.png."""
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
import swmcc_eps as SE  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]                       # categorical slots 1-4: Delta E_F
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]                                                             # ordinal blue 250-700: eps
DEF_LAB = {0.0: "clean", -0.04: "ΔE_F −40 meV", -0.07: "ΔE_F −70 meV", -0.10: "ΔE_F −100 meV"}
FEAT = ("lower side band", "upper edge 1", "upper edge 2")


def scan_rows():
    out = {}
    for f in os.listdir(S.OUT):
        if f.startswith("scan_514") and f.endswith("_full.json"):
            r = json.load(open(os.path.join(S.OUT, f)))
            if r["placed"]:
                out[(r["eps"], r["dEF"])] = r
    return out


def tip_rows():
    out = {}
    for f in os.listdir(S.OUT):
        if f.startswith("514_") and f.endswith("_full_RG.json"):
            rg = json.load(open(os.path.join(S.OUT, f)))
            p = os.path.join(S.OUT, f.replace("_RG.json", "_SF.json"))
            if rg.get("placed") and os.path.exists(p):
                out[(rg["eps"], rg["dEF"])] = (rg, json.load(open(p)))
    return out


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    sc, tp = scan_rows(), tip_rows()
    fm = np.asarray(next(iter(sc.values()))["measured_features"])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    (a, b, c), (d, e, f) = axs

    for i, dEF in enumerate(SE.DEF):
        eps = sorted(k[0] for k in sc if k[1] == dEF)
        a.plot(eps, [1e3 * sc[(x, dEF)]["RG"]["edge_rms"] for x in eps], color=CAT[i], lw=1.6, label=DEF_LAB[dEF] + ", no tip")
        c.plot(eps, [1e3 * sc[(x, dEF)]["SF"]["minimum"] for x in eps], color=CAT[i], lw=1.6, label=DEF_LAB[dEF] + ", no tip")
        te = sorted(k[0] for k in tp if k[1] == dEF)
        if te:
            a.plot(te, [1e3 * tp[(x, dEF)][0]["tip"]["edge_rms"] for x in te], "o", ms=7, color=CAT[i], mec="#fcfcfb",
                   mew=1.5, label=DEF_LAB[dEF] + ", with the tip")
            c.plot(te, [1e3 * tp[(x, dEF)][1]["tip"]["minimum"] for x in te], "o", ms=7, color=CAT[i], mec="#fcfcfb",
                   mew=1.5, label=DEF_LAB[dEF] + ", with the tip")
    a.set(xscale="log", ylim=(0, 110), xlabel="interlayer ε", ylabel="rms error of the three bulk edges (mV)",
          title="(a) (+⁵) bulk band edges")
    a.legend(fontsize=6.8, frameon=False, ncol=2)
    c.axhline(1e3 * next(iter(sc.values()))["measured_min"], color=INK, lw=1.2)
    c.text(1000, 1e3 * next(iter(sc.values()))["measured_min"] - 1, "measured", ha="right", va="top", color=INK2, fontsize=7.5)
    c.set(xscale="log", xlabel="interlayer ε", ylabel="(+⁴−) minimum (mV)", title="(c) (+⁴−) minimum")
    c.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper left")

    dEF = -0.07
    eps = sorted(k[0] for k in sc if k[1] == dEF)
    te = sorted(k[0] for k in tp if k[1] == dEF)
    for j, (idx, lab) in enumerate(zip((0, 2, 3), FEAT)):
        b.plot(eps, [1e3 * (sc[(x, dEF)]["RG"]["features"][idx] - fm[idx]) for x in eps], color=CAT[j], lw=1.6,
               label=lab + ", no tip")
        b.plot(te, [1e3 * (tp[(x, dEF)][0]["tip"]["features"][idx] - fm[idx]) for x in te], "o", ms=7, color=CAT[j],
               mec="#fcfcfb", mew=1.5, label=lab + ", with the tip")
    b.axhline(0, color=INK, lw=1.0)
    b.set(xscale="log", xlabel="interlayer ε", ylabel="model − measured (mV)", title="(b) edge offsets at ΔE_F −70 meV")
    b.legend(fontsize=6.8, frameon=False, ncol=2)

    ramp_eps = sorted({k[0] for k in tp if k[1] == dEF})
    all_eps = sorted(set(eps) | set(ramp_eps))
    cols = dict(zip(all_eps, RAMP[-len(all_eps):]))
    for x in eps:
        d.plot(np.arange(1, 7), 1e3 * np.asarray(sc[(x, dEF)]["U_RG"]), "o-", color=cols[x], lw=1.6, ms=4,
               label=f"ε {x:g}")
    d.set(xlabel="layer (1 = top)", ylabel="U − E_F, no tip (meV)", title="(d) (+⁵) layer potentials at ΔE_F −70 meV")
    d.legend(fontsize=6.8, frameon=False)

    Vr, yr = rescore.measured("514")
    Vs, ys, _ = step2.measured_sf("514")
    ws = (Vs >= -0.45) & (Vs <= 0.45)
    e.plot(Vr, yr, color=INK, lw=1.8, label="measured")
    f.plot(Vs[ws], ys[ws], color=INK, lw=1.8, label="measured")
    for x in ramp_eps:
        rg, sf = tp[(x, dEF)]
        e.plot(step1.FINE, np.asarray(rg["tip"]["didv"]) * rg["tip"]["scale"], color=cols[x], lw=1.2, label=f"ε {x:g}")
        f.plot(step1.FINE, np.asarray(sf["tip"]["didv"]) * sf["tip"]["scale"], color=cols[x], lw=1.2, label=f"ε {x:g}")
    for v in fm:
        e.axvline(v, color=GRID, lw=0.8, zorder=0)
    e.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (scaled)", title="(e) (+⁵) with the tip, ΔE_F −70 meV")
    f.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", title="(f) (+⁴−) with the tip, ΔE_F −70 meV")
    e.legend(fontsize=6.8, frameon=False, loc="upper left")
    for ax in axs.ravel():
        style(ax)
    fig.suptitle("Line 514, full SWMcC engine tight binding: interlayer ε sweep. Flat-band asymmetry |asym| ≤ 0.05 in every "
                 "cell except clean ε 4 (+0.32 / +0.35 with the tip); measured +0.23.", fontsize=9, x=0.01, ha="left",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(REPO, "examples", "stacking_wall_swmcc_eps514.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
