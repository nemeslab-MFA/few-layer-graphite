"""Figure of cpot940.py: ../examples/stacking_wall_cpot940.png."""
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
SHADES = {-0.30: "#86b6ef", -0.367: "#2a78d6", -0.45: "#104281", 0.0: "#eb6834"}
BIAS = [(-0.25, "#1c5cab"), (-0.075, "#3987e5"), (-0.025, "#86b6ef"), (0.025, "#f19a99"), (0.25, "#e34948"),
        (0.35, "#9e2b2a")]


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    runs = {round(r["cpot"], 3): r for r in C.rows()}
    pl = json.load(open(os.path.join(C.OUT, "place.json")))
    best = min(runs, key=lambda c: abs(c - pl["cpot"]))
    Vr, yr = rescore.measured(C.TAG)
    fm = np.asarray(runs[best]["measured_features"])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.15, 1))
    a, b = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    c, d, e = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    for ax, lim in ((a, (-0.45, 0.45)), (b, (-0.15, 0.10))):
        w = (Vr >= lim[0]) & (Vr <= lim[1])
        ax.plot(Vr[w], yr[w], color=INK, lw=2.0, label="measured 940 (+⁵)", zorder=6)
        f = runs[best]["free"]
        ax.plot(step1.FINE, np.asarray(f["didv"]) * f["scale"], color=MUTED, lw=1.1, ls="--", label="no tip (neutral stack)")
        for cp in sorted(runs, reverse=True):
            t = runs[cp]["tip"]
            ax.plot(step1.FINE, np.asarray(t["didv"]) * t["scale"], color=SHADES.get(cp, INK2),
                    lw=2.0 if cp == best else 1.1, label=f"with the tip, CPot {cp:+.3f} eV" + ("  (fitted)" if cp == best else ""))
        ax.axvline(fm[1], color=GRID, lw=0.9, zorder=0)
        ax.set(xlim=lim, xlabel="sample bias (V)")
    t = runs[best]["tip"]
    a.set_ylabel("dI/dV (measured units; model scaled)")
    fig.suptitle(f"Line 940 (+⁵): neutral stack (no offset, no substrate charge), ε 2.6, full SWMcC, tip R 20 nm, d 0.7 nm; "
                 f"CPot fitted to the flat band only.  CPot {best:+.3f} eV: features {C.S._mv(t['features'])} mV "
                 f"(measured {C.S._mv(fm)}), asymmetry {t['asym']:+.2f} (measured {runs[best]['measured_asym']:+.2f})",
                 fontsize=9, x=0.01, ha="left", color=INK)
    a.set_title("full spectrum", fontsize=8.5, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    b.set_title("flat-band region", fontsize=8.5, loc="left")

    lg = sorted(pl["log"], key=lambda x: x["cpot"])
    c.plot([x["cpot"] for x in lg], [1e3 * x["fb"] for x in lg], "o-", color=INK2, lw=1.5, ms=4)
    c.axhline(1e3 * pl["v_fb"], color=INK, lw=1.0, ls=":")
    c.annotate(f"measured {1e3 * pl['v_fb']:+.1f} mV", (lg[0]["cpot"], 1e3 * pl["v_fb"]), textcoords="offset points",
               xytext=(2, 4), color=INK2, fontsize=7.5)
    c.set(xlabel="CPot = W_tip − W_RG (eV)", ylabel="flat band under the apex at V = −32.5 mV (meV)",
          title="flat-band position vs contact potential")
    c.title.set_fontsize(8.5)

    sw = runs[best]["sweep"]
    V, U0, mu0 = np.asarray(sw["bias"]), np.asarray(sw["U0"]), sw["mu0"]
    L = np.arange(1, 7)
    d.axhline(0, color=MUTED, lw=0.8)
    d.plot(L, 1e3 * np.asarray(runs[best]["U_free_minus_EF"]), "o--", color=INK, lw=1.5, ms=4, label="no tip", zorder=5)
    for vb, col in BIAS:
        k = int(np.argmin(np.abs(V - vb)))
        d.plot(L, 1e3 * (U0[k] - mu0), "o-", color=col, lw=1.4, ms=3.5, label=f"tip, {1e3 * V[k]:+.0f} mV")
    d.set(xticks=L, xlabel="layer (1 = top)", ylabel="U_l − E_F under the apex (meV)",
          title=f"layer potentials, CPot {best:+.3f} eV")
    d.title.set_fontsize(8.5)
    d.legend(fontsize=7, frameon=False)

    for cp in sorted(runs, reverse=True):
        s = runs[cp]["sweep"]
        e.plot(np.asarray(s["bias"]), 1e3 * (np.asarray(s["U0"])[:, 0] - s["mu0"]), "o-", color=SHADES.get(cp, INK2),
               lw=2.0 if cp == best else 1.1, ms=3, label=f"CPot {cp:+.3f} eV")
    e.axhline(0, color=MUTED, lw=0.8)
    e.set(xlabel="sample bias (V)", ylabel="top layer U_1 − E_F under the apex (meV)", title="tip gating of the top layer")
    e.title.set_fontsize(8.5)
    e.legend(fontsize=7, frameon=False)
    for ax in (a, b, c, d, e):
        style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
