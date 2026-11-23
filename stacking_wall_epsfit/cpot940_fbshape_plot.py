"""940 (+5) flat-band peak shape: fitted CPot against no tip, overlaid.  ../examples/stacking_wall_cpot940_fbshape.png

Left: both spectra as computed (same model units, no scaling).  Right: the flat-band region with the no-tip spectrum
shifted in bias onto the fitted peak and both normalised to their peak height, plus the measured peak treated the same
way; the half widths at half maximum on the outer (more negative) and inner side quantify the asymmetry.  The right
panel uses the 0.25 mV re-evaluation of cpot940_geom_fine.py (same model, d 0.7 nm, R 20 nm), whose half widths are not
quantised by the 2.5 mV grid."""
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
import cpot940_geom_fine as GF  # noqa: E402
import rescore  # noqa: E402
import step1  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE = "#2a78d6"


def peak(V, y, lo=-0.12, hi=0.06):
    """Peak bias, height above the local floor, and half widths at half maximum (outer = negative side, inner)."""
    w = (V >= lo) & (V <= hi)
    v, z = V[w], y[w]
    k = int(np.argmax(z))
    floor = min(z[:k].min(), z[k:].min())
    half = floor + 0.5 * (z[k] - floor)
    left = v[:k][np.where(z[:k] < half)[0][-1]] if np.any(z[:k] < half) else np.nan
    right = v[k:][np.where(z[k:] < half)[0][0]] if np.any(z[k:] < half) else np.nan
    return v[k], z[k], floor, v[k] - left, right - v[k]


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    runs = {round(r["cpot"], 3): r for r in C.rows()}
    best = min(runs, key=lambda c: abs(c - json.load(open(os.path.join(C.OUT, "place.json")))["cpot"]))
    F = step1.FINE
    yt, yf = np.asarray(runs[best]["tip"]["didv"]), np.asarray(runs[best]["free"]["didv"])
    Vr, yr = rescore.measured(C.TAG)
    fine = json.load(open(os.path.join(C.OUT, "geom_fine.json")))
    VF = np.asarray(fine["V"])
    ytf = np.asarray(next(t for t in fine["tip"] if t["d"] == 0.7 and t["R"] == 20.0)["didv"])
    yff = np.asarray(fine["free"]["didv"])
    P = lambda V, y: tuple(GF.peak(V, y)[k] for k in ("v", "height", "floor", "hwhm_out", "hwhm_in"))  # noqa: E731
    pt, pf, pm = P(VF, ytf), P(VF, yff), P(Vr, yr)

    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=(1.5, 1)))
    a.plot(F, yf, color=MUTED, lw=1.6, ls="--", label="no tip")
    a.plot(F, yt, color=BLUE, lw=1.8, label=f"with the tip, CPot {best:+.3f} eV (fit)")
    a.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (model units, unscaled)")
    a.set_title("as computed, on top of each other", fontsize=9.5, loc="left")
    a.legend(fontsize=8.5, frameon=False)

    for lab, V, y, p, col, ls, lw in (("measured", Vr, yr, pm, INK, "-", 1.8),
                                      (f"no tip, shifted {1e3 * (pt[0] - pf[0]):+.0f} mV", VF + (pt[0] - pf[0]), yff, pf, MUTED, "--", 1.6),
                                      (f"tip, CPot {best:+.3f} eV", VF, ytf, pt, BLUE, "-", 1.8)):
        x = V - p[0] if lab == "measured" else V - pt[0]
        b.plot(1e3 * x, (y - p[2]) / (p[1] - p[2]), color=col, ls=ls, lw=lw,
               label=f"{lab}:  {1e3 * p[3]:.1f} / {1e3 * p[4]:.1f}")
    b.axhline(0.5, color=GRID, lw=0.9, zorder=0)
    b.axvline(0, color=GRID, lw=0.9, zorder=0)
    b.set(xlim=(-110, 70), ylim=(-0.1, 1.08), xlabel="bias from the flat-band peak (mV); negative = outer side, away from E_F",
          ylabel="(dI/dV − floor) / peak height")
    b.set_title("flat-band peak, aligned and normalised", fontsize=9.5, loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper left", title="HWHM outer / inner (mV)", title_fontsize=8,
             alignment="left")
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(REPO, "examples", "stacking_wall_cpot940_fbshape.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
    for lab, p in (("tip", pt), ("no tip", pf), ("measured", pm)):
        print(f"{lab:9s} peak {1e3 * p[0]:+6.1f} mV, HWHM outer {1e3 * p[3]:5.1f} inner {1e3 * p[4]:5.1f} mV, "
              f"ratio {p[3] / p[4]:.2f}")
