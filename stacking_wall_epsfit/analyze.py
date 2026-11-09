"""Line 940, (+5) side: summary and figure of the interlayer-permittivity x starting-doping scans.

Reads the per-cell results of scan.py (cells/: uniformly doped stack) and polar.py (polar/:
doped stack with a compensating SiO2 charge n_sub) and writes
examples/stacking_wall_epsfit_940.json (every cell, the best cells and their spectra) and
examples/stacking_wall_epsfit_940.png.  A cell counts only if all of its features are found, in
increasing order, with the flat-band peak inside its search window.

    python stacking_wall_epsfit/analyze.py
"""
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
from features import NAMES, NAMES5, features, features5                # noqa: E402

EXAMPLES = os.path.join(REPO, "examples")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
S1, S2 = "#2a78d6", "#eb6834"
EPS = [1.0, 1.8, 2.6, 4.0, 6.0, 10.0, 15.0, 25.0]
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)

st = json.load(open(os.path.join(EXAMPLES, "stacking_wall_tibb_940.json")))
VM, YM = np.asarray(st["measured"]["bias"]), np.asarray(st["measured"]["rg"], dtype=float)
FM4, FM5 = features(VM, YM, smooth=9), features5(VM, YM, smooth=9)
WIN = (VM >= -0.45) & (VM <= 0.45)
NORM = YM[WIN].max()


def load(folder):
    d = os.path.join(HERE, folder)
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))
            if f.endswith(".json") and not f.endswith(".ckpt.json")]


def valid(f):
    f = np.asarray(f, dtype=float)
    return bool(np.all(np.isfinite(f)) and np.all(np.diff(f) > 0) and f[1] > -0.115)


def rms(f, fm):
    return float(np.sqrt(np.mean((np.asarray(f) - fm) ** 2)))


def scaled(y):
    """Least-squares scale of a simulated curve onto the measured one on -0.45..0.45 V, then / NORM."""
    y = np.asarray(y, dtype=float)
    ys = np.interp(VM[WIN], FINE, y)
    return float(np.dot(ys, YM[WIN]) / np.dot(ys, ys)) * y / NORM


uni, pol = load("cells"), load("polar")
for r in uni:
    r["f"] = np.asarray(r["features"], dtype=float)
    r["ok"] = valid(r["f"])
    r["score"] = rms(r["f"], FM4) if r["ok"] else np.nan
for r in pol:
    r["f"] = np.asarray(r["features"], dtype=float)
    r["ok"] = valid(r["f"])
    r["score"] = rms(r["f"], FM5) if r["ok"] else np.nan
    ff = np.asarray(r["features_free"], dtype=float)
    r["score_free"] = rms(ff, FM5) if valid(ff) else np.nan

best_u = min((r for r in uni if r["ok"]), key=lambda r: r["score"])
fb_u = min((r for r in uni if r["ok"] and abs(r["f"][1] - FM4[1]) <= 0.005), key=lambda r: r["score"])
best_p = min((r for r in pol if r["ok"]), key=lambda r: r["score"])
near_p = sorted((r for r in pol if r["ok"]), key=lambda r: r["score"])[:4]


def grid(rows):
    dEFs = sorted({r["dEF"] for r in rows})
    M = np.full((len(dEFs), len(EPS)), np.nan)
    for r in rows:
        M[dEFs.index(r["dEF"]), EPS.index(r["eps"])] = r["score"]
    return dEFs, M


def heat(ax, rows, best, title):
    dEFs, M = grid(rows)
    vmax = 60.0
    im = ax.imshow(np.clip(1e3 * M, 0, vmax), origin="lower", cmap="Blues", aspect="auto", vmin=10, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = 1e3 * M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 40 else INK)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=MUTED)
    bi, bj = dEFs.index(best["dEF"]), EPS.index(best["eps"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=S2, lw=2.2))
    ax.set_xticks(range(len(EPS)), [f"{e:g}" for e in EPS])
    ax.set_yticks(range(len(dEFs)), [f"{1e3 * d:g}" for d in dEFs])
    ax.set_xlabel("interlayer permittivity ε")
    ax.set_ylabel("starting doping ΔE$_F$ (meV)")
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("feature rms under the tip (mV)")


def spectra(ax, curves, fm, title, legend_loc="upper right"):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)
    ax.plot(VM[WIN], YM[WIN] / NORM, color=INK, lw=1.5, label="measured dI/dV")
    for y, color, ls, label, f in curves:
        ax.plot(FINE, y, color=color, lw=1.4 if ls == "-" else 1.0, ls=ls, label=label)
        if f is not None:
            ok = np.isfinite(f)
            ax.plot(f[ok], np.interp(f[ok], FINE, y), "o", ms=4.5, color=color, mec="white", mew=0.8, zorder=5)
    ok = np.isfinite(fm)
    ax.plot(fm[ok], np.interp(fm[ok], VM[WIN], YM[WIN] / NORM), "o", ms=4.5, color=INK, mec="white", mew=0.8, zorder=5)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(0, max(1.4, 1.12 * max(float(np.max(c[0])) for c in curves)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("sample bias (V)")
    ax.set_ylabel("dI/dV (normalised)")
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc=legend_loc)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(2, 2, figsize=(14, 9.4), gridspec_kw={"width_ratios": [1.0, 1.45]})
heat(axs[0, 0], pol, best_p, "(a) polarized stack: doping + SiO₂ charge n$_{sub}$\n5 features; –: a feature missing or misassigned")
y_tip, y_free = scaled(best_p["didv"]), scaled(best_p["didv_free"])
spectra(axs[0, 1], [
    (y_tip, S1, "-", f"under the tip: ε = {best_p['eps']:g}, ΔE$_F$ = {1e3 * best_p['dEF']:g} meV, "
     f"n$_{{sub}}$ = {best_p['nsub'] / 1e13:.3f}×10$^{{13}}$ cm$^{{-2}}$", best_p["f"]),
    (y_free, S1, (0, (3, 2)), "same stack without the tip", None)],
    FM5, f"(b) best polarized stack: feature rms {1e3 * best_p['score']:.1f} mV under the tip", legend_loc="upper left")
heat(axs[1, 0], uni, best_u, "(c) uniformly doped stack, no SiO₂ charge\n4 features (no bottom band)")
spectra(axs[1, 1], [
    (scaled(best_u["didv"]), S1, "-", f"best rms: ε = {best_u['eps']:g}, ΔE$_F$ = {1e3 * best_u['dEF']:g} meV "
     f"({1e3 * best_u['score']:.1f} mV)", best_u["f"]),
    (scaled(fb_u["didv"]), S2, "-", f"flat band in place: ε = {fb_u['eps']:g}, ΔE$_F$ = {1e3 * fb_u['dEF']:g} meV "
     f"({1e3 * fb_u['score']:.1f} mV)", fb_u["f"])],
    FM4, "(d) uniformly doped stack under the tip")
fig.text(0.01, 0.005, "Tip R = 20 nm, d = 0.7 nm, no band bending at 0 V; 9 K, 10 meV broadening, 3.75 mV lock-in. "
         "Dots: lower side-band peak, flat-band peak, bottom-band onset (polarized only), upper side-band edges 1 and 2.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(os.path.join(EXAMPLES, "stacking_wall_epsfit_940.png"), dpi=160)


def cell(r, polar):
    out = dict(eps=r["eps"], dEF=r["dEF"], valid=r["ok"], rms=None if np.isnan(r["score"]) else r["score"],
               features=[None if not np.isfinite(v) else float(v) for v in r["f"]], residual=r["residual"])
    if polar:
        U = np.asarray(r["U_far"])
        out.update(nsub=r["nsub"], carriers=r["carriers"], cpot=r["cpot"], U_far_minus_EF=(U - r["mu0"]).tolist(),
                   features_free=[None if not np.isfinite(v) else float(v) for v in np.asarray(r["features_free"], dtype=float)],
                   rms_free=None if np.isnan(r["score_free"]) else r["score_free"])
    return out


prev_run = st["scan"][f"{st['best_cpot']:+.3f}"]
prev_f = features(np.asarray(st["fine_bias"]), prev_run["didv"])
summary = dict(
    measured=dict(features4=FM4.tolist(), features5=FM5.tolist(), names4=list(NAMES), names5=list(NAMES5)),
    previous=dict(features=prev_f.tolist(), rms=rms(prev_f, FM4), cpot=st["best_cpot"], eps=2.6,
                  dEF=st["settings"]["fermi_level"]),
    settings=dict(eps=EPS, tip="R = 20 nm, d = 0.7 nm, no band bending at 0 V", temperature=9.0, broadening=0.01,
                  modulation=0.00375, oxide_nm=90.0, trap_nm=2.0, grid="draft", bias_step=0.025),
    uniform=dict(cells=[cell(r, False) for r in uni], best=cell(best_u, False), flat_band_in_place=cell(fb_u, False)),
    polarized=dict(cells=[cell(r, True) for r in pol], best=cell(best_p, True), near=[cell(r, True) for r in near_p],
                   best_free=cell(min((r for r in pol if np.isfinite(r["score_free"])), key=lambda r: r["score_free"]), True)),
    spectra=dict(bias=FINE.tolist(), polarized_tip=y_tip.tolist(), polarized_free=y_free.tolist(),
                 uniform_best=scaled(best_u["didv"]).tolist(), uniform_flat_band=scaled(fb_u["didv"]).tolist()),
)
json.dump(summary, open(os.path.join(EXAMPLES, "stacking_wall_epsfit_940.json"), "w"))
print(f"polarized best: eps {best_p['eps']}, dEF {1e3 * best_p['dEF']:.0f} meV, rms {1e3 * best_p['score']:.1f} mV; "
      f"near: {[(r['eps'], round(1e3 * r['dEF']), round(1e3 * r['score'], 1)) for r in near_p]}")
print(f"uniform best: eps {best_u['eps']}, dEF {1e3 * best_u['dEF']:.0f} meV, {1e3 * best_u['score']:.1f} mV; flat band in place: "
      f"eps {fb_u['eps']}, dEF {1e3 * fb_u['dEF']:.0f} meV, {1e3 * fb_u['score']:.1f} mV")
print("valid cells: polarized", sum(r["ok"] for r in pol), "of", len(pol), "; uniform", sum(r["ok"] for r in uni), "of", len(uni))
