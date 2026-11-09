"""Line 940, (+5) side, eps = 2.6: map of the feature rms under the tip over the starting doping and
the hole density left by n_sub (fixeps.py), and the best spectrum against the measurement.

Writes examples/stacking_wall_eps26_940.json and examples/stacking_wall_eps26_940.png.
A cell counts only if all five features are found in increasing order, with the flat-band peak
inside its window and the lower side-band peak below -200 mV (above it the finder has taken a
weak structure between the side band and the flat band).
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
from features import NAMES5, features5                               # noqa: E402

EXAMPLES = os.path.join(REPO, "examples")
INK, INK2, MUTED, GRID, S1, S2 = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#2a78d6", "#eb6834"
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
st = json.load(open(os.path.join(EXAMPLES, "stacking_wall_tibb_940.json")))
VM, YM = np.asarray(st["measured"]["bias"]), np.asarray(st["measured"]["rg"], dtype=float)
FM5 = features5(VM, YM, smooth=9)
WIN = (VM >= -0.45) & (VM <= 0.45)
NORM = YM[WIN].max()

rows = []
d = os.path.join(HERE, "fixeps")
for f in sorted(os.listdir(d)):
    if f.endswith(".json") and not f.endswith(".ckpt.json"):
        rows.append(json.load(open(os.path.join(d, f))))
from scipy.signal import peak_prominences, savgol_filter      # noqa: E402


def low_prominence(V, y, f0):
    """Prominence of the lower side-band peak at f0, in units of the flat-band peak."""
    y = np.asarray(y, dtype=float) / np.nanmax(np.asarray(y)[(V > -0.12) & (V < 0.06)])
    return float(peak_prominences(y, [int(np.argmin(np.abs(V - f0)))])[0][0])


PROM_M = low_prominence(VM, savgol_filter(YM, 9, 3), FM5[0])
for r in rows:
    f = np.asarray(r["features"], dtype=float)
    r["prominence"] = low_prominence(FINE, r["didv"], f[0]) if np.isfinite(f[0]) else np.nan
    r["ok"] = bool(np.all(np.isfinite(f)) and np.all(np.diff(f) > 0) and f[1] > -0.115 and f[0] < -0.2
                   and r["prominence"] >= 0.5 * PROM_M)
    r["score"] = float(np.sqrt(np.mean((f - FM5) ** 2))) if r["ok"] else np.nan
    r["holes"] = -r["carriers_target"]
best = min((r for r in rows if r["ok"]), key=lambda r: r["score"])
# ties in the feature rms (within 1 mV) are broken by the shape residual of the whole curve
fb_best = min((r for r in rows if r["ok"] and abs(r["features"][1] - FM5[1]) <= 0.010),
              key=lambda r: (round(1e3 * r["score"]), r["residual"]))
ranked = sorted((r for r in rows if r["ok"]), key=lambda r: r["score"])


def scaled(y):
    y = np.asarray(y, dtype=float)
    ys = np.interp(VM[WIN], FINE, y)
    return float(np.dot(ys, YM[WIN]) / np.dot(ys, ys)) * y / NORM


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.0, 1.45]})
dEFs = sorted({r["dEF"] for r in rows})
holes = sorted({r["holes"] for r in rows})
M = np.full((len(dEFs), len(holes)), np.nan)
present = np.zeros_like(M, dtype=bool)
for r in rows:
    i, j = dEFs.index(r["dEF"]), holes.index(r["holes"])
    M[i, j] = r["score"]
    present[i, j] = True
ax = axs[0]
im = ax.imshow(np.clip(1e3 * M, 0, 60), origin="lower", cmap="Blues", aspect="auto", vmin=10, vmax=60)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            v = 1e3 * M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 40 else INK)
        elif present[i, j]:
            ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=MUTED)
bi, bj = dEFs.index(best["dEF"]), holes.index(best["holes"])
ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=S2, lw=2.2))
ax.set_xticks(range(len(holes)), [f"{h / 1e10:g}" for h in holes])
ax.set_yticks(range(len(dEFs)), [f"{1e3 * v:g}" for v in dEFs])
ax.set_xlabel("holes left in the stack by n$_{sub}$ (10$^{10}$ cm$^{-2}$)")
ax.set_ylabel("starting doping ΔE$_F$ (meV)")
bi2, bj2 = dEFs.index(fb_best["dEF"]), holes.index(fb_best["holes"])
ax.add_patch(plt.Rectangle((bj2 - 0.5, bi2 - 0.5), 1, 1, fill=False, ec=S1, lw=2.2, ls="--"))
ax.set_title("(a) ε = 2.6: feature rms under the tip (mV)\nframes: best (orange), best with the flat band within 10 mV (blue)",
             fontsize=10, loc="left", color=INK)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("rms (mV)")
ax = axs[1]
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)
ax.plot(VM[WIN], YM[WIN] / NORM, color=INK, lw=1.5, label="measured dI/dV")
y_tip, y_free = scaled(best["didv"]), scaled(best["didv_free"])
ax.plot(FINE, y_tip, color=S1, lw=1.5,
        label=f"under the tip: ε = 2.6, ΔE$_F$ = {1e3 * best['dEF']:g} meV, n$_{{sub}}$ = "
              f"{best['nsub'] / 1e13:.4f}×10$^{{13}}$ cm$^{{-2}}$ ({best['carriers'] / 1e10:+.0f}×10$^{{10}}$ cm$^{{-2}}$ carriers)")
ax.plot(FINE, y_free, color=S1, lw=1.0, ls=(0, (3, 2)), label="same stack without the tip")
y_fb = scaled(fb_best["didv"])
ax.plot(FINE, y_fb, color=S2, lw=1.3,
        label=f"flat band within 10 mV: ΔE$_F$ = {1e3 * fb_best['dEF']:g} meV, {fb_best['carriers'] / 1e10:+.0f}×10$^{{10}}$ cm$^{{-2}}$ "
              f"carriers ({1e3 * fb_best['score']:.1f} mV)")
f = np.asarray(best["features"])
ax.plot(f, np.interp(f, FINE, y_tip), "o", ms=4.5, color=S1, mec="white", mew=0.8, zorder=5)
f2 = np.asarray(fb_best["features"])
ax.plot(f2, np.interp(f2, FINE, y_fb), "o", ms=4.5, color=S2, mec="white", mew=0.8, zorder=5)
ax.plot(FM5, np.interp(FM5, VM[WIN], YM[WIN] / NORM), "o", ms=4.5, color=INK, mec="white", mew=0.8, zorder=5)
ax.set_xlim(-0.45, 0.45)
ax.set_ylim(0, max(1.4, 1.12 * max(y_tip.max(), y_free.max())))
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_xlabel("sample bias (V)")
ax.set_ylabel("dI/dV (normalised)")
ax.set_title(f"(b) best cell (blue, {1e3 * best['score']:.1f} mV; dashed: without the tip)\nand best with the flat band in place (orange)",
             fontsize=10, loc="left", color=INK)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.text(0.01, 0.01, "Tip R = 20 nm, d = 0.7 nm, no band bending at 0 V; 9 K, 10 meV broadening, 3.75 mV lock-in. Map: blank, not computed; "
         "–, a feature missing, misassigned or a ripple. "
         "Dots: lower side-band peak, flat-band peak, bottom-band onset, upper side-band edges 1 and 2.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(EXAMPLES, "stacking_wall_eps26_940.png"), dpi=160)

U = np.asarray(best["U_far"])
summary = dict(
    eps=2.6, measured=FM5.tolist(), names=list(NAMES5),
    best=dict(dEF=best["dEF"], nsub=best["nsub"], carriers=best["carriers"], rms=best["score"], rms_free=best["rms_free"],
              features=best["features"], features_free=best["features_free"], cpot=best["cpot"],
              U_far_minus_EF=(U - best["mu0"]).tolist(), residual=best["residual"]),
    flat_band_in_place=dict(dEF=fb_best["dEF"], nsub=fb_best["nsub"], carriers=fb_best["carriers"], rms=fb_best["score"],
                            features=fb_best["features"], residual=fb_best["residual"]),
    prominence_measured=PROM_M,
    ranked=[dict(dEF=r["dEF"], holes=r["holes"], carriers=r["carriers"], nsub=r["nsub"], rms=r["score"],
                 features=r["features"], residual=r["residual"], prominence=r["prominence"]) for r in ranked],
    cells=[dict(dEF=r["dEF"], holes=r["holes"], carriers=r["carriers"], nsub=r["nsub"], valid=r["ok"],
                rms=None if np.isnan(r["score"]) else r["score"], features=r["features"], rms_free=r["rms_free"])
           for r in rows],
    spectra=dict(bias=FINE.tolist(), tip=y_tip.tolist(), free=y_free.tolist()),
)
json.dump(summary, open(os.path.join(EXAMPLES, "stacking_wall_eps26_940.json"), "w"))
print(f"{len(rows)} cells, {sum(r['ok'] for r in rows)} valid")
for r in ranked[:6]:
    print(f"  dEF {1e3 * r['dEF']:5.0f} meV, holes {r['holes'] / 1e10:5.1f}e10 (carriers {r['carriers'] / 1e10:+6.1f}e10), "
          f"n_sub {r['nsub'] / 1e13:+.4f}e13: rms {1e3 * r['score']:.1f} mV, shape residual {r['residual']:.3f}, {np.round(1e3 * np.asarray(r['features']))}")
print(f"flat band within 10 mV: dEF {1e3 * fb_best['dEF']:.0f} meV, carriers {fb_best['carriers'] / 1e10:+.1f}e10, rms {1e3 * fb_best['score']:.1f}, "
      f"residual {fb_best['residual']:.3f}, {np.round(1e3 * np.asarray(fb_best['features']))}")
print("measured", np.round(1e3 * FM5), " best U - E_F far (meV)", np.round(1e3 * (U - best["mu0"]), 1))
