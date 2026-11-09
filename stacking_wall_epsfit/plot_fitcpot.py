"""Figure and summary of fit_cpot.py: the measured (+5) spectrum of 940 against the starting point (the
eps = 2.6 flat-band fit) and the best point of the search over Delta E_F, the hole density and the tip
work function; the search path.  Writes examples/stacking_wall_eps26_cpot_940.png and .json.
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
import fit_cpot                                                     # noqa: E402

EX = os.path.join(REPO, "examples")
INK, INK2, MUTED, GRID, S1, S2 = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#2a78d6", "#eb6834"
FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
st = json.load(open(os.path.join(EX, "stacking_wall_tibb_940.json")))
VM, YM = np.asarray(st["measured"]["bias"]), np.asarray(st["measured"]["rg"], dtype=float)
FM = features5(VM, YM, smooth=9)
WIN = (VM >= -0.45) & (VM <= 0.45)
NORM = YM[WIN].max()

log = [json.loads(l) for l in open(os.path.join(HERE, "fitcpot", "log.jsonl"))]
states = [e for e in log if e["event"] in ("start", "state", "done")]
best_p = tuple(states[-1]["p"])
cell = lambda p: json.load(open(os.path.join(HERE, "fitcpot", fit_cpot.key(tuple(p)) + ".json")))   # noqa: E731
start, best = cell(fit_cpot.START), cell(best_p)
evals = [e for e in log if e["event"] == "eval"]


def scaled(y):
    y = np.asarray(y, dtype=float)
    ys = np.interp(VM[WIN], FINE, y)
    return float(np.dot(ys, YM[WIN]) / np.dot(ys, ys)) * y / NORM


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.6, 1.0]})
ax = axs[0]
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)
ax.plot(VM[WIN], YM[WIN] / NORM, color=INK, lw=1.5, label="measured dI/dV")
for r, c, lab in ((start, S2, "start (flat-band fit, tip flat at 0 V)"), (best, S1, "best (tip work function fitted)")):
    y = scaled(r["didv"])
    ax.plot(FINE, y, color=c, lw=1.4,
            label=f"{lab}: ΔE$_F$ = {1e3 * r['dEF']:.0f} meV, {r['holes'] / 1e10:.0f}×10$^{{10}}$ holes, "
                  f"δ = {1e3 * r['delta']:+.0f} meV ({1e3 * r['rms']:.1f} mV)")
    f = np.asarray(r["features"])
    ax.plot(f, np.interp(f, FINE, y), "o", ms=4.5, color=c, mec="white", mew=0.8, zorder=5)
ax.plot(FM, np.interp(FM, VM[WIN], YM[WIN] / NORM), "o", ms=4.5, color=INK, mec="white", mew=0.8, zorder=5)
ax.set_xlim(-0.45, 0.45)
ax.set_ylim(0, 1.95)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("sample bias (V)")
ax.set_ylabel("dI/dV (normalised)")
ax.set_title("(a) line 940, (+⁵) side, ε = 2.6, under the tip", fontsize=10, loc="left", color=INK)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax = axs[1]
path = [s for s in states if s["event"] != "done"]
its = [s.get("it", -1) + 1 for s in path]
ax.plot(its, [1e3 * s["rms"] for s in path], "o-", color=S1, ms=4)
ok = [e for e in evals if e["valid"]]
ax.plot([e["it"] + 1 for e in ok], [1e3 * e["rms"] for e in ok], ".", color=MUTED, ms=5, label="points tried")
ax.set_xlabel("iteration")
ax.set_ylabel("feature rms (mV)")
ax.set_title("(b) search path (line: current best)", fontsize=10, loc="left", color=INK)
ax.grid(color=GRID, lw=0.8)
ax.legend(fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "δ: contact potential minus the value that leaves the bands flat at zero bias (δ > 0: tip work function "
         "above the sample's). Dots: lower side-band peak, flat-band peak, bottom-band onset, upper side-band edges 1 and 2.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(EX, "stacking_wall_eps26_cpot_940.png"), dpi=160)

summary = dict(measured=FM.tolist(), names=list(NAMES5),
               start={k: start[k] for k in ("p", "dEF", "holes", "delta", "nsub", "cpot", "features", "rms", "residual")},
               best={k: best[k] for k in ("p", "dEF", "holes", "delta", "nsub", "cpot", "carriers", "features", "rms", "residual",
                                          "bend_at_zero")},
               path=[dict(p=s["p"], rms=s["rms"]) for s in states], n_evaluations=len(evals))
json.dump(summary, open(os.path.join(EX, "stacking_wall_eps26_cpot_940.json"), "w"))
for name, r in (("start", start), ("best", best)):
    print(f"{name}: dEF {1e3 * r['dEF']:.0f} meV, holes {r['holes']:.2e}, delta {1e3 * r['delta']:+.1f} meV, n_sub {r['nsub']:.4e}, "
          f"CPot {1e3 * r['cpot']:.1f} meV, rms {1e3 * r['rms']:.1f} mV, residual {r['residual']:.3f}, {np.round(1e3 * np.asarray(r['features']))}")
print("measured", np.round(1e3 * FM), f"; {len(evals)} evaluations")
