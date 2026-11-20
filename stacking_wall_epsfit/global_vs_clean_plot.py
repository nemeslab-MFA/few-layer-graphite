"""Line 940, both sides of the wall, with the tip: the global-offset (per-layer doping + SiO2 charge) fits against the clean model of steps 1 and 2."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/home/user/few-layer-graphite"
sys.path.insert(0, REPO + "/stacking_wall_epsfit")
import step1
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
m = json.load(open(f"{REPO}/examples/stacking_wall_tibb_940.json"))
V = np.asarray(m["measured"]["bias"]); w = (V >= -0.45) & (V <= 0.45)
meas = {"(+5)": np.asarray(m["measured"]["rg"], float), "(+4-)": np.asarray(m["measured"]["sf"], float)}

def scaled(x, y, ym):
    yi = np.interp(V[w], x, y)
    return float(np.dot(yi, ym[w]) / np.dot(yi, yi)) * np.asarray(y)

models = []
for suf, lab, col in (("_eps26_fb", "global offset ΔE_F 60 meV, ε 2.6 (flat-band fit)", ORANGE),
                      ("_eps26_peaks", "global offset ΔE_F 40 meV, ε 2.6 (best-peak fit)", AQUA)):
    st = json.load(open(f"{REPO}/examples/stacking_wall_polar{suf}_940.json"))
    fb = np.asarray(st["fine_bias"])
    r0 = next(r for r in st["runs"].values() if abs(r["s"]) < 1e-6)
    r1 = next(r for r in st["runs"].values() if abs(r["s"] - 1) < 1e-6)
    U1 = lambda r: 1e3 * (np.nanmean(np.asarray(r["sweep"]["U_far"]), axis=0)[0] - r["sweep"]["mu0"])
    models.append((lab, col, {"(+5)": (fb, np.asarray(r0["didv"])), "(+4-)": (fb, np.asarray(r1["didv"]))}, {"(+5)": U1(r0), "(+4-)": U1(r1)}))
c1 = json.load(open(f"{REPO}/stacking_wall_epsfit/step1/940_eps02.6_n+2.00_c-0.150.json"))
c2 = json.load(open(f"{REPO}/stacking_wall_epsfit/step2/940.json"))
models.append(("clean model: neutral layers, n_sub +2×10¹³, ε 2.6, CPot −0.15 eV", BLUE,
               {"(+5)": (step1.FINE, np.asarray(c1["tip"]["didv"])), "(+4-)": (step1.FINE, np.asarray(c2["tip"]["didv"]))},
               {"(+5)": 1e3 * c1["U_minus_mu0"][0], "(+4-)": 1e3 * c2["U_minus_mu0"][0]}))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, side in zip(axs, ("(+5)", "(+4-)")):
    ax.plot(V[w], meas[side][w], color=INK, lw=1.9, label=f"measured {side}, line 940")
    for lab, col, sp, U1d in models:
        x, y = sp[side]
        ax.plot(x, scaled(x, y, meas[side]), color=col, lw=1.3, label=f"{lab}; U₁ − E_F {U1d[side]:+.0f} meV")
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set(xlim=(-0.45, 0.45), xlabel="sample bias (V)", ylabel="dI/dV (measured units)")
    ax.set_title(f"{side.replace('(+5)', '(+⁵)').replace('(+4-)', '(+⁴−)')} side, all with the tip", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=7.2, frameon=False, loc="upper center")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
fig.tight_layout()
out = f"{REPO}/examples/stacking_wall_global_vs_clean_940.png"
fig.savefig(out, dpi=160)
print("wrote", out)
