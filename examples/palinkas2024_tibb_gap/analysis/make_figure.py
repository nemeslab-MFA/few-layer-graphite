"""Figures: S7 gap of 3-, 4- and 6-layer RG from the SWMcC TB model, with the SiO2 charge only, the tip only and
tip + SiO2 charge (fitted), against the STM (S7 method) and ERS (Palinkas et al. 2024) gaps."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
import s7

INV_CM_TO_MEV = 1 / 8.0658163
ERS = {3: np.mean([5692.57, 5785.27, 5630.10, 5674.55]) * INV_CM_TO_MEV,       # Zenodo "ers all flakes" notebook
       4: np.mean([5044.64, 5032.03, 5004.45]) * INV_CM_TO_MEV, 6: 4008.35 * INV_CM_TO_MEV}
ERS_ERR = {3: 120 / 2 * INV_CM_TO_MEV, 4: 40 / np.sqrt(3) * INV_CM_TO_MEV, 6: 40 * INV_CM_TO_MEV}
STM_PAPER = {3: (777.94, 30.50), 4: (657.56, 51.87)}                        # all pixels of both maps, mean, std
SAMPLES = [(3, "3L_647", "thin/gap_3L_647.json", "map 647"), (3, "3L_280", "thin/gap_3L_280.json", "map 280"),
           (4, "4L_209", "thin/gap_4L_209.json", "map 209"), (4, "4L_660", "thin/gap_4L_660.json", "map 660"),
           (6, "6L_940", "six/gap_940.json", "line 940"), (6, "6L_514", "six/gap_514.json", "line 514")]
C = dict(tb="#7a7974", sub="#2a78d6", tipsub="#eb6834", tip="#1baf7a", stm="#0b0b0b", ers="#0b0b0b")
SER = [("tb", "SWMcC TB (no field)", "s", "none"), ("sub", "SiO$_2$ charge only", "o", None),
       ("tip", "tip only", "D", None), ("tipsub", "tip + SiO$_2$ charge (fit)", "o", None),
       ("measured", "STM (S7 fit)", "o", None)]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})


def load():
    rows = []
    for N, key, f, lab in SAMPLES:
        p = os.path.join(W, f)
        if os.path.exists(p):
            d = json.load(open(p))
            g = {k: 1e3 * d[k]["gap"] for k in d if k != "meta" and "gap" in d[k]}
            row = dict(N=N, key=key, label=lab, d=d, g=g)
            if N < 6:                                                    # d = 0.5 / 0.9 nm at the best fit
                ds = d["meta"]["d_sens"]
                row["tibb_d"] = (1e3 * ds["0.50"]["gap"] - g["sub"], 1e3 * ds["0.90"]["gap"] - g["sub"])
            else:                                                        # 6L: relative change of the line-940 geometry sweep
                gg = json.load(open(os.path.join(W, "six/geom_gaps.json")))
                ref = gg["d0.70_R020.0"]["gap"]
                row["tibb_d"] = tuple(g["tipsub"] - g["sub"] + gg[k]["gap"] - ref for k in ("d0.50_R020.0", "d0.90_R020.0"))
            dg = json.load(open(os.path.join(W, "dos_gap.json")))
            k = f"{key}_sub"
            if k in dg:
                row["dos_sub"] = 1e3 * dg[k]["gap"]
            rows.append(row)
    return rows


def fig_gaps(rows, out):
    Ns = sorted({r["N"] for r in rows})
    fig, axs = plt.subplots(1, len(Ns) + 1, figsize=(4.0 * (len(Ns) + 1), 4.6),
                            gridspec_kw=dict(width_ratios=[1] * len(Ns) + [1.25]))
    for ax, N in zip(axs, Ns):
        rs = [r for r in rows if r["N"] == N]
        ax.axhspan(ERS[N] - ERS_ERR[N], ERS[N] + ERS_ERR[N], color="#0b0b0b", alpha=0.08, lw=0)
        ax.axhline(ERS[N], color="#0b0b0b", lw=1.2, ls="-")
        ax.text(2.1, ERS[N] - 2, f"ERS\n{ERS[N]:.0f}", va="top", ha="right", fontsize=8.5, color="#52514e")
        if N in STM_PAPER:
            ax.axhline(STM_PAPER[N][0], color="#0b0b0b", lw=1.0, ls="--")
            ax.text(2.1, STM_PAPER[N][0] + 2, f"STM paper\n{STM_PAPER[N][0]:.0f}", va="bottom", ha="right", fontsize=8.5,
                    color="#52514e")
        for i, r in enumerate(rs):
            for j, (k, lab, mk, fc) in enumerate(SER):
                if k not in r["g"]:
                    continue
                x = i + (j - 2) * 0.13
                col = C["stm"] if k == "measured" else C[k]
                ax.plot(x, r["g"][k], mk, ms=8 if mk != "D" else 7, mfc=fc if fc else col, mec=col, mew=1.6,
                        label=lab if (i == 0 and ax is axs[0]) else None, zorder=3)
        ax.set_xticks(range(len(rs)), [r["label"] for r in rs])
        ax.set_xlim(-0.55, 1.55 + 0.6)
        ax.set_title(f"{N}-layer RG", fontsize=11)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel(r"$-1 \rightarrow +1$ gap (meV)")
    axs[0].plot([], [], "-", color="#0b0b0b", label="ERS (mean of flakes)")
    axs[0].plot([], [], "--", color="#0b0b0b", label="STM, paper Fig. 4d")
    # increase panel
    ax = axs[-1]
    by = {}
    for r in rows:
        by.setdefault(r["N"], []).append(r)
    first = True
    for N, rs in sorted(by.items()):
        for i, r in enumerate(rs):
            x0 = N + (i - 0.5) * 0.42
            g = r["g"]
            t = g["tipsub"] - g["sub"]
            lo, hi = sorted(r["tibb_d"])
            ax.plot([x0 - 0.09] * 2, [lo, hi], "-", color=C["tipsub"], lw=5, alpha=0.25, solid_capstyle="butt", zorder=2,
                    label="model, tip distance 0.5$-$0.9 nm" if first else None)
            ax.plot(x0 - 0.09, t, "o", ms=8, color=C["tipsub"], zorder=3,
                    label="model: S7(tip+SiO$_2$) $-$ S7(SiO$_2$ only)" if first else None)
            if "dos_sub" in r:
                ax.plot(x0 - 0.09, g["tipsub"] - r["dos_sub"], "o", ms=8, mfc="none", mec=C["tipsub"], mew=1.6, zorder=3,
                        label="model: S7(tip+SiO$_2$) $-$ DOS-peak gap (ERS analog)" if first else None)
            ax.errorbar(x0 + 0.09, g["measured"] - ERS[N], yerr=ERS_ERR[N], fmt="o", ms=8, color=C["stm"], lw=1, capsize=3,
                        zorder=3, label="data: STM (S7, this work) $-$ ERS" if first else None)
            first = False
        if N in STM_PAPER:
            ax.plot(N, STM_PAPER[N][0] - ERS[N], "s", ms=8, mfc="none", mec=C["stm"], mew=1.6, zorder=3,
                    label="data: STM (paper Fig. 4d) $-$ ERS" if N == min(by) else None)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(sorted(by), [f"{n} layers" for n in sorted(by)])
    ax.set_xlim(min(by) - 0.6, max(by) + 0.6)
    ax.set_ylabel("gap increase (meV)")
    ax.set_title("tip-induced gap increase vs STM $-$ ERS", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.04, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=200)


def fig_spectra(rows, out):
    n = len(rows)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4.3 * ((n + 1) // 2), 7.6), squeeze=False)
    for ax, r in zip(axs.T.ravel(), rows):
        d = r["d"]
        lim = (-0.6, 0.7) if r["N"] < 6 else (-0.45, 0.45)
        for k, lab, col, dy in (("measured", "STM", C["stm"], 0.0), ("tipsub", "tip + SiO$_2$ (fit)", C["tipsub"], 0.0),
                                ("sub", "SiO$_2$ only", C["sub"], -0.9)):
            V, y = np.asarray(d[k]["bias"]), np.asarray(d[k]["didv"])
            s = (V >= lim[0]) & (V <= lim[1])
            yn = y / np.percentile(y[s], 98)
            ax.plot(V[s], yn[s] + dy, color=col, lw=1.4 if k == "measured" else 1.2,
                    label=lab)
            for x in (d[k]["x0_h"], d[k]["x0_e"]):
                ax.plot([x], [np.interp(x, V, yn) + dy], "|", color=col, ms=14, mew=2)
        ax.set_title(f"{r['N']}L, {r['label']}: STM {r['g']['measured']:.0f}, fit {r['g']['tipsub']:.0f}, "
                     f"SiO$_2$ only {r['g']['sub']:.0f} meV", fontsize=8.5)
        ax.set_xlabel("sample bias (V)")
        ax.set_yticks([])
    for ax in axs.T.ravel()[n:]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=8, frameon=False, loc="upper center")
    axs[0, 0].set_ylabel("d$I$/d$V$ (norm.; SiO$_2$-only offset)")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


if __name__ == "__main__":
    rows = load()
    for r in rows:
        print(r["key"], {k: round(v, 1) for k, v in r["g"].items()})
    fig_gaps(rows, os.path.join(W, "..", "results", "fig_gap_tibb.png"))
    fig_spectra(rows, os.path.join(W, "..", "results", "fig_spectra_tibb.png"))
