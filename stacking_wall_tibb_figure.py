#!/usr/bin/env python3
"""
Figure of the tip-induced band-bending calculation (``stacking_wall_tibb.py``):
for each line spectrum, the measured dI/dV far on the (+4-) side, at the wall and
far on the (+5) side against the simulated top-layer dI/dV of the three stacks
(same tip, same doping), the contact-potential scan residuals and the layer
potentials under the apex.

    python stacking_wall_tibb_figure.py [--tags 514 940]

Writes examples/stacking_wall_tibb.png and prints the fitted parameters.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
INK, MUTED, BLUE, ORANGE, GREEN = "#1d1c1a", "#605d57", "#1f6fb2", "#d95926", "#2a8a5c"


def load(tag):
    with open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")) as fh:
        return json.load(fh)


def best_run(state, prefix):
    runs = state.get("runs", {})
    keys = [k for k in runs if k.startswith(prefix)]
    return runs[keys[0]] if keys else None


def summary_rows(tags):
    rows = []
    for tag in tags:
        st = load(tag)
        s = st["settings"]
        scan = st.get("scan", {})
        best = scan.get(f"{st.get('best_cpot', 0.0):+.3f}") if scan else None
        sf, wall = best_run(st, "sf"), best_run(st, "wall")
        rows.append({
            "tag": tag, "E_fb": s["E_fb_measured"], "fermi_level": s["fermi_level"],
            "cpot": st.get("best_cpot"), "res_rg": best["residual"] if best else None,
            "peak_rg": best["peak_sim"] if best else None,
            "res_wall": wall["residual"] if wall else None, "peak_wall": wall["peak_sim"] if wall else None,
            "res_sf": sf["residual"] if sf else None,
            "doping": st.get("doping"), "s": s["s"], "x_wall": s["x_wall"],
            "mu0_sf": sf["sweep"]["mu0"] if sf else None, "mu0_wall": wall["sweep"]["mu0"] if wall else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["514", "940"])
    ap.add_argument("--out", default=os.path.join(EXAMPLES, "stacking_wall_tibb.png"))
    args = ap.parse_args()
    tags = [t for t in args.tags if os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_tibb_{t}.json"))]
    fig, axs = plt.subplots(len(tags), 4, figsize=(16, 3.9 * len(tags)), squeeze=False)
    for row, tag in enumerate(tags):
        st = load(tag)
        meas = st["measured"]
        bias = np.asarray(meas["bias"])
        fine = np.asarray(st.get("fine_bias", []))
        scan = st.get("scan", {})
        best_key = f"{st.get('best_cpot', 0.0):+.3f}" if scan else None
        panels = [("rg", "(+5), far from the wall", scan.get(best_key) if best_key else None),
                  ("wall", f"wall, s = {st['settings']['s']:.2f} at {st['settings']['x_wall']:.0f} nm", best_run(st, "wall")),
                  ("sf", "(+4−), far from the wall", best_run(st, "sf"))]
        for col, (key, title, run) in enumerate(panels):
            ax = axs[row, col]
            y = np.asarray(meas[key])
            ok = np.isfinite(y)
            ax.plot(bias[ok], y[ok] * 1e3, color=INK, lw=1.4, label="measured dI/dV")
            if run is not None and fine.size:
                a = run["scale"]
                ax.plot(fine, a * np.asarray(run["didv"]) * 1e3, color=ORANGE, lw=1.3, label="simulated, scaled")
                ax.plot(fine, a * np.asarray(run["didv_frozen"]) * 1e3, color=ORANGE, lw=0.9, ls=(0, (3, 2)),
                        label="frozen-potential term")
                ax.set_title(f"{title}  (residual {run['residual']:.3f})", fontsize=9)
            else:
                ax.set_title(f"{title}  (not computed)", fontsize=9)
            ax.axvline(0.0, color=MUTED, lw=0.5)
            ax.set_xlim(-0.45, 0.45)
            ax.set_xlabel("sample bias (V)")
            if col == 0:
                ax.set_ylabel(f"line spectrum {tag}\nnormalised dI/dV (a.u.)")
            ax.legend(fontsize=7, frameon=False)
        # contact-potential scan and apex potentials
        ax = axs[row, 3]
        if scan:
            cp = sorted(scan.values(), key=lambda d: d["cpot"])
            ax.plot([d["cpot"] for d in cp], [d["residual"] for d in cp], "o-", color=BLUE, label="(+5) residual")
            ax.set_xlabel("tip contact potential W_tip − W_sample (eV)")
            ax.set_ylabel("relative residual, (+5) spectrum")
            ax2 = ax.twinx()
            ax2.plot([d["cpot"] for d in cp], [1e3 * d["peak_sim"] for d in cp], "s--", color=GREEN, label="simulated peak")
            ax2.axhline(1e3 * st["settings"]["E_fb_measured"], color=GREEN, lw=0.6, ls=":")
            ax2.set_ylabel("flat-band peak (mV)", color=GREEN)
            ax.set_title(f"scan; best {st.get('best_cpot', float('nan')):+.2f} eV", fontsize=9)
    fig.suptitle("Tip-induced band bending in the local approximation (bandbending engine): "
                 "same tip and doping for the three stacks", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")
    for r in summary_rows(tags):
        print(r)


if __name__ == "__main__":
    main()
