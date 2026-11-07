#!/usr/bin/env python3
"""
Waterfall plots of the band-bending comparison: for each line spectrum, the three
measured dI/dV curves (far on the (+4-) side, at the wall, far on the (+5) side) stacked
with a vertical offset on the left, and the simulated top-layer dI/dV of the three stacks
(scaled as in the fit) stacked the same way on the right.

    python stacking_wall_tibb_waterfall.py [--tags 514 940]

Writes examples/stacking_wall_tibb_waterfall.png.
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
INK, MUTED, BLUE, ORANGE, GREY = "#1d1c1a", "#605d57", "#1f6fb2", "#d95926", "#52514e"


def load(tag):
    with open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")) as fh:
        return json.load(fh)


def series(state):
    """[(label, colour, measured (bias, y), simulated (bias, a*y))] from the bottom up."""
    meas = state["measured"]
    bias = np.asarray(meas["bias"])
    fine = np.asarray(state["fine_bias"])
    s = state["settings"]
    scan = state["scan"]
    best = scan[f"{state['best_cpot']:+.3f}"]
    runs = state["runs"]
    wall = next(runs[k] for k in runs if k.startswith("wall"))
    out = []
    for key, run, label, col in (("sf", runs["sf"], "(+4−), far from the wall", BLUE),
                                 ("wall", wall, f"wall, s = {s['s']:.2f} ({s['x_wall']:.0f} nm)", GREY),
                                 ("rg", best, "(+5), far from the wall", ORANGE)):
        y = np.asarray(meas[key]) * 1e3
        out.append((label, col, (bias, y), (fine, run["scale"] * np.asarray(run["didv"]) * 1e3)))
    return out


def waterfall(ax, curves, window, step):
    for i, (label, col, (x, y)) in enumerate(curves):
        sel = (x >= window[0]) & (x <= window[1]) & np.isfinite(y)
        ax.plot(x[sel], y[sel] + i * step, color=col, lw=1.4)
        ax.text(window[1] - 0.01, i * step + 0.5 * step, label, ha="right", va="bottom", fontsize=8, color=col)
    ax.axvline(0.0, color=MUTED, lw=0.5)
    ax.set_xlim(window)
    ax.set_xlabel("sample bias (V)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["514", "940"])
    ap.add_argument("--window", type=float, nargs=2, default=(-0.45, 0.45))
    ap.add_argument("--out", default=os.path.join(EXAMPLES, "stacking_wall_tibb_waterfall.png"))
    args = ap.parse_args()
    tags = [t for t in args.tags if os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_tibb_{t}.json"))]
    fig, axs = plt.subplots(len(tags), 2, figsize=(10, 4.6 * len(tags)), squeeze=False, sharey="row")
    for row, tag in enumerate(tags):
        st = load(tag)
        ser = series(st)
        win = tuple(args.window)
        inwin = lambda c: np.nanmax(c[1][(c[0] >= win[0]) & (c[0] <= win[1])])   # noqa: E731
        top_meas = max(inwin(m) for _, _, m, _ in ser)
        top_all = max(top_meas, max(inwin(s) for _, _, _, s in ser))
        step = 0.9 * top_meas
        waterfall(axs[row, 0], [(l, c, m) for l, c, m, _ in ser], win, step)
        waterfall(axs[row, 1], [(l, c, s) for l, c, _, s in ser], win, step)
        axs[row, 0].set_title(f"line spectrum {tag}: measured dI/dV", fontsize=10)
        axs[row, 1].set_title("simulated top-layer dI/dV, same tip and doping (scaled)", fontsize=10)
        axs[row, 0].set_ylabel("normalised dI/dV, offset (a.u.)")
        axs[row, 0].set_ylim(-0.1 * step, 2 * step + 1.05 * top_all)
    fig.suptitle("Across the wall, bottom to top: (+4−) side, wall, (+5) side", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
