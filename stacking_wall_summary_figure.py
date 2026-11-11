#!/usr/bin/env python3
"""
Figure set per line spectrum: the flat-band bend at the (+4-)|(+5) wall against the models.

Top row, zoomed on the flat band: the measured dI/dV(V, x), the local approximation (the Poisson
solver's simulated dI/dV of uniform stacks placed along the fitted profile, tip included) and the
SWMcC wall model with the self-consistent far-field on-site energies (the hybrid of
examples/stacking_wall_hybrid.ipynb, no tip).  Bottom left: the flat-band peak along the line for the
three, plus the SWMcC model with the tip-induced bump at the tip positions solved
(stacking_wall_tibb_bump.py; hybrid track plus the centroid shift).  Bottom right: the bend at the
position of the measured wall spectrum split into its parts.

    python stacking_wall_summary_figure.py --tag 940 --sm4 PATH_TO_940.sm4

Writes examples/stacking_wall_summary_TAG.png.  Without --sm4 the measured map is left out (the
measured peak track then comes from the three spectra stored in stacking_wall_tibb_TAG.json).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
sys.path.insert(0, HERE)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (validated all-pairs)


def peak_track(x, E, Z, E0, half=0.09, h_min=0.15, max_jump=0.03):
    """Energy of the maximum within E0 +- half along x; NaN where the peak has faded below h_min of its
    value far on the (+5) side (the last 20 % of the x range)."""
    E, Z, x = np.asarray(E), np.asarray(Z), np.asarray(x)
    sel = (E >= E0 - half) & (E <= E0 + half)
    far = np.nanmax(Z[sel][:, x > x.max() - 0.2 * (x.max() - x.min())])
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        col = Z[sel, i]
        if np.isfinite(col).all() and col.max() > h_min * far:
            out[i] = E[sel][np.argmax(col)]
    # keep only the branch connected to the (+5) side: walk from the far end and stop at the first gap
    # (and at the first jump of more than max_jump, where the window has caught another feature)
    order = np.argsort(x)[::-1]
    gone, last = False, None
    for i in order:
        if gone or not np.isfinite(out[i]) or (last is not None and abs(out[i] - last) > max_jump):
            gone = True
            out[i] = np.nan
        else:
            last = out[i]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sm4", default=None)
    ap.add_argument("--window", type=float, default=0.15, help="half bias window around the (+5) flat band (V)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import stacking_wall as sw
    import stacking_wall_report as R

    t = args.tag
    st = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{t}.json")))
    fit = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_{t}.json")))["fit"]
    s = st["settings"]
    E_fb, x_wall, x0 = s["E_fb_measured"], s["x_wall"], fit["x0"]
    hn = R.hybrid_numbers(t, st)
    bn = R.bump_numbers(t)
    dcf = R.far_tip_shift(t) or 0.0          # the tip's shift of the (+5) reference at its own bias

    # maps
    loc = np.load(os.path.join(EXAMPLES, f"stacking_wall_tibb_map_{t}.npz"))
    with np.load(os.path.join(EXAMPLES, f"stacking_wall_hybrid_{t}.npz")) as f:
        E_h, x_h, top = f["energy"], f["x"] / 10.0 + x0, f["ldos"][-1]
    sel = (E_h >= -0.12) & (E_h <= 0.12)
    E_far = float(E_h[sel][np.argmax(top[sel][:, x_h > x_h.max() - 30.0].mean(axis=1))])
    E_h = E_h + (E_fb - E_far)                       # hybrid (+5) flat band on the measured one
    exp = sw.load_line_spectrum(args.sm4) if args.sm4 and os.path.exists(args.sm4) else None

    panels = []
    if exp is not None:
        panels.append(("measured dI/dV", exp["dist"], exp["bias"], exp["didv"], "bias (V)"))
    panels.append(("local approximation: Poisson solver, tip included", loc["dist"], loc["bias"], loc["didv"], "bias (V)"))
    panels.append(("SWMcC wall + self-consistent on-site energies", x_h, E_h, top, "E − E_F (eV), aligned"))

    xlim = (x0 - 25.0, x0 + 30.0)
    ylim = (E_fb - args.window, E_fb + args.window)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(15, 8.6))
    outer = fig.add_gridspec(2, 1, hspace=0.36)
    gs = outer[0].subgridspec(1, 3, wspace=0.28)
    gb = outer[1].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.55)
    for k, (title, x, E, Z, ylab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k + (3 - len(panels))])
        xs = (x >= xlim[0]) & (x <= xlim[1])
        es = (E >= ylim[0]) & (E <= ylim[1])
        Zw = np.asarray(Z)[np.ix_(es, xs)]
        vmax = np.nanpercentile(Zw, 99.5)
        ax.pcolormesh(np.asarray(x)[xs], np.asarray(E)[es], Zw, cmap="Greys", vmin=0.0, vmax=vmax,
                      shading="nearest", rasterized=True)
        ax.axvline(x0, color=MUTED, lw=0.8, ls="--")
        ax.axhline(E_fb, color=MUTED, lw=0.6, ls=":")
        ax.annotate("", xy=(x_wall, ylim[0]), xytext=(x_wall, ylim[0] + 0.06 * (ylim[1] - ylim[0])),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE if k else INK, lw=1.2))
        ax.set(xlim=xlim, ylim=ylim, xlabel="distance along the line (nm)", ylabel=ylab)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # peak tracks
    ax = fig.add_subplot(gb[0, 0])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if exp is not None:
        tr = peak_track(exp["dist"], exp["bias"], exp["didv"], E_fb)
        ax.plot(exp["dist"], 1e3 * tr, "o", ms=3.2, color=INK, label="measured")
    tr_loc = peak_track(loc["dist"], loc["bias"], loc["didv"], E_fb)
    ax.plot(loc["dist"], 1e3 * tr_loc, "-", color=ORANGE, lw=2, label="local approximation (tip included)")
    tr_h = peak_track(x_h, E_h, top, E_fb)
    ax.plot(x_h, 1e3 * tr_h, "-", color=BLUE, lw=2, label="SWMcC + on-site energies (no tip)")
    if bn is not None:
        pts = []
        h_max = max(r["height"] for r in bn["rows"])
        for r in bn["rows"]:
            if r.get("dcentroid") is None or r["height"] < 0.15 * h_max or abs(r["peak"] - E_fb) > 0.06:
                continue
            xa = r["x"] + x0
            base = np.interp(xa, x_h[np.isfinite(tr_h)], tr_h[np.isfinite(tr_h)])
            pts.append((xa, base + r["dcentroid"] - dcf))
        if pts:
            ax.plot([p[0] for p in pts], [1e3 * p[1] for p in pts], "D", ms=7, color=AQUA, mec="white", mew=1.0,
                    label="SWMcC + on-site energies + tip (tip also under the (+⁵) reference)", zorder=5)
    ax.axvline(x0, color=MUTED, lw=0.8, ls="--")
    ax.axvline(x_wall, color=MUTED, lw=0.8, ls=":")
    ax.text(x_wall, 0.98, " measured wall spectrum", transform=ax.get_xaxis_transform(), fontsize=8, color=INK2,
            ha="left" if E_fb < 0 else "left", va="top")
    ax.set(xlim=xlim, xlabel="distance along the line (nm)", ylabel="flat-band peak (mV)")
    ax.set_title("flat-band peak along the line (followed until it fades to 15 % of its (+⁵) height)",
                 fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right" if E_fb < 0 else "lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # bend at the measured wall position
    ax = fig.add_subplot(gb[0, 1])
    meas = hn["meas"] - E_fb
    local = hn["local"] - hn["local_far"]
    far = hn["hybrid"] - E_fb
    dc = None
    if bn is not None:
        rows = sorted([r for r in bn["rows"] if r.get("dcentroid") is not None
                       and r["height"] >= 0.15 * max(q["height"] for q in bn["rows"])
                       and abs(r["peak"] - E_fb) <= 0.06], key=lambda r: r["x"])
        if rows:
            dc = float(np.interp(x_wall - x0, [r["x"] for r in rows], [r["dcentroid"] for r in rows]))
    bars = [("measured", meas, INK), ("local approximation", local, ORANGE), ("SWMcC + on-sites", far, BLUE)]
    if dc is not None:
        bars.append(("SWMcC + on-sites + tip", far + dc - dcf, AQUA))
    ys = np.arange(len(bars))[::-1]
    for y, (lab, v, c) in zip(ys, bars):
        ax.barh(y, 1e3 * v, height=0.5, color=c)
        ax.text(1e3 * v, y, f" {1e3 * v:+.0f} mV ({100 * v / meas:.0f} %) " if lab != "measured" else f" {1e3 * v:+.0f} mV ",
                va="center", ha="left" if v > 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_yticks(ys, [b[0] for b in bars])
    lim = 1.45 * 1e3 * max(abs(b[1]) for b in bars)
    ax.set_xlim((-0.15 * lim, lim) if meas > 0 else (-lim, 0.15 * lim))
    ax.set_xlabel("bend of the flat-band peak at the wall (mV)")
    ax.set_title(f"bend at x = {x_wall:.1f} nm (s = {s['s']:.2f})", fontsize=9.5, loc="left", color=INK)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.suptitle(f"Line spectrum {t}: flat band {'empty, seen at positive bias' if E_fb > 0 else 'filled, seen at negative bias'}; "
                 f"(+⁵) peak at {1e3 * E_fb:+.0f} mV, wall centre x0 = {x0:.1f} nm (dashed)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    out = os.path.join(EXAMPLES, f"stacking_wall_summary_{t}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
