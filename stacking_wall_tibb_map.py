#!/usr/bin/env python3
"""
Simulated dI/dV(V, x) across the wall in the style of the measured line spectra.

The wall sweeps of ``stacking_wall_tibb.py`` at several (+4-) weights s (the
(+5) stack is s = 0, the (+4-) stack s = 1) are interpolated linearly in s along
the fitted profile s(x) = [1 - tanh((x - x0)/l)]/2 of the line spectrum, with
one global scale (the fit of the (+5) spectrum), and drawn next to the
measurement: the maps and their bias derivatives.

    python stacking_wall_tibb_map.py [--tags 514 940]

Reads examples/stacking_wall_tibb_TAG.json and every
examples/stacking_wall_tibb_TAG_s*.json, and examples/stacking_wall_TAG.json
for the profile; writes examples/stacking_wall_tibb_map_TAG.png and
examples/stacking_wall_tibb_map_TAG.npz.
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")


def collect(tag):
    """(s values, dI/dV curves (n_s, n_V) scaled, fine bias, state, profile fit)."""
    main = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")))
    fine = np.asarray(main["fine_bias"])
    best = main["scan"][f"{main['best_cpot']:+.3f}"]
    scale = best["scale"]
    curves = {0.0: np.asarray(best["didv"]), 1.0: np.asarray(main["runs"]["sf"]["didv"])}
    files = [os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")]
    files += sorted(glob.glob(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}_s*.json")))
    for f in files:
        st = json.load(open(f))
        for k, run in st.get("runs", {}).items():
            if k.startswith("wall_s") and run.get("s") is not None:
                curves[float(run["s"])] = np.asarray(run["didv"])
    s_vals = np.array(sorted(curves))
    D = np.array([curves[s] for s in s_vals]) * scale * 1e3
    fit = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_{tag}.json")))["fit"]
    return s_vals, D, fine, main, fit


def simulated_map(s_vals, D, dist, fit):
    """dI/dV (n_V, n_x): linear interpolation in s along the profile."""
    s_x = 0.5 * (1.0 - np.tanh((dist - fit["x0"]) / fit["ell"]))
    out = np.empty((D.shape[1], dist.size))
    for j, s in enumerate(s_x):
        i = np.searchsorted(s_vals, s)
        i = np.clip(i, 1, s_vals.size - 1)
        w = (s - s_vals[i - 1]) / (s_vals[i] - s_vals[i - 1])
        out[:, j] = (1.0 - w) * D[i - 1] + w * D[i]
    return out, s_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["514", "940"])
    ap.add_argument("--window", type=float, nargs=2, default=(-0.45, 0.45))
    ap.add_argument("--sm4", nargs="*", default=[], metavar="TAG=PATH",
                    help="line spectra for the measured panels, e.g. 514=/path/to/file.sm4")
    args = ap.parse_args()
    sm4 = dict(item.split("=", 1) for item in args.sm4)
    for tag in args.tags:
        if not os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")):
            continue
        s_vals, D, fine, main_state, fit = collect(tag)
        meas = main_state["measured"]
        bias = np.asarray(meas["bias"])
        # the measured line (bias x dist) is not stored in the json: read the sm4 when given
        exp_map = dist = None
        bias_exp = bias
        if tag in sm4:
            import stacking_wall as sw
            exp = sw.load_line_spectrum(sm4[tag])
            dist, bias_exp, exp_map = exp["dist"], exp["bias"], exp["didv"]
        if dist is None:
            dist = np.linspace(0.0, 2.0 * fit["x0"], 256)
        sim, s_x = simulated_map(s_vals, D, dist, fit)
        np.savez_compressed(os.path.join(EXAMPLES, f"stacking_wall_tibb_map_{tag}.npz"), dist=dist, bias=fine,
                            didv=sim.astype(np.float32), s=s_x, s_values=s_vals)
        lo, hi = args.window
        sel_s = (fine >= lo) & (fine <= hi)
        ncol = 2 if exp_map is not None else 1
        fig, axs = plt.subplots(2, ncol, figsize=(5.5 * ncol, 8), squeeze=False)
        col = 0
        if exp_map is not None:
            sel_e = (bias_exp >= lo) & (bias_exp <= hi)
            Z = exp_map[sel_e] * 1e3
            v = np.nanpercentile(Z, [1, 99.5])
            im = axs[0, 0].imshow(Z, aspect="auto", origin="lower", cmap="plasma",
                                  extent=[dist.min(), dist.max(), lo, hi], vmin=v[0], vmax=v[1])
            fig.colorbar(im, ax=axs[0, 0], pad=0.02).set_label("dI/dV / (I/V) (a.u.)")
            axs[0, 0].set_title(f"experiment, line spectrum {tag}", fontsize=10)
            g = np.gradient(Z, bias_exp[sel_e], axis=0)
            vg = np.nanpercentile(np.abs(g), 98)
            im = axs[1, 0].imshow(g, aspect="auto", origin="lower", cmap="coolwarm",
                                  extent=[dist.min(), dist.max(), lo, hi], vmin=-vg, vmax=vg)
            fig.colorbar(im, ax=axs[1, 0], pad=0.02).set_label("d(dI/dV)/dV (a.u.)")
            col = 1
        Z = sim[sel_s]
        v = np.nanpercentile(Z, [1, 99.5])
        im = axs[0, col].imshow(Z, aspect="auto", origin="lower", cmap="plasma",
                                extent=[dist.min(), dist.max(), lo, hi], vmin=v[0], vmax=v[1])
        fig.colorbar(im, ax=axs[0, col], pad=0.02).set_label("simulated dI/dV, scaled (a.u.)")
        axs[0, col].set_title(f"model: tip-induced band bending, s at {len(s_vals)} values interpolated", fontsize=10)
        g = np.gradient(Z, fine[sel_s], axis=0)
        vg = np.nanpercentile(np.abs(g), 98)
        im = axs[1, col].imshow(g, aspect="auto", origin="lower", cmap="coolwarm",
                                extent=[dist.min(), dist.max(), lo, hi], vmin=-vg, vmax=vg)
        fig.colorbar(im, ax=axs[1, col], pad=0.02).set_label("d(dI/dV)/dV (a.u.)")
        for ax in axs.ravel():
            ax.axvline(fit["x0"], color="w", ls="--", lw=0.8)
            ax.set_xlabel("distance (nm)")
            ax.set_ylabel("sample bias (V)")
        fig.suptitle(f"Across the wall ({tag}): same tip and doping, s(x) from the fitted profile "
                     f"(x0 = {fit['x0']:.1f} nm, l = {fit['ell']:.1f} nm); s values {', '.join(f'{s:.2f}' for s in s_vals)}",
                     fontsize=9)
        fig.tight_layout()
        out = os.path.join(EXAMPLES, f"stacking_wall_tibb_map_{tag}.png")
        fig.savefig(out, dpi=170)
        plt.close(fig)
        print(f"[{tag}] wrote {out} ({len(s_vals)} s values)")


if __name__ == "__main__":
    main()
