#!/usr/bin/env python3
"""
Top-surface LDOS across a (+4-) | (+5) stacking domain wall of six-layer
graphite, set against a measured dI/dV line spectrum.

    python stacking_wall_figure.py --sm4 LINE_SPECTRUM.sm4 --tag 514 --x-sf 22 --x-rg 55

Steps
1. ``stacking_wall.load_line_spectrum`` reads and normalises the line spectrum;
   ``mixing_fraction`` expresses every spectrum as (1 - s) S_SF + s S_RG with
   the end-member spectra averaged at dist < x_sf and dist > x_rg (nm), and
   ``fit_wall_profile`` fits s(x) with [1 + tanh((x - x0) / l)] / 2.
2. ``stacking_wall.ldos_wall`` computes the layer-resolved LDOS of the model
   with the fitted wall length l (cached in examples/stacking_wall_ldos_TAG.npz).
3. Figures: examples/stacking_wall_TAG.png (dI/dV and top-layer LDOS maps,
   their bias derivatives, the profile, spectra at three positions) and
   examples/stacking_wall_layers_TAG.png (all six layers).  The fit and the
   model parameters are written to examples/stacking_wall_TAG.json.

The model energy axis is shifted rigidly so that its flat-band peak on the
(+5) side coincides with the measured one (the sample is slightly doped);
the model x axis is shifted so that the wall centre sits at the fitted x0.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import stacking_wall as sw   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sm4", required=True, help="RHK line spectrum across one wall, (+4-) on the left")
    p.add_argument("--tag", default=None, help="label of the output files (default: last number of the file name)")
    p.add_argument("--x-sf", type=float, required=True, help="positions below this (nm) are pure (+4-)")
    p.add_argument("--x-rg", type=float, required=True, help="positions above this (nm) are pure (+5)")
    p.add_argument("--bias-window", type=float, nargs=2, default=(-0.3, 0.3), help="bias range (V) of the mixing fit")
    p.add_argument("--ell", type=float, default=None, help="wall length l (nm) instead of the fitted value")
    p.add_argument("--length", type=float, default=480.0, help="period of the plane-wave cell (nm)")
    p.add_argument("--p-cut", type=float, default=0.15, help="plane-wave cut-off (1/A)")
    p.add_argument("--ky-max", type=float, default=0.15, help="k_y range (1/A)")
    p.add_argument("--dky", type=float, default=0.001, help="k_y step (1/A)")
    p.add_argument("--sigma", type=float, default=0.005, help="Gaussian broadening (eV)")
    p.add_argument("--e-max", type=float, default=0.5, help="half-width of the energy window (eV)")
    p.add_argument("--de", type=float, default=0.0025, help="energy step (eV)")
    p.add_argument("--dx", type=float, default=0.5, help="position step of the LDOS (nm)")
    p.add_argument("--x-half", type=float, default=120.0, help="half-width of the computed position window (nm)")
    p.add_argument("--n-proc", type=int, default=None)
    p.add_argument("--recompute", action="store_true", help="ignore the cached LDOS")
    p.add_argument("--plot-bias", type=float, nargs=2, default=(-0.45, 0.45), help="bias range (V) of the maps")
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or os.path.splitext(os.path.basename(args.sm4))[0].split("_")[-1]
    os.makedirs(EXAMPLES, exist_ok=True)

    # ---- experiment ---------------------------------------------------------
    exp = sw.load_line_spectrum(args.sm4)
    bias, dist, didv = exp["bias"], exp["dist"], exp["didv"]
    s_exp, S_sf, S_rg = sw.mixing_fraction(bias, dist, didv, args.x_sf, args.x_rg, tuple(args.bias_window))
    fit = sw.fit_wall_profile(dist, s_exp, x0_guess=0.5 * (args.x_sf + args.x_rg))
    E_fb_exp = sw.flat_band_energy(bias, S_rg)
    ell_nm = fit["ell"] if args.ell is None else args.ell
    print(f"[{tag}] wall: x0 = {fit['x0']:.2f} +- {fit['x0_err']:.2f} nm, "
          f"l = {fit['ell']:.2f} +- {fit['ell_err']:.2f} nm (10-90 % width {fit['width_10_90']:.1f} nm); "
          f"flat band of the (+5) side at {E_fb_exp * 1e3:+.0f} mV; using l = {ell_nm:.2f} nm")

    # ---- model --------------------------------------------------------------
    cache = os.path.join(EXAMPLES, f"stacking_wall_ldos_{tag}.npz")
    if os.path.exists(cache) and not args.recompute:
        res = sw.load_ldos(cache)
        print(f"[{tag}] loaded {cache} (l = {res['wall_length'] / 10:.2f} nm)")
    else:
        energy = np.arange(-args.e_max, args.e_max + 1e-9, args.de)
        x_out = np.arange(-args.x_half, args.x_half + 1e-9, args.dx) * 10.0
        t0 = time.time()
        res = sw.ldos_wall(wall_length=10.0 * ell_nm, length=10.0 * args.length, p_cut=args.p_cut,
                           ky_max=args.ky_max, dky=args.dky, energy=energy, sigma=args.sigma,
                           x_out=x_out, n_proc=args.n_proc)
        print(f"[{tag}] LDOS computed in {time.time() - t0:.0f} s ({res['n_pw']} plane waves)")
        sw.save_ldos(cache, res)

    E_m, x_m, ldos = res["energy"], res["x"] / 10.0, res["ldos"]
    top = ldos[-1]
    far = x_m > 0.5 * x_m.max()
    E_fb_model = sw.flat_band_energy(E_m, top[:, far].mean(axis=1))
    dE = E_fb_exp - E_fb_model
    print(f"[{tag}] model flat band at {E_fb_model * 1e3:+.1f} meV; energy axis shifted by {dE * 1e3:+.1f} meV")

    with open(os.path.join(EXAMPLES, f"stacking_wall_{tag}.json"), "w") as fh:
        json.dump({
            "sm4": os.path.basename(args.sm4), "tag": tag, "fit": fit,
            "flat_band_experiment_V": E_fb_exp, "flat_band_model_eV": E_fb_model,
            "energy_shift_eV": dE, "wall_length_used_nm": ell_nm,
            "model": {k: res[k] for k in ("length", "p_cut", "ky_max", "dky", "sigma", "n_pw", "seq_rg", "seq_sf")},
            "parameters": res["params"],
        }, fh, indent=2, default=sw._json_default)

    # ---- figures ------------------------------------------------------------
    make_figure(tag, exp, s_exp, fit, res, dE, args)
    make_layer_figure(tag, res, dE, fit, args)


def _extent(x, y):
    return [x.min(), x.max(), y.min(), y.max()]


def _derivative(z, e):
    return np.gradient(z, e, axis=0)


def make_figure(tag, exp, s_exp, fit, res, dE, args):
    bias, dist, didv = exp["bias"], exp["dist"], exp["didv"]
    lo, hi = args.plot_bias
    E_m, x_m, top = res["energy"] + dE, res["x"] / 10.0 + fit["x0"], res["ldos"][-1]
    sel_b = (bias >= lo) & (bias <= hi)
    sel_e = (E_m >= lo) & (E_m <= hi)
    sel_x = (x_m >= dist.min()) & (x_m <= dist.max())
    exp_map = didv[sel_b]
    mod_map = top[np.ix_(sel_e, sel_x)]
    x_mod = x_m[sel_x]

    fig, axs = plt.subplots(3, 2, figsize=(11, 11.5), gridspec_kw={"height_ratios": [1.0, 1.0, 0.8]})
    cm = "plasma"

    v = np.nanpercentile(exp_map, [1, 99.5])
    im = axs[0, 0].imshow(exp_map, aspect="auto", origin="lower", cmap=cm, extent=_extent(dist, bias[sel_b]),
                          vmin=v[0], vmax=v[1])
    fig.colorbar(im, ax=axs[0, 0], pad=0.02).set_label("dI/dV / (I/V) (a.u.)")
    axs[0, 0].set_title(f"experiment, line spectrum {tag}", fontsize=10)
    axs[0, 0].set_ylabel("sample bias (V)")

    v = np.nanpercentile(mod_map, [1, 99.5])
    im = axs[0, 1].imshow(mod_map * 1e3, aspect="auto", origin="lower", cmap=cm, extent=_extent(x_mod, E_m[sel_e]),
                          vmin=v[0] * 1e3, vmax=v[1] * 1e3)
    fig.colorbar(im, ax=axs[0, 1], pad=0.02).set_label("LDOS, top layer (10$^{-3}$ eV$^{-1}$ Å$^{-2}$)")
    axs[0, 1].set_title(f"SWMcC model, l = {res['wall_length'] / 10:.1f} nm, σ = {res['sigma'] * 1e3:.0f} meV", fontsize=10)
    axs[0, 1].set_ylabel("energy (eV)")

    g_exp = _derivative(exp_map, bias[sel_b])
    vg = np.nanpercentile(np.abs(g_exp), 98)
    im = axs[1, 0].imshow(g_exp, aspect="auto", origin="lower", cmap="coolwarm", extent=_extent(dist, bias[sel_b]),
                          vmin=-vg, vmax=vg)
    fig.colorbar(im, ax=axs[1, 0], pad=0.02).set_label("d(dI/dV)/dV (a.u.)")
    axs[1, 0].set_ylabel("sample bias (V)")

    g_mod = _derivative(mod_map, E_m[sel_e])
    vg = np.nanpercentile(np.abs(g_mod), 98)
    im = axs[1, 1].imshow(g_mod, aspect="auto", origin="lower", cmap="coolwarm", extent=_extent(x_mod, E_m[sel_e]),
                          vmin=-vg, vmax=vg)
    fig.colorbar(im, ax=axs[1, 1], pad=0.02).set_label("d(LDOS)/dE (a.u.)")
    axs[1, 1].set_ylabel("energy (eV)")

    for ax in axs[:2].ravel():
        ax.axvline(fit["x0"], color="w", ls="--", lw=0.8)
        ax.set_xlim(dist.min(), dist.max())
        ax.set_xlabel("distance (nm)")

    # profile
    ax = axs[2, 0]
    ax.plot(dist, s_exp, ".", ms=3, color="#52514e", label="experiment: (+5) fraction of the spectra")
    ax.plot(dist, sw.tanh_step(dist, fit["x0"], fit["ell"], fit["lo"], fit["hi"]), "-", color="#d95926",
            label=f"tanh fit: x0 = {fit['x0']:.1f} nm, l = {fit['ell']:.1f} nm")
    ax.plot(x_m, 1.0 - res["s"], "--", color="#1f6fb2", lw=1.2, label="model: 1 - s(x)")
    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("distance (nm)")
    ax.set_ylabel("(+5) weight")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # spectra at three positions
    ax = axs[2, 1]
    positions = [fit["x0"] - 2.5 * fit["ell"], fit["x0"], fit["x0"] + 2.5 * fit["ell"]]
    colors = ["#1f6fb2", "#52514e", "#d95926"]
    for pos, col in zip(positions, colors):
        i_e = np.argmin(np.abs(dist - pos))
        i_m = np.argmin(np.abs(x_m - pos))
        y_e = didv[sel_b, i_e]
        y_m = top[sel_e, i_m]
        ax.plot(bias[sel_b], y_e / np.nanmax(didv[sel_b]), color=col, lw=1.2, label=f"exp. {dist[i_e]:.0f} nm")
        ax.plot(E_m[sel_e], y_m / np.nanmax(mod_map), color=col, lw=1.0, ls="--", label=f"model {x_m[i_m]:.0f} nm")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("sample bias / energy (V, eV)")
    ax.set_ylabel("normalised dI/dV, LDOS")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle(f"(+4−) | (+5) stacking wall, six-layer graphite — line spectrum {tag}", fontsize=11)
    fig.tight_layout()
    out = os.path.join(EXAMPLES, f"stacking_wall_{tag}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[{tag}] wrote {out}")


def make_layer_figure(tag, res, dE, fit, args):
    lo, hi = args.plot_bias
    E_m, x_m, ldos = res["energy"] + dE, res["x"] / 10.0 + fit["x0"], res["ldos"]
    sel_e = (E_m >= lo) & (E_m <= hi)
    n_layers = ldos.shape[0]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for n, ax in enumerate(axs.ravel()[:n_layers]):
        layer = n_layers - n            # paper numbering: layer 1 is the top layer
        z = _derivative(ldos[layer - 1][sel_e], E_m[sel_e])
        vg = np.nanpercentile(np.abs(z), 98)
        im = ax.imshow(z, aspect="auto", origin="lower", cmap="coolwarm", extent=_extent(x_m, E_m[sel_e]),
                       vmin=-vg, vmax=vg)
        ax.axvline(fit["x0"], color="k", ls="--", lw=0.6)
        ax.set_title(f"layer {n + 1}" + (" (top, STM)" if n == 0 else ""), fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("distance (nm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("energy (eV)")
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02).set_label("d(LDOS)/dE (a.u.)")
    fig.suptitle(f"layer-resolved d(LDOS)/dE across the wall, l = {res['wall_length'] / 10:.1f} nm ({tag})", fontsize=10)
    out = os.path.join(EXAMPLES, f"stacking_wall_layers_{tag}.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[{tag}] wrote {out}")


if __name__ == "__main__":
    main()
