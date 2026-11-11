#!/usr/bin/env python3
"""
Sensitivity of the tip-induced part of the wall bend to the tip work function.

A work-function offset delta of the tip from the value that leaves the (+5) bands flat at zero
bias (the fits give delta = 0, CPot = 0 in stacking_wall_tibb.py) enters the electrostatics only
through the tip potential U_tip = V + delta.  The tip-induced bending at bias V and offset delta is
therefore that of the fitted sweeps at bias V + delta.  For each line the k.p wall model is solved as
in stacking_wall_tibb_bump.py (far-field potentials of the hybrid plus a Gaussian bump of width
sqrt(2 R d), engine layer order reversed for the k.p model) with the tip

  * at the position of the measured wall spectrum, bump amplitude of the stack there at the measured
    wall-peak bias + delta, and
  * far on the (+5) side (+20 nm), bump amplitude at the measured (+5) flat-band bias + delta,

and once without the bump at each position.  The tip-induced part of the bend is the shift of the
peak centroid at the wall minus that on the (+5) side; the far-field part is the bend of the hybrid
map at the wall (the report's hybrid section).

    python stacking_wall_cpot_sensitivity.py --tag 940
    python stacking_wall_cpot_sensitivity.py --plot

Writes examples/stacking_wall_cpot_sensitivity_TAG.json and, with --plot,
examples/stacking_wall_cpot_sensitivity.png.
"""

import argparse
import json
import os
import time

import numpy as np

import stacking_wall as sw
from stacking_wall_tibb_bump import interp_in_s, load_series, peak

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
DELTAS = (-0.35, -0.2, -0.1, 0.0, 0.1, 0.2)          # eV; the engine sweeps cover |V + delta| <= 0.45 V
X_FAR = 20.0                                          # nm from the wall centre, (+5) side


def centroid(E, y, pk, half=0.03):
    """Centroid of the peak at pk within +-half, above the minimum of that window."""
    E, y = np.asarray(E), np.asarray(y)
    sel = (E >= pk - half) & (E <= pk + half)
    yb = y[sel] - y[sel].min()
    return float(np.sum(E[sel] * yb) / np.sum(yb))


def far_field_bend(tag, settings, fit):
    """Bend of the hybrid map at the measured wall position, its far (+5) peak aligned to the measured one (eV)."""
    E_fb, x_wall = settings["E_fb_measured"], settings["x_wall"]
    with np.load(os.path.join(EXAMPLES, f"stacking_wall_hybrid_{tag}.npz")) as f:
        E, x, top = f["energy"], f["x"] / 10.0 + fit["x0"], f["ldos"][-1]
    sel = (E >= -0.12) & (E <= 0.12)
    far = x > x.max() - 30.0
    E_far = float(E[sel][np.argmax(top[sel][:, far].mean(axis=1))])
    j = int(np.argmin(np.abs(x - x_wall)))
    return peak(E, top[:, j], E_far)[0] - E_far


def run(args):
    out_path = os.path.join(EXAMPLES, f"stacking_wall_cpot_sensitivity_{args.tag}.json")
    state = json.load(open(out_path)) if os.path.exists(out_path) else {}
    rec, settings, meas, fit = load_series(args.tag)
    E_fb, x_wall = settings["E_fb_measured"], settings["x_wall"]
    V_wall = peak(meas["bias"], meas["wall"], E_fb)[0]
    V_of_s = interp_in_s(rec, lambda r: r["U_far"] - r["mu0"])                   # engine order, top layer first
    L, ell = 10.0 * args.length, 10.0 * fit["ell"]
    w = np.sqrt(2.0 * 10.0 * args.radius * 10.0 * args.separation)            # A
    energy = np.arange(-args.e_max, args.e_max + 1e-9, 0.0025)
    positions = {"wall": (x_wall - fit["x0"], V_wall), "far": (X_FAR, E_fb)}
    state["settings"] = dict(E_fb=E_fb, V_wall=V_wall, x_wall=x_wall, x0=fit["x0"], ell=fit["ell"], x_far=X_FAR,
                             radius=args.radius, separation=args.separation, spot=w / 10.0, length=args.length,
                             dky=args.dky, p_cut=args.p_cut, sigma=args.sigma, deltas=list(DELTAS),
                             far_field_bend=far_field_bend(args.tag, settings, fit),
                             measured_bend=V_wall - E_fb)
    runs = state.setdefault("runs", {})
    for pos, (xt, V) in positions.items():
        x_tip = 10.0 * xt
        s_t = float(sw.wall_profile(x_tip, ell, L))
        for delta in (None,) + DELTAS:
            key = f"{pos}_none" if delta is None else f"{pos}_{delta:+.3f}"
            if key in runs:
                continue
            if delta is None:
                bump = np.zeros(6)
            else:
                Vt = V + delta
                bump = interp_in_s(rec, lambda r, Vt=Vt: np.array([np.interp(Vt, r["bias"], r["U0"][:, l])
                                                                    for l in range(6)]) - r["U_far"])(s_t)
            profile = lambda x: sw.wall_profile(x, ell, L)                                       # noqa: E731
            lpp = lambda x, bump=bump: (V_of_s(profile(x))                                       # noqa: E731
                                        + bump[:, None] * np.exp(-0.5 * ((np.asarray(x, dtype=float) - x_tip) / w) ** 2)[None, :])[::-1]
            t0 = time.time()
            res = sw.ldos_wall(wall_length=ell, length=L, p_cut=args.p_cut, ky_max=args.p_cut, dky=args.dky,
                               sigma=args.sigma, energy=energy, x_out=np.array([x_tip]), layer_potential_profile=lpp)
            y = res["ldos"][-1][:, 0]
            pk, h = peak(energy, y, E_fb)
            runs[key] = dict(pos=pos, delta=delta, x=xt, s=s_t, bias=V, bump=bump.tolist(), peak=pk, height=h,
                             centroid=centroid(energy, y, pk), energy=energy.tolist(), ldos=y.tolist(),
                             seconds=time.time() - t0)
            print(f"[{args.tag}] {pos} ({xt:+.1f} nm, s = {s_t:.2f}), delta {'none' if delta is None else f'{1e3 * delta:+.0f} meV'}: "
                  f"top-layer bump {1e3 * bump[0]:+6.1f} meV, peak {1e3 * pk:+6.1f}, centroid {1e3 * runs[key]['centroid']:+6.2f} meV, "
                  f"{time.time() - t0:.0f} s", flush=True)
            json.dump(state, open(out_path, "w"))
    state["summary"] = summary(state)
    json.dump(state, open(out_path, "w"))


def summary(state):
    """Per delta: top-layer bump at the wall and far, centroid shifts, tip-induced and total bend (eV)."""
    runs, s = state["runs"], state["settings"]
    out = []
    for d in s["deltas"]:
        a, b = runs.get(f"wall_{d:+.3f}"), runs.get(f"far_{d:+.3f}")
        if a is None or b is None or "wall_none" not in runs or "far_none" not in runs:
            continue
        dc_w = a["centroid"] - runs["wall_none"]["centroid"]
        dc_f = b["centroid"] - runs["far_none"]["centroid"]
        out.append(dict(delta=d, bump_wall=a["bump"][0], bump_far=b["bump"][0], dc_wall=dc_w, dc_far=dc_f,
                        tip=dc_w - dc_f, total=s["far_field_bend"] + dc_w - dc_f))
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, S1 = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#2a78d6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    tags = [t for t in ("514", "940") if os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_cpot_sensitivity_{t}.json"))]
    fig, axs = plt.subplots(1, len(tags), figsize=(6.2 * len(tags), 4.2), squeeze=False)
    for ax, t in zip(axs[0], tags):
        st = json.load(open(os.path.join(EXAMPLES, f"stacking_wall_cpot_sensitivity_{t}.json")))
        s, rows = st["settings"], st["summary"]
        d = 1e3 * np.array([r["delta"] for r in rows])
        ax.axvspan(-50, 50, color=GRID, lw=0, zorder=0)
        ax.text(0, 0.02, "fitted\n0 ± 50 meV", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INK2)
        ax.axhline(1e3 * s["measured_bend"], color=INK, lw=1.2, ls="--")
        ax.axhline(1e3 * s["far_field_bend"], color=MUTED, lw=1.0, ls=":")
        ax.axhline(0.0, color=GRID, lw=0.8)
        ax.plot(d, 1e3 * np.array([r["total"] for r in rows]), "o-", color=S1, lw=2, ms=6, mec="white", mew=1.2,
                label="model: far field + tip")
        xl = (d.min() - 20, d.max() + 20)
        ax.text(xl[1], 1e3 * s["measured_bend"], f"measured {1e3 * s['measured_bend']:+.0f} mV ", ha="right",
                va="bottom" if s["measured_bend"] > 0 else "top", fontsize=8, color=INK)
        ax.text(xl[1], 1e3 * s["far_field_bend"], f"far field only {1e3 * s['far_field_bend']:+.0f} mV ", ha="right",
                va="bottom" if s["far_field_bend"] > 0 else "top", fontsize=8, color=INK2)
        ax.set_xlim(*xl)
        ax.set_xlabel("tip work function minus sample's, δ (meV)")
        ax.set_ylabel("flat-band bend at the wall (mV)")
        ax.set_title(f"line {t}: wall peak at {1e3 * s['V_wall']:+.0f} mV, (+⁵) peak at {1e3 * s['E_fb']:+.0f} mV",
                     fontsize=10, loc="left", color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper left" if s["measured_bend"] < 0 else "lower left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(EXAMPLES, "stacking_wall_cpot_sensitivity.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--separation", type=float, default=0.7)
    ap.add_argument("--length", type=float, default=300.0, help="periodic cell (nm)")
    ap.add_argument("--dky", type=float, default=0.002)
    ap.add_argument("--p-cut", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.006)
    ap.add_argument("--e-max", type=float, default=0.35)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.tag is not None:
        run(args)
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
