"""Rescore every stored (+5) cell of line 940 (and 514 where cells exist) with the reduced targets of 2026-11-20:

  * positions: the lower side-band peak, the flat-band peak and the two upper side-band edges (features.features);
    the +69 mV onset of 940 is dropped (not a band edge; the step is seen on both sides of the wall);
  * the flat-band asymmetry only has to be present with the measured sign: mean tail height on the side away from
    E_F minus that on the side of E_F (40, 60, 80 mV from the peak, relative to the peak) above 0.05; its size is
    not fitted.

Sources: step1/ (clean model: neutral layers + SiO2 charge, CPot 0), polar/, fixeps/, fitcpot/ (global offset:
per-layer doping Delta E_F + SiO2 charge, eps = 2.6 except polar/, tip at zero band bending at zero bias plus
delta in fitcpot/).  Writes ../examples/stacking_wall_rescore.json and prints the best cells per model.

    python rescore.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import features  # noqa: E402

FINE = np.round(np.arange(-0.45, 0.45 + 1e-9, 0.0025), 5)
TAILS = (0.04, 0.06, 0.08)
ASYM_MIN = 0.05


def measured(tag):
    m = json.load(open(os.path.join(REPO, "examples", f"stacking_wall_tibb_{tag}.json")))
    V, y = np.asarray(m["measured"]["bias"]), np.asarray(m["measured"]["rg"], float)
    w = (V >= -0.45) & (V <= 0.45)
    return V[w], y[w]


def asym(V, y, E_fb):
    """Away-from-E_F minus towards-E_F mean tail height (relative to the flat-band peak)."""
    V, y = np.asarray(V), np.asarray(y)
    sel = np.abs(V - E_fb) < 0.04
    i0 = np.where(sel)[0][np.argmax(y[sel])]
    v0, pk = V[i0], y[i0]
    s = np.sign(v0) if v0 != 0 else 1.0
    inner = np.mean([np.interp(v0 - s * d, V, y) / pk for d in TAILS])
    outer = np.mean([np.interp(v0 + s * d, V, y) / pk for d in TAILS])
    return float(outer - inner)


def score(tag, y, fm):
    f = features(FINE, y)
    ok = np.isfinite(f)
    err = np.where(ok, f - fm, np.nan)
    a = asym(FINE, y, f[1])
    edges = err[[0, 2, 3]]
    return dict(features=f.tolist(), err=err.tolist(), fb_err=float(err[1]),
                edge_rms=float(np.sqrt(np.nanmean(edges ** 2))) if np.isfinite(edges).any() else 1.0,
                pos_rms=float(np.sqrt(np.nanmean(err ** 2))) if ok.any() else 1.0, missing=int((~ok).sum()),
                asym=a, asym_ok=bool(a > ASYM_MIN))


def rows():
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "step1", "*.json"))):
        if ".ckpt" in f:
            continue
        r = json.load(open(f))
        out.append(dict(model="clean", tag=r["tag"], eps=r["eps"], dEF=0.0, nsub=r["nsub"], cpot=r.get("cpot", 0.0),
                        tip=r["tip"]["didv"], free=r["free"]["didv"], U=r["U_minus_mu0"], src=os.path.basename(f)))
    for d in ("polar", "fixeps", "fitcpot"):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.json"))):
            if ".ckpt" in f:
                continue
            r = json.load(open(f))
            if "didv" not in r or "didv_free" not in r:
                continue
            U = (np.asarray(r["U_far"]) - r["mu0"]).tolist() if "U_far" in r else None
            out.append(dict(model="global", tag="940", eps=r.get("eps", 2.6), dEF=r["dEF"], nsub=r["nsub"],
                            cpot=r["cpot"], tip=r["didv"], free=r["didv_free"], U=U, src=f"{d}/{os.path.basename(f)}",
                            sigma=0.010))
    return out


if __name__ == "__main__":
    res = []
    fms = {}
    for tag in ("940", "514"):
        V, y = measured(tag)
        fm = features(V, y, smooth=9)
        fms[tag] = dict(features=fm.tolist(), asym=asym(V, y, fm[1]))
        print(f"== {tag} measured: lower {1e3 * fm[0]:+.0f}, FB {1e3 * fm[1]:+.0f}, up {1e3 * fm[2]:+.0f} / "
              f"{1e3 * fm[3]:+.0f} mV; asymmetry {fms[tag]['asym']:+.2f}")
    for r in rows():
        fm = np.asarray(fms[r["tag"]]["features"])
        r2 = {k: v for k, v in r.items() if k not in ("tip", "free")}
        for lab in ("tip", "free"):
            r2[lab] = score(r["tag"], np.asarray(r[lab]), fm)
        res.append(r2)
    json.dump(dict(measured=fms, asym_min=ASYM_MIN, rows=res), open(os.path.join(REPO, "examples", "stacking_wall_rescore.json"), "w"))

    def line(r):
        t, f = r["tip"], r["free"]
        return (f"{r['model']:6s} eps {r['eps']:4.1f} dEF {1e3 * r['dEF']:5.1f} n_sub {r['nsub'] / 1e13:+.2f} CPot {r['cpot']:+.3f} | "
                f"tip: edges rms {1e3 * t['edge_rms']:5.1f}, FB {1e3 * t['fb_err']:+5.1f}, asym {t['asym']:+.2f} "
                f"{'ok' if t['asym_ok'] else '--'}, feats {np.round(1e3 * np.asarray(t['features'])).astype(int).tolist()} | "
                f"free: edges {1e3 * f['edge_rms']:5.1f}, FB {1e3 * f['fb_err']:+5.1f}, asym {f['asym']:+.2f}  [{r['src']}]")

    for tag in ("940", "514"):
        for model in ("clean", "global"):
            rs = [r for r in res if r["tag"] == tag and r["model"] == model]
            if not rs:
                continue
            good = [r for r in rs if r["tip"]["asym_ok"] and abs(r["tip"]["fb_err"]) < 0.015 and not r["tip"]["missing"]]
            print(f"\n{tag} {model}: {len(rs)} cells, {len(good)} with the asymmetry present and FB within 15 mV (with the tip)")
            for r in sorted(good or rs, key=lambda r: r["tip"]["edge_rms"])[:6]:
                print("  " + line(r))
