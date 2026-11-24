"""HTML report of the ERS / STS gap comparison: ../report.html (standalone) and, with --artifact PATH, the same page
as a body fragment for publishing.  Reads only the stored results of this folder (no solves).

    python make_report.py [--artifact PATH]
"""
import html
import json
import os
import sys

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(W)
sys.path.insert(0, W)

INV_CM_TO_MEV = 1 / 8.0658163
ERS = {3: np.mean([5692.57, 5785.27, 5630.10, 5674.55]) * INV_CM_TO_MEV,
       4: np.mean([5044.64, 5032.03, 5004.45]) * INV_CM_TO_MEV, 6: 4008.35 * INV_CM_TO_MEV}
ERS_ERR = {3: 120 / 2 * INV_CM_TO_MEV, 4: 40 / np.sqrt(3) * INV_CM_TO_MEV, 6: 40 * INV_CM_TO_MEV}
ERS_FLAKES = {3: 4, 4: 3, 6: 1}
STM_PAPER = {3: (777.94, 30.50), 4: (657.56, 51.87)}
SAMPLES = [(3, "3L_647", "thin/gap_3L_647.json", "map 647"), (3, "3L_280", "thin/gap_3L_280.json", "map 280"),
           (4, "4L_209", "thin/gap_4L_209.json", "map 209"), (4, "4L_660", "thin/gap_4L_660.json", "map 660"),
           (6, "6L_940", "six/gap_940.json", "line 940"), (6, "6L_514", "six/gap_514.json", "line 514")]
FIT6 = {"6L_940": (-0.615, -0.5e13), "6L_514": (0.370, 0.375e13)}
# per-map S7 gaps of the STM maps (mV): archived notebook results (Zenodo "ers all flakes", splitting .nc files),
# the notebook settings rerun here (repro_s7.py), multi-start per pixel and map-averaged spectrum (pixel_s7.json)
S7_ARCHIVED = {"3L_647": 778.83, "3L_280": 755.79, "4L_209": 695.24, "4L_660": 615.98}
S7_RERUN = {"3L_647": 777.07, "3L_280": 766.87, "4L_209": 656.49, "4L_660": 659.89}

SERIES = [("tb", "SWMcC TB, no field", "square"), ("sub", "SiO₂ charge only", "circle"),
          ("tip", "tip only", "diamond"), ("tipsub", "tip + SiO₂ charge (fit)", "circle"),
          ("measured", "STM, S7 fit", "circle")]
CLS = {"tb": "s-tb", "sub": "s-sub", "tip": "s-tip", "tipsub": "s-tipsub", "measured": "s-data"}


def j(p):
    return json.load(open(os.path.join(W, p)))


def load():
    dg, gg = j("dos_gap.json"), j("six/geom_gaps.json")
    rows = []
    for N, key, f, lab in SAMPLES:
        d = j(f)
        g = {k: 1e3 * d[k]["gap"] for k in ("measured", "tb", "sub", "tip", "tipsub")}
        r = dict(N=N, key=key, label=lab, d=d, g=g, tibb=g["tipsub"] - g["sub"], excess=g["measured"] - ERS[N])
        m, t = d["measured"], d["tipsub"]
        if N < 6:
            ds = d["meta"]["d_sens"]
            r["tibb_d"] = (1e3 * ds["0.50"]["gap"] - g["sub"], 1e3 * ds["0.90"]["gap"] - g["sub"])
            r["cpot"], r["nsub"] = d["meta"]["cpot"], d["meta"]["nsub"]
            e = 1e3 * np.array([t["fb"] - m["fb"], t["x0_h"] - m["x0_h"], t["x0_e"] - m["x0_e"]])
            U0, b, mu0 = np.array(t["U0"]) - t["mu0"], np.array(t["sweep_bias"]), t["mu0"]
        else:
            ref = gg["d0.70_R020.0"]["gap"]
            r["tibb_d"] = tuple(r["tibb"] + gg[k]["gap"] - ref for k in ("d0.50_R020.0", "d0.90_R020.0"))
            r["cpot"], r["nsub"] = FIT6[key]
            e = 1e3 * np.array([t["features"][1] - m["features"][1], t["x0_h"] - m["x0_h"], t["x0_e"] - m["x0_e"]])
            s = t["sweep"]
            U0, b = np.array(s["U0"]) - s["mu0"], np.array(s["bias"])
        r["rms"] = float(np.sqrt(np.mean(e ** 2)))
        mean_at = [1e3 * np.mean([np.interp(x, b, U0[:, l]) for l in range(U0.shape[1])]) for x in (t["x0_h"], t["x0_e"])]
        r["apex_h"], r["apex_e"] = mean_at
        r["dos_sub"] = 1e3 * dg[f"{key}_sub"]["gap"]
        rows.append(r)
    return rows


def esc(s):
    return html.escape(str(s), quote=True)


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    t0 = np.ceil(lo / step) * step
    return [round(v, 6) for v in np.arange(t0, hi + 1e-9, step)]


def marker(shape, x, y, cls, tip, size=5.5):
    t = f'<title>{esc(tip)}</title>'
    if shape == "square":
        s = size * 0.9
        body = f'<rect x="{x - s:.1f}" y="{y - s:.1f}" width="{2 * s:.1f}" height="{2 * s:.1f}" class="m-open {cls}"/>'
    elif shape == "diamond":
        s = size * 1.15
        body = (f'<path d="M{x:.1f} {y - s:.1f}L{x + s:.1f} {y:.1f}L{x:.1f} {y + s:.1f}L{x - s:.1f} {y:.1f}Z" '
                f'class="m-fill {cls}"/>')
    elif shape == "osquare":
        s = size * 0.9
        body = f'<rect x="{x - s:.1f}" y="{y - s:.1f}" width="{2 * s:.1f}" height="{2 * s:.1f}" class="m-open s-data"/>'
    else:
        body = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" class="m-fill {cls}"/>'
    return f'<g class="pt" tabindex="0" data-tip="{esc(tip)}">{t}{body}</g>'


def gap_panel(N, rows):
    rs = [r for r in rows if r["N"] == N]
    Wd, Ht, L, R, T, B = 380, 300, 54, 70, 16, 44
    pw, ph = Wd - L - R, Ht - T - B
    vals = [v for r in rs for v in r["g"].values()] + [ERS[N] - ERS_ERR[N], ERS[N] + ERS_ERR[N]]
    if N in STM_PAPER:
        vals.append(STM_PAPER[N][0])
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    Y = lambda v: T + ph * (hi - v) / (hi - lo)   # noqa: E731
    out = [f'<svg viewBox="0 0 {Wd} {Ht}" role="img" aria-label="{N}-layer gaps by sample and model case">']
    for tk in nice_ticks(lo, hi):
        y = Y(tk)
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{L - 6}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">{tk:g}</text>')
    y1, y2 = Y(ERS[N] + ERS_ERR[N]), Y(ERS[N] - ERS_ERR[N])
    out.append(f'<rect x="{L}" y="{y1:.1f}" width="{pw}" height="{y2 - y1:.1f}" class="band"/>'
               f'<line x1="{L}" x2="{L + pw}" y1="{Y(ERS[N]):.1f}" y2="{Y(ERS[N]):.1f}" class="ref"/>'
               f'<text x="{L + pw + 6}" y="{Y(ERS[N]) + 4:.1f}" class="lbl">ERS {ERS[N]:.0f}</text>')
    if N in STM_PAPER:
        yp = Y(STM_PAPER[N][0])
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{yp:.1f}" y2="{yp:.1f}" class="ref dash"/>'
                   f'<text x="{L + pw + 6}" y="{yp - 3:.1f}" class="lbl">STM paper</text>'
                   f'<text x="{L + pw + 6}" y="{yp + 9:.1f}" class="lbl">{STM_PAPER[N][0]:.0f}</text>')
    for i, r in enumerate(rs):
        xc = L + pw * (0.25 + 0.5 * i)
        out.append(f'<text x="{xc:.1f}" y="{T + ph + 20}" class="cat" text-anchor="middle">{esc(r["label"])}</text>')
        for k, (key, name, shape) in enumerate(SERIES):
            x = xc + (k - 2) * 15
            v = r["g"][key]
            out.append(marker(shape, x, Y(v), CLS[key], f"{N}L {r['label']} · {name}: {v:.1f} meV"))
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{T + ph}" class="axis"/>'
               f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" class="axis"/>'
               f'<text x="14" y="{T + ph / 2:.0f}" class="axt" text-anchor="middle" '
               f'transform="rotate(-90 14 {T + ph / 2:.0f})">gap (meV)</text></svg>')
    return "".join(out)


def increase_chart(rows):
    Wd, Ht, L, R, T, B = 760, 340, 58, 16, 16, 58
    pw, ph = Wd - L - R, Ht - T - B
    hi = max(max(r["tibb_d"]) for r in rows) + 10
    lo = min(0, min(r["excess"] - ERS_ERR[r["N"]] for r in rows)) - 4
    Y = lambda v: T + ph * (hi - v) / (hi - lo)   # noqa: E731
    out = [f'<svg viewBox="0 0 {Wd} {Ht}" role="img" aria-label="Tip-induced gap increase against STM minus ERS">']
    for tk in nice_ticks(lo, hi, 6):
        y = Y(tk)
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid{" zero" if tk == 0 else ""}"/>'
                   f'<text x="{L - 6}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">{tk:g}</text>')
    groups = [3, 4, 6]
    gw = pw / len(groups)
    for gi, N in enumerate(groups):
        rs = [r for r in rows if r["N"] == N]
        gx = L + gw * (gi + 0.5)
        out.append(f'<text x="{gx:.1f}" y="{T + ph + 40}" class="cat strong" text-anchor="middle">{N} layers</text>')
        if gi:
            out.append(f'<line x1="{L + gw * gi:.1f}" x2="{L + gw * gi:.1f}" y1="{T}" y2="{T + ph}" class="sep"/>')
        if N in STM_PAPER:
            v = STM_PAPER[N][0] - ERS[N]
            out.append(marker("osquare", gx, Y(v), "s-data", f"{N}L · STM (paper, all pixels) − ERS: {v:+.1f} meV"))
        for i, r in enumerate(rs):
            x0 = gx + (i - 0.5) * gw * 0.5
            a, b = sorted(r["tibb_d"])
            out.append(f'<g class="pt" tabindex="0" data-tip="{esc(f"{N}L {r["label"]} · model, tip distance 0.5 to 0.9 nm: {a:+.0f} to {b:+.0f} meV")}">'
                       f'<rect x="{x0 - 16:.1f}" y="{Y(b):.1f}" width="12" height="{Y(a) - Y(b):.1f}" rx="2" class="range"/></g>')
            out.append(marker("circle", x0 - 10, Y(r["tibb"]), "s-tipsub",
                              f"{N}L {r['label']} · model TIBB at d = 0.7 nm: {r['tibb']:+.1f} meV"))
            e = ERS_ERR[N]
            out.append(f'<line x1="{x0 + 10:.1f}" x2="{x0 + 10:.1f}" y1="{Y(r["excess"] + e):.1f}" '
                       f'y2="{Y(r["excess"] - e):.1f}" class="err"/>')
            out.append(marker("circle", x0 + 10, Y(r["excess"]), "s-data",
                              f"{N}L {r['label']} · STM (S7 fit) − ERS: {r['excess']:+.1f} ± {e:.1f} meV"))
            out.append(f'<text x="{x0:.1f}" y="{T + ph + 18}" class="cat" text-anchor="middle">{esc(r["label"])}</text>')
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{T + ph}" class="axis"/>'
               f'<text x="14" y="{T + ph / 2:.0f}" class="axt" text-anchor="middle" '
               f'transform="rotate(-90 14 {T + ph / 2:.0f})">gap increase (meV)</text></svg>')
    return "".join(out)


def thin(V, y, n=420):
    V, y = np.asarray(V, float), np.asarray(y, float)
    if V.size > n:
        idx = np.unique(np.round(np.linspace(0, V.size - 1, n)).astype(int))
        V, y = V[idx], y[idx]
    return V, y


def spectra_panel(r):
    d = r["d"]
    Wd, Ht, L, R, T, B = 380, 250, 14, 12, 12, 36
    pw, ph = Wd - L - R, Ht - T - B
    lim = (-0.6, 0.7) if r["N"] < 6 else (-0.45, 0.45)
    X = lambda v: L + pw * (v - lim[0]) / (lim[1] - lim[0])   # noqa: E731
    curves = []
    for k, cls, off in (("measured", "s-data", 0.0), ("tipsub", "s-tipsub", 0.0), ("sub", "s-sub", -0.95)):
        V, y = np.asarray(d[k]["bias"]), np.asarray(d[k]["didv"])
        s = (V >= lim[0]) & (V <= lim[1])
        yn = y / np.percentile(y[s], 98) + off
        curves.append((k, cls, V[s], yn[s], V, y / np.percentile(y[s], 98) + off))
    ymin = min(c[3].min() for c in curves) - 0.05
    ymax = max(c[3].max() for c in curves) + 0.05
    Y = lambda v: T + ph * (ymax - v) / (ymax - ymin)   # noqa: E731
    out = [f'<svg viewBox="0 0 {Wd} {Ht}" role="img" aria-label="{r["N"]}-layer {esc(r["label"])} spectra">']
    for tk in np.arange(np.ceil(lim[0] / 0.2) * 0.2, lim[1] + 1e-9, 0.2):
        x = X(tk)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{T + ph + 15}" class="tick" text-anchor="middle">{tk:+.1f}</text>')
    for k, cls, V, yn, Vf, yf in curves:
        Vt, yt = thin(V, yn)
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(Vt, yt))
        out.append(f'<polyline points="{pts}" class="curve {cls}"/>')
        for xk, side in ((d[k]["x0_h"], "x0(−1)"), (d[k]["x0_e"], "x0(+1)")):
            yv = np.interp(xk, Vf, yf)
            out.append(f'<g class="pt" tabindex="0" data-tip="{esc(f"{side}: {1e3 * xk:+.0f} mV")}">'
                       f'<line x1="{X(xk):.1f}" x2="{X(xk):.1f}" y1="{Y(yv) - 9:.1f}" y2="{Y(yv) + 9:.1f}" '
                       f'class="x0 {cls}"/></g>')
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" class="axis"/>'
               f'<text x="{L + pw / 2:.0f}" y="{Ht - 3}" class="axt" text-anchor="middle">sample bias (V)</text></svg>')
    g = r["g"]
    cap = (f'<figcaption><b>{r["N"]}L, {esc(r["label"])}</b> <span>STM {g["measured"]:.0f}</span> '
           f'<span>fit {g["tipsub"]:.0f}</span> <span>SiO₂ only {g["sub"]:.0f} meV</span></figcaption>')
    return f'<figure class="spec">{cap}{"".join(out)}</figure>'


def f0(v, sign=False):
    return f"{v:+.0f}" if sign else f"{v:.0f}"


def build(rows):
    by = {N: [r for r in rows if r["N"] == N] for N in (3, 4, 6)}

    def rng(vals, sign=True):
        a, b = min(vals), max(vals)
        return f0(a, sign) if round(a) == round(b) else f"{f0(a, sign)} to {f0(b, sign)}"

    sub_max = max(abs(r["g"]["sub"] - r["g"]["tb"]) for r in rows)
    strip = []
    for N in (3, 4, 6):
        ex = [r["excess"] for r in by[N]] + ([STM_PAPER[N][0] - ERS[N]] if N in STM_PAPER else [])
        strip.append(f'<div class="cell"><div class="eyebrow">{N} layers</div>'
                     f'<div class="big">{rng([r["tibb"] for r in by[N]])}<small> meV</small></div>'
                     f'<div class="sub">added by the tip (model)</div>'
                     f'<div class="sub2">STM − ERS measured: <b>{rng(ex)} meV</b></div></div>')
    strip.append(f'<div class="cell quiet"><div class="eyebrow">SiO₂ charge alone</div>'
                 f'<div class="big">≤ {np.ceil(sub_max):.0f}<small> meV</small></div>'
                 f'<div class="sub">change of the gap without the tip</div>'
                 f'<div class="sub2">all six samples</div></div>')

    legend = "".join(
        f'<span class="key"><svg viewBox="-8 -8 16 16" width="16" height="16" aria-hidden="true">'
        + {"square": '<rect x="-5" y="-5" width="10" height="10" class="m-open {c}"/>',
           "diamond": '<path d="M0 -6.3L6.3 0L0 6.3L-6.3 0Z" class="m-fill {c}"/>',
           "circle": '<circle r="5.5" class="m-fill {c}"/>'}[shape].format(c=CLS[key])
        + f'</svg>{esc(name)}</span>' for key, name, shape in SERIES)
    legend += ('<span class="key"><svg viewBox="0 0 22 10" width="22" height="10" aria-hidden="true"><line x1="0" x2="22" '
               'y1="5" y2="5" class="ref"/></svg>ERS, mean of flakes ± error</span>'
               '<span class="key"><svg viewBox="0 0 22 10" width="22" height="10" aria-hidden="true"><line x1="0" x2="22" '
               'y1="5" y2="5" class="ref dash"/></svg>STM, paper Fig. 4d</span>')
    legend2 = ('<span class="key"><svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><rect x="5" y="1" width="6" '
               'height="14" rx="2" class="range"/><circle cx="8" cy="8" r="5" class="m-fill s-tipsub"/></svg>model: tip + SiO₂ '
               'minus SiO₂ only (dot d = 0.7 nm, bar 0.5 to 0.9 nm)</span>'
               '<span class="key"><svg viewBox="-8 -8 16 16" width="16" height="16" aria-hidden="true"><circle r="5.5" '
               'class="m-fill s-data"/></svg>data: STM (S7 fit) minus ERS</span>'
               '<span class="key"><svg viewBox="-8 -8 16 16" width="16" height="16" aria-hidden="true"><rect x="-5" y="-5" '
               'width="10" height="10" class="m-open s-data"/></svg>data: STM (paper, all pixels) minus ERS</span>')

    trs = []
    for r in rows:
        g = r["g"]
        a, b = r["tibb_d"]
        trs.append(f'<tr><th scope="row">{r["N"]}L {esc(r["label"])}</th><td>{g["measured"]:.1f}</td><td>{ERS[r["N"]]:.1f}</td>'
                   f'<td>{g["tb"]:.1f}</td><td>{g["sub"]:.1f}</td><td>{g["tip"]:.1f}</td><td>{g["tipsub"]:.1f}</td>'
                   f'<td class="hl">{r["tibb"]:+.1f}</td><td>{a:+.0f} / {b:+.0f}{"*" if r["N"] == 6 else ""}</td>'
                   f'<td>{g["tipsub"] - r["dos_sub"]:+.1f}</td><td class="hl2">{r["excess"]:+.1f}</td></tr>')
    fits = []
    for r in rows:
        fits.append(f'<tr><th scope="row">{r["N"]}L {esc(r["label"])}</th><td>{r["cpot"]:+.3f}</td><td>{r["nsub"] / 1e13:+.3f}</td>'
                    f'<td>{r["rms"]:.1f}</td><td>{r["g"]["tipsub"] - r["g"]["measured"]:+.1f}</td>'
                    f'<td>{"this folder: 5 × 3 grid, interpolated, verified" if r["N"] < 6 else "kandrai2026_rg_tip_gating"}</td></tr>')
    apex = []
    for r in rows:
        apex.append(f'<tr><th scope="row">{r["N"]}L {esc(r["label"])}</th><td>{r["apex_h"]:+.0f}</td><td>{r["apex_e"]:+.0f}</td>'
                    f'<td>{r["apex_e"] - r["apex_h"]:+.0f}</td><td>{r["tibb"]:+.0f}</td></tr>')
    # CPot dependence at n_sub = 0 (grid sweeps)
    sg = j("thin/sub_grid.json")
    cp_rows = []
    for c in (-0.3, -0.15, 0.0, 0.15, 0.3):
        cells = []
        for N, name in ((3, "3L_647"), (4, "4L_209")):
            p = os.path.join(W, "thin", f"sweep_N{N}_c{1e3 * c:+07.1f}_n+0.000.json")
            sw = json.load(open(p))
            f = sw["spectra"][name]["features"]
            gap = 1e3 * (f[2] - f[1])
            cells.append(f'<td>{1e3 * f[0]:+.0f}</td><td>{gap:.0f}</td><td class="hl">{gap - 1e3 * sg[f"N{N}_n+0.00"]["gap"]:+.0f}</td>')
        cp_rows.append(f'<tr><th scope="row">{c:+.2f}</th>{"".join(cells)}</tr>')
    gg = j("six/geom_gaps.json")
    ref6 = gg["d0.70_R020.0"]["gap"]
    tb6 = by[6][0]["g"]["tb"]
    geo = [f'<tr><th scope="row">d = {d:.1f} nm, R = 20 nm</th><td>{gg[f"d{d:.2f}_R020.0"]["gap"]:.0f}</td>'
           f'<td>{gg[f"d{d:.2f}_R020.0"]["gap"] - tb6:+.0f}</td></tr>' for d in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    geo += [f'<tr><th scope="row">d = 0.7 nm, R = {R:.0f} nm</th><td>{gg[f"d0.70_R{R:05.1f}"]["gap"]:.0f}</td>'
            f'<td>{gg[f"d0.70_R{R:05.1f}"]["gap"] - tb6:+.0f}</td></tr>' for R in (5, 10, 40, 80)]
    px = j("pixel_s7.json")
    s7rows = []
    for k, lab in (("3L_647", "3L map 647"), ("3L_280", "3L map 280"), ("4L_209", "4L map 209"), ("4L_660", "4L map 660")):
        s7rows.append(f'<tr><th scope="row">{lab}</th><td>{S7_ARCHIVED[k]:.0f}</td><td>{S7_RERUN[k]:.0f}</td>'
                      f'<td>{px[k]["mean"]:.0f} ± {px[k]["std"]:.0f}</td><td>{px[k]["avg_gap"]:.0f}</td>'
                      f'<td>{[r for r in rows if r["key"] == k][0]["g"]["measured"]:.0f}</td></tr>')

    dmin = min(min(r["tibb_d"]) for r in rows if r["N"] == 3)
    dmax = max(max(r["tibb_d"]) for r in rows if r["N"] == 3)

    head = f"""<title>ERS–STS Gap Comparison</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Tip-induced band bending and the STM versus ERS gap of 3-, 4- and 6-layer rhombohedral graphite.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans+Condensed:wght@500;600&display=swap">
<style>{CSS}</style>"""

    body = f"""<div class="page">
<header class="top">
  <div class="eyebrow">Rhombohedral graphite · −1 → +1 band-edge gap · STM at 9 K vs ERS at 300 K</div>
  <h1>ERS–STS Gap Comparison</h1>
  <p class="lede">STM measures the −1 → +1 gap of rhombohedral graphite larger than electronic Raman scattering
  (Pálinkás et al., <i>Carbon</i> 230, 119608, 2024: +72 meV for 3 layers, +34 meV for 4). Self-consistent
  simulations of the tip, fitted to the measured spectra, show that tip-induced band bending (TIBB) widens the STM gap by
  the same amount and with the same thickness dependence.</p>
</header>

<section class="strip" aria-label="Main result">{"".join(strip)}</section>

<section class="verdict">
  <h2>Answer</h2>
  <p>The excess of the STM gap over the ERS gap is tip-induced band bending. With the tip contact potential and the SiO₂
  trap charge fitted to each spectrum, the tip adds {rng([r["tibb"] for r in by[3]])} meV to the 3-layer gap,
  {rng([r["tibb"] for r in by[4]])} meV to the 4-layer gap and {rng([r["tibb"] for r in by[6]])} meV to the 6-layer gap.
  The SiO₂ charge alone moves the gap by at most {np.ceil(sub_max):.0f} meV. Model and measured excess agree within
  ±25 meV with no systematic sign; the one outlier is 3L map 280 (model {by[3][1]["tibb"]:+.0f}, measured
  {by[3][1]["excess"]:+.0f} meV). The paper assigned the 70 ± 27 meV trilayer excess to excitonic and polarization
  effects in ERS. These data leave room for at most about 20 meV of such corrections.</p>
</section>

<section>
  <h2>Gaps by sample</h2>
  <p class="note">S7 gap x0(+1) − x0(−1) of each spectrum. Model cases left to right: tight binding without field,
  SiO₂ charge only, tip only (fitted contact potential, no SiO₂ charge), tip and SiO₂ charge (the fit), then the
  measured STM spectrum. Hover or focus a marker for its value.</p>
  <div class="legend">{legend}</div>
  <div class="panels">
    <figure>{gap_panel(3, rows)}<figcaption>3 layers (ABC), STM maps on flake FLG_26</figcaption></figure>
    <figure>{gap_panel(4, rows)}<figcaption>4 layers (ABCA), STM maps on flake FLG_33</figcaption></figure>
    <figure>{gap_panel(6, rows)}<figcaption>6 layers, (+5) side of the stacking wall, lines 940 and 514</figcaption></figure>
  </div>
</section>

<section>
  <h2>Tip-induced increase against the measured excess</h2>
  <p class="note">Orange: gap with tip and SiO₂ charge minus gap with SiO₂ charge only, at tip–sample distance
  d = 0.7 nm; the bar spans d = 0.5 to 0.9 nm. Black: measured STM gap minus ERS gap of the same thickness; the error bar is
  the ERS error of the mean.</p>
  <div class="legend">{legend2}</div>
  <figure class="wide">{increase_chart(rows)}</figure>
</section>

<section>
  <h2>Numbers</h2>
  <div class="tablewrap"><table>
    <caption>S7 gaps in meV. TIBB = (tip + SiO₂) − (SiO₂ only). “vs DOS peaks”: tip + SiO₂ minus the
    −1/+1 total-DOS peak separation with the SiO₂ charge and no tip, the model analog of STM − ERS.
    * 6L distance range transferred from the line-940 geometry sweep.</caption>
    <thead><tr><th scope="col">sample</th><th scope="col">STM</th><th scope="col">ERS</th><th scope="col">TB</th>
    <th scope="col">SiO₂ only</th><th scope="col">tip only</th><th scope="col">tip + SiO₂</th><th scope="col">TIBB</th>
    <th scope="col">TIBB, d 0.5 / 0.9 nm</th><th scope="col">vs DOS peaks</th><th scope="col">STM − ERS</th></tr></thead>
    <tbody>{"".join(trs)}</tbody>
  </table></div>
  <p class="note">ERS: {ERS[3]:.1f} ± {ERS_ERR[3]:.1f} (3L, {ERS_FLAKES[3]} flakes), {ERS[4]:.1f} ± {ERS_ERR[4]:.1f}
  (4L, {ERS_FLAKES[4]} flakes), {ERS[6]:.1f} ± {ERS_ERR[6]:.1f} meV (6L, 1 flake). Paper STM, all pixels of both maps:
  {STM_PAPER[3][0]:.1f} (3L) and {STM_PAPER[4][0]:.1f} meV (4L), giving STM − ERS of {STM_PAPER[3][0] - ERS[3]:+.1f} and
  {STM_PAPER[4][0] - ERS[4]:+.1f} meV.</p>
</section>

<section>
  <h2>Spectra and fitted band edges</h2>
  <p class="note">Measured dI/dV (black), the fitted tip + SiO₂ model (orange) and the SiO₂-only model (blue, offset
  down). Ticks mark the S7 edge positions x0(−1) and x0(+1). Spectra normalised to their 98th percentile. The models
  carry no inhomogeneous broadening, so their peaks are sharper than the data.</p>
  <div class="specs">{"".join(spectra_panel(r) for r in rows)}</div>
</section>

<section class="prose">
  <h2>Method</h2>
  <h3>Gap definition</h3>
  <p>Supplementary S7 of the paper: around each band edge the spectrum is fitted with two linear backgrounds, a sigmoid
  step and a Gaussian, all sharing the centre <var>E</var><sub>0</sub>,</p>
  <p class="eq"><var>y</var>(<var>E</var>) = <var>y</var><sub>0</sub> + <var>s</var><sub>≶</sub>(<var>E</var> − <var>E</var><sub>0</sub>)
  + <var>A</var><sub>sigm</sub> / [exp((<var>E</var> − <var>E</var><sub>0</sub>)/<var>sp</var>) + 1]
  + <var>A</var><sub>G</sub> exp[−2 ln 2 (<var>E</var> − <var>E</var><sub>0</sub>)²/<var>w</var>²]</p>
  <p>and the gap is <var>E</var><sub>0</sub>(+1) − <var>E</var><sub>0</sub>(−1). The same code fits measured and
  simulated spectra, multi-start over <var>E</var><sub>0</sub>. Windows: 3L −0.6 to −0.15 and 0.2 to 0.7 V, 4L
  −0.6 to −0.2 and 0.2 to 0.7 V, as in the Zenodo notebook. In 6L the ±2 edges lie about 100 mV beyond ±1,
  so each window is anchored on the spectrum's −1 peak (−50 to +90 mV around it) and +1 rise (−70 to +60 mV).
  For broadened spectra <var>E</var><sub>0</sub> lands on the steep flank, 10 to 25 meV inside the DOS peak; the offset is
  common to data and model.</p>
  <h3>Model</h3>
  <p>Self-consistent Hartree plus full SWMcC tight binding (γ<sub>0</sub> to γ<sub>5</sub>, Δ′ of
  McEllistrim et al. 2023) in the bandbending engine, branch <code>lateral-solver</code>, set up as in
  kandrai2026_rg_tip_gating: neutral stack, interlayer ε<sub>z</sub> = 2.6, a 2 nm SiO₂ trap layer with fixed charge
  <var>n</var><sub>sub</sub> on 88 nm SiO₂, grounded Si gate, tip radius 20 nm at d = 0.7 nm with contact potential
  CPot, T = 9 K, WKB barrier 4.6 eV. Gaussian broadening 10 meV (3L, 4L) and 12.5 / 13.7 meV (6L), lock-in modulation of
  each map (2.5 to 5.5 mV). Cases: tight binding without field; SiO₂ charge only (tip-free solve); tip only (fitted
  CPot, <var>n</var><sub>sub</sub> = 0); tip and SiO₂ charge (the fit).</p>
  <h3>Fits</h3>
  <p>3L and 4L: flat-band peak, x0(−1) and x0(+1) of the map-averaged spectrum, on a grid of CPot ∈
  {{−0.3, −0.15, 0, 0.15, 0.3}} eV × <var>n</var><sub>sub</sub> ∈ {{−0.5, 0, 0.5}}·10¹³
  cm⁻², interpolated, then checked with a full sweep at the optimum. 6L: the published (+5) fits of
  kandrai2026_rg_tip_gating. The (CPot, <var>n</var><sub>sub</sub>) valley is degenerate, but the model gap changes by less
  than 10 meV along it.</p>
  <div class="tablewrap"><table class="narrow">
    <thead><tr><th scope="col">sample</th><th scope="col">CPot (eV)</th><th scope="col"><var>n</var><sub>sub</sub> (10¹³ cm⁻²)</th>
    <th scope="col">rms of the targets (mV)</th><th scope="col">model − STM gap (meV)</th><th scope="col">source</th></tr></thead>
    <tbody>{"".join(fits)}</tbody>
  </table></div>
</section>

<section class="prose">
  <h2>Why the tip widens the gap</h2>
  <p>At the edge biases, |V| ≈ 0.3 to 0.4 V, only the flat band can screen the tip field, and it is emptied or filled
  within tens of mV. The stack then follows the tip potential with a lever arm of about 0.1 for 3 layers, falling with
  thickness. The added gap equals the change of the layer-averaged potential under the apex between the two edge biases:</p>
  <div class="tablewrap"><table class="narrow">
    <thead><tr><th scope="col">sample</th><th scope="col">⟨U⟩ at x0(−1) (meV)</th><th scope="col">⟨U⟩ at x0(+1) (meV)</th>
    <th scope="col">difference</th><th scope="col">TIBB</th></tr></thead>
    <tbody>{"".join(apex)}</tbody>
  </table></div>
  <p>The two edges lie on opposite sides of the zero-TIBB bias, so the added gap depends only weakly on the contact
  potential. Without SiO₂ charge, a CPot sweep moves the flat band by ±50 mV but the TIBB by only about 20 meV:</p>
  <div class="tablewrap"><table class="narrow">
    <thead><tr><th scope="col" rowspan="2">CPot (eV)</th><th scope="col" colspan="3">3L (map 647 settings)</th><th scope="col" colspan="3">4L (map 209 settings)</th></tr>
    <tr><th scope="col">flat band (mV)</th><th scope="col">gap</th><th scope="col">TIBB</th><th scope="col">flat band (mV)</th><th scope="col">gap</th><th scope="col">TIBB</th></tr></thead>
    <tbody>{"".join(cp_rows)}</tbody>
  </table></div>
</section>

<section class="prose">
  <h2>Uncertainties</h2>
  <h3>Tip–sample distance</h3>
  <p>The distance is fixed at 0.7 nm, the convention of kandrai2026_rg_tip_gating, and it is the largest model uncertainty.
  Over d = 0.5 to 0.9 nm the 3-layer TIBB spans {dmin:+.0f} to {dmax:+.0f} meV. The tip radius matters little
  (6L, line 940 geometry, CPot −0.367 eV, no SiO₂ charge; TB gap {tb6:.0f} meV):</p>
  <div class="tablewrap"><table class="narrow">
    <thead><tr><th scope="col">geometry</th><th scope="col">gap (meV)</th><th scope="col">minus TB</th></tr></thead>
    <tbody>{"".join(geo)}</tbody>
  </table></div>
  <h3>S7 fit of the measured maps</h3>
  <p>The STM gap of one map depends on how the S7 fit is run, by up to 40 meV for the 4-layer maps. The model is compared
  with the map-averaged spectrum fitted by the same code (last column). The 695 mV of the paper's Fig. S7d histogram does
  not reproduce with the archived notebook settings.</p>
  <div class="tablewrap"><table class="narrow">
    <thead><tr><th scope="col">map</th><th scope="col">archived notebook</th><th scope="col">notebook settings, rerun</th>
    <th scope="col">multi-start, per pixel</th><th scope="col">rolling-averaged map spectrum</th><th scope="col">map-averaged spectrum (used)</th></tr></thead>
    <tbody>{"".join(s7rows)}</tbody>
  </table></div>
  <h3>Other limits</h3>
  <ul>
    <li>3L map 280 is not reproduced. Its gap ({by[3][1]["g"]["measured"]:.0f} meV) is below every tip model at d = 0.7 nm;
    it would need d ≈ 1 nm or a differently shaped tip. 6L line 514: model {by[6][1]["g"]["tipsub"] - by[6][1]["g"]["measured"]:+.0f} meV
    above the data.</li>
    <li>ERS is measured at room temperature on other flakes. The tight-binding DOS-peak gaps exceed ERS by
    {by[3][0]["dos_sub"] - ERS[3]:+.0f} (3L) and {by[4][0]["dos_sub"] - ERS[4]:+.0f} meV (4L) and fall
    {ERS[6] - by[6][0]["dos_sub"]:.0f} meV short of it for 6L. Corrections smaller than this, excitonic or other, cannot be
    resolved.</li>
    <li>The flat-band peak of 3L is a peak plus a sharp edge in the top-layer LDOS, a fragile fit target; it dominates the
    verified rms but does not affect the gap.</li>
  </ul>
</section>

<footer class="prose src">
  <h2>Sources</h2>
  <ul>
    <li>STM maps, ERS fits and the S7 notebook: Zenodo 10.5281/zenodo.10931524 (Pálinkás et al. 2024).</li>
    <li>6-layer spectra and fits: bandbending, branch <code>lateral-solver</code>, <code>examples/kandrai2026_rg_tip_gating</code>.</li>
    <li>This analysis: few-layer-graphite, branch <code>claude/happy-heisenberg-msa1kb</code>,
    <code>examples/palinkas2024_tibb_gap</code> (scripts, cached sweeps, figures; README lists the commands).</li>
  </ul>
</footer>
</div>
<div id="tip" class="tooltip" role="status" hidden></div>
<script>{JS}</script>"""
    return head, body


CSS = r"""
/* Layout: one reading column (68ch) with full-width figure rows; instrument-panel restraint, data in mono. */
:root{
  --bg:#f5f6f7; --surface:#ffffff; --ink:#15181c; --muted:#5b626b; --rule:#d9dde2; --grid:#e8ebee;
  --accent:#c4501b; --band:rgba(21,24,28,.07);
  --s-tb:#7a8089; --s-sub:#2a78d6; --s-tip:#1baf7a; --s-tipsub:#eb6834; --s-data:#15181c;
  --f-text:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-head:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#111316; --surface:#191c20; --ink:#e9ecef; --muted:#a2a9b1; --rule:#2d3237; --grid:#24282d;
  --accent:#e8763f; --band:rgba(233,236,239,.08);
  --s-tb:#8f959d; --s-sub:#3987e5; --s-tip:#199e70; --s-tipsub:#d95926; --s-data:#e9ecef; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#111316; --surface:#191c20; --ink:#e9ecef; --muted:#a2a9b1; --rule:#2d3237; --grid:#24282d;
  --accent:#e8763f; --band:rgba(233,236,239,.08);
  --s-tb:#8f959d; --s-sub:#3987e5; --s-tip:#199e70; --s-tipsub:#d95926; --s-data:#e9ecef; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--f-text);font-size:15px;line-height:1.55;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.page{max-width:1120px;margin:0 auto;padding-block:40px 64px;display:flex;flex-direction:column;gap:44px}
h1,h2,h3{font-family:var(--f-head);text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(30px,5vw,46px);font-weight:600;letter-spacing:-.01em}
h2{font-size:22px;font-weight:600;margin-bottom:10px}
h3{font-size:16px;font-weight:600;margin:18px 0 6px}
p{margin:0 0 10px}
.eyebrow{font-family:var(--f-mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.top{display:flex;flex-direction:column;gap:12px;max-width:72ch;min-width:0}
.lede{font-size:17px;color:var(--ink);margin:0}
.strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.strip .cell{background:var(--surface);padding:16px 18px;display:flex;flex-direction:column;gap:4px;min-width:0}
.strip .big{font-family:var(--f-mono);font-size:28px;font-weight:500;color:var(--accent);font-variant-numeric:tabular-nums;line-height:1.1}
.strip .quiet .big{color:var(--ink)}
.strip .big small{font-size:14px;color:var(--muted)}
.strip .sub{font-size:13px;color:var(--muted)}
.strip .sub2{font-size:13px;margin-top:6px;padding-top:8px;border-top:1px solid var(--rule)}
.verdict p,.prose p,.prose ul{max-width:72ch}
.prose ul{padding-left:20px;margin:0 0 10px}.prose li{margin-bottom:6px}
.note{color:var(--muted);font-size:14px;max-width:80ch}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;margin:8px 0 12px}
.key{display:inline-flex;align-items:center;gap:7px;min-width:0}.key svg{flex:none}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
figure{margin:0;min-width:0;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:10px 10px 12px}
figure svg{display:block;width:100%;height:auto}
figcaption{font-size:13px;color:var(--muted);padding:6px 4px 0}
.specs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.spec figcaption{padding:0 4px 4px;display:flex;flex-wrap:wrap;gap:4px 12px;font-family:var(--f-mono);font-size:12px}
.spec figcaption b{font-family:var(--f-text);font-size:13px;color:var(--ink);font-weight:600}
svg text{font-family:var(--f-mono);fill:var(--muted)}
svg .tick{font-size:10.5px} svg .lbl{font-size:10.5px;fill:var(--ink)} svg .cat{font-size:11.5px;fill:var(--ink)}
svg .cat.strong{font-family:var(--f-text);font-weight:600;font-size:13px} svg .axt{font-size:11px}
svg .grid{stroke:var(--grid);stroke-width:1} svg .grid.zero{stroke:var(--muted);stroke-width:1}
svg .axis{stroke:var(--muted);stroke-width:1} svg .sep{stroke:var(--rule);stroke-width:1}
svg .band{fill:var(--band)} svg .ref{stroke:var(--ink);stroke-width:1.5;fill:none} svg .ref.dash{stroke-dasharray:6 4}
svg .m-fill{stroke:var(--surface);stroke-width:2} svg .m-open{fill:var(--surface);stroke-width:2}
svg .m-fill.s-tb{fill:var(--s-tb)} svg .m-open.s-tb{stroke:var(--s-tb)}
svg .m-fill.s-sub{fill:var(--s-sub)} svg .m-fill.s-tip{fill:var(--s-tip)} svg .m-fill.s-tipsub{fill:var(--s-tipsub)}
svg .m-fill.s-data{fill:var(--s-data)} svg .m-open.s-data{stroke:var(--s-data)}
svg .range{fill:var(--s-tipsub);opacity:.28}
svg .err{stroke:var(--s-data);stroke-width:1.5}
svg .curve{fill:none;stroke-width:1.6;stroke-linejoin:round}
svg .curve.s-data{stroke:var(--s-data)} svg .curve.s-tipsub{stroke:var(--s-tipsub)} svg .curve.s-sub{stroke:var(--s-sub)}
svg .x0{stroke-width:2.5} svg .x0.s-data{stroke:var(--s-data)} svg .x0.s-tipsub{stroke:var(--s-tipsub)} svg .x0.s-sub{stroke:var(--s-sub)}
svg .pt{cursor:default;outline:none} svg .pt:focus-visible circle,svg .pt:focus-visible rect,svg .pt:focus-visible path{stroke:var(--accent);stroke-width:3}
.wide{max-width:100%}
.tablewrap{overflow-x:auto;min-width:0;border:1px solid var(--rule);border-radius:6px;background:var(--surface);margin:6px 0 12px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
table.narrow{width:auto;min-width:60%}
caption{caption-side:bottom;text-align:left;font-size:12.5px;color:var(--muted);padding:10px 12px}
th,td{padding:7px 12px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
td{font-family:var(--f-mono);font-size:13px}
thead th{font-weight:600;font-size:12px;color:var(--muted);text-align:right;border-bottom:1px solid var(--rule)}
th[scope="row"],thead th:first-child{text-align:left;font-weight:500;color:var(--ink)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
td.hl{color:var(--accent);font-weight:500} td.hl2{font-weight:500}
.eq{font-family:var(--f-mono);font-size:13.5px;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;overflow-x:auto;max-width:none}
.eq var,.prose var{font-family:var(--f-text);font-style:italic}
code{overflow-wrap:anywhere;font-family:var(--f-mono);font-size:.9em;background:var(--surface);border:1px solid var(--rule);border-radius:4px;padding:0 4px}
.src{border-top:1px solid var(--rule);padding-top:20px;font-size:14px;color:var(--muted)}
.tooltip{position:fixed;z-index:20;pointer-events:none;background:var(--ink);color:var(--bg);font-family:var(--f-mono);font-size:12px;
  padding:5px 8px;border-radius:4px;max-width:min(320px,80vw)}
@media (max-width:900px){.strip{grid-template-columns:repeat(2,minmax(0,1fr))}.panels{grid-template-columns:minmax(0,1fr)}
  .specs{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:560px){.strip{grid-template-columns:minmax(0,1fr)}.specs{grid-template-columns:minmax(0,1fr)}
  .page{padding-block:24px 48px;gap:34px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = r"""
(function(){
  var tip=document.getElementById('tip');
  function show(el,x,y){tip.textContent=el.getAttribute('data-tip');tip.hidden=false;
    var w=tip.offsetWidth,h=tip.offsetHeight,vw=window.innerWidth;
    var left=Math.min(Math.max(8,x+12),vw-w-8),top=y-h-12; if(top<8){top=y+16;}
    tip.style.left=left+'px';tip.style.top=top+'px';}
  document.addEventListener('pointermove',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
    if(el){show(el,e.clientX,e.clientY);}else{tip.hidden=true;}});
  document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
    if(el){var r=el.getBoundingClientRect();show(el,r.left+r.width/2,r.top);}});
  document.addEventListener('focusout',function(){tip.hidden=true;});
  document.addEventListener('scroll',function(){tip.hidden=true;},{passive:true});
})();
"""


if __name__ == "__main__":
    rows = load()
    head, body = build(rows)
    std = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    out = os.path.join(ROOT, "report.html")
    open(out, "w").write(std)
    print("wrote", out, f"{len(std) / 1024:.0f} kB")
    if "--artifact" in sys.argv:
        p = sys.argv[sys.argv.index("--artifact") + 1]
        open(p, "w").write(head + "\n" + body + "\n")
        print("wrote", p)
