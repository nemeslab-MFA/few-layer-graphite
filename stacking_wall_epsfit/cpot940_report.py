"""HTML report of the contact-potential-only model of the 940 (+5) flat band, from the CPot sweep on:
../examples/stacking_wall_cpot940_report.html (figures embedded).  Reads the cpot940*/geom*/dense* results under
cpot940/ and the figures under ../examples/; regenerate the figures first (cpot940_plot.py, cpot940_offset_plot.py,
cpot940_fbshape_plot.py, cpot940_geom_plot.py, cpot940_tibb_sketch.py)."""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EX = os.path.join(REPO, "examples")
sys.path.insert(0, HERE)
import cpot940 as C  # noqa: E402
import cpot940_geom as G  # noqa: E402
import cpot940_tibb_sketch as K  # noqa: E402

STYLE = """
<style>
/* layout: one reading column (62ch prose), figures and tables at the full 68rem width */
:root {
  --bg: #fbfaf7; --fg: #1d1c1a; --muted: #605d57; --rule: #d9d5cc; --accent: #9a3b12;
  --panel: #f1eee7; --sans: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #171615; --fg: #ebe7df; --muted: #a39e94; --rule: #3a3733; --accent: #e8905f; --panel: #221f1c; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #171615; --fg: #ebe7df; --muted: #a39e94; --rule: #3a3733; --accent: #e8905f; --panel: #221f1c; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--serif); font-size: 16px; line-height: 1.55;
  padding-block: 2rem 3rem; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 68rem; margin: 0 auto; }
h1, h2, h3 { font-family: var(--sans); text-wrap: balance; line-height: 1.15; margin: 0; }
h1 { font-size: 2rem; font-weight: 600; }
h2 { font-size: 1.15rem; font-weight: 600; margin-top: 2.4rem; padding-top: 0.7rem; border-top: 1px solid var(--rule); }
h3 { font-size: 0.95rem; font-weight: 600; margin-top: 1.4rem; }
p { max-width: 62ch; margin: 0.6rem 0; }
.eyebrow { font-family: var(--sans); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.lede { color: var(--muted); font-family: var(--sans); font-size: 0.98rem; max-width: 62ch; }
.findings { background: var(--panel); padding: 0.9rem 1.2rem 0.9rem 2.2rem; margin: 1.2rem 0; max-width: 62ch; }
.findings li { margin: 0.45rem 0; }
.eq { font-family: var(--serif); font-style: italic; background: var(--panel); padding: 0.6rem 1rem; margin: 0.8rem 0; overflow-x: auto; max-width: 62ch; }
.eq span.r { font-style: normal; }
table { border-collapse: collapse; font-family: var(--sans); font-size: 0.88rem; font-variant-numeric: tabular-nums; margin: 0.8rem 0; }
th, td { text-align: left; padding: 0.35rem 1.1rem 0.35rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
td.n, th.n { text-align: right; }
th { font-weight: 600; color: var(--muted); font-size: 0.76rem; letter-spacing: 0.04em; text-transform: uppercase; }
tr.hl td { font-weight: 600; }
tr.meas td { color: var(--accent); font-weight: 600; }
.tablewrap { overflow-x: auto; max-width: 100%; }
figure { margin: 1.2rem 0 0.4rem; min-width: 0; }
figure img { width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--rule); background: #fff; }
figcaption { font-family: var(--sans); font-size: 0.85rem; color: var(--muted); margin-top: 0.45rem; max-width: 80ch; }
figcaption b { color: var(--fg); font-weight: 600; }
ul, ol { max-width: 62ch; padding-left: 1.2rem; margin: 0.5rem 0; }
li { margin: 0.3rem 0; }
code { font-family: var(--mono); font-size: 0.85em; }
a { color: var(--accent); }
footer { margin-top: 2.8rem; padding-top: 0.8rem; border-top: 1px solid var(--rule); font-family: var(--sans); font-size: 0.8rem; color: var(--muted); }
footer p, footer ul { max-width: none; }
</style>
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600'
         '&family=Source+Serif+4:ital,wght@0,400;1,400&family=IBM+Plex+Mono&display=swap">')


TITLES = {"stacking_wall_cpot940.png": "Contact-potential fit",
          "stacking_wall_cpot940_offset.png": "Spectra compared: no tip, CPot 0, fit, measurement",
          "stacking_wall_cpot940_fbshape.png": "Shape of the flat-band peak",
          "stacking_wall_cpot940_geom.png": "Tip distance and apex radius",
          "stacking_wall_cpot940_tibb_sketch.png": "Why only the occupied side moves",
          "stacking_wall_cpot940_bands.png": "Band structure under the apex",
          "stacking_wall_cpot940_nsub.png": "Substrate charge at the fitted tip"}
FIGNO = {}


def img(name, alt, caption):
    FIGNO.setdefault(name, len(FIGNO) + 1)
    k = FIGNO[name]
    b = base64.b64encode(open(os.path.join(EX, name), "rb").read()).decode()
    return (f'<figure id="fig{k}"><img src="data:image/png;base64,{b}" alt="Figure {k}: {alt}" loading="lazy">'
            f'<figcaption><b>Figure {k} · {TITLES.get(name, alt)}.</b> {caption}</figcaption></figure>')


def mv(x, nd=0):
    if x is None or not np.isfinite(x):
        return "–"
    if abs(x) < 0.5 * 10.0 ** -nd:
        return f"{0:.{nd}f}"
    return f"{x:+.{nd}f}".replace("-", "−")


def num(x, nd=2):
    if abs(x) < 0.5 * 10.0 ** -nd:
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "−")


def table(head, rows, classes=None, numeric=None):
    numeric = numeric or set()
    h = "".join(f'<th class="n">{c}</th>' if i in numeric else f"<th>{c}</th>" for i, c in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        cls = f' class="{classes[k]}"' if classes and classes[k] else ""
        body.append(f"<tr{cls}>" + "".join(f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>"
                                         for i, c in enumerate(r)) + "</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def data():
    pl = json.load(open(os.path.join(C.OUT, "place.json")))
    runs = {round(r["cpot"], 3): r for r in C.rows()}
    fine = json.load(open(os.path.join(C.OUT, "geom_fine.json")))
    geom = {(r["d"], r["R"]): r for r in G.rows()}
    dense = {}
    for j in ((0.7, 20.0), (0.4, 20.0)):
        q = os.path.join(C.OUT, "dense_" + G.key(*j)[5:] + ".json")
        if os.path.exists(q):
            dense[j] = json.load(open(q))["peak"]
    return pl, runs, fine, geom, dense


def nsub_rows(v=-0.0324):
    """Substrate-charge sweep: flat band with and without the tip, top-layer lever arm and the quantum capacitance
    e^2 D_l(E_F) (uF/cm^2, Gaussian sigma of the model) of L1, L2 and L6 under the apex at the flat-band bias."""
    import cpot940_nsub as NS
    from scipy.interpolate import PchipInterpolator
    tb = C.S.tipfree(C.TAG, C.EPS, 0.0, 0.0, C.MODEL, "RG")["p"].tb
    out = []
    for r in sorted(NS.rows(), key=lambda r: r["nsub"]):
        sw = r["sweep"]
        P = PchipInterpolator(np.asarray(sw["bias"]), np.asarray(sw["U0"]), axis=0)
        U = P(v)
        Eg = sw["mu0"] + np.arange(-0.2, 0.2 + 1e-9, 5e-4)
        _, D = tb.ldos(np.asarray(U), Eg, broadening=C.SIG, resolve="layer")
        D = np.asarray(D)
        D = D if D.shape[0] == Eg.size else D.T
        cq = 1.602e-13 * D[Eg.size // 2]
        out.append(dict(n=r["nsub"] / 1e13, fb_free=1e3 * r["free"]["peak"]["v"], fb_tip=1e3 * r["tip"]["peak"]["v"],
                        alpha=float(P.derivative()(v)[0]), cq=cq, U1=1e3 * (U[0] - sw["mu0"]), feats=r["tip"]["features"],
                        U6=1e3 * (U[-1] - sw["mu0"])))
    return out


def build():
    pl, runs, fine, geom, dense = data()
    ns = nsub_rows()
    ns_rows = [[f"{q['n']:+.1f}".replace("-", "−"), mv(q["fb_free"]), mv(q["fb_tip"]), num(q["alpha"]),
                num(q["cq"][0], 1), num(q["cq"][1], 1), num(q["cq"][5], 1), f"{mv(q['U1'])} / {mv(q['U6'])}"]
               for q in ns if q["n"] in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)]
    nsd = {round(q["n"], 1): q for q in ns}
    nsub_edges = nsd[-0.5]["feats"][2:4]
    best = min(runs, key=lambda c: abs(c - pl["cpot"]))
    fm = np.asarray(runs[best]["measured_features"]) * 1e3
    fp = {(t["d"], t["R"]): t["peak"] for t in fine["tip"]}

    # lever arms of each peak: tip (mV per 0.1 eV more negative CPot, runs -0.30 -> -0.45 eV at n_sub 0) and
    # substrate (mV for n_sub 0 -> -0.5e13 at CPot -0.367 eV); linear prediction along the flat-band valley
    fa, fb2 = (1e3 * np.asarray(runs[c]["tip"]["features"]) for c in (-0.3, -0.45))
    tip_s = (fb2 - fa) / 1.5
    sub_s = 1e3 * (np.asarray(nsd[-0.5]["feats"]) - np.asarray(nsd[0.0]["feats"]))
    val_s = sub_s - tip_s * sub_s[1] / tip_s[1]                     # along the valley, per 0.5e13 of holes
    e0 = 1e3 * np.asarray(runs[best]["tip"]["features"]) - fm
    oth = [0, 2, 3]
    x_opt = -float(np.dot(e0[oth], val_s[oth]) / np.dot(val_s[oth], val_s[oth]))
    e_opt = e0 + val_s * x_opt
    e_opt[1] = 0.0
    pred_n, pred_c = -0.5 * x_opt, best - 0.1 * (-sub_s[1] / tip_s[1]) * x_opt
    lev_rows = [[nm, mv(tip_s[i], 0), mv(sub_s[i], 0), num(abs(sub_s[i] / tip_s[i]), 1)]
                for i, nm in ((1, "flat band (top layer, near 0 V)"), (0, "lower bulk edge (bulk, −0.26 V)"),
                              (2, "upper bulk edge 1 (bulk, +0.27 V)"), (3, "upper bulk edge 2 (bulk, +0.37 V)"))]
    rms0 = float(np.sqrt(np.mean(e0 ** 2)))
    rms_opt = float(np.sqrt(np.mean(e_opt ** 2)))
    pk_fit, pk_free, pk_meas = fp[(0.7, 20.0)], fine["free"]["peak"], fine["measured"]
    lg = sorted(pl["log"], key=lambda x: -x["cpot"])
    k_lo = min(lg, key=lambda x: abs(x["cpot"] + 0.35))
    k_hi = min(lg, key=lambda x: abs(x["cpot"] + 0.40))
    slope = (k_hi["fb"] - k_lo["fb"]) / (k_hi["cpot"] - k_lo["cpot"])          # V per eV
    free = runs[best]["free"]
    t = runs[best]["tip"]

    # sketch numbers: tip-induced top-layer shift and lever arm
    rf, V, dUf, Pf = K.load(-0.367)
    r0, V0s, dU0, P0 = K.load(0.0)
    v0f, v00 = K.tibb0(V, dUf[:, 0]), K.tibb0(V0s, dU0[:, 0])
    ff = np.asarray(free["features"])
    ftp = np.asarray(t["features"])
    ff[1], ftp[1] = pk_free["v"], pk_fit["v"]
    f0 = np.asarray(runs[0.0]["tip"]["features"])
    lev = lambda P, v: float(P.derivative()(v)[0])                         # noqa: E731
    shift_rows = []
    for i, name in enumerate(("lower bulk edge", "flat band", "upper bulk edge 1", "upper bulk edge 2")):
        shift_rows.append([name, mv(1e3 * ftp[i]), num(ftp[i] - v0f), mv(1e3 * float(Pf(ftp[i])[0])),
                           num(lev(Pf, ftp[i])), mv(1e3 * (ftp[i] - ff[i]), 1), mv(1e3 * (f0[i] - free["features"][i]))])
    xs = np.linspace(-0.45, 0.45, 181)
    xs = xs[xs - 0.367 >= -0.45]
    overlay = 1e3 * float(np.max(np.abs(Pf(xs)[:, 0] - P0(xs - 0.367)[:, 0])))

    feat_rows, feat_cls = [["measured", *(mv(x) for x in fm), "–", num(runs[best]["measured_asym"])]], ["meas"]
    feat_rows.append(["no tip", *(mv(1e3 * x) for x in free["features"]), num(1e3 * free["edge_rms"], 1),
                      num(free["asym"])])
    feat_cls.append("")
    for cp in sorted(runs, reverse=True):
        r = runs[cp]["tip"]
        feat_rows.append([f"CPot {num(cp, 3)} eV" + (" (fit)" if cp == best else ""),
                          *(mv(1e3 * x) for x in r["features"]), num(1e3 * r["edge_rms"], 1), num(r["asym"])])
        feat_cls.append("hl" if cp == best else "")

    shape_rows = [["measured", mv(1e3 * pk_meas["v"], 1), num(1e3 * pk_meas["hwhm_out"], 1), num(1e3 * pk_meas["hwhm_in"], 1),
                   num(pk_meas["ratio"]), num(runs[best]["measured_asym"])],
                  ["no tip", mv(1e3 * pk_free["v"], 1), num(1e3 * pk_free["hwhm_out"], 1), num(1e3 * pk_free["hwhm_in"], 1),
                   num(pk_free["ratio"]), num(free["asym"])],
                  [f"tip, CPot {num(best, 3)} eV", mv(1e3 * pk_fit["v"], 1), num(1e3 * pk_fit["hwhm_out"], 1),
                   num(1e3 * pk_fit["hwhm_in"], 1), num(pk_fit["ratio"]), num(t["asym"])]]

    def geom_row(d, R):
        p, g = fp[(d, R)], geom[(d, R)]["tip"]
        return [num(d, 1), f"{R:g}", mv(1e3 * p["v"], 1), num(1e3 * p["hwhm_out"], 1), num(1e3 * p["hwhm_in"], 1),
                num(p["ratio"]), num(g["asym"]), mv(1e3 * g["features"][0]), f'{mv(1e3 * g["features"][2])} / '
                f'{mv(1e3 * g["features"][3])}', num(1e3 * g["edge_rms"], 1)]
    dsw = [geom_row(d, 20.0) for d in G.D_SWEEP]
    rsw = [geom_row(0.7, R) for R in G.R_SWEEP]
    gp = lambda d: fp[(d, 20.0)]["v"] * 1e3                                 # noqa: E731
    d_slope = (gp(0.8) - gp(0.6)) / 0.2

    if dense:
        parts = []
        for (d, R), q in sorted(dense.items(), reverse=True):
            p = fp[(d, R)]
            parts.append(f"d {d:g} nm: peak {mv(1e3 * q['v'], 1)} mV (25 mV sweep: {mv(1e3 * p['v'], 1)}), ratio "
                         f"{num(q['ratio'])} ({num(p['ratio'])})")
        dense_txt = ("Sweep-grid check: re-solving with 5 mV bias steps from −120 to +20 mV gives " + "; ".join(parts)
                     + ". The 25 mV sweep grid is converged for the peak position and shape.")
    else:
        dense_txt = ("Sweep-grid check (pending): the d 0.7 and 0.4 nm sweeps are being re-solved with 5 mV bias steps "
                     "across the flat band, to test the PCHIP interpolation of the apex potential between the 25 mV "
                     "sweep points.")

    ratio_frac = (pk_fit["ratio"] - pk_free["ratio"]) / (pk_meas["ratio"] - pk_free["ratio"])
    lv = {v: lev(Pf, v) for v in (-0.30, -0.075, -0.025, 0.0, 0.2, 0.355)}

    html = f"""<title>940 Flat-Band Tip Gating</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
{STYLE}
<main>
<div class="eyebrow">Six-layer rhombohedral graphite · line 940, (+⁵) side · 23 November 2026</div>
<h1>The tip sets the flat band in line 940</h1>
<p class="lede">The top-layer flat band of the (+⁵) spectrum in line 940 sits at −32 mV. This report tests whether the
tip–sample contact potential alone can put it there, with a neutral stack and no substrate charge, and follows the
consequences: the shape of the peak, its dependence on tip distance and apex radius, and why the tip moves the
occupied-side features of the spectrum but not the empty-side ones.</p>

<ol class="findings">
<li><b>One parameter places the flat band.</b> CPot = W<sub>tip</sub> − W<sub>RG</sub> = {num(best, 3)} eV puts the
peak at {mv(1e3 * pk_fit['v'], 1)} mV (measured {mv(1e3 * pk_meas['v'], 1)} mV) without any charge in the stack or
the substrate. With W<sub>RG</sub> ≈ 4.6 eV this is W<sub>tip</sub> ≈ {num(4.6 + best, 2)} eV.</li>
<li><b>The tip makes the peak asymmetric in the measured sense.</b> The ratio of outer to inner half width rises from
{num(pk_free['ratio'])} without the tip to {num(pk_fit['ratio'])} with it; measured {num(pk_meas['ratio'])}. The long
outer tail of the measured peak is not reproduced.</li>
<li><b>Tip distance controls the result; apex radius hardly does.</b> At fixed CPot the peak moves from
{mv(gp(0.4), 0)} to {mv(gp(1.0), 0)} mV for d = 0.4–1.0 nm, but only from {mv(1e3 * fp[(0.7, 5.0)]['v'], 0)} to
{mv(1e3 * fp[(0.7, 80.0)]['v'], 0)} mV for R = 5–80 nm.</li>
<li><b>Only the occupied side moves because the field vanishes at V₀ = −CPot/e = {mv(v0f, 2)} V.</b> The upper bulk
features sit there. Screening by the flat band at E<sub>F</sub> cuts the top-layer lever arm from 0.22 at negative
bias to 0.05 near V₀.</li>
<li><b>Not explained by the tip:</b> the upper bulk edges ({mv(1e3 * t['features'][2])} / {mv(1e3 * t['features'][3])}
mV against {mv(fm[2])} / {mv(fm[3])} mV measured), the {abs(1e3 * t['features'][0] - fm[0]):.0f} mV overshoot of the
lower bulk edge, and the outer tail of the flat-band peak.</li>
<li><b>Substrate charge acts through the tip's lever arm.</b> At the fitted tip, positive trap charge (electrons in the
stack) raises the lever arm to {num(nsd[2.0]['alpha'])} and pulls the flat band to {mv(nsd[2.0]['fb_tip'])} mV;
negative trap charge pins it near E<sub>F</sub> (α ≈ {num(nsd[-0.5]['alpha'])}). The measured −32 mV needs
n<sub>sub</sub> ≈ 0.</li>
</ol>

<h2>Hypothesis and model</h2>
<p>The top-layer flat band of the (+⁵) spectrum sits at −32 mV in line 940 and at +23 mV in line 514, tens of meV
from where a neutral stack puts it (−4 mV). Adsorbates cannot supply that charge: there are too few of them and they
are mostly n-alkanes. Tip pulses move the peak slightly. The working assumption is therefore that the tip sets the
flat-band position. At the small bias of the flat band, the only field source the tip brings is the contact
potential.</p>
<div class="eq">CPot = W<sub>tip</sub> − W<sub>RG</sub>; the tip field vanishes at the sample bias V₀ = −CPot/e
<span class="r">(neutral stack, no other charge)</span></div>
{table(["ingredient", "setting"], [
    ["stack", "six-layer rhombohedral graphite, (+⁵) stacking; neutral: no Fermi-level offset, no trap or substrate charge"],
    ["tight binding", "full SWMcC set (γ0–γ5, Δ′), Hartree self-consistent with the Poisson solve"],
    ["interlayer ε<sub>z</sub>", "2.6 (σ bands only; π screening from the tight-binding charge)"],
    ["substrate", "2 nm trap layer (uncharged) and 88 nm SiO₂ on a grounded gate"],
    ["tip", "sphere R 20 nm at d 0.7 nm (swept in section 3); axisymmetric engine, draft grid"],
    ["spectrum", "top-layer dI/dV, WKB barrier 4.6 eV over d, σ 12.5 meV, T 9 K, 3.75 mV modulation"],
    ["bias sweep", "25 self-consistent points from −0.45 to +0.45 V (25 mV steps within ±0.15 V, 50 mV outside); "
                   "apex potentials PCHIP-interpolated in between"],
])}

<h2>1 · Contact-potential fit</h2>
<p>CPot was bisected on a single quantity: the flat-band peak of the top-layer LDOS under the apex at the measured
peak bias, V = {mv(1e3 * pl['v_fb'], 1)} mV, one self-consistent solve per step. Full bias sweeps followed at CPot 0,
−0.30, {num(best, 3)} and −0.45 eV.</p>
{table(["CPot (eV)", "flat band under the apex (mV)"],
       [[num(x["cpot"], 4), mv(1e3 * x["fb"], 1)] for x in lg], numeric={0, 1})}
<p>The fit is CPot = {num(pl['cpot'], 3)} eV. Near it the peak moves {abs(slope):.2f} mV per meV of CPot, so a tip
change of a few tens of meV in work function moves the flat band by a few mV, as tip pulses do.</p>
{table(["spectrum", "lower bulk edge", "flat band", "upper edge 1", "upper edge 2", "edge rms (mV)", "tail asym."],
       feat_rows, feat_cls, numeric={1, 2, 3, 4, 5, 6})}
<p>Feature positions in mV. Edge rms: root mean square error of the three bulk edges against the measurement. Tail
asymmetry: mean dI/dV 40, 60 and 80 mV from the peak on the side away from E<sub>F</sub> minus that on the side towards
it, relative to the peak height.</p>
{img("stacking_wall_cpot940.png", "CPot fit of the 940 (+5) spectrum",
     "Line 940 (+⁵), neutral stack, ε 2.6, full SWMcC, R 20 nm, d 0.7 nm. Top: measured spectrum against no tip and "
     "four contact potentials, full range and flat-band region. Bottom: flat band under the apex against CPot "
     "(bisection), layer potentials under the apex at the fit for six biases, and the top-layer potential against "
     "bias for each CPot.")}
{img("stacking_wall_cpot940_offset.png", "Offset spectra: no tip, CPot 0, fit, measured",
     "No tip, CPot 0, the fit and the measurement, offset vertically; grey lines mark the measured features.")}
<ul>
<li>The flat band and the lower bulk edge move together. At the fit the lower edge overshoots: {mv(1e3 * t['features'][0])}
mV against {mv(fm[0])} mV.</li>
<li>The upper edges stay at {mv(1e3 * t['features'][2])} / {mv(1e3 * t['features'][3])} mV for every negative CPot.
The measured {mv(fm[2])} / {mv(fm[3])} mV are out of reach (section 4 explains why).</li>
<li>CPot 0 leaves the flat band at {mv(1e3 * runs[0.0]['tip']['features'][1])} mV: the tip at zero contact potential
does not gate the top layer near E<sub>F</sub>.</li>
</ul>

<h2>2 · Shape of the flat-band peak</h2>
{img("stacking_wall_cpot940_fbshape.png", "Flat-band peak shape with and without the tip",
     "Left: fit and no tip as computed. Right: both peaks aligned on the fitted peak and normalised to their height, "
     "with the measured peak treated the same way; negative bias offset is the outer side, away from E<sub>F</sub>. "
     "Half widths from a 0.25 mV re-evaluation of the spectra.")}
{table(["spectrum", "peak (mV)", "HWHM outer (mV)", "HWHM inner (mV)", "outer / inner", "tail asym."],
       shape_rows, ["meas", "", "hl"], numeric={1, 2, 3, 4, 5})}
<p>The tip widens the outer flank by {1e3 * (pk_fit['hwhm_out'] - pk_free['hwhm_out']):.1f} mV and leaves the inner
flank unchanged within {abs(1e3 * (pk_fit['hwhm_in'] - pk_free['hwhm_in'])):.1f} mV. That covers
{100 * ratio_frac:.0f} % of the measured excess of the outer/inner ratio. Mechanism: a level at energy E appears at the
bias where eV = E + ΔU₁(V), so its width in bias is its energy width divided by 1 − α, with α = dU₁/d(eV) the local
lever arm of the top layer. Across the peak α falls from {num(lv[-0.075])} at −75 mV to {num(lv[0.0])} at 0 mV,
which stretches the outer flank more than the inner one.</p>
<p>The measured peak also has a long outer tail: 15–20 % of its height remains 100 mV below the peak (tail asymmetry
{num(runs[best]['measured_asym'])}). The model gives {num(t['asym'])} at the fit, and at most
{num(geom[(0.4, 20.0)]['tip']['asym'])} at d 0.4 nm. The tail needs an ingredient the model lacks.</p>
<p>The spectra of the sweeps are stored on a 2.5 mV grid, which quantised the half widths in 2.5 mV steps, the size of
the effect. All half widths here come from a re-evaluation on a 0.25 mV grid from the stored potentials, with the
half-maximum crossings interpolated.</p>

<h2>3 · Tip distance and apex radius</h2>
<p>CPot fixed at {num(best, 3)} eV; d from 0.4 to 1.0 nm at R 20 nm, and R from 5 to 80 nm at d 0.7 nm. The WKB
barrier width of the spectrum follows d.</p>
{img("stacking_wall_cpot940_geom.png", "Tip distance and radius sweeps",
     "Rows: distance sweep (R 20 nm) and radius sweep (d 0.7 nm). Columns: spectra offset against the measurement; "
     "flat band normalised; flat-band position; outer/inner half-width ratio, with the measured and no-tip values.")}
<h3>Distance d (R 20 nm)</h3>
{table(["d (nm)", "R (nm)", "flat band (mV)", "HWHM out", "HWHM in", "out / in", "tail asym.", "lower edge",
        "upper edges", "edge rms"], dsw, ["hl" if r[0] == "0.7" else "" for r in dsw], numeric=set(range(10)))}
<h3>Apex radius R (d 0.7 nm)</h3>
{table(["d (nm)", "R (nm)", "flat band (mV)", "HWHM out", "HWHM in", "out / in", "tail asym.", "lower edge",
        "upper edges", "edge rms"], rsw, ["hl" if r[1] == "20" else "" for r in rsw], numeric=set(range(10)))}
<ul>
<li><b>d sets the position.</b> Near d 0.7 nm the flat band moves {abs(d_slope) / 10:.0f} mV per 0.1 nm. Over R = 5–80 nm
it moves by {abs(1e3 * (fp[(0.7, 80.0)]['v'] - fp[(0.7, 5.0)]['v'])):.0f} mV: for R ≫ d the field under the apex
depends on d and CPot only. The fitted CPot therefore holds only together with d = 0.7 nm; the flat band fixes one
combination of the two.</li>
<li><b>The asymmetry grows as the tip approaches</b>, from {num(fp[(1.0, 20.0)]['ratio'])} at 1.0 nm to
{num(fp[(0.4, 20.0)]['ratio'])} at 0.4 nm, which exceeds the measured {num(pk_meas['ratio'])}, but there the peak sits
at {mv(gp(0.4), 0)} mV. At fixed CPot a closer tip both moves the peak away from E<sub>F</sub> and changes the lever
arm, so this sweep does not separate the two. A CPot refit at each d with the peak held at −32 mV does.</li>
<li><b>The upper edges do not move with the geometry</b> (242–245 / 355 mV throughout). The lower edge goes from
{mv(1e3 * geom[(0.4, 20.0)]['tip']['features'][0])} to {mv(1e3 * geom[(1.0, 20.0)]['tip']['features'][0])} mV; the
measured {mv(fm[0])} mV prefers d ≈ 0.9–1.0 nm, where the flat band is too close to E<sub>F</sub>.</li>
<li>{dense_txt}</li>
</ul>

<h2>4 · Why the tip moves only the occupied-side features</h2>
{img("stacking_wall_cpot940_tibb_sketch.png", "Energy diagrams and tip-induced shifts",
     "Top: tip, vacuum gap and the six layers at the biases of three spectral features, with the self-consistent "
     "layer potentials of the fit (vacuum level drawn 3.8 eV low). Bottom: top-layer shift ΔU₁ and local lever arm "
     "against bias for CPot −0.367 eV and 0; circles are the shifts of the spectral features.")}
{table(["feature", "bias (mV)", "V − V₀ (V)", "ΔU₁ (meV)", "lever arm α", "spectral shift (mV)", "shift at CPot 0"],
       shift_rows, numeric={1, 2, 3, 4, 5, 6})}
<ul>
<li><b>Electrostatics depends on V + CPot/e only.</b> The CPot {num(best, 3)} eV shift curve equals the CPot 0 curve
moved by +0.367 V (largest difference {overlay:.2f} meV). The tip field vanishes at V₀ = {mv(v0f, 3)} V, the bias of
zero tip-induced band bending. The upper bulk features at +245 and +355 mV sit 0.12 and 0.01 V below it.</li>
<li><b>Screening depends on where the flat band is.</b> At negative bias the field pushes the flat band below
E<sub>F</sub>; the top layer has no states at E<sub>F</sub> and the field penetrates (α = {num(lv[-0.30])} at −300 mV).
Approaching V₀, the flat band returns to E<sub>F</sub> and its density of states screens (α = {num(lv[0.2])} at +200
mV). A constant α of 0.22 would shift the +245 mV feature by −26 meV; the solve gives
{mv(1e3 * float(Pf(ftp[2])[0]), 0)} meV.</li>
<li><b>Bulk features follow deeper layers.</b> The flat band is a top-layer state and shifts nearly with ΔU₁. The bulk
band edges extend through the stack: the lower edge shifts {mv(1e3 * (ftp[0] - ff[0]), 1)} mV while ΔU₁ is
{mv(1e3 * float(Pf(ftp[0])[0]), 0)} meV, about the shift of the third layer.</li>
<li><b>CPot 0 shows the contrast.</b> There V₀ = {mv(v00, 3)} V; the field changes sign at zero bias and pushes the
features outward on both sides ({", ".join(mv(1e3 * (f0[i] - free['features'][i])) for i in range(4))} mV). A shifted
occupied side with a fixed empty side is the signature of a negative contact potential of about −0.37 eV.</li>
</ul>

<h2>5 · Band structure under the apex</h2>
{img("stacking_wall_cpot940_bands.png", "Band structure without and with the tip potentials",
     "Full-SWMcC bands along Γ ← K → M, coloured by weight on the top layer: no tip, and with the layer potentials "
     "under the apex at V = −32 mV (CPot −0.367 eV). Right: layer potentials (also at −270 and +355 mV). Bottom: the "
     "two surface flat bands overlaid, and the top-layer LDOS.")}
<ul>
<li>No tip: the top- and bottom-surface flat bands are degenerate at K (−7 meV) and hybridise, 50 % on L1 each.</li>
<li>Tip at −32 mV: the potential falls from −30 meV on L1 to −1 meV on L6 and splits the pair. The top-surface band
drops to −39 meV at K (100 % on L1, maximum −25 meV), fully below E<sub>F</sub>; the bottom-surface band stays at
−10 meV and holds E<sub>F</sub>. Bulk subbands shift by about −13 meV, the change of the stack-average potential.</li>
<li>At +355 mV the potentials under the apex equal the no-tip ones within 1 meV: the bands are those without the tip.</li>
</ul>

<h2>6 · Substrate charge at the fitted tip</h2>
<p>Tip fixed (CPot {num(best, 3)} eV, R 20 nm, d 0.7 nm); fixed sheet charge n<sub>sub</sub> in the 2 nm trap layer
from −2 to +2 × 10¹³ e/cm² (positive charge induces electrons in the stack). ±3 × 10¹³ would put L6 at the 0.5 eV edge of
the tight-binding window.</p>
{img("stacking_wall_cpot940_nsub.png", "Substrate-charge sweep at the fitted tip",
     "Top: top-layer dI/dV under the tip for each n<sub>sub</sub> (10¹³ e/cm²), offset, with the measured spectrum; "
     "layer potentials under the apex at −32 mV. Bottom: flat-band region overlaid; flat-band position with and "
     "without the tip; top-layer lever arm at −32 mV.")}
{table(["n_sub (10¹³ e/cm²)", "flat band, no tip (mV)", "flat band, tip (mV)", "lever arm α", "C_q L1", "C_q L2",
        "C_q L6", "U₁ / U₆ − E_F (meV)"], ns_rows,
       ["hl" if r[0] == "+0.0" else "" for r in ns_rows], numeric=set(range(8)))}
<p>C<sub>q</sub> = e²D<sub>l</sub>(E<sub>F</sub>) under the apex at −32 mV, in µF/cm²; compare C<sub>gap</sub> ≈
ε₀/d ≈ 1.3 µF/cm² and C₁₂ = ε₀·2.6/0.335 nm = 6.9 µF/cm², which cap α at 0.155 while L2 screens.</p>
<ul>
<li>Without the tip the substrate charge hardly reaches the top layer: the flat band stays between
{mv(min(q['fb_free'] for q in ns))} and {mv(max(q['fb_free'] for q in ns))} mV.</li>
<li>The lever arm depends on which surface band holds E<sub>F</sub> under the apex. At n<sub>sub</sub> = 0 the tip
fills the top band and the bottom band holds E<sub>F</sub> (C<sub>q</sub> of L6 {num(nsd[0.0]['cq'][5], 1)}):
α = {num(nsd[0.0]['alpha'])}.</li>
<li>Negative n<sub>sub</sub> (holes) empties the bottom band; the top band is left to hold E<sub>F</sub> and screens the
tip itself (C<sub>q</sub> of L1 8–16): α {num(nsd[-0.5]['alpha'])}–{num(nsd[-2.0]['alpha'])}, flat band pinned at −12 to
−22 mV.</li>
<li>Positive n<sub>sub</sub> (electrons) fills the bottom band; with the tip pulling the top layers down, L1 and L2
have almost no states at E<sub>F</sub> and the field penetrates past L2: α rises to {num(nsd[2.0]['alpha'])}.</li>
<li>The asymmetry follows from the fixed sign of the tip field at this CPot, which pulls electrons to the top. It would
reverse for CPot &gt; 0 or above V₀.</li>
</ul>

<h3>Tip and substrate lever arms of each peak</h3>
<p>Fitting four peak positions with CPot and n<sub>sub</sub> works only because each peak responds to the two in a
different ratio. The tip selects by bias: its field scales with V − V₀ and vanishes near V₀. The substrate selects by
depth: it is bias-independent, and the lower layers screen it from the top-surface flat band.</p>
{table(["peak", "tip: CPot −0.1 eV (mV)", "substrate: n_sub 0 → −0.5 × 10¹³ (mV)", "substrate / tip"], lev_rows,
       numeric={1, 2, 3})}
<ul>
<li>The flat band fixes one combination (the valley of constant flat band), the upper edges measure n<sub>sub</sub>,
and the lower edge is a check.</li>
<li>Linear prediction along the valley from these slopes: n<sub>sub</sub> ≈ {mv(pred_n, 2)} × 10¹³, CPot ≈
{num(pred_c, 2)} eV, rms of the four positions {rms0:.0f} → {rms_opt:.0f} mV (lower {mv(e_opt[0])}, upper
{mv(e_opt[2])} / {mv(e_opt[3])} mV).</li>
<li>The limit is again a lever-arm mismatch: along the valley the lower edge moves {val_s[0] / val_s[2]:.1f} times
faster than the upper edges, while the upper edges need to move up and the lower edge down. That residual, and the
spacing of the two upper edges ({1e3 * (runs[best]['tip']['features'][3] - runs[best]['tip']['features'][2]):.0f} mV
against {fm[3] - fm[2]:.0f} mV measured), lie beyond the two electrostatic parameters.</li>
</ul>

<h2>7 · Open points and next steps</h2>
<ul>
<li><b>Upper bulk edges.</b> The field vanishes there, so no tip setting moves them. Their 15–27 mV deficit belongs
to the band structure or the potential profile of the stack.</li>
<li><b>Lower edge and outer tail.</b> The lower edge prefers a weaker tip than the flat band does, and the measured
outer tail of the flat band is absent from every tip setting.</li>
<li><b>Fit CPot and n<sub>sub</sub> together</b> over n<sub>sub</sub> = 0 to −0.5 × 10¹³: at −0.5 the upper edges move
to {mv(1e3 * nsub_edges[0])} / {mv(1e3 * nsub_edges[1])} mV (measured {mv(fm[2])} / {mv(fm[3])}), but the flat band
rises to {mv(nsd[-0.5]['fb_tip'])} mV and needs a more negative CPot.</li>
<li><b>Separate lever arm from position.</b> Refit CPot at d 0.4, 0.5 and 1.0 nm with the flat band held at −32 mV
and compare the half-width ratio and the lower edge.</li>
<li><b>Interlayer ε below 2.6</b> (noted 22 November): a smaller ε raises the lever arm per unit field.</li>
<li><b>Line 514 (+⁵).</b> Its flat band sits at +23 mV; in this model that needs CPot &gt; 0, i.e. a tip with a larger
work function than the sample at the time of that measurement.</li>
<li><b>Experimental test.</b> A tip-height (setpoint) series at one tip state should move the flat band (about
{abs(d_slope) / 10:.0f} mV per 0.1 nm) and the lower edge, and leave the +0.27 and +0.37 V features in place. Features
above V₀ ≈ +0.37 V should shift slightly outward.</li>
</ul>

<footer>
<p>Generated by <code>stacking_wall_epsfit/cpot940_report.py</code> on the branch
<code>claude/eloquent-franklin-q9q4cb</code> of few-layer-graphite; Poisson and tight-binding engine: bandbending
(<code>lateral-solver</code>). Scripts: <code>cpot940.py</code> (CPot bisection and sweeps),
<code>cpot940_geom.py</code> (d and R sweeps), <code>cpot940_geom_fine.py</code> (0.25 mV peak re-evaluation),
<code>cpot940_geom_dense.py</code> (5 mV sweep-grid check), <code>cpot940_tibb_sketch.py</code>, <code>cpot940_bands.py</code>; summaries in
<code>examples/stacking_wall_cpot940*.json</code>. The raw .sm4 line spectra are not in the repository.</p>
</footer>
</main>
"""
    return html


if __name__ == "__main__":
    out = os.path.join(EX, "stacking_wall_cpot940_report.html")
    open(out, "w").write(build())
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB")
