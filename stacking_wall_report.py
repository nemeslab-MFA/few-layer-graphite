#!/usr/bin/env python3
"""
Short HTML report of the stacking-wall calculation: the model in a few lines,
the wall lengths fitted to the two line spectra and the figures written by
stacking_wall_figure.py.

    python stacking_wall_report.py [--tags 514 940] [--out examples/stacking_wall_report.html]
                                   [--page-only PATH]

``--page-only`` writes the page without the document skeleton (no doctype,
html, head or body tags) for a publisher that supplies its own; the images are
then expected next to the page under their file names.
"""

import argparse
import glob
import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")

STYLE = """
<style>
/* layout: one reading column, figures full width in a two-column grid */
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
body { background: var(--bg); color: var(--fg); font-family: var(--serif); font-size: 16px; line-height: 1.5;
  padding-block: 2rem 3rem; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 68rem; margin: 0 auto; }
h1, h2 { font-family: var(--sans); text-wrap: balance; line-height: 1.15; margin: 0; }
h1 { font-size: 2rem; font-weight: 600; }
h2 { font-size: 1.1rem; font-weight: 600; margin-top: 2rem; padding-top: 0.6rem; border-top: 1px solid var(--rule); }
p { max-width: 62ch; margin: 0.6rem 0; }
.eyebrow { font-family: var(--sans); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.lede { color: var(--muted); font-family: var(--sans); font-size: 0.95rem; }
.eq { font-family: var(--serif); font-style: italic; background: var(--panel); padding: 0.6rem 1rem; margin: 0.8rem 0; overflow-x: auto; max-width: 62ch; }
.eq span.r { font-style: normal; }
table { border-collapse: collapse; font-family: var(--sans); font-size: 0.9rem; font-variant-numeric: tabular-nums; margin: 0.8rem 0; }
th, td { text-align: left; padding: 0.35rem 1rem 0.35rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.8rem; letter-spacing: 0.04em; text-transform: uppercase; }
.tablewrap { overflow-x: auto; }
.figs { display: grid; grid-template-columns: 1fr; gap: 1.4rem; margin-top: 0.8rem; }
figure { margin: 0; min-width: 0; }
figure img { width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--rule); }
figcaption { font-family: var(--sans); font-size: 0.85rem; color: var(--muted); margin-top: 0.4rem; }
ul { max-width: 62ch; padding-left: 1.2rem; margin: 0.5rem 0; }
li { margin: 0.25rem 0; }
code { font-family: var(--mono); font-size: 0.85em; }
a { color: var(--accent); }
footer { margin-top: 2.5rem; font-family: var(--sans); font-size: 0.8rem; color: var(--muted); }
</style>
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600'
         '&family=Source+Serif+4:ital,wght@0,400;1,400&family=IBM+Plex+Mono&display=swap">')


def load(tag):
    with open(os.path.join(EXAMPLES, f"stacking_wall_{tag}.json")) as fh:
        return json.load(fh)


def _peak(bias, y, centre, half=0.09):
    bias, y = np.asarray(bias), np.asarray(y)
    sel = (bias >= centre - half) & (bias <= centre + half) & np.isfinite(y)
    return float(bias[sel][np.argmax(y[sel])])


def _minimum(bias, y, half=0.15):
    bias, y = np.asarray(bias), np.asarray(y)
    sel = (np.abs(bias) <= half) & np.isfinite(y)
    return float(bias[sel][np.argmin(y[sel])])


def _bending_at(sweep, V):
    """U_1(apex) - U_1(far) (eV) of the top layer at bias V, from a sweep record."""
    b = np.asarray(sweep["bias"])
    U0, Uf = np.asarray(sweep["U0"])[:, 0], np.asarray(sweep["U_far"])[:, 0]
    return float(np.interp(V, b, U0 - Uf))


def tibb_numbers(state):
    """Apparent flat-band peaks (measured, simulated) and the top-layer band bending under the
    apex at the flat-band bias for the (+5) stack, the wall and the (+4-) stack."""
    s = state["settings"]
    meas = state["measured"]
    bias = np.asarray(meas["bias"])
    fine = np.asarray(state.get("fine_bias", []))
    E_fb = s["E_fb_measured"]
    scan = state.get("scan", {})
    best = scan.get(f"{state.get('best_cpot', 0.0):+.3f}") if scan else None
    runs = state.get("runs", {})
    wall = next((runs[k] for k in runs if k.startswith("wall")), None)
    sf = runs.get("sf")
    out = {"E_fb": E_fb, "meas_wall_peak": _peak(bias, meas["wall"], E_fb, 0.09),
           "meas_sf_min": _minimum(bias, meas["sf"])}
    if sf is not None and fine.size:
        out["sf_min"] = _minimum(fine, sf["didv"])
    for key, run in (("rg", best), ("wall", wall), ("sf", sf)):
        if run is None or not fine.size:
            continue
        out[f"{key}_peak"] = _peak(fine, run["didv"], E_fb, 0.09)
        out[f"{key}_frozen_peak"] = _peak(fine, run["didv_frozen"], E_fb, 0.09)
        out[f"{key}_bend"] = _bending_at(run["sweep"], out[f"{key}_peak"])
        out[f"{key}_mu0"] = run["sweep"]["mu0"]
        out[f"{key}_residual"] = run["residual"]
    return out


def series_rows(tag, state):
    """(s, simulated peak (V), peak height relative to s = 0) of the wall series of one line."""
    import glob
    fine = np.asarray(state.get("fine_bias", []))
    if not fine.size or "scan" not in state:
        return []
    E_fb = state["settings"]["E_fb_measured"]
    runs = {0.0: state["scan"][f"{state['best_cpot']:+.3f}"]}
    files = [os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")]
    files += sorted(glob.glob(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}_s*.json")))
    for f in files:
        with open(f) as fh:
            for k, r in json.load(fh).get("runs", {}).items():
                if k.startswith("wall_s") and r.get("s") is not None:
                    runs[float(r["s"])] = r
    sel = (fine >= E_fb - 0.09) & (fine <= E_fb + 0.09)
    h0 = np.asarray(runs[0.0]["didv"])[sel].max()
    out = []
    for s in sorted(runs):
        y = np.asarray(runs[s]["didv"])[sel]
        out.append((s, float(fine[sel][np.argmax(y)]), float(y.max() / h0)))
    return out


def series_table(states):
    tags = list(states)
    rows = {t: series_rows(t, states[t]) for t in tags}
    s_all = sorted({s for r in rows.values() for s, _, _ in r})
    if len(s_all) < 3:
        return ""
    head = "".join(f"<th>peak {t} / height</th>" for t in tags)
    body = []
    for s in s_all:
        cells = []
        for t in tags:
            m = {a: (b, c) for a, b, c in rows[t]}
            cells.append(f"<td>{1e3 * m[s][0]:+.0f} mV / {m[s][1]:.2f}</td>" if s in m else "<td>–</td>")
        body.append(f"<tr><td>{s:.2f}</td>{''.join(cells)}</tr>")
    return (f"<div class=\"tablewrap\"><table><tr><th>s</th>{head}</tr>{''.join(body)}</table></div>"
            "<p class=\"lede\">Simulated flat-band peak of the wall stack against the (+⁴−) weight s, with its height "
            "relative to the (+⁵) stack; beyond s ≈ 0.8 the top-layer peak is gone and the maximum found is another feature.</p>")


def hybrid_numbers(tag, state):
    """Flat-band peak at the measured wall position: measured, local approximation, hybrid (V), and the
    far-field top-layer potential of the stacks at s = 0 and 0.75 (eV); None where files are missing."""
    import glob
    s = state["settings"]
    E_fb, x_wall = s["E_fb_measured"], s["x_wall"]
    out = {"meas": _peak(np.asarray(state["measured"]["bias"]), np.asarray(state["measured"]["wall"]), E_fb, 0.09),
           "E_fb": E_fb}
    loc_path = os.path.join(EXAMPLES, f"stacking_wall_tibb_map_{tag}.npz")
    hyb_path = os.path.join(EXAMPLES, f"stacking_wall_hybrid_{tag}.npz")
    fit_path = os.path.join(EXAMPLES, f"stacking_wall_{tag}.json")
    if os.path.exists(loc_path):
        with np.load(loc_path) as f:
            j = int(np.argmin(np.abs(f["dist"] - x_wall)))
            out["local"] = _peak(f["bias"], f["didv"][:, j], E_fb, 0.09)
            far_j = f["dist"] > f["dist"].max() - 10.0
            out["local_far"] = _peak(f["bias"], f["didv"][:, far_j].mean(axis=1), E_fb, 0.09)
    if os.path.exists(hyb_path) and os.path.exists(fit_path):
        with open(fit_path) as fh:
            x0 = json.load(fh)["fit"]["x0"]
        with np.load(hyb_path) as f:
            E, x, top = f["energy"], f["x"] / 10.0 + x0, f["ldos"][-1]
        far = x > x.max() - 30.0
        sel = (E >= -0.12) & (E <= 0.12)
        E_far = float(E[sel][np.argmax(top[sel][:, far].mean(axis=1))])
        E = E + (E_fb - E_far)
        j = int(np.argmin(np.abs(x - x_wall)))
        out["hybrid"] = _peak(E, top[:, j], E_fb, 0.09)
    # far-field top-layer potential of the stacks (Poisson), referred to E_F
    recs = {0.0: state["scan"][f"{state['best_cpot']:+.3f}"]}
    for f in [os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}.json")] + sorted(glob.glob(os.path.join(EXAMPLES, f"stacking_wall_tibb_{tag}_s*.json"))):
        with open(f) as fh:
            for k, r in json.load(fh).get("runs", {}).items():
                if k.startswith("wall_s") and r.get("s") is not None:
                    recs[float(r["s"])] = r
    for key, sv in (("V1_0", 0.0), ("V1_075", 0.75)):
        if sv in recs:
            sw_ = recs[sv]["sweep"]
            out[key] = float(np.mean(np.asarray(sw_["U_far"])[:, 0]) - sw_["mu0"])
    return out


def hybrid_text(states, img_prefix=""):
    nums = {t: hybrid_numbers(t, st) for t, st in states.items()}
    if any("hybrid" not in v for v in nums.values()):
        return ""
    t1, t2 = list(nums)[:2]
    a, b = nums[t1], nums[t2]
    mv = lambda v: f"{1e3 * v:+.0f}"   # noqa: E731
    return f"""<p>The two calculations are complementary: the k·p wall model has the wall and its fringes but no electrostatics,
the Poisson solver has the electrostatics but, being axisymmetric, no wall. The notebook
<code>examples/stacking_wall_hybrid.ipynb</code> joins them: it reads the Poisson outputs, refers the far-field
layer potentials of every stack to the common Fermi level, V<sub>l</sub> = U<sub>far,l</sub> − μ0, places them
along the measured line through the fitted profile s(x), and feeds them into the k·p wall model as
position-dependent on-site energies (<code>layer_potential_profile</code> of <code>stacking_wall.ldos_wall</code>).
The tip-induced part of the potential moves with the tip and is left out of the maps (the notebook adds it at
chosen tip positions on request). The fringes of the (+⁴−) side return, together with the band shift of the
stacks.</p>
<p>At the position of the measured wall spectrum the flat-band peak sits at {mv(a['meas'])} mV measured,
{mv(a['local'])} mV in the local approximation and {mv(a['hybrid'])} mV in the hybrid for {t1}; {mv(b['meas'])},
{mv(b['local'])} and {mv(b['hybrid'])} mV for {t2}. The top-layer potential of the stacks moves from
{mv(a['V1_0'])} to {mv(a['V1_075'])} meV ({t1}) and from {mv(b['V1_0'])} to {mv(b['V1_075'])} meV ({t2}) between
s = 0 and 0.75, and the k·p flat band near the wall follows it. Measured from the far (+⁵) peak of each
description, the bend at the wall is {mv(a['hybrid'] - a['E_fb'])} mV in the hybrid against
{mv(a['local'] - a['local_far'])} mV in the local approximation and {mv(a['meas'] - a['E_fb'])} mV measured ({t1}),
{mv(b['hybrid'] - b['E_fb'])}, {mv(b['local'] - b['local_far'])} and {mv(b['meas'] - b['E_fb'])} mV ({t2}). The two
descriptions of the wall agree to within {abs(1e3 * (a['hybrid'] - a['E_fb'] - a['local'] + a['local_far'])):.0f} mV
({t1}) and {abs(1e3 * (b['hybrid'] - b['E_fb'] - b['local'] + b['local_far'])):.0f} mV ({t2}): the k·p wall, with its
fringes and remote hoppings, changes little of what the far-field potentials of the stacks do to the flat band, and
both fall short of the measured bend, by a third in {t1} and by more than half in {t2}.</p>
<p><b>Correction (2026-11-09).</b> The first version of this section passed the engine's layer potentials to the
k·p model in the wrong order: the engine numbers the layers from the top, the k·p model from the bottom, so the k·p
top layer carried the potential of the bottom layer, which barely moves across the wall. The hybrid then bent by
+10 mV ({t1}) and 0 mV ({t2}) at the wall, and the shortfall against the local approximation was put down to the
k·p flat-band state spreading into the lower layers. That explanation was wrong. The maps, the tip-induced bump
and the k·p spectra of the lateral solve below are recomputed with the order fixed
(<code>stacking_wall.ldos_wall</code> now states the order; a regression test checks it).</p>
<div class="figs">
<figure><img src="{img_prefix}stacking_wall_hybrid_{t1}.png" alt="hybrid map, {t1}">
<figcaption>Line spectrum {t1}: measured dI/dV(V, x), the local approximation and the hybrid, with their bias
derivatives; the hybrid energy axis is shifted so that its (+⁵) flat band far from the wall sits at the measured
energy.</figcaption></figure>
<figure><img src="{img_prefix}stacking_wall_hybrid_{t2}.png" alt="hybrid map, {t2}">
<figcaption>Line spectrum {t2}, the same.</figcaption></figure>
</div>
<figure><img src="{img_prefix}stacking_wall_hybrid_peaks.png" alt="flat-band peak along the line">
<figcaption>Apparent flat-band energy along the line: measured, local approximation and hybrid, followed until
the peak has faded to 15 % of its far value.</figcaption></figure>
"""


def bump_numbers(tag):
    """Per tip position: s, top-layer bump (eV), peak with and without the bump (eV, aligned)."""
    path = os.path.join(EXAMPLES, f"stacking_wall_bump_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        st = json.load(fh)
    s = st["settings"]
    runs = st["runs"]
    hyb_path = os.path.join(EXAMPLES, f"stacking_wall_hybrid_{tag}.npz")
    shift = 0.0
    if os.path.exists(hyb_path):
        with np.load(hyb_path) as f:
            E, x, top = f["energy"], f["x"] / 10.0 + s["x0"], f["ldos"][-1]
        far = x > x.max() - 30.0
        sel = (E >= -0.12) & (E <= 0.12)
        shift = s["E_fb"] - float(E[sel][np.argmax(top[sel][:, far].mean(axis=1))])
    def centroid(run):
        E, y = np.asarray(run["energy"]), np.asarray(run["ldos"])
        sel = (E >= run["peak"] - 0.03) & (E <= run["peak"] + 0.03)
        yb = y[sel] - y[sel].min()
        return float(np.sum(E[sel] * yb) / np.sum(yb))
    rows = []
    for r in sorted((r for r in runs.values() if r["variant"] == "bump"), key=lambda r: r["x"]):
        ref = runs.get(f"{r['x']:+.1f}_none")
        rows.append({"x": r["x"], "s": r["s"], "bump": r["bump"][0], "peak": r["peak"] + shift,
                     "peak_ref": (ref["peak"] + shift) if ref else None, "height": r["height"],
                     "height_ref": ref["height"] if ref else None,
                     "dcentroid": (centroid(r) - centroid(ref)) if ref else None})
    return {"settings": s, "rows": rows}


def far_tip_shift(tag):
    """Centroid shift of the (+5) flat-band peak by the tip at the measured (+5) peak bias, contact potential 0
    (stacking_wall_cpot_sensitivity_TAG.json, tip 20 nm from the wall on the (+5) side), in eV; None if missing."""
    path = os.path.join(EXAMPLES, f"stacking_wall_cpot_sensitivity_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        runs = json.load(fh).get("runs", {})
    a, b = runs.get("far_+0.000"), runs.get("far_none")
    return (a["centroid"] - b["centroid"]) if (a and b) else None


def bump_text(states, img_prefix=""):
    nums = {t: bump_numbers(t) for t in states}
    if any(v is None or not v["rows"] for v in nums.values()):
        return ""
    table = []
    for t, v in nums.items():
        h_max = max(r["height"] for r in v["rows"])
        for r in v["rows"]:
            if r["height"] < 0.15 * h_max or abs(r["peak"] - v["settings"]["E_fb"]) > 0.06:
                table.append(f"<tr><td>{t}</td><td>{r['x']:+.0f} nm</td><td>{r['s']:.2f}</td><td>{1e3 * r['bump']:+.0f} meV</td>"
                             f"<td colspan=\"3\">flat-band peak faded (height below 15 % of the far value)</td></tr>")
                continue
            ref = f"{1e3 * r['peak_ref']:+.0f}" if r["peak_ref"] is not None else "–"
            extra = f"{1e3 * (r['peak'] - r['peak_ref']):+.1f}" if r["peak_ref"] is not None else "–"
            dc = f"{1e3 * r['dcentroid']:+.1f}" if r.get("dcentroid") is not None else "–"
            table.append(f"<tr><td>{t}</td><td>{r['x']:+.0f} nm</td><td>{r['s']:.2f}</td><td>{1e3 * r['bump']:+.0f} meV</td>"
                         f"<td>{ref} / {1e3 * r['peak']:+.0f} mV</td><td>{extra} mV</td><td>{dc} mV</td></tr>")
    s0 = next(iter(nums.values()))["settings"]
    biases = ", ".join(f"{1e3 * v['settings']['bias']:+.0f} mV ({t})" for t, v in nums.items())
    extras = {t: max((abs(r["dcentroid"]) for r in v["rows"]
                      if r.get("dcentroid") is not None and r["height"] >= 0.15 * max(q["height"] for q in v["rows"])
                      and abs(r["peak"] - v["settings"]["E_fb"]) <= 0.06), default=0.0)
              for t, v in nums.items()}
    worst = ", ".join(f"{1e3 * e:.1f} mV ({t})" for t, e in extras.items())
    # fraction of the top-layer bump that the flat band follows, and the bend at the measured wall position
    alive = {t: [r for r in v["rows"] if r.get("dcentroid") is not None and abs(r["bump"]) > 1e-3
                 and r["height"] >= 0.15 * max(q["height"] for q in v["rows"])
                 and abs(r["peak"] - v["settings"]["E_fb"]) <= 0.06] for t, v in nums.items()}
    follow = [r["dcentroid"] / r["bump"] for rows in alive.values() for r in rows]
    at_wall = []
    for t, v in nums.items():
        hn = hybrid_numbers(t, states[t])
        if "hybrid" not in hn or not alive[t]:
            continue
        xw = states[t]["settings"]["x_wall"] - v["settings"]["x0"]
        rows = sorted(alive[t], key=lambda r: r["x"])
        dc = float(np.interp(xw, [r["x"] for r in rows], [r["dcentroid"] for r in rows]))
        far, meas = hn["hybrid"] - hn["E_fb"], hn["meas"] - hn["E_fb"]
        dcf = far_tip_shift(t)
        at_wall.append(dict(t=t, xw=xw, dc=dc, dcf=dcf, far=far, meas=meas,
                            net=(dc - dcf) if dcf is not None else None,
                            total=(far + dc - dcf) if dcf is not None else None))
    wall_txt = ""
    if len(at_wall) == 2 and all(r["net"] is not None for r in at_wall):
        p, q = at_wall
        wall_txt = (f" At the position of the measured wall spectrum ({p['xw']:+.1f} nm from the wall centre in {p['t']}, "
                    f"{q['xw']:+.1f} nm in {q['t']}) the bump shifts the centroid by {1e3 * p['dc']:+.1f} and "
                    f"{1e3 * q['dc']:+.1f} mV. The bend, however, is measured against the (+⁵) peak, which the tip also "
                    f"shifts, at its own bias: by {1e3 * p['dcf']:+.1f} mV ({p['t']}) and {1e3 * q['dcf']:+.1f} mV ({q['t']}) "
                    f"with the tip 20 nm into the (+⁵) side (<code>stacking_wall_cpot_sensitivity.py</code>, δ = 0). There "
                    f"the flat band is a top-layer state and follows the whole bump; inside the wall it follows about half. "
                    f"The tip's contribution to the bend is the difference, {1e3 * p['net']:+.1f} mV ({p['t']}) and "
                    f"{1e3 * q['net']:+.1f} mV ({q['t']}): the hybrid bends by {1e3 * p['far']:+.0f} mV without and "
                    f"{1e3 * p['total']:+.0f} mV with the tip's own field in {p['t']} (measured {1e3 * p['meas']:+.0f} mV), "
                    f"{1e3 * q['far']:+.0f} and {1e3 * q['total']:+.0f} mV in {q['t']} (measured {1e3 * q['meas']:+.0f} mV). "
                    f"In this model the tip moves the flat band nearly as much far from the wall as at it, and its own "
                    f"field is not what bends the band.")
    follow_txt = (f" The k·p flat band follows {100 * min(follow):.0f} to {100 * max(follow):.0f} % of the top-layer bump, "
                  f"more on the (+⁵) side, where the flat-band state lives on the top layer, than inside the wall."
                  if follow else "")
    return f"""<h3>The tip-induced bump</h3>
<p>The Poisson solver also gives the potential under the apex, U0,l(V) − U_far,l, which moves with the tip. It is
added to the hybrid as a Gaussian bump of width w = √(2Rd) = {s0['spot']:.1f} nm centred on the tip, with the
amplitude of the stack at s(x_tip) at the bias where the flat-band peak is measured at the wall ({biases}), and the
wall model is solved once per tip position (cell {s0['length']:.0f} nm, k_y step {s0['dky']} Å⁻¹, {1e3 * s0['sigma']:.0f} meV
broadening; <code>stacking_wall_tibb_bump.py</code>). The table gives the apparent flat-band peak under the tip
without and with the bump (the energy grid of the peak is 2.5 meV; the centroid of the peak resolves below that);
the largest change the bump makes to the peak centroid is {worst}, against a measured bend of 35 to 55 mV.{follow_txt}{wall_txt}
(Before the layer-order correction the bump sat on the bottom layer and moved the peak by at most 2.5 mV; the
version of 2026-11-10 then added the shift at the wall to the bend without subtracting the shift of the (+⁵) reference,
which overstated the tip's contribution as +4 and −6 mV.)</p>
<div class="tablewrap"><table>
<tr><th>line</th><th>tip from the wall centre</th><th>s at the tip</th><th>top-layer bump</th><th>peak without / with bump</th><th>change of the maximum</th><th>change of the centroid</th></tr>
{''.join(table)}
</table></div>
<figure><img src="{img_prefix}stacking_wall_bump.png" alt="top-layer LDOS under the tip with and without the bump">
<figcaption>Top-layer LDOS under the tip for the hybrid without (blue) and with (orange) the tip-induced bump at
each tip position, and the resulting flat-band peak against the tip position.</figcaption></figure>
"""


def cpot_numbers(states):
    """Per line: the model bend against the work-function offset delta (stacking_wall_cpot_sensitivity.py),
    its slope near zero and the offset that would give the measured bend; None if a file is missing."""
    lines = {}
    for t in states:
        path = os.path.join(EXAMPLES, f"stacking_wall_cpot_sensitivity_{t}.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            st = json.load(fh)
        if not st.get("summary"):
            return None
        s, rows = st["settings"], sorted(st["summary"], key=lambda r: r["delta"])
        d = np.array([r["delta"] for r in rows])
        tot = np.array([r["total"] for r in rows])
        tip = np.array([r["tip"] for r in rows])
        meas = s["measured_bend"]
        near = np.abs(d) <= 0.1 + 1e-9
        slope = float(np.polyfit(d[near], tot[near], 1)[0])                       # eV per eV
        # offset that would give the measured bend (interpolated where the curve crosses it; no extrapolation),
        # and the largest bend in the measured direction within the computed range
        need, extrap = None, False
        for i in range(len(d) - 1):
            if (tot[i] - meas) * (tot[i + 1] - meas) <= 0 and tot[i] != tot[i + 1]:
                need = float(d[i] + (meas - tot[i]) * (d[i + 1] - d[i]) / (tot[i + 1] - tot[i]))
        k = int(np.argmax(np.sign(meas) * tot))
        z = next(r for r in rows if abs(r["delta"]) < 1e-9)
        lines[t] = dict(s=s, slope=slope, need=need, extrap=extrap, zero=z, rows=rows, d=d, tot=tot, tip=tip, meas=meas,
                        best=float(tot[k]), best_delta=float(d[k]))
    return lines


def cpot_sensitivity_text(states, img_prefix=""):
    """Subsection: how the tip-induced part of the wall bend depends on the tip work function, and why the
    fits put the offset at zero (stacking_wall_cpot_sensitivity.py)."""
    lines = cpot_numbers(states)
    if lines is None:
        return ""
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    t1, t2 = list(lines)[:2]
    # evidence for CPot = 0
    scan = {}
    for t, st in states.items():
        res = {float(k): v["residual"] for k, v in st.get("scan", {}).items()}
        scan[t] = res
    scan_txt = "; ".join(f"{t}: {res.get(0.0, float('nan')):.2f} at 0 against {res.get(-0.1, float('nan')):.2f} at −0.1 eV and "
                         f"{res.get(0.1, float('nan')):.2f} at +0.1 eV" for t, res in scan.items())
    eps_txt = ""
    p = os.path.join(EXAMPLES, "stacking_wall_eps26_cpot_940.json")
    if os.path.exists(p):
        with open(p) as fh:
            eps_txt = (f"; the ε = 2.6 polarized-stack fit of 940 with the offset free puts it at "
                       f"{mv(json.load(fh)['best']['delta'])} meV")
    a = bend_asymmetry(states)
    asym_txt = (f"; and if one tip recorded both lines, the asymmetry of the two bends gives {mv(a['C'])} meV "
                f"({mv(a['pos'][0])} to {mv(a['pos'][1])} meV within half a bias step)" if a else "")
    rows_html = []
    for i, r1 in enumerate(lines[t1]["rows"]):
        r2 = next((r for r in lines[t2]["rows"] if abs(r["delta"] - r1["delta"]) < 1e-9), None)
        if r2 is None:
            continue
        cells = "".join(f"<td>{mv(r['bump_wall'])} / {mv(r['bump_far'])} meV</td><td>{mv(r['tip'])} mV</td>"
                        f"<td>{mv(r['total'])} mV</td>" for r in (r1, r2))
        rows_html.append(f"<tr><td>{mv(r1['delta'])} meV</td>{cells}</tr>")

    def need_txt(t):
        v = lines[t]
        if v["need"] is not None:
            return f"δ = {mv(v['need'])} meV"
        return (f"no offset in the computed range (the largest bend is {mv(v['best'])} mV, at δ = {mv(v['best_delta'])} meV, "
                f"{100 * v['best'] / v['meas']:.0f} % of the measured)")

    a1, a2 = lines[t1], lines[t2]
    far_shift = {t: next((r["dc_far"] for r in v["rows"] if abs(r["delta"] + 0.2) < 1e-9), None) for t, v in lines.items()}
    res02 = {t: (scan[t].get(-0.2), scan[t].get(0.2)) for t in scan}
    opposite = (a1["need"] is not None and a2["need"] is not None and a1["need"] * a2["need"] < 0)
    return f"""<h3>Sensitivity to the tip work function</h3>
<p>The tip-induced numbers above take the tip work function equal to the sample's: CPot = 0, so the tip does not
bend the (+⁵) bands at zero bias. That value is fitted, not assumed. The contact-potential scan of the (+⁵) spectra
has its lowest residual at zero for both lines ({scan_txt}), which leaves CPot = 0 ± 0.05 eV{eps_txt}{asym_txt}.</p>
<p>An offset δ of the tip work function from the sample's enters the electrostatics only through the tip potential,
U<sub>tip</sub> = V + δ, so the tip-induced bending at bias V is that of the fitted sweeps at V + δ. The k·p wall
model with the Gaussian bump was solved for δ from {mv(min(a1['d']))} to {mv(max(a1['d']))} meV (the bias range of the
sweeps sets the limits), with the tip at the position of the measured wall spectrum and far on the (+⁵) side, each at
its measured peak bias (<code>stacking_wall_cpot_sensitivity.py</code>). The tip-induced part of the bend is the
centroid shift at the wall minus that on the (+⁵) side; added to the far-field bend of the hybrid
({mv(a1['s']['far_field_bend'])} mV in {t1}, {mv(a2['s']['far_field_bend'])} mV in {t2}) it gives the model bend in the
table and the figure. Near δ = 0 the bend changes by {1e2 * a1['slope']:.0f} mV ({t1}) and {1e2 * a2['slope']:.0f} mV
({t2}) per 100 meV of offset, so the fitted ±50 meV moves it by at most {abs(50 * a1['slope']):.0f} and
{abs(50 * a2['slope']):.0f} mV. The measured bend ({mv(a1['meas'])} and {mv(a2['meas'])} mV) is reached with
{need_txt(t1)} in {t1} and {need_txt(t2)} in {t2}{': offsets of opposite sign, so no single tip gives both' if opposite else ''}.
The tip moves the (+⁵) reference with the wall: there the flat band is a top-layer state and follows the whole
tip-induced potential, inside the wall it follows about half, so an offset that pulls the wall peak further also
pulls the reference, and the bend changes much less than either peak.
The lines were recorded on different days, possibly with different tips, but each line's own (+⁵) fit holds its tip
near zero: the residual at −0.2 / +0.2 eV is {res02[t1][0]:.2f} / {res02[t1][1]:.2f} in {t1} and
{res02[t2][0]:.2f} / {res02[t2][1]:.2f} in {t2}, against {scan[t1].get(0.0):.2f} and {scan[t2].get(0.0):.2f} at zero.
{f"An offset of −0.2 eV would also move the (+⁵) peak itself under the tip by {mv(far_shift[t1])} mV ({t1}) and {mv(far_shift[t2])} mV ({t2}) at fixed doping. " if None not in far_shift.values() else ""}The
vacuum work-function difference of a Pt tip (+1.05 eV) lies far outside both the computed range and what the spectra
allow.</p>
<div class="tablewrap"><table>
<tr><th>δ</th><th>{t1}: top-layer bump, wall / (+⁵)</th><th>{t1}: tip part</th><th>{t1}: bend</th>
<th>{t2}: top-layer bump, wall / (+⁵)</th><th>{t2}: tip part</th><th>{t2}: bend</th></tr>
{''.join(rows_html)}
</table></div>
<figure><img src="{img_prefix}stacking_wall_cpot_sensitivity.png" alt="model bend at the wall against the tip work-function offset">
<figcaption>Flat-band bend at the measured wall position against the offset δ of the tip work function from the
sample's: k·p wall model with the far-field potentials of the hybrid and the tip-induced bump (blue), the far-field
part alone (dotted) and the measurement (dashed). Grey band: the range of δ allowed by the (+⁵) fits.</figcaption></figure>
"""


def lateral_numbers(tag):
    """Numbers of the lateral (3D) solve: the (+4-) far field, the tip-free step against the
    columns, the apex bending at the k.p positions against the local approximation, the width
    of the tip-induced part and the k.p peaks (eV, aligned like the bump table)."""
    path = os.path.join(EXAMPLES, f"stacking_wall_lateral_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        st = json.load(fh)
    s, runs, kp = st["settings"], st["runs"], st.get("kp", {})
    if not runs:
        return None
    cols = np.asarray(st["columns"])                                   # (n_s, 6)
    s_grid = np.asarray(s["s_grid"])
    ell = s["ell"]
    r0 = min(runs.values(), key=lambda r: abs(r["x_tip"]))
    x, U0 = np.asarray(r0["x"]), np.asarray(r0["U0"])
    col_x = np.array([np.interp(0.5 * (1 - np.tanh(x / ell)), s_grid, cols[:, l]) for l in range(6)])
    step_dev = float(np.max(np.abs(U0 - col_x)))
    # width of the tip-induced part of the top layer with the tip at the wall centre
    dU = np.asarray(r0["U"])[0] - U0[0]
    dU = dU * np.sign(dU[int(np.argmin(np.abs(x - r0["x_tip"])))])     # bending down counts as positive here
    half = 0.5 * dU.max()
    above = x[dU >= half]
    fwhm = float(above.max() - above.min()) if above.size else float("nan")
    left = float(r0["x_tip"] - above.min()) if above.size else float("nan")
    right = float(above.max() - r0["x_tip"]) if above.size else float("nan")
    hyb_path = os.path.join(EXAMPLES, f"stacking_wall_hybrid_{tag}.npz")
    shift = 0.0
    if os.path.exists(hyb_path):
        with np.load(hyb_path) as f:
            E, xh, top = f["energy"], f["x"] / 10.0 + s["x0"], f["ldos"][-1]
        far = xh > xh.max() - 30.0
        sel = (E >= -0.12) & (E <= 0.12)
        shift = s["E_fb"] - float(E[sel][np.argmax(top[sel][:, far].mean(axis=1))])
    def centroid(run):
        E, y = np.asarray(run["energy"]), np.asarray(run["ldos"])
        sel = (E >= run["peak"] - 0.03) & (E <= run["peak"] + 0.03)
        yb = y[sel] - y[sel].min()
        return float(np.sum(E[sel] * yb) / np.sum(yb))
    rows = []
    for r in sorted(runs.values(), key=lambda r: r["x_tip"]):
        a, b = kp.get(f"{r['x_tip']:+.1f}_tip"), kp.get(f"{r['x_tip']:+.1f}_none")
        rows.append(dict(x=r["x_tip"], s=r["s"], dU=r["dU_apex"], dU_local=r["dU_local"], dU_uniform=r.get("dU_uniform"),
                         peak=(a["peak"] + shift) if a else None, peak_ref=(b["peak"] + shift) if b else None,
                         height=a["height"] if a else None, height_ref=b["height"] if b else None,
                         dcentroid=(centroid(a) - centroid(b)) if (a and b) else None))
    return dict(settings=s, sf_top=float(cols[-1, 0]), step_dev=step_dev, fwhm=fwhm, left=left, right=right,
                rows=rows, n_positions=len(runs), seconds=float(np.mean([r["seconds"] for r in runs.values()])),
                nodes=int(np.mean([r["n_free"] for r in runs.values()])))


def lateral_text(states, img_prefix=""):
    nums = {t: lateral_numbers(t) for t in states}
    if any(v is None for v in nums.values()):
        return ""
    table = []
    for t, v in nums.items():
        h_max = max((r["height"] for r in v["rows"] if r["height"] is not None), default=0.0)
        for r in v["rows"]:
            if r["peak"] is None:
                continue                                             # positions without k.p spectra are in the figure only
            faded = r["height"] < 0.15 * h_max or abs(r["peak"] - v["settings"]["E_fb"]) > 0.06
            peaks = ("flat-band peak faded" if faded else
                     f"{1e3 * r['peak_ref']:+.0f} / {1e3 * r['peak']:+.0f} mV (centroid {1e3 * r['dcentroid']:+.1f})")
            ref = r["dU_uniform"] if r["dU_uniform"] is not None else r["dU_local"]
            table.append(f"<tr><td>{t}</td><td>{r['x']:+.0f} nm</td><td>{r['s']:.2f}</td>"
                         f"<td>{1e3 * ref[0]:+.1f} / {1e3 * r['dU'][0]:+.1f} meV</td>"
                         f"<td>{1e3 * ref[1]:+.1f} / {1e3 * r['dU'][1]:+.1f} meV</td><td>{peaks}</td></tr>")
    t1, t2 = list(nums)[:2]
    a, b = nums[t1], nums[t2]
    sa, sb = a["settings"], b["settings"]
    rel, rel_engine = [], []
    for v in nums.values():
        for r in v["rows"]:
            ref = r["dU_uniform"] if r["dU_uniform"] is not None else r["dU_local"]
            if abs(ref[0]) > 1e-3:
                rel.append(r["dU"][0] / ref[0] - 1.0)
            if r["dU_uniform"] is not None and abs(r["dU_uniform"][0]) > 1e-3:
                rel_engine.append(r["dU_local"][0] / r["dU_uniform"][0] - 1.0)
    rel_txt = f"{100 * min(rel):+.0f} to {100 * max(rel):+.0f} %" if rel else "–"
    engine_txt = (f" (The engine's local-approximation sweeps differ from the uniform-stack solves of the same code by "
                  f"{100 * min(rel_engine):+.0f} to {100 * max(rel_engine):+.0f} %: their fixed charge per layer differs by 20 % "
                  f"in the top layer, and the bending scales with the inverse density of states at E<sub>F</sub>, which on the "
                  f"flank of the flat band changes by about 20 % per meV of Fermi-level position. That sensitivity is a property "
                  f"of the rigid-band model, not of the geometry.)" if rel_engine else "")
    kp_rows = [r for v in nums.values() for r in v["rows"] if r["peak"] is not None and r["peak_ref"] is not None
               and r["height"] >= 0.15 * max(q["height"] for q in v["rows"] if q["height"] is not None)
               and abs(r["peak"] - v["settings"]["E_fb"]) <= 0.06]
    worst = max((abs(r["peak"] - r["peak_ref"]) for r in kp_rows), default=None)
    worst_c = max((abs(r["dcentroid"]) for r in kp_rows if r["dcentroid"] is not None), default=None)
    worst_txt = (f"the maximum of the k·p flat-band peak under the tip moves by at most {1e3 * worst:.1f} mV (its energy grid is "
                 f"2.5 mV) and its centroid by at most {1e3 * worst_c:.1f} mV"
                 if worst is not None and worst_c is not None else "the k·p spectra are not finished")
    # tip-free k.p peaks of the lateral potentials against those of the hybrid (bump runs without the bump), and the
    # Gaussian bump's largest centroid change, at the same positions
    offs, bump_c = [], []
    for t, v in nums.items():
        bn = bump_numbers(t)
        if bn is None:
            continue
        by_x = {r["x"]: r for r in bn["rows"]}
        for r in kp_rows:
            if r in v["rows"] and r["x"] in by_x and by_x[r["x"]]["peak_ref"] is not None:
                offs.append(r["peak_ref"] - by_x[r["x"]]["peak_ref"])
                if by_x[r["x"]].get("dcentroid") is not None:
                    bump_c.append(abs(by_x[r["x"]]["dcentroid"]))
    if offs and (all(o < 0 for o in offs) or all(o > 0 for o in offs)):
        offs_txt = (f"{1e3 * min(map(abs, offs)):.1f} to {1e3 * max(map(abs, offs)):.1f} mV "
                    f"{'below' if offs[0] < 0 else 'above'}")
    elif offs:
        offs_txt = f"within {1e3 * max(map(abs, offs)):.1f} mV of"
    else:
        offs_txt = "a few mV from"
    bump_txt = (f", against {1e3 * max(bump_c):.1f} mV for the Gaussian bump at the same positions: the bump was an adequate "
                f"stand-in for the lateral tip potential" if bump_c else "")
    apex = [abs(r["dU"][0]) for v in nums.values() for r in v["rows"] if r["peak"] is not None]
    abs_lat = max(abs(r["dU"][0] - r["dU_uniform"][0]) for v in nums.values() for r in v["rows"] if r["dU_uniform"] is not None)
    return f"""<h3>The lateral problem</h3>
<p>The local approximation was dropped last. The Poisson equation was solved in three dimensions with the wall in
it (bandbending branch <code>lateral-solver</code>, <code>engine/lateral.py</code>; driver
<code>stacking_wall_lateral.py</code>): a Cartesian finite-volume grid with mirror symmetry across the line, the
same hyperbolic tip (R = {sa['radius']:.0f} nm, d = {sa['separation']:.1f} nm, CPot = 0) as a Dirichlet body, the
six layers as rigid-band charge sheets whose density follows the zero-field layer DOS of the local stack
(<code>WallTB</code> tables at s = 0, 0.15, …, 1 along the fitted profile), the oxide and the gate below, and far-field
boundaries from the tip-free columns of the local stack plus the engine's axisymmetric tip field. One energy
reference serves all stacks: the fixed charge per layer makes the (+⁵) stack neutral with its Fermi level at
μ0 = {1e3 * sa['mu0']:+.0f} meV ({t1}) and {1e3 * sb['mu0']:+.0f} meV ({t2}) from its neutrality point, as measured, and
the (+⁴−) stack, joined to it, shares that Fermi level and the gate; its top-layer potential far from the wall comes
out at {1e3 * a['sf_top']:+.0f} meV ({t1}) and {1e3 * b['sf_top']:+.0f} meV ({t2}). The local approximation, with its
own Fermi level per stack and the engine's slightly different layer charges, placed the (+⁴−)-side potentials a few meV
away, which is why the tip-free k·p peaks of the table sit {offs_txt} those of the hybrid. The tip sits at the
bias of the measured wall peak, {1e3 * sa['bias']:+.0f} mV ({t1}) and {1e3 * sb['bias']:+.0f} mV ({t2}), and was placed
at {a['n_positions']} positions across the wall; a solve has about {a['nodes'] / 1e3:.0f} thousand unknowns and takes
{a['seconds']:.0f} s (Newton with algebraic-multigrid conjugate gradients).</p>
<p>Three results. First, without the tip the built-in step of the layer potentials across the wall (top layer
0 to {1e3 * a['sf_top']:+.0f} meV in {t1}, 0 to {1e3 * b['sf_top']:+.0f} meV in {t2}) follows the columns of the local
stack to within {1e3 * a['step_dev']:.1f} meV ({t1}, wall length {sa['ell']:.1f} nm) and {1e3 * b['step_dev']:.1f} meV
({t2}, {sb['ell']:.1f} nm): the stack screens laterally over a length well below the wall width, so the local treatment
of the far field was adequate. Second, the tip-induced bending under the apex differs from the local approximation
by {rel_txt}, at most {1e3 * abs_lat:.1f} meV, across the positions solved (table and figure; the reference is the
uniform stack at s(x_tip) solved with the same code): the field of the tip averages the local response over its
footprint, which smooths the kinks of the local curve where the top-layer density of states changes quickly with s,
and reaches into the weakly screening (+⁴−) region, but the result stays a few-meV effect.{engine_txt} The tip-induced part of the top-layer potential with
the tip at the wall centre has a full width at half maximum of {a['fwhm']:.0f} nm ({t1}; {a['left']:.0f} nm towards
(+⁴−), {a['right']:.0f} nm towards (+⁵)) against the 12.5 nm of the Gaussian bump used above. Third, fed into the
k·p wall model in place of the Gaussian bump, {worst_txt}{bump_txt}. The conclusion of the local approximation
stands: with the lateral problem solved, the self-consistent tip-induced bending of the top layer at the bias of the
wall peak is {1e3 * min(apex):.0f} to {1e3 * max(apex):.0f} meV at the k·p positions, it moves the flat band under the tip
by up to {1e3 * worst_c:.0f} mV in the measured direction, most of which it also does to the (+⁵) reference at its own
bias, and the measured 35 to 55 mV bend is not a tip-gating effect of this model.</p>
<div class="tablewrap"><table>
<tr><th>line</th><th>tip from the wall centre</th><th>s at the tip</th><th>apex bending, layer 1: uniform / lateral</th><th>layer 2: uniform / lateral</th><th>k·p peak tip-free / with the tip (change)</th></tr>
{''.join(table)}
</table></div>
<figure><img src="{img_prefix}stacking_wall_lateral.png" alt="lateral solve: layer potentials, apex bending and k.p peak against the tip position">
<figcaption>Lateral solve. Left: layer potentials along the line without the tip (blue) and the top layer with the
tip at several positions (orange). Middle: tip-induced bending under the apex against the tip position, lateral
solve (solid) and local approximation (dashed), layers 1 and 2. Right: flat-band peak under the tip from the k·p
model with the lateral potentials, with and without the tip, and the Gaussian-bump result.</figcaption></figure>
"""


def epsfit_text(img_prefix=""):
    """Section: refit of the (+5) spectrum of 940 over interlayer eps and starting doping, uniformly
    doped and polarized (examples/stacking_wall_epsfit_940.json, stacking_wall_epsfit/)."""
    path = os.path.join(EXAMPLES, "stacking_wall_epsfit_940.json")
    if not os.path.exists(path):
        return ""
    with open(path) as fh:
        s = json.load(fh)
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    m4, m5 = s["measured"]["features4"], s["measured"]["features5"]
    prev, ub, uf = s["previous"], s["uniform"]["best"], s["uniform"]["flat_band_in_place"]
    pb = s["polarized"]["best"]
    near = [n for n in s["polarized"]["near"] if (n["eps"], n["dEF"]) != (pb["eps"], pb["dEF"])][:2]
    U = pb["U_far_minus_EF"]
    n_valid = sum(c["valid"] for c in s["polarized"]["cells"])
    dev_u = [abs(1e3 * (uf['features'][k] - m4[k])) for k in (0, 2, 3)]
    dev_p = [abs(1e3 * (a - b)) for a, b in zip(pb['features'], m5)]
    names = ["lower side-band peak", "flat-band peak", "bottom-band onset", "upper side-band edge 1", "upper side-band edge 2"]
    rows = "".join(f"<tr><td>{n}</td><td>{mv(m)} mV</td><td>{mv(t)} mV</td><td>{mv(f)} mV</td></tr>"
                   for n, m, t, f in zip(names, m5, pb["features"], pb["features_free"]))
    near_txt = " and ".join(f"ε = {n['eps']:g}, ΔE<sub>F</sub> = {1e3 * n['dEF']:.0f} meV ({1e3 * n['rms']:.1f} mV)" for n in near)
    return f"""<h3>Refit of the (+⁵) spectrum of 940: interlayer ε, starting doping and SiO₂ charge</h3>
<p>The (+⁵) fit above varied only the contact potential, with ε = 2.6 between the layers and the doping set by the
tip-free flat-band position. It puts the flat band within 8 mV, but not the side bands: the lower side-band peak,
the flat-band peak and the two upper side-band edges (measured {mv(m4[0])}, {mv(m4[1])}, {mv(m4[2])} and {mv(m4[3])} mV)
miss by {1e3 * prev['rms']:.0f} mV rms. Two models were therefore scanned over the interlayer permittivity ε (1 to 25)
and the starting doping ΔE<sub>F</sub> (the Fermi level of the field-free doped stack above its neutrality point), 48
self-consistent tip sweeps each, with the same tip set to bend nothing at zero bias, as the asymmetry of the bends
suggests (<code>stacking_wall_epsfit/</code>; the draft Poisson grid, which agrees with the normal one to 0.01 meV).</p>
<p><b>Uniformly doped stack.</b> With no other charge, the doped stack is field-free far from the tip. The best match is
{1e3 * ub['rms']:.0f} mV at ΔE<sub>F</sub> = {1e3 * ub['dEF']:.0f} meV, but with the flat band at {mv(ub['features'][1])} mV;
with the flat band in place (ε = {uf['eps']:g}, ΔE<sub>F</sub> = {1e3 * uf['dEF']:.0f} meV) it is {1e3 * uf['rms']:.0f} mV,
with the three side-band features {min(dev_u):.0f} to {max(dev_u):.0f} mV too low. ε changes this by a few mV at most.
A field-free stack centres its side bands on the flat band, while the measured lower side-band peak and first upper
edge are centred {1e3 * (0.5 * (m4[0] + m4[2]) - m4[1]):.0f} mV above it: the measured spectrum asks for a field across
the stack.</p>
<p><b>Polarized stack.</b> The doping is then partly compensated by a fixed charge n<sub>sub</sub> in the top 2 nm of
the SiO₂, chosen for every (ε, ΔE<sub>F</sub>) as the value whose tip-free spectrum fits best, and the fit takes a fifth
feature, the onset of the empty bottom-surface band ({mv(m5[2])} mV). The best match under the tip is
{1e3 * pb['rms']:.1f} mV at ε = {pb['eps']:g}, ΔE<sub>F</sub> = {1e3 * pb['dEF']:.0f} meV and n<sub>sub</sub> =
{f"{pb['nsub'] / 1e13:.3f}".replace("-", "−")}×10¹³ cm⁻² ({1e3 * pb['rms_free']:.1f} mV for the same stack without the tip). The minimum is a
valley rather than a point, {near_txt} come within 0.3 mV, so the scan fixes the combination of a higher doping and a
larger ε, not each. Far from the tip the stack is neutral (within {abs(pb['carriers']) / 1e10:.0f}×10¹⁰ cm⁻²) and
polarized: the top layer {mv(U[0])} meV and the bottom layer {mv(U[-1])} meV from E<sub>F</sub>, the filled top flat
band below E<sub>F</sub> and the empty bottom one above it. Four of the five features fall within {min(dev_p[1:]):.0f} to
{max(dev_p[1:]):.0f} mV; the lower
side-band peak stays {abs(1e3 * (pb['features'][0] - m5[0])):.0f} mV too deep under the tip, where the tip bends the top
layer by tens of meV at negative bias (it is {abs(1e3 * (pb['features_free'][0] - m5[0])):.0f} mV too shallow without the
tip). At low ε the weakly screening stack lets the tip pull the flat band far too deep or out of its search window
({len(s['polarized']['cells']) - n_valid} of the {len(s['polarized']['cells'])} cells lose a feature, dashes in the map). The fitted ε of 10 to 25 is not that of graphite's
σ bands (2.6); it stands for screening the model lacks, such as states in the gap or charges that move. The wall and
(+⁴−) calculations above keep the uniformly doped far field.</p>
<div class="tablewrap"><table>
<tr><th>feature</th><th>measured</th><th>polarized stack under the tip</th><th>same stack without the tip</th></tr>
{rows}
</table></div>
<figure><img src="{img_prefix}stacking_wall_epsfit_940.png" alt="interlayer permittivity and starting doping scans of the (+5) spectrum of 940">
<figcaption>(a) Feature rms under the tip of the polarized stack over ε and ΔE<sub>F</sub> (n<sub>sub</sub> fitted per cell
without the tip); the frame marks the best cell. (b) The best polarized stack under the tip (solid) and without it
(dashed) against the measured (+⁵) spectrum of 940. (c, d) The same for the uniformly doped stack: the best cell and the
best cell with the flat band in place. Dots: the fitted features.</figcaption></figure>
"""


def eps26_fit_text(img_prefix=""):
    """Section: refit of the (+5) spectrum of 940 at eps = 2.6 with Delta E_F and n_sub free
    (stacking_wall_epsfit/fixeps.py, analyze_fixeps.py; examples/stacking_wall_eps26_940.json)."""
    path = os.path.join(EXAMPLES, "stacking_wall_eps26_940.json")
    if not os.path.exists(path):
        return ""
    with open(path) as fh:
        s = json.load(fh)
    p15 = os.path.join(EXAMPLES, "stacking_wall_epsfit_940.json")
    b15 = json.load(open(p15))["polarized"]["best"] if os.path.exists(p15) else None
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    sci = lambda v: f"{v / 1e13:.3f}".replace("-", "−")  # noqa: E731
    m, b, fb = s["measured"], s["best"], s["flat_band_in_place"]
    valid = [c for c in s["cells"] if c["valid"]]
    few = [c["features"][0] for c in valid if c["holes"] <= 1.01e11]
    many = [c["features"][0] for c in valid if c["holes"] >= 1.49e12]
    fb_many = [c["features"][1] for c in valid if c["holes"] >= 5.9e11]
    dEFs = sorted({c["dEF"] for c in s["cells"]})
    holes = sorted({c["holes"] for c in s["cells"]})
    hole_of = lambda r: next(c["holes"] for c in s["cells"] if abs(c["nsub"] - r["nsub"]) < 1e6)   # noqa: E731
    names = ["lower side-band peak", "flat-band peak", "bottom-band onset", "upper side-band edge 1", "upper side-band edge 2"]
    cols = [("best by the peak positions", b["features"]), ("best with the flat band within 10 mV", fb["features"])]
    if b15:
        cols.append(("ε = 15 (polarized fit above)", b15["features"]))
    head = "".join(f"<th>{c[0]}</th>" for c in cols)
    body = "".join(f"<tr><td>{n}</td><td>{mv(m[k])} mV</td>" + "".join(f"<td>{mv(c[1][k])} mV</td>" for c in cols) + "</tr>"
                   for k, n in enumerate(names))
    rms_row = "<tr><td>rms of the five</td><td></td>" + "".join(
        f"<td>{1e3 * r:.1f} mV</td>" for r in [b["rms"], fb["rms"]] + ([b15["rms"]] if b15 else [])) + "</tr>"
    return f"""<h3>Refit at ε = 2.6: starting doping and SiO₂ charge</h3>
<p>The polarized fit above needs an interlayer permittivity of 10 to 25, far above the 2.6 of graphite's σ bands; values
above about 3 stand for screening the model does not contain. The large ε also suppresses the bend at the wall: in the
local approximation most of the bend is the charge balance of the wall stack, a field between its layers whose
potential steps scale as 1/ε (tip-free, from s = 0 to 0.75 the top layer of the uniformly doped stack moves by about
−44 meV at ε = 2.6 and −19 meV at ε = 15, that of the polarized stack by −68 and −13 meV). ε was therefore fixed at 2.6
and the starting doping ΔE<sub>F</sub> and the SiO₂ charge n<sub>sub</sub> fitted under the same tip, set to bend
nothing at zero bias (<code>stacking_wall_epsfit/fixeps.py</code>). n<sub>sub</sub> enters through the carrier
density it leaves in the stack: for each ΔE<sub>F</sub> the neutral point is found by bisection, and n<sub>sub</sub>
then leaves {holes[0] / 1e10:g} to {holes[-1] / 1e10:g}×10¹⁰ holes per cm² (ΔE<sub>F</sub> =
{1e3 * dEFs[0]:.0f} to {1e3 * dEFs[-1]:.0f} meV, {len(s['cells'])} sweeps). Cells whose lower side-band "peak" is a ripple
(prominence below half the measured {s['prominence_measured']:.3f} of the flat-band peak) or where a feature is missing are
not counted.</p>
<p>The holes screen the tip. With up to 10¹¹ holes the tip pushes the lower side band to {mv(min(few))} to
{mv(max(few))} mV; with 1.5×10¹² it stays near its tip-free position, {mv(min(many))} to {mv(max(many))} mV
(measured {mv(m[0])} mV). But the holes empty part of the top flat band and pin it near E<sub>F</sub>: from
6×10¹¹ holes on, the flat-band peak sits at {mv(min(fb_many))} to {mv(max(fb_many))} mV against the measured
{mv(m[1])} mV. No cell gets both. The best match of the five peak positions is {1e3 * b['rms']:.1f} mV
(ΔE<sub>F</sub> = {1e3 * b['dEF']:.0f} meV, n<sub>sub</sub> = {sci(b['nsub'])}×10¹³ cm⁻², {hole_of(b) / 1e10:g}×10¹⁰
holes), with the flat band {abs(1e3 * (b['features'][1] - m[1])):.0f} mV too shallow and the whole curve far off
(shape residual {b['residual']:.2f}). With the flat band within 10 mV of the measured one the best is
{1e3 * fb['rms']:.1f} mV (ΔE<sub>F</sub> = {1e3 * fb['dEF']:.0f} meV, n<sub>sub</sub> = {sci(fb['nsub'])}×10¹³ cm⁻²,
{hole_of(fb) / 1e10:g}×10¹⁰ holes), which is also the best overall shape at ε = 2.6 (residual {fb['residual']:.2f}); its
lower side band is {abs(1e3 * (fb['features'][0] - m[0])):.0f} mV too deep, as at ε = 15. This flat-band fit is
{1e3 * (fb['rms'] - b15['rms']) if b15 else 0:.1f} mV worse than the ε = 15 fit by the peak positions; it is the
parameter set carried across the wall below.</p>
<div class="tablewrap"><table>
<tr><th>feature</th><th>measured</th>{head}</tr>
{body}
{rms_row}
</table></div>
<figure><img src="{img_prefix}stacking_wall_eps26_940.png" alt="eps = 2.6 refit of the (+5) spectrum of 940">
<figcaption>(a) Feature rms under the tip at ε = 2.6 over the starting doping and the hole density left by
n<sub>sub</sub>; frames: the best cell (orange) and the best with the flat band within 10 mV (blue). (b) The two cells
under the tip against the measured (+⁵) spectrum of 940 (dashed: the best cell without the tip).</figcaption></figure>
"""


def eps26_series_text(img_prefix=""):
    """Section: line spectrum 940 with the eps = 2.6 flat-band fit, local approximation
    (stacking_wall_polar_series.py --set eps26_fb)."""
    n = polar_series_numbers("_eps26_fb")
    if n is None:
        return ""
    st, rows = n["state"], n["rows"]
    p = st["params"]
    fit = st["fit"]
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    mv1 = lambda v: ("0.0" if abs(v) < 5e-5 else f"{1e3 * v:+.1f}".replace("-", "−"))  # noqa: E731
    dx = lambda s: (0.0 if abs(s - 0.5) < 1e-9 else fit["ell"] * np.arctanh(1.0 - 2.0 * s)) if 0.0 < s < 1.0 else None  # noqa: E731
    walls = [r["s"] for r in rows if 0.0 < r["s"] < 1.0]
    table = "".join(
        f"<tr><td>{r['s']:g}</td><td>{'' if dx(r['s']) is None else ('0 nm' if dx(r['s']) == 0.0 else f'{dx(r[chr(115)]):+.1f} nm'.replace('-', '−'))}</td>"
        f"<td>{mv1(r['U1'])} meV</td>"
        + ("<td colspan=\"3\">flat band faded</td></tr>" if r["faded"] else
           f"<td>{mv1(r['peak'])} mV</td><td>{100 * r['height'] / rows[0]['height']:.0f} %</td><td>{mv1(r['bend_pk'])} meV</td></tr>")
        for r in rows)
    alive = [r for r in rows if not r["faded"]]
    last = max(alive, key=lambda r: r["s"])
    r0, r1 = rows[0], rows[-1]
    bend = {k: st[f"extreme_{k}"] - st[f"far_{k}"] for k in ("measured", "polarized")}
    dist = np.asarray(st["track"]["dist"])
    e = {k: np.asarray(st["edge"][k]) for k in ("measured", "polarized")}
    far_l = dist < 10.0
    j5 = int(np.argmin(np.abs(dist - (fit["x0"] - 5.0))))
    j2 = int(np.argmin(np.abs(dist - (fit["x0"] - 2.0))))
    holes = -json.load(open(os.path.join(EXAMPLES, "stacking_wall_eps26_940.json")))["flat_band_in_place"]["carriers"]
    tf = st.get("tipfree_peaks", {})
    # the uniformly doped stack of the first calculation: mean potential of the top three layers, s = 0 and 0.75
    main = json.load(open(os.path.join(EXAMPLES, "stacking_wall_tibb_940.json")))
    u_runs = {0.0: main["scan"][f"{main['best_cpot']:+.3f}"]}
    for f in [os.path.join(EXAMPLES, "stacking_wall_tibb_940.json")] + sorted(glob.glob(os.path.join(EXAMPLES, "stacking_wall_tibb_940_s*.json"))):
        for k, r in json.load(open(f)).get("runs", {}).items():
            if k.startswith("wall_s"):
                u_runs[float(r["s"])] = r
    um = lambda r: float(np.mean(np.asarray(r["sweep"]["U_far"])[0][:3] - r["sweep"]["mu0"]))   # noqa: E731
    uni_mean = um(u_runs[0.75]) - um(u_runs[0.0])
    return f"""<h3>Line spectrum 940 with the ε = 2.6 flat-band fit</h3>
<p>The ε = 2.6 cell with the flat band in place (ΔE<sub>F</sub> = {1e3 * p['dEF']:.0f} meV, n<sub>sub</sub> =
{f"{p['nsub'] / 1e13:.3f}".replace("-", "−")}×10¹³ cm⁻², {holes / 1e10:.0f}×10¹⁰ holes) was carried across the wall in
the local approximation as above: the same fixed charge per layer, n<sub>sub</sub> and tip for every stack. The
{len(walls)} wall stacks are spaced by about 1 nm of the fitted profile within 3 nm of the wall centre and more widely
beyond (s = {", ".join(f"{s:g}" for s in sorted(walls))}).</p>
<p>The charge balance of the wall stack is three times larger than at ε = 15: far from the tip the top layer moves
from {mv1(r0['U1'])} meV ((+⁵)) to {mv1(r1['U1'])} meV ((+⁴−)). The flat-band peak hardly follows. Under the tip it
stays at {mv1(min(r['peak'] for r in alive))} to {mv1(max(r['peak'] for r in alive))} mV up to s = {last['s']:g}
({abs(dx(last['s'])):.0f} nm on the (+⁴−) side of the centre), a bend of {mv1(bend['polarized'])} mV against the measured
{mv1(bend['measured'])} mV, while its height falls to {100 * last['height'] / r0['height']:.0f} % of the (+⁵) value; without
the tip it moves from {mv1(tf[f"{r0['s']:.2f}"])} to {mv1(tf[f"{last['s']:.2f}"])} mV. The reason is where the shift goes.
With E<sub>F</sub> in the gap of a neutral, polarized stack the wall stack cannot move as a whole: its charge rearrangement
is a dipole inside the stack. Between s = 0 and s = {last['s']:g} the top layer goes down by
{abs(1e3 * (last['U'][0] - r0['U'][0])):.0f} meV while layers 3 to 6 go up by
{1e3 * min(last['U'][k] - r0['U'][k] for k in range(2, 6)):.0f} to {1e3 * max(last['U'][k] - r0['U'][k] for k in range(2, 6)):.0f} meV,
and the mean of the top three layers, over which the flat band near its peak is spread, moves by only
{mv1(np.mean(last['U'][:3]) - np.mean(r0['U'][:3]))} meV. In the uniformly doped stack of the first calculation
E<sub>F</sub> lies in the bands and each wall stack takes its own Fermi level against the fixed charges; there all layers
move down together, the mean of the top three by {mv1(uni_mean)} meV at s = 0.75, and the flat band follows. The tip
does not add to the bend: its bending of the top layer at the peak bias falls from {mv1(r0['bend_pk'])} meV on the (+⁵)
side to {mv1(last['bend_pk'])} meV at s = {last['s']:g}, so the peak under the tip moves even less than without it. On
the (+⁴−) side the lowest edge above E<sub>F</sub> lies at
{mv(e['polarized'][far_l].mean())} mV far from the wall (measured {mv(e['measured'][far_l].mean())} mV). The measured edge
is already at {mv(e['measured'][j5])} mV 5 nm before the centre, where the model is still at {mv(e['polarized'][j5])} mV;
the model's edge rises only within the last 3 nm, to {mv(e['polarized'][j2])} mV 2 nm before the centre.</p>
<div class="tablewrap"><table>
<tr><th>s</th><th>distance from the wall centre</th><th>top layer far from the tip, U<sub>1</sub> − E<sub>F</sub></th>
<th>flat-band peak</th><th>its height (of the (+⁵) peak)</th><th>tip-induced bending at the peak bias</th></tr>
{table}
</table></div>
<div class="figs">
<figure><img src="{img_prefix}stacking_wall_polar_eps26_fb_map_940.png" alt="line spectrum 940, measured and the eps = 2.6 flat-band fit">
<figcaption>Line spectrum 940: measured dI/dV(V, x) and its bias derivative (left) against the ε = 2.6 flat-band fit in
the local approximation (right).</figcaption></figure>
<figure><img src="{img_prefix}stacking_wall_polar_eps26_fb_spectra_940.png" alt="spectra of the eps = 2.6 wall stacks and the flat-band peak along the line">
<figcaption>Left: top-layer dI/dV of the ε = 2.6 wall stack at each s under the same tip. Right: the flat-band peak
along the line, measured and for the four models (the ε = 2.6 flat-band fit, the uniformly doped stack of the first
calculation, the polarized stack at ε = 15).</figcaption></figure>
</div>
"""


def eps26_peaks_series_text(img_prefix=""):
    """Section: line spectrum 940 with the eps = 2.6 best peak fit, local approximation
    (stacking_wall_polar_series.py --set eps26_peaks)."""
    n = polar_series_numbers("_eps26_peaks")
    if n is None:
        return ""
    st, rows = n["state"], n["rows"]
    p, fit = st["params"], st["fit"]
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    mv1 = lambda v: ("0.0" if abs(v) < 5e-5 else f"{1e3 * v:+.1f}".replace("-", "−"))  # noqa: E731
    dx = lambda s: (0.0 if abs(s - 0.5) < 1e-9 else fit["ell"] * np.arctanh(1.0 - 2.0 * s)) if 0.0 < s < 1.0 else None  # noqa: E731
    table = "".join(
        f"<tr><td>{r['s']:g}</td><td>{'' if dx(r['s']) is None else ('0 nm' if dx(r['s']) == 0.0 else f'{dx(r[chr(115)]):+.1f} nm'.replace('-', '−'))}</td>"
        f"<td>{mv1(r['U1'])} meV</td>"
        + ("<td colspan=\"3\">flat band faded</td></tr>" if r["faded"] else
           f"<td>{mv1(r['peak'])} mV</td><td>{100 * r['height'] / rows[0]['height']:.0f} %</td><td>{mv1(r['bend_pk'])} meV</td></tr>")
        for r in rows)
    alive = [r for r in rows if not r["faded"]]
    r0, last = rows[0], max(alive, key=lambda r: r["s"])
    faded_s = min(r["s"] for r in rows if r["faded"])
    tf = st.get("tipfree_peaks", {})
    tip_vs_free = max(abs(r["peak"] - tf[f"{r['s']:.2f}"]) for r in alive if f"{r['s']:.2f}" in tf)
    bend = {k: st[f"extreme_{k}"] - st[f"far_{k}"] for k in ("measured", "polarized")}
    dist = np.asarray(st["track"]["dist"])
    e = {k: np.asarray(st["edge"][k]) for k in ("measured", "polarized")}
    far_l = dist < 10.0
    j5 = int(np.argmin(np.abs(dist - (fit["x0"] - 5.0))))
    j2 = int(np.argmin(np.abs(dist - (fit["x0"] - 2.0))))
    eps26 = json.load(open(os.path.join(EXAMPLES, "stacking_wall_eps26_940.json")))["best"]
    return f"""<h3>Line spectrum 940 with the ε = 2.6 best peak fit</h3>
<p>The ε = 2.6 cell with the lowest rms of the five peak positions (ΔE<sub>F</sub> = {1e3 * p['dEF']:.0f} meV,
n<sub>sub</sub> = {f"{p['nsub'] / 1e13:.3f}".replace("-", "−")}×10¹³ cm⁻², {-eps26['carriers'] / 1e12:.1f}×10¹² holes) was
carried across the wall on the same s grid as the flat-band fit, with the same construction. Its flat-band peak does not
move: it stays at {mv1(min(r['peak'] for r in alive))} to {mv1(max(r['peak'] for r in alive))} mV up to s = {last['s']:g}
(a bend of {mv1(bend['polarized'])} mV against the measured {mv1(bend['measured'])} mV) and fades from there, to
{100 * last['height'] / r0['height']:.0f} % of its (+⁵) height at s = {last['s']:g} and below the background at
s = {faded_s:g}. {"The peaks with and without the tip are identical on the 2.5 mV grid" if tip_vs_free < 1e-6 else f"The peaks with and without the tip differ by at most {1e3 * tip_vs_free:.1f} mV"}, and the tip bends the
top layer at the peak bias by {mv1(min(r['bend_pk'] for r in alive))} to {mv1(max(r['bend_pk'] for r in alive))} meV:
the holes put states at E<sub>F</sub>, which screen the tip and hold the flat band at E<sub>F</sub>, while the top layer
far from the tip moves from {mv1(r0['U1'])} to {mv1(last['U1'])} meV and the mean of the top three layers by
{mv1(np.mean(last['U'][:3]) - np.mean(r0['U'][:3]))} meV. Neither ε = 2.6 parameter set bends the flat band: with
E<sub>F</sub> in the gap (flat-band fit) the charge rearrangement of the wall is a dipole the flat band barely sees, with
holes at E<sub>F</sub> (best peak fit) the band stays at E<sub>F</sub>. On the (+⁴−) side the lowest edge above
E<sub>F</sub> lies at {mv(e['polarized'][far_l].mean())} mV far from the wall (measured {mv(e['measured'][far_l].mean())} mV);
5 nm before the centre the model is at {mv(e['polarized'][j5])} mV against the measured {mv(e['measured'][j5])} mV, and it
rises to {mv(e['polarized'][j2])} mV 2 nm before the centre.</p>
<div class="tablewrap"><table>
<tr><th>s</th><th>distance from the wall centre</th><th>top layer far from the tip, U<sub>1</sub> − E<sub>F</sub></th>
<th>flat-band peak</th><th>its height (of the (+⁵) peak)</th><th>tip-induced bending at the peak bias</th></tr>
{table}
</table></div>
<div class="figs">
<figure><img src="{img_prefix}stacking_wall_polar_eps26_peaks_map_940.png" alt="line spectrum 940, measured and the eps = 2.6 best peak fit">
<figcaption>Line spectrum 940: measured dI/dV(V, x) and its bias derivative (left) against the ε = 2.6 best peak fit in
the local approximation (right).</figcaption></figure>
<figure><img src="{img_prefix}stacking_wall_polar_eps26_peaks_spectra_940.png" alt="spectra of the eps = 2.6 best-peak wall stacks and the flat-band peak along the line">
<figcaption>Left: top-layer dI/dV of the ε = 2.6 best-peak wall stack at each s under the same tip. Right: the flat-band
peak along the line, measured and for all models.</figcaption></figure>
</div>
"""


def polar_series_numbers(suffix=""):
    """Numbers of a polarized-stack line spectrum (stacking_wall_polar_series.py), or None."""
    path = os.path.join(EXAMPLES, f"stacking_wall_polar{suffix}_940.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        st = json.load(fh)
    rows = []
    for k, r in sorted(st["runs"].items(), key=lambda kv: kv[1]["s"]):
        sw = r["sweep"]
        V, U0, Uf = np.asarray(sw["bias"]), np.asarray(sw["U0"]), np.asarray(sw["U_far"])
        pk = st["peaks_s"][f"{r['s']:.2f}"]
        h = st.get("heights", {}).get(f"{r['s']:.2f}", 1.0)
        rows.append(dict(s=r["s"], U1=float(Uf[0, 0] - sw["mu0"]), U=(Uf[0] - sw["mu0"]).tolist(), mu0=sw["mu0"], peak=pk,
                         height=h, faded=h < 0.1,
                         bend0=float(np.interp(0.0, V, U0[:, 0] - Uf[:, 0])),
                         bend_pk=float(np.interp(pk, V, U0[:, 0] - Uf[:, 0]))))
    return dict(state=st, rows=rows)


def polar_series_text(img_prefix=""):
    """Section: line spectrum 940 with the best polarized stack, local approximation."""
    n = polar_series_numbers()
    if n is None:
        return ""
    st, rows = n["state"], n["rows"]
    p = st["params"]
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    mv1 = lambda v: ("0.0" if abs(v) < 5e-5 else f"{1e3 * v:+.1f}".replace("-", "−"))  # noqa: E731
    s_list = ", ".join(f"{r['s']:g}" for r in rows)
    table = "".join(f"<tr><td>{r['s']:g}</td><td>{mv1(r['U1'])} meV</td><td>{mv1(r['bend0'])} meV</td>"
                    + ("<td colspan=\"3\">flat band faded</td></tr>" if r["faded"] else
                       f"<td>{mv1(r['peak'])} mV</td><td>{100 * r['height'] / rows[0]['height']:.0f} %</td><td>{mv1(r['bend_pk'])} meV</td></tr>")
                    for r in rows)
    bend = {k: st[f"extreme_{k}"] - st[f"far_{k}"] for k in ("measured", "polarized", "uniform")}
    alive = [r for r in rows if not r["faded"]]
    r0 = rows[0]
    rs = min(alive, key=lambda r: r["peak"])
    d_pk, d_tip, d_u1 = rs["peak"] - r0["peak"], rs["bend_pk"] - r0["bend_pk"], rs["U1"] - r0["U1"]
    half = next((r["s"] for r in alive if r["height"] < 0.5 * r0["height"]), None)
    gone = next((r["s"] for r in rows if r["faded"]), None)
    dist = np.asarray(st["track"]["dist"])
    e = {k: np.asarray(st["edge"][k]) for k in ("measured", "polarized", "uniform")}
    j36 = int(np.argmin(np.abs(dist - (st["fit"]["x0"] - 5.0))))
    s36 = 0.5 * (1 - np.tanh((dist[j36] - st["fit"]["x0"]) / st["fit"]["ell"]))
    j15 = int(np.argmin(np.abs(dist - 15.0)))
    s15 = 0.5 * (1 - np.tanh((dist[j15] - st["fit"]["x0"]) / st["fit"]["ell"]))
    far_l = dist < 10.0
    reading = (
        f"The polarized stack starts the flat band at the measured energy and bends it the measured way, but by "
        f"{100 * bend['polarized'] / bend['measured']:.0f} % of the measured bend; the uniformly doped stack bent it by "
        f"{100 * bend['uniform'] / bend['measured']:.0f} %, from a start {abs(1e3 * (st['far_uniform'] - st['far_measured'])):.0f} mV too low. "
        f"Of the {mv1(d_pk)} mV between s = 0 and s = {rs['s']:g}, {mv1(d_tip)} meV is the change of the tip-induced band bending "
        f"at the peak bias ({mv1(r0['bend_pk'])} meV on the (+⁵) stack, {mv1(rs['bend_pk'])} meV at s = {rs['s']:g}) and "
        f"{mv1(d_u1)} meV the lower top-layer potential of the wall stack far from the tip. With E<sub>F</sub> in the gap of "
        f"the polarized stack the tip is screened less well, and the tip-induced part, a few meV in the uniformly doped "
        f"stack, becomes the larger one. Beyond that the flat band splits into the peak near −30 mV and a shoulder near "
        f"−70 mV and fades (below half its (+⁵) height from s = {half:g}, gone at s = {gone:g}), so the model loses it near "
        f"the wall centre, where the measured peak still reaches {mv(st['extreme_measured'])} mV. On the (+⁴−) side the "
        f"lowest edge above E<sub>F</sub> lies at {mv(e['polarized'][far_l].mean())} mV in the model and "
        f"{mv(e['measured'][far_l].mean())} mV in the measurement within 10 nm of the end of the line; the uniformly doped "
        f"stack has no edge below {mv(e['uniform'][far_l].mean())} mV there. The measured edge then rises to "
        f"{mv(e['measured'][j36])} mV {st['fit']['x0'] - dist[j36]:.0f} nm from the wall centre (model "
        f"{mv(e['polarized'][j36])} mV), over a stretch where s(x) only falls from {s15:.2f} to {s36:.2f}: a rise the local "
        f"approximation cannot follow.")
    return f"""<h3>Line spectrum 940 with the polarized stack</h3>
<p>The best polarized stack of the refit (ε = {p['eps']:g}, ΔE<sub>F</sub> = {1e3 * p['dEF']:.0f} meV,
n<sub>sub</sub> = {f"{p['nsub'] / 1e13:.3f}".replace("-", "−")}×10¹³ cm⁻²) and its tip (no band bending at zero bias on the (+⁵)
side) were carried across the wall in the local approximation, as the uniformly doped stack above: the wall stack at
s = {s_list}, each with the same fixed charge per layer (that of the doped (+⁵) stack), the same n<sub>sub</sub> and the
same tip, and the top-layer dI/dV of each stack placed along the fitted profile s(x) with one global scale, the fit of
the (+⁵) spectrum (<code>stacking_wall_polar_series.py</code>; 23 biases per sweep, 25 mV steps within 75 mV of zero and
50 mV beyond, which reproduces a full 25 mV sweep within 0.2 % of the peak). Far from the tip every stack keeps its own
Fermi level against the same fixed charges; the tip bends the top layer by at most
{max(abs(r['bend0']) for r in rows) * 1e3:.1f} meV at zero bias on any of them.</p>
<p>The flat-band peak along the line: measured {mv1(st['far_measured'])} mV far on the (+⁵) side and
{mv1(st['extreme_measured'])} mV at its deepest ({st['extreme_dist_measured']:.0f} nm), a bend of {mv1(bend['measured'])} mV;
polarized stack {mv1(st['far_polarized'])} and {mv1(st['extreme_polarized'])} mV ({st['extreme_dist_polarized']:.0f} nm),
{mv1(bend['polarized'])} mV; uniformly doped stack {mv1(st['far_uniform'])} and {mv1(st['extreme_uniform'])} mV
({st['extreme_dist_uniform']:.0f} nm), {mv1(bend['uniform'])} mV. {reading}</p>
<div class="tablewrap"><table>
<tr><th>s</th><th>top layer far from the tip, U<sub>1</sub> − E<sub>F</sub></th><th>tip-induced bending at 0 V</th>
<th>flat-band peak</th><th>its height (of the (+⁵) peak)</th><th>tip-induced bending at the peak bias</th></tr>
{table}
</table></div>
<div class="figs">
<figure><img src="{img_prefix}stacking_wall_polar_map_940.png" alt="line spectrum 940, measured and polarized-stack model">
<figcaption>Line spectrum 940: measured dI/dV(V, x) and its bias derivative (left) against the polarized-stack model
in the local approximation (right).</figcaption></figure>
<figure><img src="{img_prefix}stacking_wall_polar_spectra_940.png" alt="spectra of the polarized wall stacks and the flat-band peak along the line">
<figcaption>Left: top-layer dI/dV of the polarized wall stack at each s under the same tip. Right: the flat-band peak
along the line while it is visible, measured, polarized stack and the uniformly doped stack of the series above.</figcaption></figure>
</div>
"""


def bend_asymmetry(states):
    """Contact potential from the two measured bends in the tip-gating picture: a feature at
    energy E0 is seen at the bias V where V (1 - alpha) = E0 + alpha CPot, with one lever arm
    alpha (times r for the second line) and one CPot for both lines; E0 the far-field flat band."""
    rows = {}
    for t, st in states.items():
        s, meas = st["settings"], st["measured"]
        bias = np.asarray(meas["bias"])
        n = tibb_numbers(st)
        rows[t] = dict(E0=s["E_fb_measured"], V=n["meas_wall_peak"], step=float(np.median(np.diff(bias))),
                       sim_bend=(n["wall_peak"] - n["rg_peak"]) if "wall_peak" in n and "rg_peak" in n else None,
                       tibb=n.get("wall_bend"))
    if len(rows) < 2:
        return None
    t1, t2 = list(rows)[:2]
    up, down = (t1, t2) if rows[t1]["E0"] > 0 else (t2, t1)

    def fit(V1, E1, V2, E2, r=1.0):
        a = (V2 - E2 - r * (V1 - E1)) / (r * (V2 - V1))
        return a, (V1 * (1.0 - a) - E1) / a

    u, d = rows[up], rows[down]
    alpha, C = fit(u["V"], u["E0"], d["V"], d["E0"])
    h = 0.5 * max(u["step"], d["step"])
    pos = [fit(u["V"] + a, u["E0"] + b, d["V"] + c, d["E0"] + e)[1]
           for a in (-h, h) for b in (-h, h) for c in (-h, h) for e in (-h, h)]
    lev = [fit(u["V"], u["E0"], d["V"], d["E0"], r)[1] for r in (0.8, 1.25)]
    pred = {cp: (alpha * (u["V"] + cp), alpha * (d["V"] + cp)) for cp in (-0.05, 0.0, 0.05)}
    return dict(up=up, down=down, rows=rows, alpha=alpha, C=C, pos=(min(pos), max(pos)), lev=(min(lev), max(lev)),
                h=h, pred=pred)


def asymmetry_text(states):
    """Paragraph: what the different size of the upward and downward bends says about CPot."""
    a = bend_asymmetry(states)
    if a is None:
        return ""
    u, d = a["rows"][a["up"]], a["rows"][a["down"]]
    bu, bd = u["V"] - u["E0"], d["V"] - d["E0"]
    mv = lambda v: f"{1e3 * v:+.0f}".replace("-", "−")   # noqa: E731
    pred = "; ".join(f"CPot {mv(cp) if cp else '0'} meV: {mv(p[0])} and {mv(p[1])} mV" for cp, p in a["pred"].items())
    sim = ""
    if u["sim_bend"] is not None and d["sim_bend"] is not None and u["tibb"] is not None and d["tibb"] is not None:
        sim = (f" The self-consistent model at CPot = 0 also bends the band further down than up (simulated "
               f"{mv(d['sim_bend'])} against {mv(u['sim_bend'])} mV, ratio {abs(d['sim_bend'] / u['sim_bend']):.2f}), "
               f"through the charge balance of the wall stack, which grows with the distance of the far-field flat band "
               f"from E<sub>F</sub> ({abs(d['E0'] / u['E0']):.2f} times larger in {a['down']}).")
    return f"""<p><b>The asymmetry of the bends.</b> The flat band bends up by {1e3 * abs(bu):.0f} mV in {a['up']} (from
{mv(u['E0'])} mV far on the (+⁵) side to {mv(u['V'])} mV at the wall) and down by {1e3 * abs(bd):.0f} mV in {a['down']} (from
{mv(d['E0'])} to {mv(d['V'])} mV): the downward bend is {abs(bd / bu):.2f} times the upward one. This does not
point to a contact potential. Per volt of bias the two bends are the same, {bu / u['V']:.2f} and {bd / d['V']:.2f}:
the flat band of {a['down']} lies further from E<sub>F</sub> and is seen at a larger bias, so it bends more. In the
tip-gating picture a feature at energy E<sub>0</sub> appears at the bias V with V(1 − α) = E<sub>0</sub> + α CPot;
one lever arm α and one CPot for both lines fit the two bends exactly with α = {a['alpha']:.2f} and CPot =
{mv(a['C'])} meV (no band bending at {mv(-a['C'])} mV). Moving each of the four positions by half a bias step
(±{1e3 * a['h']:.2f} mV) gives CPot from {mv(a['pos'][0])} to {mv(a['pos'][1])} meV, and lever arms differing by up to
25 % between the two walls from {mv(a['lev'][0])} to {mv(a['lev'][1])} meV. The ratio of the bends is very sensitive
to CPot (upward and downward bend at the measured biases: {pred}): a tip whose work function exceeded the sample's
would favour the upward bend, a lower one the downward bend. Read this way, tip and sample work functions agree
within about 10 meV, in line with CPot = 0 ± 0.05 eV from the (+⁵) spectra and far from the +1.05 eV of a vacuum Pt
tip. Two qualifications. The fit assumes one tip for both lines, which were recorded on different days; with
different tips each line gives only the bound above. And it needs {100 * a['alpha']:.0f} % of the bias to drop in the
sample, while the self-consistent calculation bends the top layer at the wall by only {mv(u['tibb'])} and
{mv(d['tibb'])} meV at the peak bias.{sim} Under either mechanism the asymmetry follows from the different doping of
the two regions and needs no contact potential.</p>
"""


def workfunction_text(states):
    """Paragraph on the tip work function: what entered the calculation and what the data say."""
    rows = {}
    for t, st in states.items():
        s = st["settings"]
        scan = {d["cpot"]: d for d in st.get("scan", {}).values()}
        slope = None
        if 0.1 in scan and -0.1 in scan:
            slope = (scan[0.1]["peak_sim"] - scan[-0.1]["peak_sim"]) / 0.2          # V per V
        meas = st["measured"]
        wall_peak = _peak(np.asarray(meas["bias"]), np.asarray(meas["wall"]), s["E_fb_measured"], 0.09)
        rows[t] = dict(cpot=st.get("best_cpot", 0.0), E_fb=s["E_fb_measured"], wall=wall_peak,
                       mu0=s["fermi_level"], slope=slope)
    if len(rows) < 2:
        return ""
    t1, t2 = list(rows)[:2]
    a, b = rows[t1], rows[t2]
    up, down = (a, b) if a["E_fb"] > 0 else (b, a)          # the line seen at positive bias bends up
    slope_txt = ", ".join(f"{1e3 * r['slope'] * 0.1:.0f} mV per 0.1 eV ({t})" for t, r in rows.items() if r["slope"])
    return f"""<p><b>Tip work function.</b> No absolute work function enters the electrostatics. The engine takes the
contact potential CPot = W<sub>tip</sub> − W<sub>ref</sub>, with W<sub>ref</sub> the depth of the stack's
charge-neutrality point below vacuum, and places the tip at U<sub>tip</sub> = V + CPot; a positive CPot acts like a
positive bias and bends the bands up at V = 0. CPot was the fit parameter of the (+⁵) spectra, scanned in 0.1 eV
steps, and came out at {a['cpot']:+.2f} eV for {t1} and {b['cpot']:+.2f} eV for {t2}; the simulated flat-band peak
moves by {slope_txt}, so the fit holds it to about ±0.05 eV. The reference is the neutrality point, not the Fermi
level: the sample's own work function differs from W<sub>ref</sub> by the doping offset, {1e3 * a['mu0']:+.0f} meV
({t1}) and {1e3 * b['mu0']:+.0f} meV ({t2}), so tip and sample work functions are equal within a few tens of meV.
The 4.6 eV barrier of the simulated spectra enters only the WKB transmission factor, not the band bending.</p>
<p>The data constrain the same quantity without any solver. In the tip-gating picture a feature measured at bias
V shifts in proportion to V + CPot, so the flat band seen at positive bias bends up only if
CPot &gt; −V and the one seen at negative bias bends down only if CPot &lt; −V: with the far-field flat bands at
{1e3 * up['E_fb']:+.0f} and {1e3 * down['E_fb']:+.0f} mV this is {-1e3 * up['E_fb']:+.0f} &lt; CPot &lt;
{-1e3 * down['E_fb']:+.0f} meV, with the peaks at the wall ({1e3 * up['wall']:+.0f} and {1e3 * down['wall']:+.0f} mV)
{-1e3 * up['wall']:+.0f} &lt; CPot &lt; {-1e3 * down['wall']:+.0f} meV. A tip whose work function exceeded the
sample's by more than about 0.1 eV would make the downward bend impossible in that picture, one below it by more
than about 0.05 eV the upward bend; this holds separately for the two tips if they differed. In the self-consistent
calculation, however, most of the simulated bend comes from the charge balance of the wall stack, whose sign follows
the filling of the flat band and not the tip: there a large contact potential would move both flat bands the same
way (and the far-field peaks with them) instead of forbidding one bend. Measuring the same wall with tips of
different work function separates the two: the tip-gating bend changes sign when the flat-band voltage crosses the
flat-band bias, the charge-balance bend does not. For comparison, the engine's authors found an effective contact
potential of about +0.1 eV for Au and PtIr tips on rhombohedral graphite, and vacuum work-function differences
(Pt: +1.05 eV) overestimate the tip field several times.</p>
{asymmetry_text(states)}"""


def tibb_text(states, img_prefix=""):
    """Interpretation of the band-bending results, with the numbers of the result files."""
    n = {t: tibb_numbers(st) for t, st in states.items()}
    if any("wall_peak" not in v for v in n.values()):
        return "<p>The wall and (+⁴−) calculations are not finished.</p>"
    def shifts(t):
        v = n[t]
        return (1e3 * (v["meas_wall_peak"] - v["E_fb"]), 1e3 * (v["wall_peak"] - v["rg_peak"]),
                1e3 * v["rg_bend"], 1e3 * v["wall_bend"], 1e3 * (v["wall_mu0"] - v["rg_mu0"]),
                1e3 * v["meas_sf_min"], 1e3 * v["sf_min"], 1e3 * v["sf_bend"])
    rows = {t: shifts(t) for t in n}
    t1, t2 = list(n)[:2]
    a, b = rows[t1], rows[t2]
    return f"""<p><b>What the calculation gives.</b> With the tip and the doping fixed on the (+⁵) side, the
simulated flat-band peak at the wall moves in the measured direction for both fillings: up by
{a[1]:+.0f} mV in spectrum {t1} (measured {a[0]:+.0f} mV) and down by {b[1]:+.0f} mV in spectrum {t2}
(measured {b[0]:+.0f} mV), so the sign follows the filling as observed, with {100 * a[1] / a[0]:.0f} % and
{100 * b[1] / b[0]:.0f} % of the measured size. The tip, however, is not what moves it in this model: the
band bending under the apex at the flat-band bias is {a[2]:+.0f} meV far on the (+⁵) side and {a[3]:+.0f} meV at
the wall in {t1}, {b[2]:+.0f} and {b[3]:+.0f} meV in {t2}, so the tip-induced part of the shift is at most
a few meV. At s = 0.75 the top layer keeps about a quarter of its flat-band states (4 × 10¹³ eV⁻¹ cm⁻² at
E<sub>F</sub> with the 10 meV width), still a far better screen than the 0.013 F m⁻² of the vacuum gap, and
the self-consistent solution confirms the pinning. The simulated shift comes instead from the charge balance of
the wall stack: with the same fixed charge per layer, a top layer that has lost most of its flat-band states
must move its bands to hold that charge ({a[4]:+.0f} meV in {t1}, {b[4]:+.0f} meV in {t2}, up for the empty
band and down for the filled one), and the remaining top-layer weight sits on the far flank of the band, which
adds the rest. The frozen-potential term alone (dashed) reproduces the full simulated peak position.</p>
<p><b>The (+⁴−) side.</b> The simulated (+⁴−) spectra have the measured V shape (residuals
{n[t1]['sf_residual']:.2f} and {n[t2]['sf_residual']:.2f}), but their minimum sits at {a[6]:+.0f} mV ({t1})
and {b[6]:+.0f} mV ({t2}) where the measured minima are at {a[5]:+.0f} and {b[5]:+.0f} mV: the same fixed
charge per layer puts the (+⁴−) Fermi level 30 to 40 meV from its neutrality point, while the real
stacking-fault region sits near neutrality because its compensating charge can move into the buried flat band.
The per-layer charge was the only way to place the (+⁵) flat band at the measured energy in this
γ0/γ1 model (substrate traps alone pin it at E<sub>F</sub>); transferred to the other stacks it overstates
their band offsets, so the intrinsic shift it produces at the wall is an upper estimate.</p>
<p><b>Across the wall.</b> The wall stack was then solved at s = 0.15, 0.30, 0.45, 0.60, 0.75 and 0.90 with
the same tip and doping (50 mV bias steps), and the simulated top-layer dI/dV was placed along the measured line
through the fitted profile s(x) = [1 − tanh((x − x0)/l)]/2, interpolated linearly between the s values with one
global scale. The simulated maps reproduce the sequence of the measurement: the flat band of the (+⁵) side bends
up (514) or down (940) as it fades into the wall, the (+⁵) gap edges at ±0.2 to 0.3 V give way to the V-shaped
spectrum of the (+⁴−) side, and there are no fringes, since the local approximation has no lateral scattering.
Two differences stand out: the simulated flat band disappears over a shorter distance than the measured one,
because in the γ0/γ1 model the top layer keeps most of its flat-band weight until s ≈ 0.75, and the simulated
(+⁴−) side keeps the Dirac-point offset discussed above.</p>
{series_table(states)}
<div class="figs">
<figure><img src="{img_prefix}stacking_wall_tibb_map_514.png" alt="simulated line spectrum across the wall, 514">
<figcaption>Line spectrum 514: measured dI/dV(V, x) and its bias derivative (left) against the simulated
top-layer dI/dV of the wall stack along the fitted profile (right).</figcaption></figure>
<figure><img src="{img_prefix}stacking_wall_tibb_map_940.png" alt="simulated line spectrum across the wall, 940">
<figcaption>Line spectrum 940, the same.</figcaption></figure>
</div>"""


def load_states(tags):
    """stacking_wall_tibb_TAG.json of every line that has one."""
    states = {}
    for t in tags:
        path = os.path.join(EXAMPLES, f"stacking_wall_tibb_{t}.json")
        if os.path.exists(path):
            with open(path) as fh:
                states[t] = json.load(fh)
    return states


def tibb_setup_text(states, img_prefix=""):
    """The self-consistent model and the fit of the (+5) spectra (stacking_wall_tibb.py)."""
    rows = []
    fmt = lambda d, k, f="%+.0f", scale=1e3: (f % (scale * d[k])) if k in d else "–"   # noqa: E731
    for t, st in states.items():
        n = tibb_numbers(st)
        rows.append(
            f"<tr><td>{html.escape(t)}</td><td>{st.get('best_cpot', float('nan')):+.2f} eV</td>"
            f"<td>{1e3 * st['settings']['fermi_level']:+.0f} meV</td>"
            f"<td>{1e3 * n['E_fb']:+.0f} / {fmt(n, 'rg_peak')} mV</td>"
            f"<td>{1e3 * n['meas_wall_peak']:+.0f} / {fmt(n, 'wall_peak')} mV</td>"
            f"<td>{fmt(n, 'rg_bend')} / {fmt(n, 'wall_bend')} / {fmt(n, 'sf_bend')} meV</td>"
            f"<td>{fmt(n, 'rg_residual', '%.2f', 1)} / {fmt(n, 'wall_residual', '%.2f', 1)} / {fmt(n, 'sf_residual', '%.2f', 1)}</td></tr>")
    s0 = next(iter(states.values()))["settings"]
    return f"""<p>Laterally uniform six-layer stacks under the tip, each solved self-consistently (Poisson equation with
Hartree + γ0/γ1 tight-binding layers, <code>GrapheneMultilayer</code> of the
<a href="https://github.com/nemeslab-MFA/bandbending">bandbending</a> engine): the (+⁵) stack, the wall stack with the
top dimer shared between the (+⁵) bond (weight 1 − s) and the (+⁴−) bond (weight s; s = {s0['s']:.2f} at the position of
the measured wall spectrum), and the (+⁴−) stack. The wall Hamiltonian closes a bond loop whose flux is the phase of
the structure factor, so it is evaluated with a complex Hamiltonian averaged over that phase (<code>WallTB</code> in
<code>stacking_wall_tibb.py</code>; it reproduces the engine's end members exactly and a brute-force zone sum to
1 %). Tip radius {s0['radius']:.0f} nm, separation {s0['separation']:.1f} nm, barrier {s0['barrier']:.1f} eV,
{s0['temperature']:.0f} K, Gaussian width {1e3 * s0['broadening']:.0f} meV of the tight-binding occupation and LDOS (the
bare γ0/γ1 flat band is 6 meV wide; with this Gaussian the simulated (+⁵) peak is 34 to 35 mV wide at half
maximum, the measured one 42 to 45 mV), lock-in {1e3 * s0['modulation']:.2f} mV; 90 nm SiO₂ over the
grounded Si. The doping is set from the measured flat-band energy of the (+⁵) side (the isolated stack is neutral with
E<sub>F</sub> that far from its flat band); the fixed charge per layer this implies is used unchanged for the wall and
the (+⁴−) stack. The tip contact potential is the value, scanned in 0.1 eV steps, at which the simulated (+⁵)
spectrum (<code>tunneling_spectrum</code>: frozen-potential term plus band-bending term) is closest to the
measurement, one scale factor free.</p>
<div class="tablewrap"><table>
<tr><th>line spectrum</th><th>contact potential</th><th>E<sub>F</sub> − E<sub>CNP</sub></th><th>flat band (+⁵): measured / simulated</th><th>flat band at the wall: measured / simulated</th><th>U<sub>1</sub>(apex) − U<sub>1</sub>(far) at the peak: (+⁵) / wall / (+⁴−)</th><th>residual (+⁵) / wall / (+⁴−)</th></tr>
{''.join(rows)}
</table></div>
<figure><img src="{img_prefix}stacking_wall_tibb.png" alt="measured and simulated dI/dV at the three positions">
<figcaption>Measured dI/dV (black) and simulated top-layer dI/dV (orange, scaled; dashed: frozen-potential term) far
on the (+⁵) side, at the wall and far on the (+⁴−) side, with the contact-potential scan.</figcaption></figure>
"""


def status_numbers(states):
    """Per line: measured bend at the wall spectrum and the model bends (eV), the tip's lever arm on the top
    layer, the tip-induced parts, the work-function offset that would give the measured bend; and the largest
    bends of the polarized stacks of 940."""
    out = {}
    for t, st in states.items():
        s = st["settings"]
        hn = hybrid_numbers(t, st)
        E_fb = s["E_fb_measured"]
        d = dict(E_fb=E_fb, V_wall=hn["meas"], meas=hn["meas"] - E_fb, s=s["s"], x_wall=s["x_wall"])
        if "local" in hn and "local_far" in hn:
            d["local"] = hn["local"] - hn["local_far"]
        if "hybrid" in hn:
            d["swmcc"] = hn["hybrid"] - E_fb
        bn = bump_numbers(t)
        if bn and bn["rows"]:
            x0 = bn["settings"]["x0"]
            h_max = max(r["height"] for r in bn["rows"])
            rows = sorted([r for r in bn["rows"] if r.get("dcentroid") is not None and r["height"] >= 0.15 * h_max
                           and abs(r["peak"] - E_fb) <= 0.06], key=lambda r: r["x"])
            if rows:
                xs = [r["x"] for r in rows]
                d["tip_bump"] = float(np.interp(s["x_wall"] - x0, xs, [r["dcentroid"] for r in rows]))
                d["bump_top"] = float(np.interp(s["x_wall"] - x0, xs, [r["bump"] for r in rows]))
                d["lever"] = d["bump_top"] / bn["settings"]["bias"]
        ln = lateral_numbers(t)
        if ln:
            x0 = ln["settings"]["x0"]
            h = [r["height"] for r in ln["rows"] if r["height"] is not None]
            rows = sorted([r for r in ln["rows"] if r.get("dcentroid") is not None and r["height"] >= 0.15 * max(h)
                           and abs(r["peak"] - E_fb) <= 0.06], key=lambda r: r["x"])
            if rows:
                d["tip_lateral"] = float(np.interp(s["x_wall"] - x0, [r["x"] for r in rows], [r["dcentroid"] for r in rows]))
        dcf = far_tip_shift(t)
        if dcf is not None:
            d["dcf"] = dcf
            for k in ("tip_bump", "tip_lateral"):
                if k in d:
                    d[k + "_net"] = d[k] - dcf
        out[t] = d
    cp = cpot_numbers(states)
    if cp:
        for t, v in cp.items():
            out[t]["need"], out[t]["need_extrap"] = v["need"], v["extrap"]
            out[t]["wf_best"], out[t]["wf_best_delta"] = v["best"], v["best_delta"]
            z = v["zero"]
            out[t]["lever_far"] = z["bump_far"] / v["s"]["E_fb"]
    pol = {}
    for suf in ("", "_eps26_fb", "_eps26_peaks"):
        n = polar_series_numbers(suf)
        if n:
            pol[suf] = n["state"]["extreme_polarized"] - n["state"]["far_polarized"]
    return out, pol


def _mv(v, nd=0):
    return (f"{1e3 * v:+.{nd}f}" if v is not None else "–").replace("-", "−")


def abstract_text(states):
    nums, pol = status_numbers(states)
    sep = separation_numbers(states)
    if len(nums) < 2 or sep is None:
        return ""
    up = next(t for t, v in nums.items() if v["E_fb"] > 0)
    dn = next(t for t, v in nums.items() if v["E_fb"] < 0)
    u, d = nums[up], nums[dn]
    su, sd = sep[up], sep[dn]
    a = bend_asymmetry(states)
    levs = [abs(v[k]) for v in sep.values() for k in ("lever_wall", "lever_rg") if v.get(k)]
    cp = cpot_numbers(states)
    p1 = f"""<p>Across the (+⁴−)|(+⁵) wall of six-layer rhombohedral graphite the flat-band peak bends away from
E<sub>F</sub> as it fades: up by {_mv(u['meas'])} mV in line {up}, where the band is empty, and down by
{_mv(d['meas'])} mV in line {dn}, where it is filled, at the position of the wall spectrum (s = {u['s']:.2f} of the fitted
profile). The measurement looks like tip gating: per volt of bias the two bends are the same (a lever arm of
{a['alpha']:.2f} for both lines with one contact potential of {_mv(a['C'])} meV), and in {dn} an edge seen at positive bias
on the stacking-fault side rises by 50 to 90 mV next to the same wall where the flat band, seen at negative bias,
sinks. A capacitor estimate gives both bends if the density of states at E<sub>F</sub> under the tip falls to about
5 × 10¹² eV⁻¹ cm⁻² at the wall while the (+⁵) side stays pinned.</p>"""
    p2 = f"""<p>The self-consistent model (Poisson equation with γ0/γ1 tight-binding stacks, tip work function fitted to the
(+⁵) spectra, equal to the sample's within 0.05 eV) bends the flat band the measured way, but not by tip gating.
Computed from the same self-consistent stacks with and without the tip, the bend at the wall spectrum is
{_mv(su['wall']['tip'])} and {_mv(sd['wall']['tip'])} mV with the tip and {_mv(su['wall']['free'])} and
{_mv(sd['wall']['free'])} mV without it: it is charge redistribution in the wall stack, which has lost most of its
top-layer flat-band states and must move its bands to hold the same charge. The tip shifts the peaks
({_mv(su['ref_shift'], 1)} and {_mv(sd['ref_shift'], 1)} mV on the (+⁵) stack), but by about as much at the wall as on the
(+⁵) side, since its lever arm on the top layer is {min(levs):.2f} to {max(levs):.2f} on both; the bend keeps
{_mv(su['wall']['tip'] - su['wall']['free'], 1)} and {_mv(sd['wall']['tip'] - sd['wall']['free'], 1)} mV of it. The SWMcC wall
model with the self-consistent on-site energies gives the same picture ({_mv(u['swmcc'])} and {_mv(d['swmcc'])} mV, the
tip adding {_mv(u.get('tip_bump_net'), 1)} and {_mv(d.get('tip_bump_net'), 1)} mV), with the fringes of the (+⁴−) side.
{(f"In the SWMcC model with the tip-induced potential no tip work function within {_mv(min(cp[up]["d"]))} to {_mv(max(cp[up]["d"]))} meV of the sample's reaches the measured bend (the largest bends are {_mv(cp[up]['best'])} and {_mv(cp[dn]['best'])} mV).") if cp else ""}</p>"""
    wide_txt = ""
    if "wide" in su and "wide" in sd:
        wu, wd = su["wide"], sd["wide"]
        ru = next((r for r in wu["rows"] if abs(r["s"] - su["wall"]["s"]) < 1e-6), None)
        rd = next((r for r in wd["rows"] if abs(r["s"] - sd["wall"]["s"]) < 1e-6), None)
        if ru and rd:
            wide_txt = (f" Widening the Gaussian of the occupation and the LDOS from 10 meV to {1e3 * wu['sigma']:.1f} and "
                        f"{1e3 * wd['sigma']:.1f} meV, which gives the measured flat-band width (42 to 45 mV against 34 to 35 mV), "
                        f"leaves the lever arm at the wall at {abs(wu['lever_wall']):.2f} and {abs(wd['lever_wall']):.2f} and "
                        f"gives bends of {_mv(ru['tip'])} and {_mv(rd['tip'])} mV with the tip, {_mv(ru['free'])} and "
                        f"{_mv(rd['free'])} mV without it.")
    p3 = f"""<p>A bend from charge redistribution belongs to the sample: its sign follows the filling of the flat band, so it
appears in both directions whatever the tip, which answers why the tip and sample work functions would otherwise have
to agree within about 25 meV. In size it reaches {100 * su['wall']['tip'] / u['meas']:.0f} % ({up}) and
{100 * sd['wall']['tip'] / d['meas']:.0f} % ({dn}) of the measured bend, and it depends on the doping model: with the doping
compensated by charge in the SiO₂ (the polarized stack that fits the (+⁵) spectrum of {dn} best at ε = 2.6) the
wall's charge rearrangement becomes a dipole inside the stack and the bend falls to {_mv(pol.get('_eps26_fb'), 1)} mV.
No model run produces the rising edge on the (+⁴−) side.{wide_txt}</p>"""
    p4 = """<p>The status: charge redistribution gives the sign and about half the size of the bend; the rest, and the
rising (+⁴−) edge, need a tip lever arm of about 0.6 at the wall. The model cannot get there by thinning the top layer:
as long as the layer below has states at E<sub>F</sub>, it holds the top layer through the interlayer capacitance and caps
the lever arm near 0.16, where the model sits, with or without a flat band widened to the measured width. A lever arm of
0.6 needs the tip's field to reach through the whole stack, with no layer under the apex holding states at E<sub>F</sub>
(section 9). The measurement that separates the two mechanisms is the bend against the tip height at one wall: a
tip-gating bend scales with the gap capacitance, a charge-redistribution bend does not.</p>"""
    return p1 + p2 + p3 + p4


def status_table(states):
    nums, pol = status_numbers(states)
    if len(nums) < 2:
        return ""
    sep = separation_numbers(states)
    if sep:
        for t, v in sep.items():
            nums[t]["sep_tip"], nums[t]["sep_free"] = v["wall"]["tip"], v["wall"]["free"]
            nums[t]["lever"] = v["lever_wall"]
            nums[t]["lever_far"] = v["lever_rg"]
            if "wide" in v:
                r = next((q for q in v["wide"]["rows"] if abs(q["s"] - v["wall"]["s"]) < 1e-6), None)
                if r:
                    nums[t]["wide_tip"], nums[t]["wide_free"], nums[t]["wide_sigma"] = r["tip"], r["free"], v["wide"]["sigma"]
    t1, t2 = list(nums)[:2]
    a, b = nums[t1], nums[t2]
    al = bend_asymmetry(states)

    def row(label, f, note=""):
        return f"<tr><td>{label}</td><td>{f(a)}</td><td>{f(b)}</td><td>{note}</td></tr>"

    pct = lambda v, k: f"{_mv(v.get(k))} mV ({100 * v[k] / v['meas']:.0f} %)" if v.get(k) is not None else "–"   # noqa: E731
    tot = lambda v: (v["swmcc"] + v["tip_bump_net"]) if ("swmcc" in v and "tip_bump_net" in v) else None   # noqa: E731
    rows = [
        row("flat-band peak (+⁵) / at the wall spectrum, measured", lambda v: f"{_mv(v['E_fb'])} / {_mv(v['V_wall'])} mV"),
        row("<b>measured bend</b>", lambda v: f"<b>{_mv(v['meas'])} mV</b>", f"at s = {a['s']:.2f} of the fitted profile"),
        row("lever arm of the tip on the top layer at the wall: needed / model", lambda v: f"{al['alpha']:.2f} / {abs(v.get('lever', float('nan'))):.2f}"
            if al else "–", "needed: one lever arm and one contact potential for both lines; model: top-layer bending at the wall-peak bias over that bias"),
        row("local approximation with the tip / without the tip", lambda v: (f"{_mv(v['sep_tip'], 1)} / {_mv(v['sep_free'], 1)} mV "
            f"({100 * v['sep_tip'] / v['meas']:.0f} / {100 * v['sep_free'] / v['meas']:.0f} %)") if v.get("sep_tip") is not None else pct(v, "local"),
            "same self-consistent stacks; centroids against the own (+⁵) peak"),
        row("local approximation, flat band as wide as measured: with / without the tip", lambda v: (f"{_mv(v['wide_tip'], 1)} / {_mv(v['wide_free'], 1)} mV "
            f"(σ = {1e3 * v['wide_sigma']:.1f} meV)") if v.get("wide_tip") is not None else "running",
            "σ = 10 meV above"),
        row("SWMcC wall + self-consistent on-site energies (no tip)", lambda v: pct(v, "swmcc"), "far-field charge balance"),
        row("lever arm of the tip on the top layer far on the (+⁵) side, at its bias", lambda v: f"{abs(v.get('lever_far', float('nan'))):.2f}"),
        row("tip-induced shift of the peak at the wall / of the (+⁵) peak", lambda v: f"{_mv(v.get('tip_bump'), 1)} / {_mv(v.get('dcf'), 1)} mV",
            "SWMcC with the Gaussian bump, contact potential 0, each at its measured bias"),
        row("tip-induced part of the bend: Gaussian bump / three-dimensional solve", lambda v: f"{_mv(v.get('tip_bump_net'), 1)} / {_mv(v.get('tip_lateral_net'), 1)} mV",
            "shift at the wall minus shift of the (+⁵) reference"),
        row("SWMcC + on-site energies + tip", lambda v: (f"{_mv(tot(v))} mV ({100 * tot(v) / v['meas']:.0f} %)" if tot(v) is not None else "–")),
        row("tip work function − sample's for the measured bend", lambda v: (f"{_mv(v['need'])} meV" if v.get("need") is not None else
            (f"not reached; largest {_mv(v['wf_best'])} mV at {_mv(v['wf_best_delta'])} meV" if v.get("wf_best") is not None else "–")),
            "offsets −350 to +200 meV computed; the (+⁵) fits allow 0 ± 50 meV"),
    ]
    if pol:
        lab = {"": "polarized stack, ε = 15 (unphysical permittivity)", "_eps26_fb": "polarized stack, ε = 2.6, flat-band fit",
               "_eps26_peaks": "polarized stack, ε = 2.6, best peak fit"}
        for suf, v in pol.items():
            rows.append(f"<tr><td>{lab[suf]}</td><td>–</td><td>{_mv(v, 1)} mV</td><td>local approximation, largest bend along the line</td></tr>")
    return f"""<div class="tablewrap"><table>
<tr><th>flat-band bend at the wall</th><th>{t1}</th><th>{t2}</th><th>note</th></tr>
{''.join(rows)}
</table></div>"""


def doping_extra_text(img_prefix=""):
    """The work-function fit at eps = 2.6 and the hole-side van Hove observation."""
    p = os.path.join(EXAMPLES, "stacking_wall_eps26_cpot_940.json")
    if not os.path.exists(p):
        return ""
    with open(p) as fh:
        c = json.load(fh)
    st, be = c["start"], c["best"]
    return f"""<h3>Tip work function as a third fit parameter (ε = 2.6)</h3>
<p>With the offset δ of the tip work function from the sample's free next to ΔE<sub>F</sub> and the hole density
(<code>stacking_wall_epsfit/fit_cpot.py</code>, compass search from the flat-band fit) the five peak positions of the
(+⁵) spectrum of 940 improve from {1e3 * st['rms']:.1f} to {1e3 * be['rms']:.1f} mV rms at ΔE<sub>F</sub> =
{1e3 * be['dEF']:.1f} meV, {be['holes'] / 1e10:.0f}×10¹⁰ holes and δ = {_mv(be['delta'])} meV: the tip work function
stays at the sample's.</p>
<figure><img src="{img_prefix}stacking_wall_eps26_cpot_940.png" alt="eps 2.6 fit with the tip work function free">
<figcaption>Line 940, (+⁵) side, ε = 2.6: measured dI/dV against the start and the best point of the three-parameter
search, and the search path.</figcaption></figure>
<p><b>Observation (2026-11-09).</b> In the ε = 2.6 best-peak fit (ΔE<sub>F</sub> = 40 meV, about 1.5 × 10¹² holes) the
spectrum computed without the tip has a weak hole-side van Hove maximum near −330 mV that disappears once the tip is
in the solve, while the structure near −215 mV survives (at −222 mV). The measured spectra lack the −330 mV
maximum as well: the negative-bias spectrum is shaped by the tip's field.</p>"""


def separation_numbers(states):
    """With and without the tip, per line: bends at every s (centroids against the own (+5) peak), the tip's shift
    of the (+5) reference, the lever arm at the wall, and the same at the wider Gaussian; None if not computed."""
    path = os.path.join(EXAMPLES, "stacking_wall_separation.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    sep = data.get("separation", {})
    if not all(t in sep for t in states):
        return None
    out = {}
    for t, st in states.items():
        d = sep[t]
        rows = d["rows"]
        r0 = rows[0]
        h0 = {k: r0[k]["height"] for k in ("tip", "free")}
        tab = []
        for r in rows:
            alive = all(r[k]["height"] > 0.15 * h0[k] and abs(r[k]["peak"] - r0[k]["peak"]) < 0.06 for k in ("tip", "free"))
            tab.append(dict(s=r["s"], alive=alive, tip=r["tip"]["centroid"] - r0["tip"]["centroid"],
                            free=r["free"]["centroid"] - r0["free"]["centroid"]))
        sw_ = min(tab, key=lambda q: abs(q["s"] - d["s_wall"]))
        s = st["settings"]
        Vw = d["measured_wall"]
        def lever(sweep, V):
            Vb, U0, Uf = np.asarray(sweep["bias"]), np.asarray(sweep["U0"]), np.asarray(sweep["U_far"])
            return float(np.interp(V, Vb, U0[:, 0] - Uf[:, 0])) / V
        wall_rec = st["runs"].get(f"wall_s{d['s_wall']:.2f}")
        rg_rec = st["scan"][f"{st['best_cpot']:+.3f}"]
        o = dict(tab=tab, wall=sw_, ref_shift=r0["tip"]["centroid"] - r0["free"]["centroid"], meas=Vw - d["E_fb"],
                 sigma=d["broadening"], lever_wall=lever(wall_rec["sweep"], Vw) if wall_rec else None,
                 lever_rg=lever(rg_rec["sweep"], d["E_fb"]), cal=data.get("calibration", {}).get(t))
        wp = os.path.join(EXAMPLES, f"stacking_wall_separation_wide_{t}.json")
        if os.path.exists(wp):
            with open(wp) as fh:
                w = json.load(fh)
            if w.get("rows"):
                wr = w["rows"]
                w0 = wr[0]
                o["wide"] = dict(sigma=w["settings"]["sigma"], fwhm=w.get("fwhm_rg"),
                                 rows=[dict(s=r["s"], tip=r["tip"]["centroid"] - w0["tip"]["centroid"],
                                            free=r["free"]["centroid"] - w0["free"]["centroid"],
                                            h_tip=r["tip"]["height"] / w0["tip"]["height"]) for r in wr[1:]],
                                 ref_shift=w0["tip"]["centroid"] - w0["free"]["centroid"],
                                 lever_wall=lever(w["runs"][f"{d['s_wall']:.2f}"]["sweep"], Vw) if f"{d['s_wall']:.2f}" in w["runs"] else None,
                                 lever_rg=lever(w["runs"]["0.00"]["sweep"], d["E_fb"]))
        out[t] = o
    return out


def separation_text(states, img_prefix=""):
    nums = separation_numbers(states)
    if nums is None:
        return ""
    t1, t2 = list(nums)[:2]
    a, b = nums[t1], nums[t2]
    mv = lambda v, nd=0: _mv(v, nd)   # noqa: E731
    rows = []
    for t, v in nums.items():
        for q in v["tab"]:
            if not q["alive"]:
                rows.append(f"<tr><td>{t}</td><td>{q['s']:.2f}</td><td colspan=\"3\">flat-band peak faded</td></tr>")
                continue
            rows.append(f"<tr><td>{t}</td><td>{q['s']:.2f}</td><td>{mv(q['tip'], 1)} mV</td><td>{mv(q['free'], 1)} mV</td>"
                        f"<td>{mv(q['tip'] - q['free'], 1)} mV</td></tr>")
    wide_txt = ""
    if "wide" in a and "wide" in b:
        wa, wb = a["wide"], b["wide"]
        def at(w, sv):
            return next((r for r in w["rows"] if abs(r["s"] - sv) < 1e-6), None)
        ra, rb = at(wa, a["wall"]["s"]), at(wb, b["wall"]["s"])
        wrows = []
        for t, v in nums.items():
            for r in v["wide"]["rows"]:
                wrows.append(f"<tr><td>{t}</td><td>{1e3 * v['wide']['sigma']:.1f} meV</td><td>{r['s']:.2f}</td>"
                             f"<td>{mv(r['tip'], 1)} mV</td><td>{mv(r['free'], 1)} mV</td><td>{mv(r['tip'] - r['free'], 1)} mV</td>"
                             f"<td>{100 * r['h_tip']:.0f} %</td></tr>")
        if ra and rb:
            wide_txt = f"""<p><b>A flat band as wide as the measured one.</b> With the Gaussian of 10 meV used so far the
simulated (+⁵) peak is {1e3 * a['cal']['rows'][0]['fwhm']:.0f} and {1e3 * b['cal']['rows'][0]['fwhm']:.0f} mV wide at half
maximum, the measured one {1e3 * a['cal']['measured_fwhm']:.0f} ({t1}) and {1e3 * b['cal']['measured_fwhm']:.0f} mV ({t2}).
σ = {1e3 * wa['sigma']:.1f} and {1e3 * wb['sigma']:.1f} meV reproduce the measured widths
(<code>stacking_wall_separation.py calibrate</code>), and the (+⁵) stack and the wall stacks were solved again with
that σ in the occupation as well as in the spectrum, so that the screening sees the wider band (contact potential 0,
biases ±0.2 V in 25 mV steps; <code>stacking_wall_separation.py wide</code>). The lever arm of the tip on the top
layer at the wall goes from {abs(a['lever_wall']):.2f} to {abs(wa['lever_wall']):.2f} ({t1}) and from
{abs(b['lever_wall']):.2f} to {abs(wb['lever_wall']):.2f} ({t2}); on the (+⁵) side, at its own bias, from
{abs(a['lever_rg']):.2f} to {abs(wa['lever_rg']):.2f} and from {abs(b['lever_rg']):.2f} to {abs(wb['lever_rg']):.2f}.
At s = {a['wall']['s']:.2f} the flat band bends by {mv(ra['tip'])} mV with the tip and {mv(ra['free'])} mV without it in
{t1} (σ = 10 meV: {mv(a['wall']['tip'])} and {mv(a['wall']['free'])} mV), {mv(rb['tip'])} and {mv(rb['free'])} mV in {t2}
({mv(b['wall']['tip'])} and {mv(b['wall']['free'])} mV); measured {mv(a['meas'])} and {mv(b['meas'])} mV.</p>
<div class="tablewrap"><table>
<tr><th>line</th><th>σ</th><th>s</th><th>bend with the tip</th><th>without the tip</th><th>tip-induced part</th><th>peak height (of (+⁵))</th></tr>
{''.join(wrows)}
</table></div>"""
    return f"""<p>Every stack of the local approximation was evaluated twice from the same self-consistent sweep: with
the layer potentials under the apex, U0(V), as in the maps, and with the far-field potentials U<sub>far</sub> of the
same stack at every bias, which contain the charge redistribution of the stack but no tip
(<code>stacking_wall_separation.py</code>). The bend is the shift of the flat-band peak (its centroid) from that of the
(+⁵) stack under the same conditions. At the position of the wall spectrum (s = {a['wall']['s']:.2f}) the bend is
{mv(a['wall']['tip'])} mV with the tip and {mv(a['wall']['free'])} mV without it in {t1}, {mv(b['wall']['tip'])} and
{mv(b['wall']['free'])} mV in {t2} (measured {mv(a['meas'])} and {mv(b['meas'])} mV): the bend of the model is charge
redistribution, and the tip changes it by {mv(a['wall']['tip'] - a['wall']['free'], 1)} and
{mv(b['wall']['tip'] - b['wall']['free'], 1)} mV. The tip does move the peaks, by {mv(a['ref_shift'], 1)} and
{mv(b['ref_shift'], 1)} mV on the (+⁵) stack at its bias, but by about as much at the wall, so the bend, a difference,
keeps little of it. The lever arm of the tip on the top layer is {abs(a['lever_rg']):.2f} and {abs(b['lever_rg']):.2f}
on the (+⁵) stack at its bias and {abs(a['lever_wall']):.2f} and {abs(b['lever_wall']):.2f} at the wall at the
wall-peak bias.</p>
<div class="tablewrap"><table>
<tr><th>line</th><th>s</th><th>bend with the tip</th><th>without the tip</th><th>tip-induced part</th></tr>
{''.join(rows)}
</table></div>
{wide_txt}
<figure><img src="{img_prefix}stacking_wall_separation.png" alt="bend with and without the tip against s">
<figcaption>Flat-band peak against that of the (+⁵) stack for the stacks of the local approximation, with the tip
(orange) and without it (blue), σ = 10 meV (lines) and the σ that matches the measured width (open markers); star:
the measured bend at the wall spectrum.</figcaption></figure>
<p><b>Why the bend needs no tuned work function in this picture.</b> A bend from charge redistribution belongs to
the sample: its sign follows the filling of the flat band, up where the band is empty and down where it is filled,
whatever the tip. Pure tip gating with a lever arm confined to the wall gives a bend whose sign is that of the bias
measured from the zero-TIBB bias, −CPot; seeing an upward bend at +23 mV and a downward one at −32 mV then requires
|CPot| below about 25 meV. Even when the tip does gate, with the same lever arm α at the wall and on the (+⁵) side,
the apparent energies are V = (E + α CPot)/(1 − α), and the bend, V<sub>wall</sub> − V<sub>(+⁵)</sub> =
(E<sub>wall</sub> − E<sub>(+⁵)</sub>)/(1 − α), no longer contains CPot: a uniform lever arm only magnifies the intrinsic
shift. The contact potential matters only through a lever arm that differs between the wall and the (+⁵) side, and
that is the situation the self-consistent model does not produce.</p>"""


def zero_bias_numbers(states):
    """dI/dV near zero bias (within 6 mV): wall over (+5), measured and simulated (sigma = 10 meV), and the
    (+5) zero-bias value over the (+5) flat-band peak."""
    out = {}
    for t, m in states.items():
        Vm, fb = np.asarray(m["measured"]["bias"]), np.asarray(m["fine_bias"])
        rg_m, w_m = np.asarray(m["measured"]["rg"], float), np.asarray(m["measured"]["wall"], float)
        rg_s = np.asarray(m["scan"][f"{m['best_cpot']:+.3f}"]["didv"])
        w_rec = m["runs"].get(f"wall_s{m['settings']['s']:.2f}")
        if w_rec is None:
            continue
        w_s = np.asarray(w_rec["didv"])
        z = lambda V, y: float(np.mean(y[np.abs(V) <= 0.006]))   # noqa: E731
        E_fb = m["settings"]["E_fb_measured"]
        pk = lambda V, y: float(np.max(y[np.abs(V - E_fb) <= 0.03]))   # noqa: E731
        out[t] = dict(wall_m=z(Vm, w_m) / z(Vm, rg_m), wall_s=z(fb, w_s) / z(fb, rg_s),
                      tail_m=z(Vm, rg_m) / pk(Vm, rg_m), tail_s=z(fb, rg_s) / pk(fb, rg_s))
    return out


def lever_arm_text(states):
    nums = separation_numbers(states)
    zb = zero_bias_numbers(states)
    eps0, d, c, eps = 8.854e-12, 0.7e-9, 0.335e-9, 2.6
    cg, ci = eps0 / d, eps0 * eps / c
    cap = lambda cg_, ci_: cg_ / (cg_ + ci_)   # noqa: E731
    lev = ""
    if nums:
        vals = [abs(v["lever_wall"]) for v in nums.values() if v.get("lever_wall")] + [abs(v["lever_rg"]) for v in nums.values()]
        lev = f"{min(vals):.2f} to {max(vals):.2f}"
    wide = ""
    if nums and all("wide" in v for v in nums.values()):
        wide = (" Widening the flat band to the measured width changes the lever arm by at most "
                f"{max(abs(abs(v['wide']['lever_wall']) - abs(v['lever_wall'])) for v in nums.values()):.2f} at the wall and "
                f"{max(abs(abs(v['wide']['lever_rg']) - abs(v['lever_rg'])) for v in nums.values()):.2f} on the (+⁵) side: E<sub>F</sub> "
                "lies 22 to 33 meV from the band, so a wider Gaussian puts more of the band's tail at E<sub>F</sub>, and in any "
                "case the cap, not the top layer, sets α.")
    data = ""
    if len(zb) == 2:
        t1, t2 = list(zb)
        data = (f"<p><b>What the spectra say about the states at E<sub>F</sub>.</b> On the (+⁵) side the measured dI/dV at zero bias "
                f"is {zb[t1]['tail_m']:.2f} ({t1}) and {zb[t2]['tail_m']:.2f} ({t2}) of the flat-band peak, the model's "
                f"{zb[t1]['tail_s']:.2f} and {zb[t2]['tail_s']:.2f}: the model does not overstate the weight at E<sub>F</sub>. At the "
                f"wall the measured zero-bias dI/dV is {zb[t1]['wall_m']:.2f} and {zb[t2]['wall_m']:.2f} of the (+⁵) value, the "
                f"model's {zb[t1]['wall_s']:.2f} and {zb[t2]['wall_s']:.2f}: the measured top layer is less depleted at "
                "E<sub>F</sub> than the model's, not more. Neither points to the missing depletion (both are top-layer "
                "LDOS; the layers below are not seen by the tip).</p>")
    return f"""<p>The top layer under the apex sits between the tip, coupled through the vacuum gap,
C<sub>gap</sub> ≈ ε₀/d ≈ {1e3 * cg:.0f} mF m⁻² at d = 0.7 nm, and two capacitances that hold its potential: its own
quantum capacitance e²D<sub>1</sub>(E<sub>F</sub>) (16 mF m⁻² per 10¹³ eV⁻¹ cm⁻²) and the layer below, through the
interlayer gap, C<sub>int</sub> = ε₀ε/c ≈ {1e3 * ci:.0f} mF m⁻² (ε = 2.6, c = 0.335 nm), in series with that layer's own
quantum capacitance. The lever arm is α ≈ C<sub>gap</sub>/(C<sub>gap</sub> + e²D<sub>1</sub> + C<sub>below</sub>). As
long as layer 2 has states at E<sub>F</sub>, C<sub>below</sub> ≈ C<sub>int</sub> and α ≤ C<sub>gap</sub>/(C<sub>gap</sub> +
C<sub>int</sub>) ≈ {cap(cg, ci):.2f}, whatever the top layer does. The model's {lev} sit at this cap: at the wall the
flat-band states leave the top layer for layer 2 and screen the tip from 0.34 nm below.{wide} The measured bends need
α ≈ 0.6, C<sub>below</sub> + e²D<sub>1</sub> ≈ 9 mF m⁻²: about eight interlayer gaps in series, more than the stack has.
The tip's field would have to reach through the whole six-layer stack, with no layer under the apex holding states
at E<sub>F</sub>. Ways α could be larger:</p>
<ul>
<li><b>A stack without states at E<sub>F</sub> under the tip.</b> A gap at E<sub>F</sub> through all layers: a correlated gap
of the flat band, or the gap a displacement field opens in a rhombohedral stack near neutrality (the field-induced gap
is in the γ0/γ1 model and appears where E<sub>F</sub> is in the gap, as in the polarized stacks, where the tip's share grew
to −22 meV at s = 0.45 for ε = 15; exchange is not in the model).</li>
<li><b>A smaller interlayer permittivity.</b> ε = 1 instead of 2.6 lowers C<sub>int</sub> to {1e3 * eps0 / c:.0f} mF m⁻² and
raises the cap to {cap(cg, eps0 / c):.2f}; values below the σ-band 2.6 have no physical basis.</li>
<li><b>A closer tip.</b> d = 0.4 nm raises C<sub>gap</sub> to {1e3 * eps0 / 0.4e-9:.0f} mF m⁻² and the cap to
{cap(eps0 / 0.4e-9, ci):.2f}. The tip radius changes the extent of the gated region more than α under the apex.</li>
<li><b>A lever arm larger at the wall than on the (+⁵) side.</b> This is what tip gating needs: a uniform α moves the
(+⁵) reference as well and leaves only the magnified intrinsic shift (section 6). The (+⁵) spectra limit α there: in
the fits of 940 (section 8) the lower side band at −256 mV comes out 36 to 41 mV too deep under the tip, so the
model's tip already bends the (+⁵) surface at that bias more than the measurement shows, not less.</li>
</ul>
{data}"""


def open_issues_text(states):
    return """<ul>
<li><b>Width of the flat band.</b> The bare γ0/γ1 flat band is 6 meV wide; a Gaussian of 10 meV in the occupation and
the LDOS gives a simulated (+⁵) peak of 34 to 35 mV against 42 to 45 mV measured. Section 6 widens the Gaussian to the
measured width. A Gaussian keeps the band's weight and centre; γ3, γ4 and Δ′ (the bandbending plan's phase P5) would
also reshape the band and move weight between layers.</li>
<li><b>Doping model.</b> The charge-balance shift exists with a fixed charge per layer (uniform doping) and nearly
vanishes with the doping compensated in the SiO₂ (polarized stack), which fits the (+⁵) spectrum of 940 better at
ε = 2.6. The per-layer charge also puts the (+⁴−) Fermi level 30 to 40 meV from its neutrality point, where the
measurement has it near neutrality; the charge-balance shift it produces is an upper estimate.</li>
<li><b>The (+⁴−)-side edge.</b> In 940 the lowest edge above E<sub>F</sub> on the stacking-fault side rises from +66 to
+143 mV within 5 nm of the wall centre, over a stretch where s(x) only falls from 1.00 to 0.86. No model run follows
it; it rises at positive bias next to the wall where the flat band sinks at negative bias, which a rigid potential
shift cannot do and tip gating does.</li>
<li><b>Tests on the instrument.</b> A tip-gating bend scales with the gap capacitance and with the tip work function;
a charge-balance bend does not. The bend against the setpoint (tip height) at one wall, and at the same wall with a
tip of different work function, separates them.</li>
<li><b>Different tips.</b> 514 and 940 were recorded on different days. One contact potential for both lines is an
assumption of the asymmetry fit; each line's own (+⁵) fit gives 0 ± 0.05 eV on its own.</li>
</ul>"""


def tip_gating_text():
    return """<p>Close to the wall the flat-band peak does not only fade, it moves: in spectrum 514, where the band is
mostly empty and seen at positive bias, it climbs from +22 mV to about +55 mV; in spectrum 940, where it is mostly
filled and seen at negative bias, it sinks from −32 mV to about −88 mV. The SWMcC model without electrostatics shifts
it by only +5 meV in both cases, so the bend is not in the band structure of the wall.</p>
<p>The first explanation considered was tip gating. The tunnel junction is a capacitor with the sample as one plate;
how much of the bias drops inside the sample depends on how easily the surface under the tip piles up or sheds
charge. Far from the wall the flat band, about 10¹³ states per cm² within some 10 meV, makes the top layer a metal
plate: the whole bias falls across the vacuum gap and the peak sits at its true energy. At the wall the flat-band
states leave the top layer (on the stacking-fault side they sit in layer 2). If what is left under the tip were the
Dirac cone of the (+⁴−) surface near its node, a fraction f of the bias would drop inside the sample and shift the
local bands by f·eV: up at positive bias, down at negative bias, so that an empty band (seen at positive bias) bends
up and a filled one bends down, with V<sub>app</sub> = E<sub>FB</sub> / [e(1 − f)].</p>
<div class="tablewrap"><table>
<tr><th>line spectrum</th><th>flat band far from the wall</th><th>at one third peak height</th><th>shift</th><th>f = 1 − V<sub>far</sub>/V<sub>app</sub></th></tr>
<tr><td>514 (band mostly empty)</td><td>+22 mV</td><td>+49 to +60 mV</td><td>+27 to +38 mV</td><td>0.54 to 0.62</td></tr>
<tr><td>940 (band mostly filled)</td><td>−32 mV</td><td>−88 mV</td><td>−56 mV</td><td>0.63</td></tr>
</table></div>
<p>Scale: a 0.7 nm gap gives C<sub>gap</sub> = ε₀/d ≈ 13 mF m⁻². The intact flat band has C<sub>q</sub> = e²D ≈ 1.6 F m⁻²,
so f ≈ 0.01: the band is pinned. f ≈ 0.6 needs C<sub>q</sub> ≈ 9 mF m⁻², D(E<sub>F</sub>) ≈ 5 × 10¹² eV⁻¹ cm⁻², the value
of the (+⁴−) surface cone 10 to 20 meV from its Dirac point. The estimate treats the top layer alone; the layers
below hold its potential as well (section 9), so the same holds for them. Both lines give the same f with opposite bias, and in 940
an edge on the stacking-fault side at positive bias rises by 50 to 90 mV next to the same wall where the flat band at
negative bias sinks; a rigid offset of the bands, such as a polarization of the (+⁴−) stack, would move both the same
way. The estimate leaves open whether the density of states under the tip really falls that far while the flat-band
peak is still visible. The self-consistent calculation below tests it, and at the fitted work function it does not
find it: the top layer keeps about a quarter of its flat-band states at s = 0.75 and the tip moves the peak by a few
millivolts.</p>"""


def page(tags, img_prefix=""):
    infos = [load(t) for t in tags]
    states = load_states(tags)
    rows = []
    for d in infos:
        f = d["fit"]
        rows.append(
            f"<tr><td>{html.escape(d['sm4'])}</td>"
            f"<td>{f['x0']:.1f} ± {f['x0_err']:.1f} nm</td>"
            f"<td>{f['ell']:.1f} ± {f['ell_err']:.1f} nm</td>"
            f"<td>{f['width_10_90']:.1f} nm</td>"
            f"<td>{1e3 * d['flat_band_experiment_V']:+.0f} mV</td>"
            f"<td>{1e3 * d['energy_shift_eV']:+.1f} meV</td></tr>")
    m = infos[0]["model"]
    p = infos[0]["parameters"]
    figs = []
    for d in infos:
        t = d["tag"]
        figs.append(
            f'<figure><img src="{img_prefix}stacking_wall_{t}.png" alt="dI/dV and model LDOS, line spectrum {t}">'
            f"<figcaption>Line spectrum {t}, l = {d['fit']['ell']:.1f} nm. Top: measured dI/dV(V, x) and top-layer "
            f"LDOS(E, x). Middle: bias derivatives. Bottom: fitted profile and spectra at three positions.</figcaption></figure>")
    summary_figs = []
    for t in tags:
        if os.path.exists(os.path.join(EXAMPLES, f"stacking_wall_summary_{t}.png")):
            summary_figs.append(
                f'<figure><img src="{img_prefix}stacking_wall_summary_{t}.png" alt="flat-band bend at the wall, line {t}">'
                f"<figcaption>Line spectrum {t}. Top, zoomed on the flat band: measured dI/dV, the local approximation "
                f"(Poisson solver, uniform stacks along the fitted profile, tip included) and the SWMcC wall model with the "
                f"self-consistent on-site energies (no tip); arrows mark the measured wall spectrum. Bottom left: the "
                f"flat-band peak along the line, with the SWMcC model plus the tip-induced bump at the tip positions solved "
                f"(diamonds). Bottom right: the bend at the wall spectrum, each model against its own (+⁵) peak "
                f"(<code>stacking_wall_summary_figure.py</code>).</figcaption></figure>")
    has_tibb = bool(states)
    tibb_parts = ""
    if has_tibb:
        tibb_parts = f"""
<h2>3. Self-consistent electrostatics: the model, the (+⁵) fit and the tip work function</h2>
{tibb_setup_text(states, img_prefix)}
{workfunction_text(states)}
<h2>4. Across the wall in the local approximation</h2>
{tibb_text(states, img_prefix)}
<h2>5. The SWMcC wall model with the self-consistent on-site energies</h2>
{hybrid_text(states, img_prefix)}
<h2>6. Charge redistribution and tip-induced bending separated</h2>
{separation_text(states, img_prefix)}
<h2>7. The tip's own field in the SWMcC model</h2>
{bump_text(states, img_prefix)}
{cpot_sensitivity_text(states, img_prefix)}
{lateral_text(states, img_prefix)}
<h2>8. The doping model of 940</h2>
{epsfit_text(img_prefix)}
{polar_series_text(img_prefix)}
{eps26_fit_text(img_prefix)}
{eps26_series_text(img_prefix)}
{eps26_peaks_series_text(img_prefix)}
{doping_extra_text(img_prefix)}
<h2>9. How could the lever arm be larger?</h2>
{lever_arm_text(states)}
<h2>10. Open issues</h2>
{open_issues_text(states)}"""

    return f"""<title>Stacking Wall LDOS</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
{STYLE}
<main>
<div class="eyebrow">Six-layer graphite · (+⁴−) | (+⁵) domain wall · STM line spectra 514 and 940</div>
<h1>Flat-band bending at a stacking domain wall: what the models reproduce</h1>
<p class="lede">Status report on the bend of the flat-band peak at the (+⁴−)|(+⁵) wall (flake MK_ABC_FLG_25, 9 K):
the SWMcC wall model, the self-consistent Poisson solver of the bandbending engine, and their combination.</p>

<h2>Abstract</h2>
{abstract_text(states) if has_tibb else ""}

<h2>Status in numbers</h2>
{status_table(states) if has_tibb else ""}
<div class="figs">
{''.join(summary_figs)}
</div>

<h2>1. The measurement and the SWMcC wall model</h2>
<p>The k·p Slonczewski–Weiss–McClure Hamiltonian is linear in the momentum from the valley centre,
<i>H</i><sub>S</sub>(<b>p</b>) = <i>C</i><sub>S</sub> + <i>p<sub>x</sub></i><i>X</i><sub>S</sub> + <i>p<sub>y</sub></i><i>Y</i><sub>S</sub>,
for both stackings S = (+⁵) ABCABC and (+⁴−) ABCABA. Across the wall every matrix element that differs
(the γ1 dimer of the top layer, γ3, γ4, γ2, γ5 and the Δ′ terms of layers 5 and 6) follows one profile:</p>
<div class="eq">H(x, <b>p</b>) = [1 − s(x)] H<sub>(+⁵)</sub>(<b>p</b>) + s(x) H<sub>(+⁴−)</sub>(<b>p</b>),&nbsp;&nbsp;
s(x) = [1 − tanh(x / l)] / 2<span class="r"> &nbsp;(stacking fault at x &lt; 0)</span></div>
<p>For every conserved <i>k<sub>y</sub></i> the problem is diagonalised in plane waves along the wall normal
(cell {m['length'] / 10:.0f} nm, |q| ≤ {m['p_cut']} Å⁻¹, {m['n_pw']} waves per orbital; the second wall that
periodicity needs is {m['length'] / 20:.0f} nm away). The layer-resolved LDOS(E, x) sums the
<i>k<sub>y</sub></i> mesh (step {m['dky']:.3f} Å⁻¹) with spin, both valleys and a Gaussian of
σ = {1e3 * m['sigma']:.0f} meV. Parameters of McEllistrim et al.: γ0 = {p['gamma0']:.3f}, γ1 = {p['gamma1']:.3f},
γ2 = {p['gamma2']:.3f}, γ3 = {p['gamma3']:.3f}, γ4 = {p['gamma4']:.3f}, γ5 = {p['gamma5']:.3f},
Δ′ = {p['Delta_prime']:.3f} eV. Without a wall the layer-summed LDOS equals the k·p DOS of the repository to 1 %,
and a monolayer gives 2|E| / (πħ²v²) to 10⁻⁴. The k·p arrays are numbered from the bottom layer (the top-layer LDOS
is the last), the engine's from the top; potentials passed from one to the other are reversed.</p>
<p>Wall length: every spectrum of a line is written as (1 − s) S<sub>(+⁴−)</sub>(V) + s S<sub>(+⁵)</sub>(V) with the
end-member spectra averaged far from the wall (least squares over −0.3 … 0.3 V), and s(x) is fitted with the same
tanh step. The flat-band weight within ±25 meV gives the same lengths within 0.5 nm. The model energy axis is shifted
rigidly so that its (+⁵) flat band sits on the measured one.</p>
<div class="tablewrap"><table>
<tr><th>line spectrum</th><th>wall centre x0</th><th>l (tanh)</th><th>10–90 % width</th><th>flat band (+⁵)</th><th>model shift</th></tr>
{''.join(rows)}
</table></div>
<div class="figs">
{''.join(figs)}
</div>
<ul>
<li>The model has the flat-band peak on the (+⁵) surface; on the (+⁴−) surface it is absent and sits in layer 2
(layer-resolved maps in <code>examples/stacking_wall_layers_*.png</code>). The (+⁵) gap faces the V-shaped surface
band of (+⁴−).</li>
<li>Quasiparticle-interference fringes fan out from the wall on the (+⁴−) top layer only, as in Fig. 2 and 3 of the
paper; none inside the (+⁵) gap.</li>
<li>The fitted l is an apparent LDOS width that includes the lateral extent of the surface states, not the
structural soliton width.</li>
<li>Linear interpolation passes through a midpoint with both dimer couplings at γ1/2 rather than the saddle-point
stacking of the real shift path.</li>
<li>The k·p trigonal-warping legs extend to large momentum, so the LDOS is reliable within about ±0.4 eV.</li>
</ul>

<h2>2. The tip-gating estimate</h2>
{tip_gating_text()}
{tibb_parts}
<footer>Kandrai et al., arXiv:2610.20905 (2026) · few-layer-graphite, branch claude/eloquent-franklin-q9q4cb ·
line spectra 514 and 940 are not part of the repository.</footer>
</main>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=["514", "940"])
    ap.add_argument("--out", default=os.path.join(EXAMPLES, "stacking_wall_report.html"))
    ap.add_argument("--page-only", default=None, metavar="PATH")
    args = ap.parse_args()
    body = page(args.tags)
    head, _, rest = body.partition("<main>")
    with open(args.out, "w") as fh:
        fh.write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                 '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                 + head + "</head>\n<body>\n<main>" + rest + "</body>\n</html>\n")
    print(f"report written to {args.out}")
    if args.page_only:
        with open(args.page_only, "w") as fh:
            fh.write(body)
        print(f"page without skeleton written to {args.page_only}")


if __name__ == "__main__":
    main()
