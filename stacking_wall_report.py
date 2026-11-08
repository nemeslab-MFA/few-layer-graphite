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
    out = {"meas": _peak(np.asarray(state["measured"]["bias"]), np.asarray(state["measured"]["wall"]), E_fb, 0.09)}
    loc_path = os.path.join(EXAMPLES, f"stacking_wall_tibb_map_{tag}.npz")
    hyb_path = os.path.join(EXAMPLES, f"stacking_wall_hybrid_{tag}.npz")
    fit_path = os.path.join(EXAMPLES, f"stacking_wall_{tag}.json")
    if os.path.exists(loc_path):
        with np.load(loc_path) as f:
            j = int(np.argmin(np.abs(f["dist"] - x_wall)))
            out["local"] = _peak(f["bias"], f["didv"][:, j], E_fb, 0.09)
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
    return f"""<h2>Hybrid: the wall model with the Poisson potentials</h2>
<p>The two calculations are complementary: the k·p wall model has the wall and its fringes but no electrostatics,
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
{mv(b['local'])} and {mv(b['hybrid'])} mV for {t2}. The hybrid bends less than the local approximation although
it uses the same potentials: the top-layer potential of the stacks moves from {mv(a['V1_0'])} to
{mv(a['V1_075'])} meV ({t1}) and from {mv(b['V1_0'])} to {mv(b['V1_075'])} meV ({t2}) between s = 0 and 0.75,
and in the γ0/γ1 model the top-layer flat band is a pure layer-1 state that takes the whole shift, while in the
k·p model the flat-band state near the wall spreads into the layers whose potentials barely move and follows only
part of it.</p>
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
    return f"""<h3>The tip-induced bump</h3>
<p>The Poisson solver also gives the potential under the apex, U0,l(V) − U_far,l, which moves with the tip. It is
added to the hybrid as a Gaussian bump of width w = √(2Rd) = {s0['spot']:.1f} nm centred on the tip, with the
amplitude of the stack at s(x_tip) at the bias where the flat-band peak is measured at the wall ({biases}), and the
wall model is solved once per tip position (cell {s0['length']:.0f} nm, k_y step {s0['dky']} Å⁻¹, {1e3 * s0['sigma']:.0f} meV
broadening; <code>stacking_wall_tibb_bump.py</code>). The table gives the apparent flat-band peak under the tip
without and with the bump (the energy grid of the peak is 2.5 meV; the centroid of the peak resolves below that);
the largest change the bump makes to the peak centroid is {worst}, against a measured bend of 35 to 55 mV. Within
the self-consistent model the tip's own potential does not move the flat band at the wall; the bend it leaves
unexplained is not a tip effect that the local approximation missed.</p>
<div class="tablewrap"><table>
<tr><th>line</th><th>tip from the wall centre</th><th>s at the tip</th><th>top-layer bump</th><th>peak without / with bump</th><th>change of the maximum</th><th>change of the centroid</th></tr>
{''.join(table)}
</table></div>
<figure><img src="{img_prefix}stacking_wall_bump.png" alt="top-layer LDOS under the tip with and without the bump">
<figcaption>Top-layer LDOS under the tip for the hybrid without (blue) and with (orange) the tip-induced bump at
each tip position, and the resulting flat-band peak against the tip position.</figcaption></figure>
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
    rows = []
    for r in sorted(runs.values(), key=lambda r: r["x_tip"]):
        a, b = kp.get(f"{r['x_tip']:+.1f}_tip"), kp.get(f"{r['x_tip']:+.1f}_none")
        rows.append(dict(x=r["x_tip"], s=r["s"], dU=r["dU_apex"], dU_local=r["dU_local"], dU_uniform=r.get("dU_uniform"),
                         peak=(a["peak"] + shift) if a else None, peak_ref=(b["peak"] + shift) if b else None,
                         height=a["height"] if a else None, height_ref=b["height"] if b else None))
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
                     f"{1e3 * r['peak_ref']:+.0f} / {1e3 * r['peak']:+.0f} mV ({1e3 * (r['peak'] - r['peak_ref']):+.1f})")
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
    worst_txt = (f"the largest change of the k·p peak under the tip is {1e3 * worst:.1f} mV (grid 2.5 mV)"
                 if worst is not None else "the k·p spectra are not finished")
    return f"""<h3>The lateral problem</h3>
<p>The local approximation was dropped last. The Poisson equation was solved in three dimensions with the wall in
it (bandbending branch <code>lateral-solver</code>, <code>engine/lateral.py</code>; driver
<code>stacking_wall_lateral.py</code>): a Cartesian finite-volume grid with mirror symmetry across the line, the
same hyperbolic tip (R = {sa['radius']:.0f} nm, d = {sa['separation']:.1f} nm, CPot = 0) as a Dirichlet body, the
six layers as rigid-band charge sheets whose density follows the zero-field layer DOS of the local stack
(<code>WallTB</code> tables at s = 0, 0.15, …, 1 along the fitted profile), the oxide and the gate below, and far-field
boundaries from the tip-free columns of the local stack plus the engine's axisymmetric tip field. One energy
reference serves all stacks: the fixed charge per layer makes the (+⁵) stack neutral with its Fermi level
{1e3 * sa['mu0']:+.0f} meV ({t1}) and {1e3 * sb['mu0']:+.0f} meV ({t2}) below its neutrality point, as measured, and
the (+⁴−) stack, joined to it, shares that Fermi level and the gate; its top-layer potential far from the wall comes
out at {1e3 * a['sf_top']:+.0f} meV ({t1}) and {1e3 * b['sf_top']:+.0f} meV ({t2}) (the local approximation, with its
own Fermi level per stack and the engine's slightly different layer charges, gave about 4 meV more). The tip sits at the
bias of the measured wall peak, {1e3 * sa['bias']:+.0f} mV ({t1}) and {1e3 * sb['bias']:+.0f} mV ({t2}), and was placed
at {a['n_positions']} positions across the wall; a solve has about {a['nodes'] / 1e3:.0f} thousand unknowns and takes
{a['seconds']:.0f} s (Newton with algebraic-multigrid conjugate gradients).</p>
<p>Three results. First, without the tip the built-in step of the layer potentials across the wall follows the
columns of the local stack to within {1e3 * max(a['step_dev'], b['step_dev']):.1f} meV: the stack screens laterally
over a length far below the {sa['ell']:.0f} to {sb['ell']:.0f} nm of the wall, so the local treatment of the far field
was exact for practical purposes. Second, the tip-induced bending under the apex differs from the local approximation
by {rel_txt} at the positions studied (table; the reference is the uniform stack at s(x_tip) solved with the same
code): the field of the tip over the fading flat band does reach into the (+⁴−) region, but the gain is a fraction of a
few-meV effect.{engine_txt} The tip-induced part of the top-layer potential with
the tip at the wall centre has a full width at half maximum of {a['fwhm']:.0f} nm ({t1}; {a['left']:.0f} nm towards
(+⁴−), {a['right']:.0f} nm towards (+⁵)) against the 12.5 nm of the Gaussian bump used above. Third, fed into the
k·p wall model in place of the Gaussian bump, {worst_txt}. The conclusion of the local approximation stands: the
self-consistent tip-induced band bending at the flat-band bias is a few meV also when the lateral problem is solved,
and the measured 35 to 55 mV bend is not a tip-gating effect of this model.</p>
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
"""


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
    return f"""{workfunction_text(states)}
<p><b>What the calculation gives.</b> With the tip and the doping fixed on the (+⁵) side, the
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
</div>
{hybrid_text(states, img_prefix)}
{bump_text(states, img_prefix)}
{lateral_text(states, img_prefix)}
<p><b>Reading.</b> In the local approximation the self-consistent tip-induced band bending is too small to
bend the flat band by the measured 35 to 55 mV while the peak is still visible; for that the density of
states at E<sub>F</sub> under the tip would have to fall to the (+⁴−) level, about 5 × 10¹² eV⁻¹ cm⁻², and
only on the (+⁴−) stack itself does the tip reach 15 to 30 meV of bending. {reading_lateral(states)}The fit of the (+⁵) spectra leaves the contact
potential within about ±0.05 eV of zero, consistent with the sign reversal of the bend between the two
measurements.</p>"""


def reading_lateral(states):
    """One sentence of the reading paragraph: the lateral problem, solved or still open."""
    nums = {t: lateral_numbers(t) for t in states}
    if any(v is None for v in nums.values()):
        return ("What the local approximation cannot hold is the tip's field reaching across the wall into the (+⁴−) "
                "region, where screening is weak, while the tip sits over the fading flat band; that lateral problem, "
                "a Poisson solution with the wall in it, is the natural next step and is outside the axisymmetric engine. ")
    rel = max(abs(r["dU"][0] / (r["dU_uniform"] or r["dU_local"])[0] - 1.0) for v in nums.values() for r in v["rows"]
              if abs((r["dU_uniform"] or r["dU_local"])[0]) > 1e-3)
    return (f"The tip's field reaching across the wall into the weakly screening (+⁴−) region was then checked with the "
            f"three-dimensional solve: it changes the bending under the apex by at most {100 * rel:.0f} %, so the lateral "
            "problem does not rescue the tip-gating explanation either. ")


def tibb_section(tags, img_prefix=""):
    """Section on the tip-induced band-bending calculation (stacking_wall_tibb.py), or ''."""
    states = {}
    for t in tags:
        path = os.path.join(EXAMPLES, f"stacking_wall_tibb_{t}.json")
        if os.path.exists(path):
            with open(path) as fh:
                states[t] = json.load(fh)
    if not states:
        return ""
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
    return f"""
<h2>Self-consistent tip-induced band bending (bandbending engine)</h2>
<p>Three laterally uniform six-layer stacks under the same tip, each solved self-consistently
(Hartree + γ0/γ1 tight binding, <code>GrapheneMultilayer</code> of the
<a href="https://github.com/nemeslab-MFA/bandbending">bandbending</a> engine): the (+⁵) stack far from the
wall, the wall itself with the top dimer shared between the (+⁵) bond (weight 1 − s) and the (+⁴−) bond
(weight s, s = {s0['s']:.2f} at the position of the bend), and the (+⁴−) stack. The wall Hamiltonian closes
a bond loop whose flux is the phase of the structure factor, so it is evaluated with a complex Hamiltonian
averaged over that phase (<code>WallTB</code> in <code>stacking_wall_tibb.py</code>; it reproduces the
engine's end members exactly and a brute-force zone sum to 1 %). Tip radius {s0['radius']:.0f} nm,
separation {s0['separation']:.1f} nm, barrier {s0['barrier']:.1f} eV, {s0['temperature']:.0f} K, Gaussian
width {1e3 * s0['broadening']:.0f} meV of the tight-binding occupation and LDOS (the γ0/γ1 flat band is
6 meV wide, the measured one 26 mV), lock-in {1e3 * s0['modulation']:.2f} mV; 90 nm SiO₂ over the
grounded Si. The doping is set from the measured flat-band energy of the (+⁵) side (the isolated stack is
neutral with E<sub>F</sub> that far from its flat band); the fixed charge per layer this implies is used
unchanged for the wall and the (+⁴−) stack, and the tip contact potential W<sub>tip</sub> − W<sub>sample</sub>
is the value, scanned in 0.1 eV steps, at which the simulated (+⁵) spectrum (<code>tunneling_spectrum</code>:
frozen-potential term plus band-bending term) is closest to the measurement far from the wall, one scale
factor free.</p>
<div class="tablewrap"><table>
<tr><th>line spectrum</th><th>contact potential</th><th>E<sub>F</sub> − E<sub>CNP</sub></th><th>flat band (+⁵): measured / simulated</th><th>flat band at the wall: measured / simulated</th><th>U<sub>1</sub>(apex) − U<sub>1</sub>(far) at the peak: (+⁵) / wall / (+⁴−)</th><th>residual (+⁵) / wall / (+⁴−)</th></tr>
{''.join(rows)}
</table></div>
<figure><img src="{img_prefix}stacking_wall_tibb.png" alt="measured and simulated dI/dV at the three positions">
<figcaption>Measured dI/dV (black) and simulated top-layer dI/dV (orange, scaled; dashed: frozen-potential
term) far on the (+⁵) side, at the wall and far on the (+⁴−) side, with the contact-potential scan.</figcaption></figure>
{tibb_text(states, img_prefix)}
"""


def page(tags, img_prefix=""):
    infos = [load(t) for t in tags]
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

    return f"""<title>Stacking Wall LDOS</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
{STYLE}
<main>
<div class="eyebrow">Six-layer graphite · (+⁴−) | (+⁵) domain wall · SWMcC model against STM</div>
<h1>Top-surface LDOS across a single-layer stacking shift</h1>
<p class="lede">The Hamiltonian of hexalayer rhombohedral graphite is interpolated into that of the
stacking fault across one wall, with the wall length taken from the measured dI/dV line spectra
(flake MK_ABC_FLG_25, 9 K). Code: <code>stacking_wall.py</code>, <code>stacking_wall_figure.py</code>.</p>

<h2>Model</h2>
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
and a monolayer gives 2|E| / (πħ²v²) to 10⁻⁴.</p>

<h2>Wall length from the measurements</h2>
<p>Every spectrum of a line is written as (1 − s) S<sub>(+⁴−)</sub>(V) + s S<sub>(+⁵)</sub>(V) with the end-member
spectra averaged far from the wall (least squares over −0.3 … 0.3 V), and s(x) is fitted with the same tanh step.
The flat-band weight within ±25 meV gives the same lengths within 0.5 nm. The model energy axis is shifted
rigidly so that its (+⁵) flat band sits on the measured one.</p>
<div class="tablewrap"><table>
<tr><th>line spectrum</th><th>wall centre x0</th><th>l (tanh)</th><th>10–90 % width</th><th>flat band (+⁵)</th><th>model shift</th></tr>
{''.join(rows)}
</table></div>

<h2>Figures</h2>
<div class="figs">
{''.join(figs)}
</div>

<h2>What the model reproduces, and its limits</h2>
<ul>
<li>Flat-band peak on the (+⁵) surface; on the (+⁴−) surface it is absent and sits in layer 2 (layer-resolved
maps in <code>examples/stacking_wall_layers_*.png</code>). The (+⁵) gap faces the V-shaped surface band of (+⁴−).</li>
<li>Quasiparticle-interference fringes fan out from the wall on the (+⁴−) top layer only, as in Fig. 2 and 3
of the paper; none inside the (+⁵) gap.</li>
<li>The fitted l is an apparent LDOS width that includes the lateral extent of the surface states, not the
structural soliton width.</li>
<li>Linear interpolation passes through a midpoint with both dimer couplings at γ1/2 rather than the
saddle-point stacking of the real shift path; no layer-dependent doping is included.</li>
<li>The k·p trigonal-warping legs extend to large momentum, so the LDOS is reliable within about ±0.4 eV.</li>
</ul>
<h2>Why the flat band bends at the wall</h2>
<p>Close to the wall the flat-band peak does not only fade, it moves: in spectrum 514, where the band is
mostly empty and seen at positive bias, it climbs from +22 mV to about +55 mV; in spectrum 940, where it is
mostly filled and seen at negative bias, it sinks from −32 mV to about −88 mV. The electrostatics-free model
shifts it by only +5 meV in both cases, so the bend is not in the band structure. It is the tip gating the
sample.</p>
<p>The tunnel junction is a capacitor with the sample as one plate. How much of the bias drops across the
vacuum gap and how much inside the sample depends on how easily the surface under the tip can pile up or
shed charge. A large density of states at the Fermi level means a tiny change of potential moves a lot of
charge, so the surface screens the tip almost perfectly and the whole bias falls across the gap. The flat
band is exactly such a reservoir, about 10¹³ states per cm² within some 10 meV: far from the wall the
top layer behaves like a metal plate, the bands inside do not move, and the peak sits at its true energy.
At the wall the flat-band states leave the top layer (they sit in layer 2 on the stacking-fault side).
What is left under the tip is the Dirac cone of the (+⁴−) surface near its node, a hundred times fewer
states. The surface turns from a metal plate into a thin semiconductor, the field of the tip now reaches
into it, and a fraction f of the bias drops inside the sample: the local bands shift rigidly by f·eV.</p>
<p>The direction follows the sign of the bias, not the filling of the band. At positive sample bias the tip
is negative with respect to the sample; it pushes electrons away and lifts the local bands, so a state that
is truly at E<sub>FB</sub> is only reached at a larger bias and the peak appears higher. At negative bias
the tip pulls electrons in, the bands bend down and the peak appears lower. The filling of the flat band
only decides which sign of bias is needed to see it: an empty band is seen at positive bias and bends up,
a filled one at negative bias and bends down. The effect feeds back on itself, because the bending moves
the peak further from zero bias, where it is measured, so V<sub>app</sub> = E<sub>FB</sub> / [e(1 − f)].
Both spectra give the same f ≈ 0.6 where the peak has lost two thirds of its height, with opposite bias;
in 940 a feature at +69 mV on the stacking-fault side bends up by 50 to 90 mV next to the same wall where
the flat band bends down, which rules out a fixed offset such as the polarization of the (+⁴−) stack.</p>
<div class="tablewrap"><table>
<tr><th>line spectrum</th><th>flat band far from the wall</th><th>at one third peak height</th><th>shift</th><th>f = 1 − V<sub>far</sub>/V<sub>app</sub></th></tr>
<tr><td>514 (band mostly empty)</td><td>+22 mV</td><td>+49 to +60 mV</td><td>+27 to +38 mV</td><td>0.54 to 0.62</td></tr>
<tr><td>940 (band mostly filled)</td><td>−32 mV</td><td>−88 mV</td><td>−56 mV</td><td>0.63</td></tr>
</table></div>
<p>Scale: a 0.7 nm gap gives C<sub>gap</sub> = ε₀/d ≈ 13 mF m⁻². The intact flat band has
C<sub>q</sub> = e²D ≈ 1.6 F m⁻², so f ≈ 0.01 and the bending stays below 1 mV: the band is pinned. f ≈ 0.6 needs
C<sub>q</sub> ≈ 9 mF m⁻², i.e. D(E<sub>F</sub>) ≈ 5 × 10¹² eV⁻¹ cm⁻², the value of the (+⁴−) surface cone
10 to 20 meV from its Dirac point. The sign reversal between −30 and +70 mV in spectrum 940 also puts the
flat-band voltage of the junction, set by the work-function difference of tip and sample, within a few tens
of mV of zero.</p>
{tibb_section(tags, img_prefix)}
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
