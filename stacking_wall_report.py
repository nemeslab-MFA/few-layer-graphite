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
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.2rem; margin-top: 0.8rem; }
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
    out = {"E_fb": E_fb, "meas_wall_peak": _peak(bias, meas["wall"], E_fb, 0.09)}
    for key, run in (("rg", best), ("wall", wall), ("sf", sf)):
        if run is None or not fine.size:
            continue
        out[f"{key}_peak"] = _peak(fine, run["didv"], E_fb, 0.09)
        out[f"{key}_frozen_peak"] = _peak(fine, run["didv_frozen"], E_fb, 0.09)
        out[f"{key}_bend"] = _bending_at(run["sweep"], out[f"{key}_peak"])
        out[f"{key}_mu0"] = run["sweep"]["mu0"]
        out[f"{key}_residual"] = run["residual"]
    return out


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
TIBB_TEXT
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
