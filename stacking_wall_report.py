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
