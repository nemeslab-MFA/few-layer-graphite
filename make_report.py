#!/usr/bin/env python3
"""
Build an HTML report that overlays the SWMcC tight-binding model (and the
paper's k.p model) on the figures of McEllistrim, Garcia-Ruiz, Goodwin and
Fal'ko, arXiv:2302.07374v1 (2023):

  * Fig. 1 band panels (3D view, azimuth 0): the p_y = 0 cut of the model drawn
    in the panel's own axes (E/gamma1 versus p_x/p_c);
  * Fig. 2 ARPES constant-momentum cuts through K along p_y: model bands on the
    simulated intensity maps;
  * Fig. 7 band panels: paper panel beside the model cut in the same units;
  * Fig. 1 and Fig. 7 DOS panels: model curves on the published curves;
  * tables of band-edge energies and DOS deviations.

Usage:
    python make_report.py --pdf arXiv-2302.07374v1.pdf [--out report] [--dk 0.001]

Requires PyMuPDF, Pillow, NumPy and Matplotlib.
"""
import argparse
import base64
import html
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import multilayer_graphene as mlg  # noqa: E402
import paper_figures as pf  # noqa: E402
import compare_with_mcellistrim  # noqa: E402

PAPER = "McEllistrim, Garcia-Ruiz, Goodwin and Fal'ko, arXiv:2302.07374v1 (2023)"
COL_TB = "#111111"
COL_KP = "#eb6834"
HALO = dict(color="white", linewidth=2.4, alpha=0.85)


# =============================================================================
# model helpers
# =============================================================================

def hbar_v():
    p = mlg.get_parameters()
    return 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma0"]


def p_c():
    return mlg.gamma1 / hbar_v()


def bands_along(seq, direction, t, model="tb"):
    """Band energies (eV) along K + t * direction (t in 1/A)."""
    K = mlg.K_point(+1)
    d = np.asarray(direction, dtype=float)
    kx, ky = K[0] + t * d[0], K[1] + t * d[1]
    if model == "tb":
        return mlg.eigenvalues(kx, ky, stacking_type=seq)
    return mlg.eigenvalues(t * d[0], t * d[1], stacking_type=seq, model="kp")


def dos_curve(seq, e_lim, model, dk, bins):
    return mlg.calculate_density_of_states((-e_lim, e_lim), stacking_type=seq, model=model,
                                           dk=dk, num_energy_points=bins)


def halo_plot(ax, x, y, color, lw=1.0, ls="-", zorder=5, label=None):
    ax.plot(x, y, zorder=zorder, **HALO)
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, zorder=zorder + 1, label=label)


# =============================================================================
# figures
# =============================================================================

def fig_bands_fig1(img, panels, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 7.0))
    t = np.linspace(-3.0, 3.0, 900)          # the paper plots the disk |p| <= 3 p_c
    for ax, pan in zip(axes, panels):
        xa, xb, _, _ = pan["crop"]
        c, r0, sg, sp = pan["axis_col"], pan["row0"], pan["px_per_gamma1"], pan["px_per_pc"]
        x0 = int(max(xa, c - 3.35 * sp)); x1 = int(min(xb, c + 3.35 * sp))
        y0 = int(max(0, r0 - 2.3 * sg)); y1 = int(min(img.shape[0], r0 + 2.3 * sg))
        crop = img[y0:y1, x0:x1]
        extent = [(x0 - c) / sp, (x1 - c) / sp, (r0 - y1) / sg, (r0 - y0) / sg]
        ax.imshow(crop, extent=extent, aspect="auto", interpolation="bilinear")
        E = bands_along(pan["name"], (1.0, 0.0), t * p_c()) / mlg.gamma1
        for b in range(E.shape[1]):
            halo_plot(ax, t, E[:, b], COL_TB, lw=0.9)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("p_x / p_c"); ax.set_title(f"{pan['name']}: Fig. 1 panel, model cut at p_y = 0")
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax.set_yticks([-2, -1, 0, 1, 2])
    axes[0].set_ylabel("E / γ1")
    fig.tight_layout()
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 88})
    plt.close(fig)


def fig_arpes_fig2(img, panels, out):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6))
    axes = axes.ravel()
    t = np.linspace(-0.28, 0.28, 700)
    for ax, pan in zip(axes, panels):
        x0, x1, y0, y1 = pan["frame"]
        crop = img[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
        extent = [(x0 + 2 - pan["k_col"]) / pan["px_per_invA"], (x1 - 1 - pan["k_col"]) / pan["px_per_invA"],
                  float(pan["emap"](y1 - 1)), float(pan["emap"](y0 + 2))]
        ax.imshow(crop, extent=extent, aspect="auto", interpolation="bilinear")
        E = bands_along(pan["name"], (0.0, 1.0), t)
        for b in range(E.shape[1]):
            halo_plot(ax, t, E[:, b], COL_TB, lw=0.9)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.axhline(0.0, color="#52514e", linewidth=0.5, linestyle=":")
        ax.set_title(f"{pan['name']}: Fig. 2 cut along p_y through K")
        ax.set_xlabel("p_y (1/Å)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Energy (eV)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 88})
    plt.close(fig)


def fig_bands_fig7(img, cells, out):
    names = list(pf.FIG7_LAYOUT.keys())
    fig, axes = plt.subplots(4, 4, figsize=(14, 15.5))
    t = np.linspace(-3.0, 3.0, 800)
    for i, name in enumerate(names):
        xa, xb, ya, yb = cells[name]
        crop = img[ya:yb, xa:xa + int(0.52 * (xb - xa))]
        ax_img = axes[i // 2, 2 * (i % 2)]
        ax_mod = axes[i // 2, 2 * (i % 2) + 1]
        ax_img.imshow(crop, interpolation="bilinear")
        ax_img.set_axis_off()
        ax_img.set_title(f"{name}: paper, Fig. 7")
        E = bands_along(name, (1.0, 0.0), t * p_c()) / mlg.gamma1
        for b in range(E.shape[1]):
            ax_mod.plot(t, E[:, b], color="#2a78d6", linewidth=0.9)
        lim = 2.0 if len(name) == 3 else 2.5
        ax_mod.set_xlim(-2.7, 2.7); ax_mod.set_ylim(-lim, lim)
        ax_mod.set_xlabel("p_x / p_c"); ax_mod.set_ylabel("E / γ1")
        ax_mod.set_title(f"{name}: model, p_y = 0 cut")
        ax_mod.axhline(0, color="#52514e", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)


def fig_dos_overlay(img, panels, out, dk, bins, ncol, figsize, upscale=False, deviations=None):
    n = len(panels)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    for i, pan in enumerate(panels):
        ax = axes[i // ncol, i % ncol]
        x0, x1, y0, y1 = pan["frame"]
        crop = img[y0:y1 + 1, x0:x1 + 1]
        extent = [float(pan["xmap"](x0)) / 1e14, float(pan["xmap"](x1)) / 1e14,
                  float(pan["ymap"](y1)), float(pan["ymap"](y0))]
        ax.imshow(crop, extent=extent, aspect="auto", interpolation="nearest" if upscale else "bilinear")
        e_lim = max(abs(extent[2]), abs(extent[3]))
        d_kp = dos_curve(pan["name"], e_lim, "kp", dk, bins)
        d_tb = dos_curve(pan["name"], e_lim, "tb", dk, bins)
        halo_plot(ax, d_kp["dos_per_cm2"] / 1e14, d_kp["energy"], COL_KP, lw=1.0, ls="--", label="k·p, Eq. (1)")
        halo_plot(ax, d_tb["dos_per_cm2"] / 1e14, d_tb["energy"], COL_TB, lw=1.0, label="tight binding")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{pan['name']}")
        ax.set_xlabel("DOS (10¹⁴ eV⁻¹ cm⁻²)")
        if i % ncol == 0:
            ax.set_ylabel("Energy (eV)")
        if i == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True)
        if deviations is not None:
            deviations[pan["name"]] = (pan["data"], d_kp, d_tb)
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 88})
    plt.close(fig)


# =============================================================================
# numbers for the tables
# =============================================================================

def paper_features(data, min_width, merge=0.02):
    E, _, lo, hi = data.T
    width = hi - lo
    picked = []
    for i in np.argsort(width)[::-1]:
        if width[i] < min_width:
            break
        if all(abs(E[i] - E[j]) > merge for j in picked):
            picked.append(i)
    return np.sort(E[picked])


def local_shift(E_model, dos_model, E_paper, dos_paper, e0, half=0.05, s_max=0.03, step=0.001):
    grid = np.arange(e0 - half, e0 + half + 1e-9, step)
    p = np.interp(grid, E_paper, dos_paper); p = p - p.mean()
    best = None
    for s in np.arange(-s_max, s_max + 1e-9, step):
        m = np.interp(grid + s, E_model, dos_model); m = m - m.mean()
        c = (m * p).sum() / np.sqrt((m * m).sum() * (p * p).sum() + 1e-30)
        if best is None or c > best[0]:
            best = (c, s)
    return best


def band_edge_rows():
    p = mlg.get_parameters()
    g1, dp = p["gamma1"], p["Delta_prime"]
    phi = (1 + np.sqrt(5)) / 2
    labels = {
        "ABAB": [("φγ1 + √3Δ'", phi * g1 + np.sqrt(3) * dp), ("γ1/φ + √2Δ'", g1 / phi + np.sqrt(2) * dp),
                 ("−γ1/φ + √2Δ'", -g1 / phi + np.sqrt(2) * dp), ("−φγ1 + √3Δ'", -phi * g1 + np.sqrt(3) * dp)],
        "ABCA": [("γ1 + Δ'", g1 + dp), ("−γ1 + Δ'", -g1 + dp)],
        "ABCB": [("√2γ1 + 3Δ'/2", np.sqrt(2) * g1 + 1.5 * dp), ("γ1 + Δ'", g1 + dp),
                 ("−γ1 + Δ'", -g1 + dp), ("−√2γ1 + 3Δ'/2", -np.sqrt(2) * g1 + 1.5 * dp)],
    }
    return labels


# =============================================================================
# HTML
# =============================================================================

def img_tag(path, alt):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    mime = "image/jpeg" if path.endswith((".jpg", ".jpeg")) else "image/png"
    return f'<img src="data:{mime};base64,{b64}" alt="{html.escape(alt)}">'


def table(headers, rows, caption=None):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f'<div class="tablewrap"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


STYLE = """
<style>
/* layout: one reading column, figures full width of the column, tables scroll */
:root {
  --bg: #faf9f6; --fg: #1c1b18; --muted: #615c52; --line: #d8d3c7; --panel: #f1eee6;
  --accent: #1f5fa8; --accent-2: #c9561f;
  --font-body: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --font-ui: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181b; --fg: #ebe8e0; --muted: #aaa59a; --line: #393d44; --panel: #1e2226;
    --accent: #7fb0ea; --accent-2: #f0905f; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #16181b; --fg: #ebe8e0; --muted: #aaa59a; --line: #393d44; --panel: #1e2226;
  --accent: #7fb0ea; --accent-2: #f0905f; color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 17px;
       line-height: 1.55; margin: 0; padding-block: 2rem 4rem; padding-inline: 16px; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1, h2, h3 { font-family: var(--font-ui); line-height: 1.2; text-wrap: balance; }
h1 { font-size: 2rem; font-weight: 600; margin: 0 0 .4rem; }
h2 { font-size: 1.35rem; font-weight: 600; margin: 2.6rem 0 .8rem; border-top: 1px solid var(--line); padding-top: 1.2rem; }
h3 { font-size: 1.08rem; font-weight: 600; margin: 1.8rem 0 .5rem; }
p { max-width: 70ch; }
.eyebrow { font-family: var(--font-ui); font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.lede { font-size: 1.1rem; color: var(--muted); max-width: 70ch; }
.key { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 1.4rem 0; }
.key div { background: var(--panel); border: 1px solid var(--line); padding: .8rem 1rem; }
.key b { display: block; font-family: var(--font-ui); font-size: 1.3rem; font-weight: 600; }
.key span { font-family: var(--font-ui); font-size: .85rem; color: var(--muted); }
figure { margin: 1.2rem 0 1.6rem; }
figure img { width: 100%; height: auto; display: block; background: #fff; border: 1px solid var(--line); }
figcaption { font-family: var(--font-ui); font-size: .9rem; color: var(--muted); margin-top: .5rem; max-width: 80ch; }
.tablewrap { overflow-x: auto; margin: 1rem 0 1.4rem; }
table { border-collapse: collapse; font-family: var(--font-ui); font-size: .88rem; font-variant-numeric: tabular-nums; min-width: 420px; }
caption { text-align: left; color: var(--muted); padding-bottom: .4rem; font-size: .85rem; }
th, td { border-bottom: 1px solid var(--line); padding: .35rem .7rem; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid var(--fg); font-weight: 600; }
code, pre { font-family: var(--font-mono); font-size: .85em; }
pre { background: var(--panel); border: 1px solid var(--line); padding: .8rem 1rem; overflow-x: auto; }
ul { max-width: 75ch; }
li { margin: .25rem 0; }
a { color: var(--accent); }
.swatch { display: inline-block; width: 1.6em; height: .25em; vertical-align: middle; margin-right: .3em; }
</style>
"""


def build_html(ctx, standalone):
    s = []
    if standalone:
        s.append('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                 '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">')
    s.append("<title>SWMcC model vs McEllistrim 2023</title>")
    s.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600'
             '&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono&display=swap">')
    s.append(STYLE)
    if standalone:
        s.append("</head><body>")
    p = mlg.get_parameters()
    s.append('<div class="wrap">')
    s.append('<p class="eyebrow">Few-layer graphene · tight-binding model check</p>')
    s.append("<h1>SWMcC model vs McEllistrim 2023</h1>")
    s.append(f'<p class="lede">Band structures and densities of states of the rewritten '
             f'Slonczewski–Weiss–McClure tight-binding model, drawn over the published panels of {html.escape(PAPER)}. '
             f'All model curves use the parameters quoted in the paper.</p>')
    s.append('<div class="key">')
    for b, t in ctx["key"]:
        s.append(f"<div><b>{html.escape(b)}</b><span>{html.escape(t)}</span></div>")
    s.append("</div>")

    s.append("<h2>Model and parameters</h2>")
    s.append("<p>The lattice Hamiltonian couples the sublattices A<sub>n</sub>, B<sub>n</sub> of N layers with the "
             "couplings γ0 … γ5 and the on-site energy Δ' (Δ' per vertical γ1 bond). Expanded to first order around K it "
             "becomes Eq. (1) of the paper; the k·p curves below are that expansion evaluated with the same code path. "
             "Energies in eV, p<sub>c</sub> = γ1/v.</p>")
    s.append(table(["parameter", "value", "origin"], ctx["param_rows"], "Parameters (McEllistrim et al. 2023)"))

    s.append("<h2>Band structures</h2>")
    s.append("<h3>Tetralayers, Fig. 1 of the paper</h3>")
    s.append("<p>The published panels are 3D views (azimuth 0) of E(p<sub>x</sub>, p<sub>y</sub>) over the disk "
             "|p| ≤ 3p<sub>c</sub>, with the energy axis through the origin and the p<sub>x</sub> axis on the grey plane. "
             "Their tick marks give the pixel scales, so the model's p<sub>y</sub> = 0 cut can be drawn in the panel's "
             "own coordinates. Black lines with a white halo: tight-binding bands at p<sub>y</sub> = 0. The surfaces are "
             "drawn over the whole disk, so the lines run along the middle of each coloured band; band edges on the "
             "axis (e.g. the 2<sup>±</sup> minima of ABCA at Δ' ± γ1) and the Mexican-hat minima off the axis are the "
             "points to compare.</p>")
    s.append(f'<figure>{ctx["img_bands1"]}<figcaption>Fig. 1 band panels of the paper with the model cut E(p_x, 0)/γ1 overlaid. '
             f'Scales: {ctx["fig1_scale"]}.</figcaption></figure>')
    s.append("<h3>ARPES constant-momentum cuts, Fig. 2 of the paper</h3>")
    s.append("<p>Row 1 of Fig. 2 shows simulated ARPES intensity for cuts through K<sub>+</sub> along p<sub>y</sub> "
             "(ω = 70 eV, Γ = 60 meV). The intensity maxima follow the bands, so this is a direct overlay: energy from the "
             "axis labels, momentum from the 0.1 Å⁻¹ scale bar, K at the mirror-symmetry column of each map. Only band "
             "positions are compared; the intensity (layer attenuation and photoelectron phases) is not modelled here. "
             "ABCB and ABAC are the same crystal turned upside down and have identical bands.</p>")
    s.append(f'<figure>{ctx["img_arpes"]}<figcaption>Fig. 2 cuts (ω = 70 eV) with the tight-binding bands along p_y '
             f'through K overlaid (all 2N bands; the maps show occupied states only).</figcaption></figure>')
    s.append("<h3>Trilayers and pentalayers, Fig. 7 of the paper</h3>")
    s.append("<p>The Fig. 7 panels are small 3D views with boxed axes, so no pixel-exact overlay is attempted; each "
             "panel is shown next to the model cut at p<sub>y</sub> = 0 in the same units.</p>")
    s.append(f'<figure>{ctx["img_bands7"]}<figcaption>Fig. 7 band panels (left of each pair) and the model (right), '
             f'E/γ1 versus p_x/p_c.</figcaption></figure>')
    s.append(table(["stack", "paper label", "label value (eV)", "step in the paper's DOS (eV)", "model E(K) (eV)"],
                   ctx["edge_rows"], "Band edges at p = 0: the paper's labels, the step energies read from its DOS curves, and the model"))

    s.append("<h2>Density of states</h2>")
    s.append("<p>The DOS is integrated with a linear-interpolation triangle scheme on a k mesh around K (spacing "
             f"{ctx['dk']} Å⁻¹), with spin and valley degeneracy, in the paper's units of eV⁻¹ cm⁻². The published curves are "
             "the blue lines of the raster panels; the frames and tick marks give the pixel scales. "
             '<span class="swatch" style="background:#111"></span>tight binding, '
             '<span class="swatch" style="background:#eb6834"></span>k·p (Eq. 1, dashed).</p>')
    s.append("<h3>Tetralayers, Fig. 1</h3>")
    s.append(f'<figure>{ctx["img_dos1"]}<figcaption>DOS panels of Fig. 1 with the model curves overlaid.</figcaption></figure>')
    s.append("<h3>Trilayers and pentalayers, Fig. 7</h3>")
    s.append(f'<figure>{ctx["img_dos7"]}<figcaption>DOS panels of Fig. 7 with the model curves overlaid. The ABA panel is '
             f'plotted on a 10¹³ scale in the paper and lies a factor of two below the model and below the other panels.</figcaption></figure>')
    s.append("<h3>Replotted comparison</h3>")
    s.append(f'<figure>{ctx["img_clean"]}<figcaption>The same curves replotted from the digitised data: paper (green), '
             f'k·p (orange), tight binding (blue).</figcaption></figure>')
    s.append(table(["stack", "figure", "model"] + [f"E = {e:+.1f} eV" for e in ctx["dev_energies"]],
                   ctx["dev_rows"], "Relative deviation of the model DOS from the published curve, away from singularities"))
    s.append(table(["stack", "figure", "feature (eV)", "shift (meV)", "correlation"], ctx["shift_rows"],
                   "Position of each step or van Hove singularity: shift that maximises the local cross-correlation of model (k·p) and paper"))

    s.append("<h2>Notes</h2>")
    s.append("<ul>")
    for n in ctx["notes"]:
        s.append(f"<li>{n}</li>")
    s.append("</ul>")
    s.append("<h3>Reproduce</h3>")
    s.append("<pre>pip install numpy matplotlib pillow pymupdf pytest\n"
             "python -m pytest tests -q\n"
             "python compare_with_mcellistrim.py\n"
             "python make_report.py --pdf arXiv-2302.07374v1.pdf --out report</pre>")
    s.append("</div>")
    if standalone:
        s.append("</body></html>")
    return "\n".join(s)


# =============================================================================
# main
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="arXiv:2302.07374v1 PDF")
    ap.add_argument("--out", default="report")
    ap.add_argument("--dk", type=float, default=1.0e-3)
    ap.add_argument("--bins", type=int, default=800)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.out, "figures")
    mlg.use_paper_parameters()

    paths = pf.extract_figures(args.pdf, figdir)
    img1, img2, img7 = (pf.load_rgb(paths[k]) for k in ("fig1", "fig2", "fig7"))

    band1 = pf.fig1_band_panels(img1)
    dos1 = pf.fig1_dos_panels(img1)
    arpes = pf.fig2_arpes_panels(img2)
    dos7 = pf.fig7_dos_panels(img7)
    cells7 = pf.fig7_cells(img7)

    f_bands1 = os.path.join(figdir, "bands_fig1_overlay.jpg")
    f_arpes = os.path.join(figdir, "bands_fig2_arpes_overlay.jpg")
    f_bands7 = os.path.join(figdir, "bands_fig7_side_by_side.jpg")
    f_dos1 = os.path.join(figdir, "dos_fig1_overlay.jpg")
    f_dos7 = os.path.join(figdir, "dos_fig7_overlay.jpg")
    f_clean = os.path.join(figdir, "dos_comparison_replot.png")

    print("bands: Fig. 1 overlay"); fig_bands_fig1(img1, band1, f_bands1)
    print("bands: Fig. 2 ARPES overlay"); fig_arpes_fig2(img2, arpes, f_arpes)
    print("bands: Fig. 7 side by side"); fig_bands_fig7(img7, cells7, f_bands7)
    curves = {}
    print("DOS: Fig. 1 overlay"); fig_dos_overlay(img1, dos1, f_dos1, args.dk, args.bins, 3, (13.5, 6.4), deviations=curves)
    print("DOS: Fig. 7 overlay"); fig_dos_overlay(img7, dos7, f_dos7, args.dk, args.bins, 4, (14, 9.5), upscale=True, deviations=curves)
    print("DOS: replot"); compare_with_mcellistrim.main(["--dk", str(args.dk), "--bins", str(args.bins), "--out", f_clean])

    # ---- tables ------------------------------------------------------------
    p = mlg.get_parameters()
    param_rows = [
        ("v", "1.00 × 10⁶ m/s", "paper; γ0 = %.4f eV" % p["gamma0"]),
        ("v3", "0.10 × 10⁶ m/s", "paper; γ3 = %.4f eV" % p["gamma3"]),
        ("v4", "0.022 × 10⁶ m/s", "paper; γ4 = %.4f eV" % p["gamma4"]),
        ("γ1", "0.390 eV", "paper"), ("γ2", "−0.017 eV", "paper"), ("γ5", "0.038 eV", "paper"),
        ("Δ'", "0.025 eV", "paper, per γ1 bond"), ("a", "2.46 Å", "paper"),
        ("p_c = γ1/v", "%.4f Å⁻¹" % p_c(), "derived"),
    ]
    edge_rows = []
    for seq, labels in band_edge_rows().items():
        data = [d for d in dos1 if d["name"] == seq][0]["data"]
        feats = paper_features(data, 0.15e14)
        EK = np.linalg.eigvalsh(mlg.hamiltonian(*mlg.K_point(), stacking_type=seq))
        for text, val in labels:
            near = feats[np.abs(feats - val) < 0.05]
            step = f"{near[np.argmin(np.abs(near - val))]:+.3f}" if near.size else "–"
            model = EK[np.argmin(np.abs(EK - val))]
            edge_rows.append((seq, text, f"{val:+.3f}", step, f"{model:+.3f}"))
    dev_energies = (-0.7, -0.6, -0.5, 0.5, 0.6, 0.7)
    dev_rows, shift_rows = [], []
    for name, (data, d_kp, d_tb) in curves.items():
        fig = 1 if name in pf.FIG1_STACKS else 7
        e_max = np.abs(data[:, 0]).max()
        for label, d in (("k·p", d_kp), ("tight binding", d_tb)):
            cells = []
            for e in dev_energies:
                if abs(e) > e_max - 0.03:
                    cells.append("–"); continue
                mine = np.interp(e, d["energy"], d["dos_per_cm2"])
                theirs = np.interp(e, data[:, 0], data[:, 1])
                cells.append(f"{100 * (mine / theirs - 1):+.1f} %")
            dev_rows.append((name, f"Fig. {fig}", label, *cells))
        feats = paper_features(data, 0.25e14 if len(name) <= 4 else 0.6e14)
        for e0 in feats[np.abs(feats) > 0.03]:
            corr, s = local_shift(d_kp["energy"], d_kp["dos_per_cm2"], data[:, 0], data[:, 1], e0)
            shift_rows.append((name, f"Fig. {fig}", f"{e0:+.3f}", f"{1000 * s:+.0f}", f"{corr:.2f}"))
    fig1_scale = "; ".join(f"{b['name']}: {b['px_per_gamma1']:.0f} px per γ1, {b['px_per_pc']:.1f} px per p_c" for b in band1)

    ctx = {
        "key": [("≤ 5 meV", "band-edge and van Hove energies vs Fig. 1 (≤ 7 meV vs Fig. 7)"),
                ("+1 … +3 %", "k·p DOS vs Fig. 1, electron side"),
                ("+6 … +9 %", "k·p DOS vs Fig. 1, hole side"),
                ("+1 … +6 %", "k·p DOS vs Fig. 7 (3 and 5 layers)")],
        "param_rows": param_rows, "edge_rows": edge_rows, "dev_rows": dev_rows,
        "dev_energies": dev_energies, "shift_rows": shift_rows, "dk": args.dk, "fig1_scale": fig1_scale,
        "img_bands1": img_tag(f_bands1, "Fig. 1 band panels with model overlay"),
        "img_arpes": img_tag(f_arpes, "Fig. 2 ARPES cuts with model bands"),
        "img_bands7": img_tag(f_bands7, "Fig. 7 band panels beside the model"),
        "img_dos1": img_tag(f_dos1, "Fig. 1 DOS panels with model overlay"),
        "img_dos7": img_tag(f_dos7, "Fig. 7 DOS panels with model overlay"),
        "img_clean": img_tag(f_clean, "Replotted DOS comparison"),
        "notes": [
            "Eq. (1) of arXiv v1 contains two misprints: the layer-3 on-site term of H<sub>ABCB</sub> must be Δ'(1 − σ<sub>z</sub>) and the (2,3) block of H<sub>ABAC</sub> must be V<sub>AB</sub><sup>†</sup>; the band-edge labels of Fig. 1 (√2γ1 + 3Δ'/2) and the digitised steps confirm the corrected form used here.",
            "The labels γ1/φ + √2Δ' and φγ1 + √3Δ' of Fig. 1(a) are first-order estimates that omit γ5; the plotted steps and the model agree with each other (0.256, 0.691, −0.226, −0.571 eV).",
            "The tight-binding DOS is about 2 % above the k·p DOS at |E| = 0.5–0.7 eV because the lattice band velocity decreases away from K.",
            "The hole-side excess of the model over Fig. 1 is the electron–hole asymmetry of γ4, which the Fig. 1 curves lack: their hole/electron DOS ratio at ±0.5 eV (1.07, 1.08, 1.05 for ABAB, ABCA, ABCB) is that of the model with γ4 = 0 (1.06, 1.08, 1.05), whereas the printed v4 gives 1.14, 1.14, 1.13. Fig. 7 contains the asymmetry (1.13–1.18, model 1.12–1.15). Fig. 1 was therefore computed without γ4, or with γ4 of the opposite chirality, which has the same effect on the DOS; with γ4 = 0 the residual is a uniform +4–5 % on both sides. The model keeps the printed v4, which follows from the geometry.",
            "Fig. 2 overlays use the paper's ξ = +1 valley and the cut along p<sub>y</sub>; band energies along this cut are symmetric in p<sub>y</sub>, so the mirror-symmetry column of each map locates K.",
            "The 3D overlays of Fig. 1 assume an orthographic view with azimuth 0, which the horizontal p<sub>x</sub> axis and vertical energy axis of the panels imply; small perspective distortions are possible.",
            "Digitisation: Fig. 1 at 1.8 meV and 9 × 10¹¹ eV⁻¹ cm⁻² per pixel, Fig. 7 at 7.5 meV and 2.5 × 10¹² eV⁻¹ cm⁻² per pixel.",
        ],
    }
    with open(os.path.join(args.out, "index.html"), "w") as fh:
        fh.write(build_html(ctx, standalone=True))
    with open(os.path.join(args.out, "artifact.html"), "w") as fh:
        fh.write(build_html(ctx, standalone=False))
    print(f"report written to {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
