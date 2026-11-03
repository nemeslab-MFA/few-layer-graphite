#!/usr/bin/env python3
"""
Electronic Raman scattering (ERS) background of bulk graphite: density of
states of hexagonal (Bernal, AB) and rhombohedral (ABC) graphite.

In graphitic systems the ERS continuum is made of interband electron-hole
excitations with negligible momentum transfer and a nearly constant matrix
element, which connect the mirror bands n- -> n+ (Kashuba and Fal'ko, PRB 80,
241404 (2009); Ponosov et al., PRB 91, 195435 (2015), Eq. (1); Garcia-Ruiz et
al., Nano Lett. 19, 6152 (2019), Eq. (5)).  The spectral density at Raman
shift w therefore follows the density of states at E_F -+ hbar w / 2, so that a
spectrometer covering w_min ... w_max probes |E - E_F| between hbar w_min / 2
and hbar w_max / 2.

The bulk curves are followed by the DOS of finite slabs (20, 50 and 100
layers by default) of both stackings, which show how the subbands and the
surface band of a film approach the bulk.

A Raman shift is the difference between an electron and a hole energy, so the
last sections work on that difference itself: the joint density of states of
the mirror pairs n- -> n+ of a slab.  It is set against ERS spectra calculated
for whole films (examples/ers_calculations/*.dat) and drawn for the thicker
slabs and for the bulk (integrated over the three-dimensional zone), as
computed and broadened to the measured ERS line width (500 1/cm on the
Raman-shift axis).

Writes five figures, three CSV tables and an HTML report (LaTeX set by MathJax):
    <prefix>_comparison.png       DOS of both polytypes on one graph
    <prefix>_raman_window.png     the same, with the accessible energies shaded
    <prefix>_slabs.png            DOS of the finite slabs over the bulk curves
    <prefix>_ers_calculation.png  calculated ERS spectra and mirror-pair spectra
    <prefix>_raman_axis.png       ERS estimate of slabs and bulk against Raman shift
    <prefix>.csv                  E - E_F and the two bulk DOS curves
    <prefix>_slabs.csv            E - E_F and the DOS of every slab
    <prefix>_pairs.csv            excitation energy and the mirror-pair spectra
    <prefix>_report.html          figures, numbers and method on one page

Usage:
    python graphite_ers_background.py [--raman-min 80] [--raman-max 6200]
                                      [--slabs 20 50 100] [--linewidth 500]
                                      [--ers-spectra FILE ...]
                                      [--prefix examples/graphite_ers_dos]
                                      [--from-csv examples/graphite_ers_dos.csv]
"""
import argparse
import glob
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import multilayer_graphene as mlg
import bulk_graphite as bg

EV_PER_INVERSE_CM = 1.239841984e-4      # h c
K_BOLTZMANN = 8.617333262e-5            # eV / K
STACKS = (("ab", "hexagonal (AB)", "#2a78d6"), ("abc", "rhombohedral (ABC)", "#eb6834"))
# the 2 x 2 rhombohedral problem is cheap and its Dirac points need the finer mesh
REFINE_FACTOR = {"ab": 1, "abc": 4}
SLAB_PATTERN = {"ab": "aba", "abc": "abc"}      # periodic patterns of multilayer_graphene
# excitation energy (eV) below which a slab spectrum uses the sigma_z matrix elements: needed where the bilayer-like
# subbands of a hexagonal slab overlap; in rhombohedral slabs the mirror pairs carry the weight at all energies
WEIGHTED_BELOW = {"ab": 0.06, "abc": 0.0}
ERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "ers_calculations")
BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
INK, INK_SOFT, GRID, SURFACE, WINDOW = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#eda100"


def compute(e_max=1.0, n_kz=240, dk=2.0e-3, refine=8, bin_width=1.0e-3, sigma=0.0):
    """DOS of both polytypes on a common grid of E - E_F (eV)."""
    pad = 0.1                           # room for the Fermi levels of the two polytypes
    energy = np.arange(-e_max, e_max + 0.5 * bin_width, bin_width)
    table = {"energy": energy}
    info = {"parameters": mlg.get_parameters(), "d_layer": bg.D_LAYER, "n_kz": n_kz, "dk": dk,
            "bin_width": bin_width, "gaussian_sigma": sigma,
            "atoms_per_cm3": bg.atomic_density() * 1.0e24}
    for key, label, _ in STACKS:
        d = bg.calculate_bulk_density_of_states(
            (-e_max - pad, e_max + pad), key, n_kz=n_kz, dk=dk, refine=refine * REFINE_FACTOR[key],
            num_energy_points=int(round(2.0 * (e_max + pad) / bin_width)),
            gaussian_sigma=sigma, verbose=True)
        table[key] = np.interp(energy, d["energy"] - d["fermi_energy"], d["dos_per_atom"])
        info[key] = {"label": label, "fermi_energy_eV": d["fermi_energy"],
                     "carrier_density_cm3": d["carrier_density_cm3"],
                     "refine": refine * REFINE_FACTOR[key]}
    return table, info


def save_csv(path, table, info):
    with open(path, "w") as fh:
        fh.write(f"# metadata: {json.dumps(info, sort_keys=True)}\n")
        fh.write("energy_minus_EF_eV,dos_hexagonal_states_per_eV_atom,dos_rhombohedral_states_per_eV_atom\n")
        for row in zip(table["energy"], table["ab"], table["abc"]):
            fh.write(",".join(f"{v:.8g}" for v in row) + "\n")


def load_csv(path):
    with open(path) as fh:
        info = json.loads(fh.readline().split("# metadata:", 1)[1])
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return {"energy": data[:, 0], "ab": data[:, 1], "abc": data[:, 2]}, info


def _slab_dos(job):
    """DOS per atom of one slab on a grid of E - E_F and its Fermi level; runs in a worker process."""
    n_layers, key, energy, dk, k_max, params = job
    mlg.set_parameters(**params)
    bin_width = energy[1] - energy[0]
    e_lim = energy[-1] + 0.05           # room for the Fermi level of the slab
    d = mlg.calculate_density_of_states((-e_lim, e_lim), n_layers, SLAB_PATTERN[key], dk=dk, k_max=k_max,
                                        num_energy_points=int(round(2.0 * e_lim / bin_width)))
    # charge neutrality: n_layers of the 2 n_layers bands are filled inside the k window
    net = d["cumulative_per_area"] - n_layers * (2.0 * k_max) ** 2 / np.pi ** 2
    fermi = float(np.interp(0.0, net, d["energy_edges"]))
    return np.interp(energy, d["energy"] - fermi, d["dos_per_unit_cell"] / (2.0 * n_layers)), fermi


def _run_in_workers(function, jobs):
    """
    Map a function over jobs in parallel worker processes, with one BLAS thread
    each, set before the workers import NumPy: threaded LAPACK calls from
    several processes at once slow each other down a hundredfold on matrices
    of this size.
    """
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(BLAS_THREAD_VARIABLES, "1"))
    try:
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(function, jobs))
    finally:
        for name, value in saved.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value


def compute_slabs(layers, e_max=0.5, dk=1.0e-3, k_max=0.18, bin_width=1.0e-3):
    """
    DOS per atom of finite slabs of both stackings on a common grid of
    E - E_F (eV), from ``multilayer_graphene.calculate_density_of_states``.
    The slabs are diagonalised in parallel worker processes.
    """
    layers = sorted(int(n) for n in layers)
    energy = np.arange(-e_max, e_max + 0.5 * bin_width, bin_width)
    jobs = [(n, key, energy, dk, k_max, mlg.get_parameters()) for n in reversed(layers) for key, _, _ in STACKS]
    results = _run_in_workers(_slab_dos, jobs)
    slabs = {"energy": energy}
    info = {"layers": layers, "dk": dk, "k_max": k_max, "bin_width": bin_width,
            "parameters": mlg.get_parameters(), "fermi_energy_eV": {}}
    for (n, key, *_), (dos, fermi) in sorted(zip(jobs, results), key=lambda item: item[0][:2]):
        slabs[f"{key}_{n}"] = dos
        info["fermi_energy_eV"][f"{key}_{n}"] = fermi
        print(f"slab DOS: {n:4d} layers {SLAB_PATTERN[key]}, E_F = {1e3 * fermi:+.2f} meV")
    return slabs, info


def _save_table(path, table, info, header):
    columns = list(table)
    with open(path, "w") as fh:
        fh.write(f"# metadata: {json.dumps({**info, 'columns': columns}, sort_keys=True)}\n")
        fh.write(",".join(header) + "\n")
        for row in zip(*(table[c] for c in columns)):
            fh.write(",".join(f"{v:.8g}" for v in row) + "\n")


def _load_table(path):
    with open(path) as fh:
        info = json.loads(fh.readline().split("# metadata:", 1)[1])
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    return {c: data[:, i] for i, c in enumerate(info.pop("columns"))}, info


def save_slabs_csv(path, slabs, info):
    _save_table(path, slabs, info, ["energy_minus_EF_eV"]
                + [f"dos_{c}_layers_states_per_eV_atom" for c in list(slabs)[1:]])


def load_slabs_csv(path):
    return _load_table(path)


def save_pairs_csv(path, pairs, info):
    _save_table(path, pairs, info, ["excitation_energy_eV"]
                + [f"mirror_pairs_{c}{'' if c.endswith('bulk') else '_layers'}_per_eV_atom" for c in list(pairs)[1:]])


def load_pairs_csv(path):
    return _load_table(path)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def _draw_dos(ax, table, info, xlim, linewidth=1.8):
    sel = (table["energy"] >= xlim[0]) & (table["energy"] <= xlim[1])
    for key, label, color in STACKS:
        ax.plot(table["energy"][sel], 100.0 * table[key][sel], color=color, linewidth=linewidth, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 1.06 * 100.0 * max(table[key][sel].max() for key, _, _ in STACKS))
    ax.set_xlabel("E − E$_F$ (eV)", color=INK)
    ax.set_ylabel("DOS (10$^{-2}$ states eV$^{-1}$ atom$^{-1}$)", color=INK)
    per_cm3 = info["atoms_per_cm3"] * 1.0e-2 / 1.0e21
    right = ax.secondary_yaxis("right", functions=(lambda y: y * per_cm3, lambda y: y / per_cm3))
    right.set_ylabel("DOS (10$^{21}$ eV$^{-1}$ cm$^{-3}$)", color=INK)
    right.tick_params(colors=INK_SOFT, labelsize=9)
    right.spines["right"].set_color(INK_SOFT)
    _style(ax)


def plot_comparison(table, info, path, e_max=1.0, inset=0.1):
    """Figure 1: the two DOS curves, with a close-up of the Fermi level."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    _draw_dos(ax, table, info, (-e_max, e_max))
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, labelcolor=INK)
    ax.set_title("Bulk graphite: density of states of the SWMcC model", color=INK, fontsize=11, loc="left")

    box = ax.inset_axes([0.36, 0.50, 0.30, 0.32])
    sel = np.abs(table["energy"]) <= inset
    for key, _, color in STACKS:
        box.plot(1.0e3 * table["energy"][sel], 100.0 * table[key][sel], color=color, linewidth=1.5)
    box.set_xlim(-1.0e3 * inset, 1.0e3 * inset)
    box.set_ylim(0.0, 1.1 * 100.0 * max(table[key][sel].max() for key, _, _ in STACKS))
    box.set_xlabel("E − E$_F$ (meV)", color=INK_SOFT, fontsize=8, labelpad=1)
    _style(box)
    box.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def plot_raman_window(table, info, path, raman_min, raman_max, e_max=0.5):
    """Figure 2: the DOS with the energies reached by the spectrometer shaded."""
    e_lo = 0.5 * raman_min * EV_PER_INVERSE_CM
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    fig, ax = plt.subplots(figsize=(7.2, 4.9), facecolor=SURFACE)
    _draw_dos(ax, table, info, (-e_max, e_max))
    for sign in (-1.0, 1.0):
        ax.axvspan(sign * e_lo, sign * e_hi, color=WINDOW, alpha=0.16, linewidth=0)
        ax.axvline(sign * e_hi, color=INK_SOFT, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.plot([], [], color=WINDOW, alpha=0.35, linewidth=8, solid_capstyle="butt",
            label=f"within reach of {raman_min:g}–{raman_max:g} cm$^{{-1}}$:  "
                  f"|E − E$_F$| = {1e3 * e_lo:.0f}–{1e3 * e_hi:.0f} meV")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9), frameon=False, fontsize=9, labelcolor=INK)
    _wavenumber_axis(ax)
    _raman_span(ax, raman_max)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def _wavenumber_axis(ax):
    """
    Top axis with the energy from the Fermi level in 1/cm.  A Raman shift is
    the difference between an electron and a hole energy, so it reads on this
    axis as the distance between the two.
    """
    top = ax.secondary_xaxis("top", functions=(lambda e: e / EV_PER_INVERSE_CM, lambda w: w * EV_PER_INVERSE_CM))
    top.set_xlabel("E − E$_F$ (cm$^{-1}$)", color=INK)
    top.tick_params(colors=INK_SOFT, labelsize=9)
    top.spines["top"].set_color(INK_SOFT)
    return top


def _raman_span(ax, raman_max, height=0.93):
    """Arrow from E_F - hbar w / 2 to E_F + hbar w / 2 for the largest Raman shift of the instrument."""
    half = 0.5 * raman_max * EV_PER_INVERSE_CM
    ax.annotate("", xy=(half, height), xytext=(-half, height), xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops={"arrowstyle": "<->", "color": INK_SOFT, "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0})
    ax.annotate(f"Raman shift of {raman_max:g} cm$^{{-1}}$: from a hole to an electron energy",
                xy=(0.0, height), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color=INK_SOFT)


def bulk_on_slab_axis(slabs, slab_info, table, info, key, n):
    """Bulk DOS at the same absolute energy as the slab curve, on the slab's axis of E - E_F."""
    shift = slab_info["fermi_energy_eV"][f"{key}_{n}"] - info[key]["fermi_energy_eV"]
    return np.interp(slabs["energy"] + shift, table["energy"], table[key])


def plot_slabs(slabs, slab_info, table, info, path, raman_min, raman_max, e_max=0.5, zoom=0.1):
    """
    Figure 3: DOS of the finite slabs, one row per thickness.  Left: the range
    of Figure 2 with the bulk curves for reference; right: close-up of the
    Fermi level.  Energies are measured from the Fermi level of each slab and
    the bulk curves are drawn at the same absolute energy.
    """
    layers = slab_info["layers"]
    e_lo = 0.5 * raman_min * EV_PER_INVERSE_CM
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    energy = slabs["energy"]
    wide, near = np.abs(energy) <= e_max, np.abs(energy) <= zoom
    keys = [key for key, _, _ in STACKS]
    y_wide = 1.06 * 100.0 * max(table[key][np.abs(table["energy"]) <= e_max].max() for key in keys)
    y_near = 1.08 * 100.0 * max(slabs[f"{key}_{n}"][near].max() for key in keys for n in layers)

    fig, axes = plt.subplots(len(layers), 2, figsize=(10.4, 0.6 + 3.0 * len(layers)), facecolor=SURFACE,
                             squeeze=False, gridspec_kw={"width_ratios": [1.65, 1.0]})
    for row, n in enumerate(layers):
        left, right = axes[row]
        bulk = {key: bulk_on_slab_axis(slabs, slab_info, table, info, key, n) for key in keys}
        for ax, sel, scale in ((left, wide, 1.0), (right, near, 1.0e3)):
            for key, label, color in STACKS:
                ax.plot(scale * energy[sel], 100.0 * slabs[f"{key}_{n}"][sel], color=color, linewidth=1.2,
                        label=f"{label}, {n} layers" if ax is left else None)
                ax.plot(scale * energy[sel], 100.0 * bulk[key][sel], color=INK, linewidth=0.8,
                        linestyle=(0, (4, 2)), label="bulk" if ax is left and key == "abc" else None)
            for sign in (-1.0, 1.0):
                ax.axvspan(scale * sign * e_lo, scale * sign * min(e_hi, 2.0 * e_max), color=WINDOW, alpha=0.16, linewidth=0)
            _style(ax)
        left.set_xlim(-e_max, e_max)
        left.set_ylim(0.0, y_wide)
        right.set_xlim(-1.0e3 * zoom, 1.0e3 * zoom)
        right.set_ylim(0.0, y_near)
        left.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9), frameon=False, fontsize=8.5, labelcolor=INK)
        _raman_span(left, raman_max)
        left.set_ylabel("DOS (10$^{-2}$ states eV$^{-1}$ atom$^{-1}$)", color=INK)
        if row == len(layers) - 1:
            left.set_xlabel("E − E$_F$ (eV)", color=INK)
            right.set_xlabel("E − E$_F$ (meV)", color=INK)
    _wavenumber_axis(axes[0, 0])
    axes[0, 1].set_title("close-up of the Fermi level", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def linewidth_broadening(x, y, fwhm):
    """
    Convolve a curve on a uniform grid with a Gaussian of full width at half
    maximum ``fwhm`` (in the unit of x).  The kernel is normalised on the grid
    and the curve is continued by its end values, so a constant stays constant;
    values closer than about two widths to the ends depend on that continuation.
    """
    h = x[1] - x[0]
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    m = int(np.ceil(5.0 * sigma / h))
    kernel = np.exp(-0.5 * (np.arange(-m, m + 1) * h / sigma) ** 2)
    return np.convolve(np.pad(y, m, mode="edge"), kernel / kernel.sum(), mode="valid")


def load_ers_spectrum(path):
    """
    Read a calculated ERS spectrum: '#' header lines 'Name:' (the stacking
    sequence), 'NOL:' (number of layers) and 'Calculated peaks [meV]:', then
    two columns, Raman shift (meV) and intensity (arbitrary units).

    Returns a dict with 'name', 'layers', 'peaks' (eV), 'shift' (eV),
    'intensity' and 'key': 'abc', 'ab' or None for any other stacking.
    """
    header, rows = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                header[key.strip().lower()] = value.strip()
            elif line:
                rows.append([float(v) for v in line.split()[:2]])
    data = np.array(rows)
    name = header["name"].upper()
    peaks = []
    for key, value in header.items():
        if key.startswith("calculated peaks"):
            peaks = [1.0e-3 * float(v) for v in value.replace(",", " ").split() if v.replace(".", "", 1).isdigit()]
    steps = mlg.stacking_steps(name)
    kind = "abc" if np.all(steps == steps[0]) else "ab" if np.all(steps[1:] == -steps[:-1]) else None
    return {"path": path, "name": name, "layers": int(header.get("nol", len(name))), "peaks": peaks,
            "shift": 1.0e-3 * data[:, 0], "intensity": data[:, 1], "key": kind}


def sigma_z_pair_spectrum(sequence, edges, fermi, dk=1.0e-3, k_max=0.045):
    """
    ERS spectral density of a slab with the matrix elements of Garcia-Ruiz et
    al. (2019), Eq. (5): the sum over all pairs of a filled band v and an empty
    band c of |<c| sigma_z |v>|^2 delta(hbar w - e_c + e_v), per atom, spin and
    valley included, from a k window of half-width k_max around K.

    The cost grows with the square of the number of bands, so this is meant
    for low excitation energies, which come from a small window.  Returns the
    spectrum on the bins of ``edges`` and the lowest pair energy on the border
    of the window: the spectrum is complete below that energy.
    """
    params = mlg.get_parameters()
    sequence = mlg.stacking_sequence(None, sequence)
    n_layers = len(sequence)
    m = int(np.ceil(k_max / dk))
    t = np.linspace(-k_max, k_max, 2 * m + 1)
    half = t[m:]                                        # p_y >= 0; the other half is its mirror image
    px, py = np.meshgrid(t, half, indexing="xy")
    K = mlg.K_point()
    kx, ky = px.ravel() + K[0], py.ravel() + K[1]
    sigma_z = np.tile([1.0, -1.0], n_layers)
    E = np.empty((kx.size, 2 * n_layers))
    weight = np.empty((kx.size, n_layers, n_layers), dtype=np.float32)
    step = max(1, int(2 ** 27 // (16 * (2 * n_layers) ** 2)))
    for i in range(0, kx.size, step):
        E[i:i + step], V = np.linalg.eigh(mlg.hamiltonian(kx[i:i + step], ky[i:i + step], None, sequence))
        weight[i:i + step] = np.abs(np.einsum("kav,a,kac->kvc", V[:, :, :n_layers].conj(), sigma_z,
                                              V[:, :, n_layers:])) ** 2
    tri = mlg._square_mesh_triangles(t.size, half.size)
    area = (t[1] - t[0]) ** 2                           # two mirror-image triangles of area h^2 / 2

    cumulative = np.zeros(edges.size)
    for v in range(n_layers - 1, -1, -1):
        if E[:, n_layers].min() - E[:, v].max() > edges[-1]:
            break
        for c in range(n_layers):
            lower, upper = E[:, v], E[:, n_layers + c]
            excitation = upper - lower
            if excitation.min() > edges[-1]:
                break
            allowed = ((lower < fermi) & (upper > fermi)) * weight[:, v, c]
            cumulative += mlg._cumulative_states_triangles(excitation[tri], area * allowed[tri].mean(axis=1), edges)
    outer = np.ones((half.size, t.size), dtype=bool)
    outer[:-1, 1:-1] = False
    outer = outer.ravel()
    lowest_on_border = float((E[outer, n_layers] - E[outer, n_layers - 1]).min())
    per_atom = 4.0 / (2.0 * np.pi) ** 2 * mlg.unit_cell_area(params["a"]) / (2.0 * n_layers)
    return np.diff(cumulative) / np.diff(edges) * per_atom, lowest_on_border


def mirror_pair_spectrum(sequence, edges, dk=1.0e-3, k_max=0.2, weighted_below=0.0):
    """
    ERS spectral density of a slab: the joint density of states of the mirror
    pairs n- -> n+, the n-th band below and the n-th band above the middle of
    the spectrum, with the occupation of T = 0,

        J(w) = sum_n  int d^2p / (2 pi)^2  theta(E_F - e_n-) theta(e_n+ - E_F)
               delta(hbar w - e_n+(p) + e_n-(p)),

    per atom, spin and valley included.  The Raman shift is an energy
    difference between a state above and a state below the Fermi level, and
    this spectrum is taken on that difference directly.  It is Eq. (5) of
    Garcia-Ruiz et al. (2019) with the matrix element |<n+| sigma_z |n->|^2 of
    a mirror pair set to 1.

    Counting the n-th band from the middle finds the mirror partner as long as
    the subbands keep their order.  In a hexagonal slab they do not near the
    Fermi level: its bilayer-like subbands overlap there, and bands that belong
    to different subbands get paired, which adds excitations that the matrix
    element forbids (below about 27 meV for 15 layers).  ``weighted_below``
    (eV) replaces the spectrum below that energy by the full sum of
    ``sigma_z_pair_spectrum``, blended in over the upper quarter of the range.
    Above it the two agree to about 3 %, and in rhombohedral slabs to 1.5 % at
    all energies (checked in the tests).

    sequence: explicit stacking sequence; edges: bin edges of the excitation
    energy hbar w (eV).  Uses the parameters of ``multilayer_graphene``.
    Returns the spectrum (pairs per eV per atom) and the Fermi level (eV).
    """
    params = mlg.get_parameters()
    sequence = mlg.stacking_sequence(None, sequence)
    n_layers = len(sequence)
    n_side = int(np.ceil(2.0 * k_max / dk)) + 1
    t = np.linspace(-k_max, k_max, n_side)
    E = mlg._valley_window_eigenvalues(t, None, sequence).reshape(n_side * n_side, -1)
    tri = mlg._square_mesh_triangles(n_side, n_side)
    area = 0.5 * (t[1] - t[0]) ** 2

    # charge neutrality: n_layers of the 2 n_layers bands are filled inside the k window
    levels = np.linspace(-0.3, 0.3, 1201) + params["E0"]
    filled = sum(mlg._cumulative_states_triangles(E[tri, b], area, levels) for b in range(2 * n_layers))
    fermi = float(np.interp(n_layers * (2.0 * k_max) ** 2, filled, levels))

    border = np.ones((n_side, n_side), dtype=bool)
    border[1:-1, 1:-1] = False
    border = border.ravel()
    cumulative = np.zeros(edges.size)
    lowest_on_border = np.inf
    for n in range(n_layers):
        lower, upper = E[:, n_layers - 1 - n], E[:, n_layers + n]
        excitation = upper - lower
        # a triangle counts with the fraction of its corners at which the pair is allowed
        allowed = ((lower < fermi) & (upper > fermi)).astype(float)
        cumulative += mlg._cumulative_states_triangles(excitation[tri], area * allowed[tri].mean(axis=1), edges)
        lowest_on_border = min(lowest_on_border, excitation[border].min())
    if lowest_on_border < edges[-1]:
        print(f"Warning: pairs on the border of the k window start at {lowest_on_border:.3f} eV, inside the "
              f"range of excitation energies; increase k_max (currently {k_max:.3f} 1/A).")
    per_atom = 4.0 / (2.0 * np.pi) ** 2 * mlg.unit_cell_area(params["a"]) / (2.0 * n_layers)
    spectrum = np.diff(cumulative) / np.diff(edges) * per_atom

    low = int(np.searchsorted(edges, weighted_below + 1e-12, side="right"))     # edges up to weighted_below
    if weighted_below > 0.0 and low > 1:
        weighted, complete_below = sigma_z_pair_spectrum(sequence, edges[:low], fermi, dk)
        if complete_below < edges[low - 1]:
            print(f"Warning: the window of the sigma_z-weighted spectrum holds all pairs only below "
                  f"{complete_below:.3f} eV, less than weighted_below = {weighted_below:g} eV.")
        centres = 0.5 * (edges[:low - 1] + edges[1:low])
        blend = np.clip((weighted_below - centres) / (0.25 * weighted_below), 0.0, 1.0)
        spectrum[:low - 1] = blend * weighted + (1.0 - blend) * spectrum[:low - 1]
    return spectrum, fermi


def _pair_job(job):
    """Mirror-pair spectrum of one slab; runs in a worker process."""
    n_layers, key, edges, dk, k_max, params = job
    mlg.set_parameters(**params)
    return mirror_pair_spectrum(mlg.stacking_sequence(n_layers, SLAB_PATTERN[key]), edges, dk, k_max,
                                weighted_below=WEIGHTED_BELOW[key])


def compute_pair_spectra(layers, shift_max=1.0, step=3.0e-3, dk=1.0e-3, k_max=0.2):
    """
    Mirror-pair spectra (pairs per eV per atom) of slabs of both stackings on a
    common grid of excitation energies hbar w = step, 2 step, ... (eV); see
    ``mirror_pair_spectrum``.  The bin around w = 0 is left out: half of it lies
    at negative energies.  The slabs are diagonalised in parallel.
    """
    layers = sorted(int(n) for n in layers)
    shift = np.arange(step, shift_max, step)
    edges = np.concatenate([shift - 0.5 * step, [shift[-1] + 0.5 * step]])
    jobs = [(n, key, edges, dk, k_max, mlg.get_parameters()) for n in reversed(layers) for key, _, _ in STACKS]
    results = _run_in_workers(_pair_job, jobs)
    pairs = {"shift": shift}
    info = {"layers": layers, "dk": dk, "k_max": k_max, "step": step, "parameters": mlg.get_parameters(),
            "weighted_below_eV": WEIGHTED_BELOW, "fermi_energy_eV": {}}
    for (n, key, *_), (spectrum, fermi) in sorted(zip(jobs, results), key=lambda item: item[0][:2]):
        pairs[f"{key}_{n}"] = spectrum
        info["fermi_energy_eV"][f"{key}_{n}"] = fermi
        print(f"mirror-pair spectrum: {n:4d} layers {SLAB_PATTERN[key]}, E_F = {1e3 * fermi:+.2f} meV")
    return pairs, info


def _bulk_pair_job(job):
    """Mirror-pair spectrum of one bulk polytype; runs in a worker process."""
    key, edges, fermi, n_kz, dk, refine, params = job
    mlg.set_parameters(**params)
    return bg.calculate_bulk_pair_spectrum(edges, key, fermi, n_kz=n_kz, dk=dk, refine=refine,
                                           verbose=True)["pairs_per_atom"]


def compute_bulk_pair_spectra(pairs, pair_info, info, n_kz=240, dk=2.0e-3, refine=8):
    """
    Add the mirror-pair spectra of the two bulk polytypes ('ab_bulk',
    'abc_bulk', pairs per eV per atom) to a table of slab spectra, on its grid
    of excitation energies; see ``bulk_graphite.calculate_bulk_pair_spectrum``.
    The Fermi levels are those of the bulk DOS in ``info``.
    """
    shift = pairs["shift"]
    step = shift[1] - shift[0]
    edges = np.concatenate([shift - 0.5 * step, [shift[-1] + 0.5 * step]])
    jobs = [(key, edges, info[key]["fermi_energy_eV"], n_kz, dk, refine * REFINE_FACTOR[key], mlg.get_parameters())
            for key, _, _ in STACKS]
    for (key, *_), spectrum in zip(jobs, _run_in_workers(_bulk_pair_job, jobs)):
        pairs[f"{key}_bulk"] = spectrum
    pair_info["bulk"] = {"n_kz": n_kz, "dk": dk, "refine": {key: refine * REFINE_FACTOR[key] for key, _, _ in STACKS}}
    return pairs, pair_info


def bulk_pair_rule(table, shift):
    """
    The mirror-pair spectrum of the bulk (pairs per eV per atom) as it follows
    from the DOS alone: for bands that are mirror images about the Fermi level
    the joint density of states at hbar w is D(E_F + hbar w / 2) / 2; the
    electron and the hole side are averaged.
    """
    return {key: 0.25 * (np.interp(0.5 * shift, table["energy"], table[key])
                         + np.interp(-0.5 * shift, table["energy"], table[key])) for key, _, _ in STACKS}


def scale_to_model(shift, intensity, model, raman_min, raman_max, linewidth):
    """
    Factor that puts a calculated spectrum in arbitrary units on the scale of
    a model spectrum: a least-squares fit of the two curves, broadened to the
    ERS linewidth (1/cm), inside the Raman window.
    """
    window = (shift >= raman_min * EV_PER_INVERSE_CM) & (shift <= raman_max * EV_PER_INVERSE_CM)
    width = linewidth * EV_PER_INVERSE_CM
    calculated, reference = (linewidth_broadening(shift, y, width)[window] for y in (intensity, model))
    return float((calculated * reference).sum() / (calculated * calculated).sum())


def _meV_axis(ax):
    """Top axis in meV for a bottom axis in 1/cm."""
    top = ax.secondary_xaxis("top", functions=(lambda w: w * EV_PER_INVERSE_CM * 1.0e3,
                                                lambda e: e / (EV_PER_INVERSE_CM * 1.0e3)))
    top.set_xlabel("ħω (meV)", color=INK)
    top.tick_params(colors=INK_SOFT, labelsize=9)
    top.spines["top"].set_color(INK_SOFT)
    return top


def plot_ers_calculation(spectra, pairs, path, raman_min, raman_max, linewidth):
    """
    Figure 4: calculated ERS spectra with the mirror-pair spectra of the slab
    model, one row per thickness.  Left: as computed; right: broadened to the
    ERS linewidth.  Each calculated spectrum is scaled by one factor.
    """
    layers = sorted({s["layers"] for s in spectra})
    shift = pairs["shift"]
    x = shift / EV_PER_INVERSE_CM
    width = linewidth * EV_PER_INVERSE_CM
    fig, axes = plt.subplots(len(layers), 2, figsize=(10.6, 0.8 + 3.5 * len(layers)), facecolor=SURFACE, squeeze=False)
    for row, n_layers in enumerate(layers):
        for col, ax in enumerate(axes[row]):
            def drawn(y):
                return linewidth_broadening(shift, y, width) if col else y
            for spectrum in (s for s in spectra if s["layers"] == n_layers and s["key"]):
                intensity = np.interp(shift, spectrum["shift"], spectrum["intensity"])
                scale = scale_to_model(shift, intensity, pairs[f"{spectrum['key']}_{n_layers}"], raman_min, raman_max, linewidth)
                ax.plot(x, 100.0 * scale * drawn(intensity), color=INK, alpha=0.3, linewidth=4.5, solid_capstyle="round",
                        label=f"calculated ERS, {dict((k, l) for k, l, _ in STACKS)[spectrum['key']]}, scaled")
                if col == 0:
                    for n, peak in enumerate(spectrum["peaks"], start=1):
                        ax.annotate(str(n), (peak / EV_PER_INVERSE_CM, 100.0 * scale * np.interp(peak, shift, intensity)),
                                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5, color=INK_SOFT)
            for key, label, color in STACKS:
                ax.plot(x, 100.0 * drawn(pairs[f"{key}_{n_layers}"]), color=color, linewidth=1.3,
                        label=f"mirror pairs, {label}")
            ax.axvspan(raman_min, raman_max, color=WINDOW, alpha=0.16, linewidth=0)
            ax.set_xlim(0.0, x[-1])
            ax.set_ylim(0.0, None)
            _style(ax)
            ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK,
                      title=f"{n_layers} layers" + (f", line width {linewidth:g} cm$^{{-1}}$" if col else ", as computed"),
                      title_fontproperties={"weight": "bold", "size": 9})
            if row == len(layers) - 1:
                ax.set_xlabel("Raman shift (cm$^{-1}$)", color=INK)
            if row == 0:
                _meV_axis(ax)
        axes[row, 0].set_ylabel("ERS spectral density (10$^{-2}$ pairs eV$^{-1}$ atom$^{-1}$)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def plot_pair_spectra(pairs, table, path, raman_min, raman_max, linewidth, layers):
    """
    Figure 5: the ERS estimate on the Raman-shift axis, one panel for the bulk
    and one per slab.  Thin lines: mirror-pair spectra as computed; thick
    lines: broadened to the ERS linewidth.  Dashed in the slab panels: the
    broadened bulk curves; dotted in the bulk panel: what the DOS alone gives.
    """
    shift = pairs["shift"]
    x = shift / EV_PER_INVERSE_CM
    width = linewidth * EV_PER_INVERSE_CM
    bulk = {key: pairs[f"{key}_bulk"] for key, _, _ in STACKS}
    rule = bulk_pair_rule(table, shift)
    panels = ["bulk"] + list(layers)
    ncol = min(2, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 0.7 + 3.4 * nrow), facecolor=SURFACE, squeeze=False)
    y_max = 1.08 * 100.0 * max(linewidth_broadening(shift, bulk[key], width).max() for key in bulk)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.axis("off")
            continue
        panel = panels[i]
        for key, label, color in STACKS:
            raw = bulk[key] if panel == "bulk" else pairs[f"{key}_{panel}"]
            ax.plot(x, 100.0 * raw, color=color, linewidth=0.7, alpha=0.45)
            ax.plot(x, 100.0 * linewidth_broadening(shift, raw, width), color=color, linewidth=1.9, label=label)
            if panel != "bulk":
                ax.plot(x, 100.0 * linewidth_broadening(shift, bulk[key], width), color=INK, linewidth=0.8,
                        linestyle=(0, (4, 2)), label="bulk" if key == "abc" else None)
            else:
                ax.plot(x, 100.0 * linewidth_broadening(shift, rule[key], width), color=INK, linewidth=0.9,
                        linestyle=(0, (1, 2)), label="from the DOS" if key == "abc" else None)
        ax.axvspan(raman_min, raman_max, color=WINDOW, alpha=0.16, linewidth=0)
        ax.set_xlim(0.0, x[-1])
        ax.set_ylim(0.0, y_max)
        _style(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK,
                  title="bulk" if panel == "bulk" else f"{panel} layers", title_fontproperties={"weight": "bold", "size": 9})
        if i // ncol == 0:
            _meV_axis(ax)
        if i + ncol >= len(panels):
            ax.set_xlabel("Raman shift (cm$^{-1}$)", color=INK)
        if i % ncol == 0:
            ax.set_ylabel("ERS spectral density (10$^{-2}$ pairs eV$^{-1}$ atom$^{-1}$)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def window_mean(energy, dos, raman_min, raman_max):
    """Mean DOS over the energies E_F -+ hbar w / 2 reached by the spectrometer."""
    sel = (np.abs(energy) >= 0.5 * raman_min * EV_PER_INVERSE_CM) & (np.abs(energy) <= 0.5 * raman_max * EV_PER_INVERSE_CM)
    return float(dos[sel].mean())


def dos_at_shift(table, shift):
    """DOS (states/(eV atom)) of both polytypes at E_F -+ hbar w / 2, averaged over the two signs."""
    e = 0.5 * shift * EV_PER_INVERSE_CM
    return tuple(0.5 * (np.interp(e, table["energy"], table[key]) + np.interp(-e, table["energy"], table[key]))
                 for key, _, _ in STACKS)


def shift_rows(table, raman_min, raman_max):
    """(Raman shift, |E - E_F| in meV, DOS hexagonal, DOS rhombohedral, ratio) across the window."""
    shifts = sorted({raman_min, raman_max} | {v for v in (200.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
                                               if raman_min < v < raman_max})
    rows = []
    for shift in shifts:
        hexa, rhombo = dos_at_shift(table, shift)
        rows.append((shift, 0.5e3 * shift * EV_PER_INVERSE_CM, hexa, rhombo, rhombo / hexa))
    return rows


def crossing_shift(table):
    """Raman shift (1/cm) above which the rhombohedral DOS exceeds the hexagonal one."""
    shifts = np.arange(100.0, 2.0 * table["energy"].max() / EV_PER_INVERSE_CM, 10.0)
    diff = np.array([np.subtract(*dos_at_shift(table, w)) for w in shifts])
    change = np.nonzero((diff[:-1] > 0.0) & (diff[1:] <= 0.0))[0]
    return float(shifts[change[0] + 1]) if change.size else np.nan


def report(table, info, raman_min, raman_max):
    for key, label, _ in STACKS:
        print(f"{label:19s} E_F = {1e3 * info[key]['fermi_energy_eV']:+7.2f} meV, "
              f"n = p = {info[key]['carrier_density_cm3']:.2e} cm^-3, "
              f"DOS(E_F) = {np.interp(0.0, table['energy'], table[key]):.2e} states/(eV atom)")
    print("DOS at E_F -+ hbar w / 2, averaged over the two signs (10^-2 states/(eV atom)):")
    print(f"{'Raman shift (1/cm)':>19s} {'|E-E_F| (meV)':>14s} {'hexagonal':>10s} {'rhombohedral':>13s} {'ratio':>6s}")
    for shift, e, hexa, rhombo, ratio in shift_rows(table, raman_min, raman_max):
        print(f"{shift:19.0f} {e:14.1f} {100 * hexa:10.3f} {100 * rhombo:13.3f} {ratio:6.2f}")
    print(f"the two curves cross at a Raman shift of {crossing_shift(table):.0f} 1/cm")


def _listing(values):
    """'a, b and c' for a list of strings."""
    return ", ".join(values[:-1]) + " and " + values[-1] if len(values) > 1 else values[0]


def ers_peak(n, n_layers, gamma1):
    """
    n- -> n+ excitation energy (eV) of an n_layers rhombohedral film:
    2 gamma1 sin(q_n) with q_n = (n + 1/2) pi / (n_layers + 1/2), the edge of a
    pair of subbands of the chain with couplings hbar v p and gamma1.  For
    n << n_layers this is the 4 gamma1 sin(q_n / 2) of Garcia-Ruiz et al.
    (2019), Eq. (1).
    """
    return 2.0 * gamma1 * np.sin((n + 0.5) * np.pi / (n_layers + 0.5))


def subband_peak_pair(energy, dos, centre, expected, half_width):
    """
    Separation (eV) of the DOS peaks found within half_width of
    centre -+ expected / 2: the -n and +n subband peaks of a rhombohedral slab.
    """
    found = []
    for target in (centre - 0.5 * expected, centre + 0.5 * expected):
        sel = np.abs(energy - target) <= half_width
        found.append(energy[sel][np.argmax(dos[sel])])
    return found[1] - found[0]


# LaTeX fragments of the report; the page loads MathJax to typeset them (SVG
# output, which is one script and fetches no font files at run time)
MATHJAX = ('<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>'
           '<style>mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; }</style>')
CM1 = r"\(\mathrm{cm^{-1}}\)"
CM3 = r"\(\mathrm{cm^{-3}}\)"
PER_ANGSTROM = r"\(\text{Å}^{-1}\)"
PER_EV_ATOM = r"\(\mathrm{eV^{-1}\,atom^{-1}}\)"
EF = r"\(E_F\)"
DE = r"\(|E - E_F|\)"
PAIR = r"\(n^- \to n^+\)"


def _sci(x, digits=1):
    """A number as LaTeX in scientific notation."""
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"


def _signed(text):
    """Typographic minus for a number that is not set as math."""
    return text.replace("-", "−")


def _slab_remarks(slabs, slab_info, table, info, raman_min, raman_max):
    layers = slab_info["layers"]
    par = info["parameters"]
    energy = slabs["energy"]
    e_hi = 0.5 * raman_max * EV_PER_INVERSE_CM
    hv = 1.5 * (par["a"] / np.sqrt(3.0)) * par["gamma0"]
    listing = _listing
    names = listing([str(n) for n in layers])

    sel = (np.abs(energy) >= 0.03) & (np.abs(energy) <= e_hi)
    deviation = [np.abs(slabs[f"ab_{n}"][sel] / bulk_on_slab_axis(slabs, slab_info, table, info, "ab", n)[sel] - 1.0).max()
                 for n in layers]
    envelope = [slabs[f"abc_{n}"][sel].mean() / bulk_on_slab_axis(slabs, slab_info, table, info, "abc", n)[sel].mean()
                for n in layers]
    first = [ers_peak(1, n, par["gamma1"]) / EV_PER_INVERSE_CM for n in layers]
    spacing = [4.0 * np.pi * par["gamma1"] / (2 * n + 1) / EV_PER_INVERSE_CM for n in layers]
    offset = [info["abc"]["fermi_energy_eV"] - slab_info["fermi_energy_eV"][f"abc_{n}"] for n in layers]
    surface_states = 2.0 * (par["gamma1"] / hv) ** 2 * mlg.unit_cell_area(par["a"]) / np.pi

    thin = layers[0]
    merged = [str(n) for n, w in zip(layers, spacing) if w < 500.0]
    merging = (f"The ladders of the {_listing([n + '-' for n in merged[:-1]] + merged[-1:])}-layer slabs "
               "are finer than that and merge" if merged
               else "A ladder finer than that merges")
    check = ""
    if 2 * thin + 1 < 80:       # subbands far enough apart to find the peaks by their expected position
        pairs = [(subband_peak_pair(energy, slabs[f"abc_{thin}"], offset[0], ers_peak(n, thin, par["gamma1"]),
                                    np.pi * par["gamma1"] / (2 * thin + 1) / 2.0), ers_peak(n, thin, par["gamma1"]))
                 for n in (1, 2)]
        check = (f" In the {thin}-layer DOS the first two pairs of peaks are "
                 f"{pairs[0][0] * 1e3:.0f} and {pairs[1][0] * 1e3:.0f} meV apart; the formula gives "
                 f"{pairs[0][1] * 1e3:.0f} and {pairs[1][1] * 1e3:.0f} meV.")

    return (
        "<ul>"
        "<li><b>Hexagonal slabs follow the bulk curve.</b> Between 30 meV and the upper edge of the window the "
        f"largest deviation is {listing([f'{100 * d:.1f} %' for d in deviation])} for {names} layers. "
        "The discrete subbands show only as small steps within about 20 meV of the Fermi level.</li>"
        "<li><b>Rhombohedral slabs resolve the bulk continuum into a ladder of subbands.</b> Each subband starts with a "
        r"step and a van Hove peak. The \(-n\) and \(+n\) peaks pair into ERS peaks at"
        r"\[\hbar\omega_n \approx 2\gamma_1 \sin\!\left[\frac{\left(n + \frac{1}{2}\right)\pi}{N + \frac{1}{2}}\right],\]"
        r"which for \(n \ll N\) is the \(4\gamma_1 \sin[(n + \tfrac{1}{2})\pi/(2N + 1)]\) of García-Ruiz et al. (2019). "
        f"The first one lies at {listing([f'{w:.0f}' for w in first])} {CM1} for {names} layers, and at the bottom of "
        r"the ladder the peaks follow each other every \(4\pi\gamma_1/(2N + 1) =\) "
        f"{listing([f'{w:.0f}' for w in spacing])} {CM1}.{check}</li>"
        "<li><b>Thick rhombohedral slabs give a bulk-like background.</b> The measured ERS peaks of thin films are about "
        f"500 {CM1} wide at room temperature (Pálinkás et al. 2024). {merging} into a "
        "continuum that follows the envelope of the DOS, and that envelope is the bulk curve: averaged between 30 meV and "
        f"the upper edge of the window the slab DOS is {listing([f'{100 * r:.1f} %' for r in envelope])} of the bulk DOS "
        f"for {names} layers.</li>"
        "<li><b>Surface band.</b> The plateau about 30 meV wide at the Fermi level of the rhombohedral slabs is the surface "
        r"band. It holds \(2(\gamma_1/\hbar v)^2 A_\mathrm{cell}/\pi =\) "
        f"{surface_states:.3f} states per unit cell of the film (two surfaces, "
        r"spin and valley included) whatever the thickness, so its height per atom falls as \(1/N\).</li>"
        "<li><b>Fermi level of the rhombohedral slabs.</b> The surface band pins the Fermi level "
        f"{listing([f'{1e3 * o:.0f}' for o in offset])} meV below the bulk Dirac energy for {names} layers. The dashed bulk "
        f"curves are drawn at the same absolute energy, which puts their minimum to the right of zero. The {PAIR} "
        f"pairs are centred near the Dirac energy and not on {EF}, so for the subbands the shaded "
        r"window should be read as shifted by that amount; the sections below take the pair energy itself. "
        r"At low temperature the pairs with \(\hbar\omega/2\) smaller "
        f"than the offset (shifts below about {round(2.0 * min(offset) / EV_PER_INVERSE_CM, -1):.0f} {CM1}) are Pauli "
        "blocked. The offset belongs to the non-interacting model, which leaves out the charge redistribution between "
        "surface and interior.</li>"
        "</ul>")


def _slab_section(slabs, slab_info, table, info, raman_min, raman_max, fig_slabs):
    """Report section on the finite slabs: figure, table and remarks."""
    from make_report import img_tag, table as html_table

    layers = slab_info["layers"]
    e_lo, e_hi = (0.5e3 * w * EV_PER_INVERSE_CM for w in (raman_min, raman_max))
    names = _listing([str(n) for n in layers])
    rows = []
    for n in layers:
        hexa, rhombo = (window_mean(slabs["energy"], slabs[f"{key}_{n}"], raman_min, raman_max) for key, _, _ in STACKS)
        rows.append((f"{n} layers", _signed(f"{slab_info['fermi_energy_eV'][f'ab_{n}'] * 1e3:+.1f}"),
                     _signed(f"{slab_info['fermi_energy_eV'][f'abc_{n}'] * 1e3:+.1f}"),
                     f"{100 * hexa:.3f}", f"{100 * rhombo:.3f}", f"{rhombo / hexa:.3f}"))
    hexa, rhombo = (window_mean(table["energy"], table[key], raman_min, raman_max) for key, _, _ in STACKS)
    rows.append(("bulk", _signed(f"{info['ab']['fermi_energy_eV'] * 1e3:+.1f}"),
                 _signed(f"{info['abc']['fermi_energy_eV'] * 1e3:+.1f}"),
                 f"{100 * hexa:.3f}", f"{100 * rhombo:.3f}", f"{rhombo / hexa:.3f}"))
    return [
        f"<h2>Finite slabs: {names} layers</h2>",
        r"<p>The same tight-binding model for slabs of \(N\) layers with free surfaces "
        r"(<code>multilayer_graphene.py</code>, \(2N\) bands), hexagonal (ABAB…) and rhombohedral (ABCABC…). The DOS is "
        r"divided by the \(2N\) atoms of the cell, so it can be laid over the bulk curves, and each slab has its own Fermi "
        "level from charge neutrality.</p>",
        f"<figure>{img_tag(fig_slabs, 'DOS of finite slabs of hexagonal and rhombohedral graphite')}"
        f"<figcaption>Figure 3. DOS per atom of slabs of {names} layers: hexagonal (blue) and rhombohedral (orange), "
        "energies measured from the Fermi level of each slab. Dashed: the bulk curves of Figure 1 at the same absolute "
        "energy. Left: the energy range of Figure 2, with the same shading and the same arrow for the largest Raman "
        "shift. Right: the 100 meV around the Fermi level, on a common vertical scale.</figcaption></figure>",
        html_table(
            ["", rf"{EF} hexagonal (meV)", rf"{EF} rhombohedral (meV)", "hexagonal", "rhombohedral",
             "rhombohedral / hexagonal"],
            rows,
            f"Fermi level on the model's energy scale, and DOS averaged over {DE} = {e_lo:.0f}–{e_hi:.0f} meV "
            rf"(both signs) in \(10^{{-2}}\) states {PER_EV_ATOM}"),
        _slab_remarks(slabs, slab_info, table, info, raman_min, raman_max),
        "<p>Numerical details: in-plane mesh around K with spacing "
        f"{slab_info['dk']:g} {PER_ANGSTROM} and half-width {slab_info['k_max']:g} {PER_ANGSTROM}, triangle interpolation, "
        f"{slab_info['bin_width'] * 1e3:g} meV bins, no smoothing. "
        r"Only \(p_y \ge 0\) is diagonalised; the other half follows from \(E(p_x, -p_y) = E(p_x, p_y)\). "
        "For 20 layers, halving the mesh spacing changes single 1 meV bins at subband edges by up to "
        rf"\(0.04\times10^{{-2}}\) states {PER_EV_ATOM} and the rest of the "
        "curves by less; the thicker slabs were not repeated on a finer mesh.</p>",
    ]


def peak_contrast(x, y, lo, hi):
    """
    Largest peak-to-valley contrast 1 - y_min / y_peak among the local maxima
    of y with lo <= |x| <= hi, each compared with the next local minimum on
    its far side from x = 0 (where a subband peak decays before the next
    subband starts).  0 if the curve has no such peak.
    """
    best = 0.0
    inner = np.arange(1, x.size - 1)
    peaks = inner[(y[inner] > y[inner - 1]) & (y[inner] >= y[inner + 1]) & (np.abs(x[inner]) >= lo) & (np.abs(x[inner]) <= hi)]
    for i in peaks:
        step = 1 if x[i] > 0 else -1
        j = i
        while 0 < j + step < x.size - 1 and y[j + step] <= y[j] and abs(x[j + step]) <= hi:
            j += step
        if j != i and 0 < j + step < x.size - 1 and y[j + step] > y[j]:
            best = max(best, 1.0 - y[j] / y[i])
    return best


def _onset(shift, spectrum, fraction=0.25):
    """Excitation energy (eV) at which a spectrum reaches the given fraction of its value at 60 meV."""
    return shift[np.nonzero(spectrum[1:] > fraction * np.interp(0.06, shift, spectrum))[0][0] + 1]


def _ers_section(spectra, pairs, raman_min, raman_max, linewidth, fig_ers):
    """Report section that sets calculated ERS spectra against the mirror-pair spectra of the slab model."""
    from make_report import img_tag, table as html_table

    gamma1 = mlg.get_parameters()["gamma1"]
    shift = pairs["shift"]
    width = linewidth * EV_PER_INVERSE_CM
    window = (shift >= raman_min * EV_PER_INVERSE_CM) & (shift <= raman_max * EV_PER_INVERSE_CM)
    kinds = {"abc": "rhombohedral", "ab": "hexagonal"}
    layers = sorted({s["layers"] for s in spectra})
    out = [
        f"<h2>Calculated ERS of {'a ' if len(layers) == 1 else ''}{_listing([str(n) for n in layers])}-layer film"
        f"{'' if len(layers) == 1 else 's'}</h2>",
        "<p>A Raman shift is the difference between the energy of a state above and a state below the Fermi level. "
        "The ERS spectra calculated for whole films are therefore compared with a quantity taken on that difference: "
        f"the joint density of states of the mirror pairs {PAIR} of the slab model of this report, the n-th band below "
        r"and the n-th band above the middle of the spectrum, with the occupation of \(T = 0\):</p>"
        r"\[J(\omega) = \frac{g_s g_v}{N_\mathrm{at}} \sum_n \int \frac{d^2p}{(2\pi)^2}\,"
        r"\theta\!\left(E_F - \varepsilon_{n^-}\right)\theta\!\left(\varepsilon_{n^+} - E_F\right)"
        r"\delta\!\left(\hbar\omega - \varepsilon_{n^+}(\mathbf{p}) + \varepsilon_{n^-}(\mathbf{p})\right).\]"
        r"<p>This is Eq. (5) of García-Ruiz et al. (2019) with the matrix element \(|\langle n^+|\sigma_z|n^-\rangle|^2\) "
        r"of a mirror pair set to 1. For bands that are mirror images about the Fermi level it reduces to "
        r"\(D(E_F \pm \hbar\omega/2)/2\), the working rule used for the bulk.</p>",
        f"<figure>{img_tag(fig_ers, 'Calculated ERS spectra and mirror-pair spectra of the slab model')}"
        "<figcaption>Figure 4. Calculated ERS spectrum (grey band) and mirror-pair spectra of the slab model: "
        "rhombohedral (orange) and hexagonal (blue). Left: as computed, with the pairs "
        r"\(-n \to +n\) of the calculated peaks numbered. Right: broadened to an ERS line of "
        f"{linewidth:g} {CM1}. The calculated spectrum is in arbitrary units and is scaled by one factor; the Raman "
        f"shifts of {raman_min:g}–{raman_max:g} {CM1} are shaded.</figcaption></figure>",
    ]
    for n_layers in layers:
        remarks = []
        supplied = [s for s in spectra if s["layers"] == n_layers and s["key"]]
        for spectrum in supplied:
            model = pairs[f"{spectrum['key']}_{n_layers}"]
            intensity = np.interp(shift, spectrum["shift"], spectrum["intensity"])
            scaled = scale_to_model(shift, intensity, model, raman_min, raman_max, linewidth) * intensity
            kind = kinds[spectrum["key"]]
            mismatch = np.abs(scaled - model)[window].mean() / model[window].mean()
            broad = [linewidth_broadening(shift, y, width)[window] for y in (scaled, model)]
            rows, worst = [], 0.0
            for n, peak in enumerate(spectrum["peaks"], start=1):
                near = np.abs(shift - peak) <= 0.009
                found = shift[near][np.argmax(model[near])]
                formula = ers_peak(n, n_layers, gamma1)
                worst = max(worst, abs(formula - peak))
                rows.append((str(n), f"{peak * 1e3:.0f}", f"{peak / EV_PER_INVERSE_CM:.0f}", f"{found * 1e3:.0f}",
                             f"{formula * 1e3:.0f}"))
            out.append(html_table(
                ["n", "calculated ERS (meV)", f"calculated ERS ({CM1})", "mirror pairs (meV)",
                 r"\(2\gamma_1 \sin q_n\) (meV)"],
                rows,
                rf"Peaks \(-n \to +n\) of the {n_layers}-layer {kind} film ({spectrum['name']}): from the header of "
                "the calculated spectrum, from the mirror-pair spectrum on the same "
                f"{1e3 * (shift[1] - shift[0]):g} meV grid, and from the closed form with "
                r"\(q_n = (n + \tfrac{1}{2})\pi/(N + \tfrac{1}{2})\)"))
            remarks += [
                f"<li><b>The {n_layers}-layer {kind} calculation is reproduced.</b> With one scale factor the two spectra "
                f"differ by {100 * mismatch:.1f} % on average over the bins inside the window, and by "
                f"{100 * np.abs(broad[0] - broad[1]).mean() / broad[1].mean():.1f} % after broadening. Both rise near "
                f"{1e3 * _onset(shift, scaled):.0f} meV ({_onset(shift, scaled) / EV_PER_INVERSE_CM:.0f} {CM1}): below "
                "that the pairs of the surface band are blocked by its occupation.</li>",
                r"<li><b>Peak positions.</b> The peaks follow \(\hbar\omega_n \approx 2\gamma_1 \sin q_n\) to within "
                f"{1e3 * worst:.0f} meV. Each one sits at the top of a step, the edge of its pair of subbands.</li>"]
        broad = {key: linewidth_broadening(shift, pairs[f"{key}_{n_layers}"], width) for key in kinds}
        picks = [w for w in (1000.0, 2000.0, 3000.0, 5000.0) if raman_min < w < raman_max]
        ratio = [f"{np.interp(w * EV_PER_INVERSE_CM, shift, broad['abc'] / broad['ab']):.2f}" for w in picks]
        contrast = {key: 100 * peak_contrast(shift, broad[key], raman_min * EV_PER_INVERSE_CM,
                                             raman_max * EV_PER_INVERSE_CM) for key in kinds}
        missing = [kinds[key] for key in kinds if key not in {s["key"] for s in supplied}]
        remarks.append(
            f"<li><b>Rhombohedral against hexagonal at {n_layers} layers.</b> "
            + (f"No calculated spectrum was supplied for the {_listing(missing)} film; its curve is the mirror-pair "
               "spectrum alone. " if missing else "")
            + f"Broadened to {linewidth:g} {CM1}, the rhombohedral / hexagonal ratio of the mirror-pair spectra is "
            f"{_listing(ratio)} at {_listing([f'{w:.0f}' for w in picks])} {CM1}. The strongest peak that remains "
            f"stands {contrast['abc']:.0f} % above the minimum after it in the rhombohedral film and "
            f"{contrast['ab']:.0f} % in the hexagonal one.</li>")
        out.append("<ul>" + "".join(remarks) + "</ul>")
    return out


def _pair_section(pairs, pair_info, table, raman_min, raman_max, linewidth, layers, fig_pairs):
    """Report section on the ERS estimate on the Raman-shift axis: thicker slabs, bulk and the line width."""
    from make_report import img_tag, table as html_table

    shift = pairs["shift"]
    width = linewidth * EV_PER_INVERSE_CM
    thermal = 2.0 * np.log(3.0 + 2.0 * np.sqrt(2.0)) * K_BOLTZMANN * 300.0      # FWHM of -df/dE at 300 K
    lo, hi = raman_min * EV_PER_INVERSE_CM, raman_max * EV_PER_INVERSE_CM
    bulk = {key: pairs[f"{key}_bulk"] for key, _, _ in STACKS}
    rule = {key: linewidth_broadening(shift, y, width) for key, y in bulk_pair_rule(table, shift).items()}
    raw = {"bulk": bulk, **{n: {key: pairs[f"{key}_{n}"] for key, _, _ in STACKS} for n in pair_info["layers"]}}
    broad = {p: {key: linewidth_broadening(shift, y, width) for key, y in curves.items()} for p, curves in raw.items()}
    panels = ["bulk"] + list(layers)
    every = pair_info["layers"]
    names = _listing([str(n) for n in layers])
    all_names = _listing([str(n) for n in every])

    shifts = sorted({raman_min, raman_max} | {v for v in (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0)
                                             if raman_min < v < raman_max})
    rows = [[f"{w:.0f}", f"{w * EV_PER_INVERSE_CM * 1e3:.0f}"]
            + [f"{np.interp(w * EV_PER_INVERSE_CM, shift, broad[p]['abc'] / broad[p]['ab']):.2f}" for p in panels]
            for w in shifts]
    contrast = {key: [f"{100 * peak_contrast(shift, broad[n][key], lo, hi):.0f} %" for n in every] for key, _, _ in STACKS}
    upper = (shift >= 2000.0 * EV_PER_INVERSE_CM) & (shift <= hi)
    to_bulk = {key: [f"{100 * broad[n][key][upper].mean() / broad['bulk'][key][upper].mean():.1f} %" for n in every]
               for key, _, _ in STACKS}
    onset = [f"{_onset(shift, raw[n]['abc']) / EV_PER_INVERSE_CM:.0f}" for n in every]
    inside = (shift >= lo) & (shift <= hi)
    rule_error = {key: 100 * np.abs(rule[key] / broad["bulk"][key] - 1.0)[inside] for key, _, _ in STACKS}
    above = (shift >= 500.0 * EV_PER_INVERSE_CM) & (shift <= hi)
    rule_excess = 100 * (rule["ab"] / broad["bulk"]["ab"] - 1.0)
    rule_at_min = np.interp(lo, shift, bulk_pair_rule(table, shift)["ab"] / bulk["ab"])
    half_plateau = [_onset(shift, raw[p]["ab"], 0.5) / EV_PER_INVERSE_CM for p in ["bulk"] + list(every)]
    rule_ratio = [np.interp(w * EV_PER_INVERSE_CM, shift, rule["abc"] / rule["ab"]) for w in (1000.0, 3000.0)]
    true_ratio = [np.interp(w * EV_PER_INVERSE_CM, shift, broad["bulk"]["abc"] / broad["bulk"]["ab"]) for w in (1000.0, 3000.0)]

    return [
        "<h2>ERS estimate on the Raman-shift axis</h2>",
        "<p>The mirror-pair spectrum is not limited to the thickness of the calculation above. Figure 5 shows it for "
        f"slabs of {names} layers, against the Raman shift itself, and for the bulk. The bulk spectrum is the same "
        "integral over the three-dimensional Brillouin zone: the pair energies of the four bands of hexagonal graphite "
        r"(two pairs) and of the two bands of rhombohedral graphite (one pair), on the \(k_z\) slices and meshes of the "
        "bulk DOS, with the Fermi levels found there.</p>"
        f"<p>The ERS peaks of thin rhombohedral films are about 500 {CM1} wide at room temperature, which Pálinkás et al. "
        rf"(2024) relate to a thermal width of \(3k_B T =\) {5 * round(0.6 * K_BOLTZMANN * 300.0 / EV_PER_INVERSE_CM):d} {CM1}. "
        "The thick curves are the spectra convolved with a Gaussian of that width on the Raman-shift axis, "
        rf"\(\Gamma = {linewidth:g}\) {CM1} at half maximum. Smearing the single-particle energies with the derivative "
        f"of the Fermi function at 300 K would be {thermal * 1e3:.0f} meV wide and, because a shift is the difference "
        f"of two such energies, {2.0 * thermal / EV_PER_INVERSE_CM:.0f} {CM1} on this axis: "
        f"{2.0 * thermal / EV_PER_INVERSE_CM / linewidth:.0f} times the measured line, enough to erase the peaks that "
        "are observed in films of up to about 20 layers.</p>",
        f"<figure>{img_tag(fig_pairs, 'ERS estimate of bulk graphite and of finite slabs on the Raman-shift axis')}"
        "<figcaption>Figure 5. ERS estimate against the Raman shift: hexagonal (blue) and rhombohedral (orange), for the "
        f"bulk and for slabs of {names} layers. Thin lines: mirror-pair spectra as computed. Thick lines: broadened "
        f"to an ERS line of {linewidth:g} {CM1}. Dashed in the slab panels: the broadened bulk curves. Dotted in the "
        r"bulk panel: \(\tfrac{1}{4}\left[D(E_F + \hbar\omega/2) + D(E_F - \hbar\omega/2)\right]\) from the DOS, "
        f"broadened. The Raman shifts of {raman_min:g}–{raman_max:g} {CM1} are shaded.</figcaption></figure>",
        html_table(
            [f"Raman shift ({CM1})", r"\(\hbar\omega\) (meV)"] + ["bulk" if p == "bulk" else f"{p} layers" for p in panels],
            rows,
            f"Rhombohedral / hexagonal ratio of the ERS estimate, broadened to {linewidth:g} cm⁻¹"),
        "<ul>"
        "<li><b>Which films show peaks.</b> In the broadened rhombohedral spectra the strongest remaining peak stands "
        f"{_listing(contrast['abc'])} above the minimum that follows it for {all_names} layers; "
        + ("the hexagonal spectra have no peak" if set(contrast["ab"]) == {"0 %"}
           else f"in the hexagonal spectra {_listing(contrast['ab'])}")
        + ". A peak survives when the pairs are further apart than the line is wide, "
        r"that is while \(4\pi\gamma_1/(2N + 1)\) exceeds \(\Gamma\).</li>"
        "<li><b>Thick films follow the bulk.</b> Averaged between 2000 "
        f"{CM1} and the upper edge of the window, the broadened rhombohedral spectrum is {_listing(to_bulk['abc'])} of "
        f"the bulk spectrum for {all_names} layers, and the hexagonal one {_listing(to_bulk['ab'])}.</li>"
        "<li><b>How good the DOS rule was.</b> The rule of the first sections, "
        r"\(J(\omega) \simeq \tfrac{1}{4}\left[D(E_F + \hbar\omega/2) + D(E_F - \hbar\omega/2)\right]\), "
        "assumes bands that mirror about the Fermi level. Inside the window it differs from the bulk pair spectrum by "
        f"{rule_error['abc'].mean():.1f} % on average and {rule_error['abc'].max():.1f} % at most for rhombohedral "
        f"graphite (broadened curves). For hexagonal graphite it is {rule_excess[above].mean():.1f} % too high on "
        f"average above 500 {CM1}, where the hole side carries more states than the electron side, and it misses "
        f"the fall of the spectrum at low shifts: at {raman_min:g} {CM1} it gives {rule_at_min:.1f} times the pair "
        "spectrum as computed. The rhombohedral / hexagonal ratio at 1000 and 3000 "
        f"{CM1} is {true_ratio[0]:.2f} and {true_ratio[1]:.2f} from the pair spectra, against "
        f"{rule_ratio[0]:.2f} and {rule_ratio[1]:.2f} from the rule.</li>"
        "<li><b>Low shifts.</b> The spectra of the rhombohedral slabs start near "
        f"{_listing(onset)} {CM1} for {all_names} layers: the surface band pins the Fermi level and blocks its own "
        "pairs below that. Bulk rhombohedral graphite has no such threshold. In hexagonal graphite the spectrum "
        f"falls off at low shifts in the bulk and in the slabs alike, to half of its plateau near "
        + (f"{half_plateau[0]:.0f} {CM1} in the bulk and in every slab"
           if len({round(w) for w in half_plateau}) == 1 else
           f"{half_plateau[0]:.0f} {CM1} in the bulk and {_listing([f'{w:.0f}' for w in half_plateau[1:]])} {CM1} "
           f"for {all_names} layers")
        + ": with the occupation of T = 0 the pairs of low energy are blocked in the electron and "
        f"hole pockets of the semimetal. Ponosov et al. see no such gap at room temperature down to 50 {CM1}.</li>"
        "<li><b>What the line width stands for.</b> In Eq. (1) temperature enters through the occupation factors and "
        "does not move a pair excitation, so the width is not derived here: it is the measured one, set with "
        "<code>--linewidth</code>.</li>"
        "</ul>",
        "<p>Numerical details: for the slabs, in-plane mesh around K with spacing "
        f"{pair_info['dk']:g} {PER_ANGSTROM} and half-width {pair_info['k_max']:g} {PER_ANGSTROM}, triangle "
        f"interpolation of the pair energies, bins of {pair_info['step'] * 1e3:g} meV. A triangle counts with the "
        "fraction of its corners at which the pair is allowed by the occupation. Counting bands from the middle of the "
        "spectrum finds the mirror partner only while the subbands keep their order, which fails near the Fermi level "
        "of a hexagonal slab, where its bilayer-like subbands overlap. The hexagonal spectra below "
        f"{1e3 * pair_info['weighted_below_eV']['ab']:g} meV are therefore the sum over all pairs of bands weighted by "
        r"\(|\langle c|\sigma_z|v\rangle|^2\), from a window of half-width 0.045 "
        f"{PER_ANGSTROM}. For the bulk, "
        rf"{pair_info['bulk']['n_kz']} values of \(k_z\) and a mesh spacing of {pair_info['bulk']['dk']:g} {PER_ANGSTROM}, "
        f"with the cells near the Fermi level subdivided {pair_info['bulk']['refine']['ab']} times (hexagonal) or "
        f"{pair_info['bulk']['refine']['abc']} times (rhombohedral); the subdivision is used for excitation energies "
        "below 0.12 eV. With these default settings, doubling the number of "
        r"\(k_z\) values, doubling the subdivision or halving the mesh spacing changes the bulk pair spectrum by less "
        "than 0.3 %. For 15-layer slabs the mirror-pair spectrum is within 1.5 % of that weighted sum at all "
        "energies for rhombohedral stacking and within about 3 % above 27 meV for hexagonal stacking; for the bulk, "
        "where the pairs are "
        r"taken at the same \(k_z\), within 0.5 % for both.</p>",
    ]


def write_report(path, table, info, raman_min, raman_max, figures, slabs=None, slab_info=None,
                 spectra=None, pairs=None, pair_info=None, linewidth=500.0, standalone=True):
    """
    Self-contained HTML page (figures embedded) in the style of make_report.py.
    ``figures`` maps 'comparison', 'window', 'slabs', 'ers' and 'pairs' to the
    image files of the sections that are present.  The mathematics is written
    in LaTeX and typeset by MathJax, which the page loads from a CDN.  With
    standalone=False the document skeleton (doctype, html, head and body tags)
    is left out, for a publisher that adds its own.
    """
    from make_report import STYLE, img_tag, table as html_table

    rows = shift_rows(table, raman_min, raman_max)
    cross = crossing_shift(table)
    e_lo, e_hi = (0.5e3 * w * EV_PER_INVERSE_CM for w in (raman_min, raman_max))
    fermi = {key: (info[key]["fermi_energy_eV"], info[key]["carrier_density_cm3"],
                   float(np.interp(0.0, table["energy"], table[key]))) for key, _, _ in STACKS}
    par = info["parameters"]
    swmcc_delta = 2.0 * par["Delta_prime"] + par["gamma2"] - par["gamma5"]
    overlap = 4.0 * par["gamma4"] / par["gamma0"] * (par["gamma1"] * par["gamma3"] / par["gamma0"] + 0.5 * par["gamma2"])
    dirac = par["Delta_prime"] + 2.0 * par["gamma1"] * par["gamma4"] / par["gamma0"]

    s = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'] if standalone else []
    s += ["<title>Graphite ERS background</title>",
          '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600'
          '&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono&display=swap">',
          STYLE, "<style>.words td:last-child, .words th:last-child { text-align: left; white-space: normal; }</style>",
          MATHJAX]
    s += ["</head><body>"] if standalone else []
    s += ['<div class="wrap">',
          '<p class="eyebrow">Bulk graphite · SWMcC model · density of states</p>',
          "<h1>ERS background of bulk graphite</h1>",
          '<p class="lede">Density of states of pure hexagonal (AB) and rhombohedral (ABC) bulk graphite, as an '
          "estimate of the electronic Raman scattering (ERS) continuum of the two polytypes within the "
          f"{raman_min:g}–{raman_max:g} {CM1} range of the spectrometer.</p>",
          '<div class="key">',
          f"<div><b>{rows[0][4]:.2f} → {rows[-1][4]:.2f}</b><span>rhombohedral / hexagonal DOS from "
          f"{raman_min:g} to {raman_max:g} {CM1}</span></div>",
          f"<div><b>{cross:.0f}</b><span>Raman shift in {CM1} at which the two DOS curves cross</span></div>",
          f"<div><b>{e_lo:.0f}–{e_hi:.0f} meV</b><span>{DE} probed by the spectrometer, "
          r"\(\hbar\omega/2\)</span></div>",
          f"<div><b>{fermi['abc'][2] / fermi['ab'][2]:.3f}</b><span>rhombohedral / hexagonal DOS at the Fermi level</span></div>",
          "</div>"]

    s.append("<h2>From the ERS signal to the density of states</h2>")
    s.append(
        "<p>In graphene, few-layer graphene and graphite the ERS continuum consists of interband electron–hole "
        "excitations with negligible momentum transfer. They are created through a virtual intermediate state, appear "
        r"in crossed polarisation (\(A_{2g}\) symmetry), and the matrix element is nearly independent of energy "
        "(Kashuba and Fal'ko 2009; Ponosov et al. 2015). Ponosov et al. model the spectrum of bulk graphite with a "
        "constant matrix element. In the limit of small damping their Eq. (1) reads</p>"
        r"\[I(\omega) \propto \frac{1}{1 - e^{-\hbar\omega/k_B T}} \sum_{c,v} \int d^3p\,"
        r"\left[f(\varepsilon_v) - f(\varepsilon_c)\right]"
        r"\delta\!\left(\hbar\omega - \varepsilon_c(\mathbf{p}) + \varepsilon_v(\mathbf{p})\right).\]"
        "<p>García-Ruiz et al. (2019), Eq. (5), find that the Raman vertex is proportional to "
        r"\((\mathbf{l} \times \mathbf{l}')_z\,\sigma_z\), "
        f"so the excitations that count connect each valence band to its mirror conduction band, {PAIR}, "
        "and the spectral density reflects the joint density of states of electrons and holes. For bands that are mirror "
        r"images about the Fermi level, \(\varepsilon_c = -\varepsilon_v\), the joint density of states at "
        r"\(\hbar\omega\) is half the density of states at \(\hbar\omega/2\). This gives the working rule of this report:</p>"
        r"\[I_\mathrm{ERS}(\omega) \propto D\!\left(E_F \pm \frac{\hbar\omega}{2}\right).\]"
        r"<p>A Raman shift \(\omega\) therefore reads the DOS at an energy \(\hbar\omega/2\) from the Fermi level, and "
        f"{raman_min:g}–{raman_max:g} {CM1} ({raman_min * EV_PER_INVERSE_CM * 1e3:.1f}–"
        f"{raman_max * EV_PER_INVERSE_CM * 1e3:.0f} meV) corresponds to {DE} = {e_lo:.1f}–{e_hi:.0f} meV. "
        "The same reading underlies the layer-number assignment of Pálinkás et al. (2024), where the ERS peaks of "
        r"rhombohedral films sit at the separation of the \(-n\) and \(+n\) DOS peaks.</p>")

    s.append("<h2>Model</h2>")
    s.append(
        "<p>Both crystals are described by the Slonczewski–Weiss–McClure tight-binding model of this repository "
        "(<code>multilayer_graphene.py</code>, parameters of McEllistrim et al. 2023) extended to an infinite stack "
        r"(<code>bulk_graphite.py</code>): every hop from layer \(n\) to layer \(n + m\) carries the phase "
        r"\(e^{i m k_z d}\), "
        rf"\(d = {info['d_layer']}\) Å. The two polytypes use the same couplings and have the same atomic density, "
        f"{_sci(info['atoms_per_cm3'], 3)} {CM3}, so their DOS can be compared per atom or per volume.</p>")
    s.append('<div class="words">' + html_table(
        ["coupling", "value (eV)", "role"],
        [(r"\(\gamma_0\)", f"{par['gamma0']:.4f}", r"intralayer nearest neighbour (\(v = 1.0\times10^{6}\) m/s)"),
         (r"\(\gamma_1\)", f"{par['gamma1']:.3f}", "vertical dimer coupling"),
         (r"\(\gamma_2\)", _signed(f"{par['gamma2']:.3f}"), "next-nearest layer, through a hexagon centre"),
         (r"\(\gamma_3\)", f"{par['gamma3']:.4f}", "skew, non-dimer sites (trigonal warping)"),
         (r"\(\gamma_4\)", f"{par['gamma4']:.4f}", "skew, dimer to non-dimer (electron–hole asymmetry)"),
         (r"\(\gamma_5\)", f"{par['gamma5']:.3f}", "next-nearest layer, along a dimer chain"),
         (r"\(\Delta'\)", f"{par['Delta_prime']:.3f}", r"on-site energy per \(\gamma_1\) bond")],
        "Couplings, identical for the two polytypes") + "</div>")
    s.append(
        "<ul>"
        "<li><b>Hexagonal graphite.</b> Two layers per cell. On the K–H line the model gives the SWMcC energies"
        r"\[E_{1,2} = 2\Delta' - \gamma_5 + \tfrac{1}{2}\gamma_5\Gamma^2 \pm \gamma_1\Gamma, \qquad"
        r" E_3 = -\gamma_2 + \tfrac{1}{2}\gamma_2\Gamma^2, \qquad \Gamma = 2\cos(k_z d),\]"
        r"that is the standard model with \(\Delta = 2\Delta' + \gamma_2 - \gamma_5 ="
        rf" {swmcc_delta * 1e3:.0f}\) meV. "
        r"At each \(k_z\) the bands are those of a bilayer with coupling \(\gamma_1\Gamma\), which leaves a finite DOS "
        "at the Fermi level.</li>"
        r"<li><b>Rhombohedral graphite.</b> One layer per primitive cell, every atom has one \(\gamma_1\) bond. "
        r"The bands are Dirac cones whose apex moves on a spiral of radius \(\gamma_1/\hbar v\) around the K–H line "
        "(McClure 1969). To first order in the small couplings the apex energy is"
        r"\[E_D(k_z) \simeq \Delta' + \frac{2\gamma_1\gamma_4}{\gamma_0}"
        r" + \frac{2\gamma_4}{\gamma_0}\left(\frac{\gamma_1\gamma_3}{\gamma_0} + \frac{\gamma_2}{2}\right)\cos(3k_z d),\]"
        f"a mean of {dirac * 1e3:.1f} meV and a total width of {overlap * 1e3:.1f} meV, which is the band overlap of "
        "the semimetal.</li>"
        "</ul>")

    s.append("<h2>Density of states of the two polytypes</h2>")
    s.append(f"<figure>{img_tag(figures['comparison'], 'DOS of hexagonal and rhombohedral bulk graphite versus energy')}"
             "<figcaption>Figure 1. DOS of hexagonal (blue) and rhombohedral (orange) bulk graphite, energies measured "
             "from the Fermi level of each crystal. Inset: the 100 meV around the Fermi level. Spin and valley "
             "degeneracy included; the right axis gives the same curves per volume.</figcaption></figure>")
    s.append(html_table(
        ["", "hexagonal (AB)", "rhombohedral (ABC)"],
        [("Fermi level on the model's energy scale (meV)", _signed(f"{fermi['ab'][0] * 1e3:+.1f}"),
          _signed(f"{fermi['abc'][0] * 1e3:+.1f}")),
         (rf"electrons = holes at \(T = 0\) ({CM3})", _sci(fermi["ab"][1]), _sci(fermi["abc"][1])),
         (f"DOS at the Fermi level (states {PER_EV_ATOM})", _sci(fermi["ab"][2]), _sci(fermi["abc"][2]))],
        "Fermi-level quantities from charge neutrality"))
    s.append(
        "<ul>"
        "<li>Hexagonal graphite has a finite DOS at the Fermi level and a slow, nearly linear rise on both sides. "
        f"The structure within about 25 meV of {EF} comes from the low-energy bands along K–H, whose energy "
        r"varies by \(2|\gamma_2|\) between K and H and which \(\gamma_3\) warps trigonally; the kinks near "
        r"\(-0.69\) and \(+0.87\) eV are the edges of the split bands at K (\(\pm 2\gamma_1\)).</li>"
        "<li>Rhombohedral graphite has the linear DOS of graphene, per layer, and it vanishes at the Fermi level to "
        f"within the {overlap * 1e3:.1f} meV band overlap.</li>"
        "<li>The hole side of hexagonal graphite carries 25–40 % more states than the electron side "
        r"(\(\gamma_4\) and \(\Delta'\)), while "
        r"rhombohedral graphite is symmetric to 0.2 %. The two curves therefore cross at \(+0.21\) eV for electrons "
        "and do not cross for holes within 1 eV.</li>"
        "</ul>")

    s.append("<h2>Energies accessible to the Raman spectrometer</h2>")
    s.append(f"<figure>{img_tag(figures['window'], 'DOS with the energy range of the Raman spectrometer shaded')}"
             f"<figcaption>Figure 2. The same DOS with the energies reached by Raman shifts of {raman_min:g}–{raman_max:g} "
             f"{CM1} shaded: {DE} = {e_lo:.0f}–{e_hi:.0f} meV on both sides of the Fermi level. "
             f"The top axis gives the same energies in {CM1}. A Raman shift is the distance on it between a hole and "
             "an electron energy; the arrow marks the largest one.</figcaption></figure>")
    s.append(html_table(
        [f"Raman shift ({CM1})", f"{DE} (meV)", "hexagonal", "rhombohedral", "rhombohedral / hexagonal"],
        [(f"{w:.0f}", f"{e:.1f}", f"{100 * h:.3f}", f"{100 * r:.3f}", f"{q:.2f}") for w, e, h, r, q in rows],
        r"DOS at \(E_F \mp \hbar\omega/2\), averaged over the two signs, in \(10^{-2}\) states " + PER_EV_ATOM))
    s.append(
        "<ul>"
        f"<li>Across the window the hexagonal DOS grows by a factor of {rows[-1][2] / rows[0][2]:.1f}, the rhombohedral "
        f"DOS by a factor of {rows[-1][3] / rows[0][3]:.0f}.</li>"
        f"<li>Below about 1000 {CM1} the rhombohedral crystal has less than half the DOS of the hexagonal "
        f"one; the ratio is {rows[0][4]:.2f} at {raman_min:g} {CM1}.</li>"
        f"<li>The curves meet at {cross:.0f} {CM1}; above it the rhombohedral DOS is the larger one, by "
        f"{100 * (rows[-1][4] - 1):.0f} % at {raman_max:g} {CM1}.</li>"
        "</ul>")

    s.append("<h2>What the estimate leaves out</h2>")
    s.append(
        "<ul>"
        "<li><b>Thermal factors.</b> For mirror bands the occupation factor of Eq. (1) is "
        r"\(\tanh(\hbar\omega/4k_B T)\), and the Stokes prefactor is \(\left[1 - e^{-\hbar\omega/k_B T}\right]^{-1}\). "
        r"Their product rises from \(1/4\) at \(\omega \to 0\) to 1 above roughly \(\hbar\omega = 8k_B T\) "
        f"(1700 {CM1} at 300 K). It multiplies both polytypes alike, so the ratios above are unaffected.</li>"
        f"<li><b>Spectral shape at large shifts.</b> Ponosov et al. measure a broad maximum near 3000 {CM1} in "
        "graphite and in graphene, where the joint density of states keeps rising, and discuss a phonon-assisted channel "
        "as a possible reason. The DOS is a safer guide to the ratio of the two polytypes than to the absolute line shape.</li>"
        r"<li><b>Mirror symmetry.</b> The rule \(D(E_F \pm \hbar\omega/2)\) assumes "
        r"\(\varepsilon_c = -\varepsilon_v\). The table "
        "averages the electron and hole sides, which differ by 25–40 % in hexagonal graphite inside the window.</li>"
        "<li><b>Fermi surface of hexagonal graphite.</b> Electron and hole pockets block the excitations below a few "
        f"tens of meV at low temperature; at room temperature Ponosov et al. find no gap down to 50 {CM1}.</li>"
        "<li><b>Surfaces and stacking faults.</b> Figures 1 and 2 are the bulk DOS. Finite rhombohedral films add the "
        "surface flat band and the subband edges that produce the ERS peaks of Pálinkás et al. (2024)"
        + (", shown for thick slabs below" if slabs is not None else "")
        + "; stacking faults add their own features.</li>"
        r"<li><b>Parameters.</b> McClure (1969), with \(|\gamma_4| \approx 0.25\)–\(0.28\) eV and further "
        "next-nearest-plane couplings, "
        "estimated a band overlap of 14–26 meV for rhombohedral graphite and a Fermi-level DOS of 0.07 of the hexagonal "
        rf"value. With the present couplings (\(\gamma_4 = {par['gamma4']:.3f}\) eV) the overlap is "
        f"{overlap * 1e3:.1f} meV and the "
        f"ratio {fermi['abc'][2] / fermi['ab'][2]:.3f}. The difference is confined to {DE} of the order of "
        f"the overlap, below the {e_lo:.0f} meV where the window starts.</li>"
        "</ul>")

    s.append("<h2>Numerical details and checks</h2>")
    s.append(
        "<ul>"
        rf"<li>\(k_z\) is sampled at {info['n_kz']} values over half of its period. For each value the in-plane "
        f"integral uses linear interpolation on triangles of a square mesh around K (spacing {info['dk']:g} "
        f"{PER_ANGSTROM}). Cells in "
        "which a band comes within 0.12 eV of the Fermi level are subdivided "
        f"{info['ab']['refine']} times (hexagonal) or {info['abc']['refine']} times (rhombohedral). Energy bins are "
        f"{info['bin_width'] * 1e3:g} meV wide, with no further smoothing." + ("" if not info["gaussian_sigma"] else
        f" A Gaussian of {info['gaussian_sigma'] * 1e3:g} meV was applied.") + "</li>"
        r"<li>With these default settings, halving the mesh spacing, doubling the number of \(k_z\) values or "
        "doubling the subdivision changes the hexagonal DOS by at most 0.1 % at all energies. The rhombohedral DOS "
        r"changes by less than 0.1 % for \(|E - E_F| > 3\) meV and by 0.7 % inside.</li>"
        r"<li>The Fermi level follows from charge neutrality at \(T = 0\). The carrier density of hexagonal graphite, "
        f"{_sci(fermi['ab'][1])} {CM3} of each sign, compares with the "
        r"\(25\times10^{-6}\) carriers per atom (\(2.8\times10^{18}\) "
        f"{CM3}) that McClure (1969) quotes for hexagonal graphite.</li>"
        "<li>Tests (<code>tests/test_bulk_graphite.py</code>): the K–H energies of hexagonal graphite equal the SWMcC "
        "expressions; bulk Bloch states solve the finite-stack Hamiltonian of <code>multilayer_graphene.py</code> in its "
        r"interior to \(10^{-12}\) eV; with \(\gamma_0\) and \(\gamma_1\) only, the rhombohedral DOS equals that of "
        r"graphene and the hexagonal DOS equals the \(k_z\) average of bilayers.</li>"
        "</ul>")
    s.append("<h3>Reproduce</h3>")
    s.append("<pre>python -m pytest tests -q\n"
             f"python graphite_ers_background.py --raman-min {raman_min:g} --raman-max {raman_max:g}"
             + (f" --slabs {' '.join(str(n) for n in slab_info['layers'])}" if slabs is not None else "") + "\n"
             "python graphite_ers_background.py --from-csv examples/graphite_ers_dos.csv --raman-min 50   # replot only</pre>")

    if slabs is not None:
        s.extend(_slab_section(slabs, slab_info, table, info, raman_min, raman_max, figures["slabs"]))
    if spectra:
        s.extend(_ers_section(spectra, pairs, raman_min, raman_max, linewidth, figures["ers"]))
    if pairs is not None:
        layers = slab_info["layers"] if slabs is not None else pair_info["layers"]
        s.extend(_pair_section(pairs, pair_info, table, raman_min, raman_max, linewidth, layers, figures["pairs"]))

    s.append("<h2>References</h2>")
    s.append(
        "<ul>"
        "<li>O. Kashuba and V. I. Fal'ko, Signature of electronic excitations in the Raman spectrum of graphene, "
        "Phys. Rev. B 80, 241404 (2009).</li>"
        '<li>Yu. S. Ponosov, A. V. Ushakov and S. V. Streltsov, Electronic Raman scattering in graphite and single-layer '
        'and few-layer graphene, Phys. Rev. B 91, 195435 (2015), <a href="https://doi.org/10.1103/PhysRevB.91.195435">'
        "doi:10.1103/PhysRevB.91.195435</a>.</li>"
        "<li>A. García-Ruiz, S. Slizovskiy, M. Mucha-Kruczyński and V. I. Fal'ko, Spectroscopic signatures of electronic "
        'excitations in Raman scattering in thin films of rhombohedral graphite, Nano Lett. 19, 6152 (2019), '
        '<a href="https://doi.org/10.1021/acs.nanolett.9b02196">doi:10.1021/acs.nanolett.9b02196</a>.</li>'
        "<li>A. Pálinkás et al., Identification of graphite with perfect rhombohedral stacking by electronic Raman "
        'scattering, Carbon 230, 119608 (2024), <a href="https://doi.org/10.1016/j.carbon.2024.119608">'
        "doi:10.1016/j.carbon.2024.119608</a>.</li>"
        "<li>J. W. McClure, Electron energy band structure and electronic properties of rhombohedral graphite, "
        "Carbon 7, 425 (1969).</li>"
        "<li>A. McEllistrim, A. Garcia-Ruiz, Z. A. H. Goodwin and V. I. Fal'ko, Spectroscopic signatures of tetralayer "
        "graphene polytypes, Phys. Rev. B 107, 155147 (2023), arXiv:2302.07374.</li>"
        "</ul>")
    s.append("</div></body></html>" if standalone else "</div>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(s))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raman-min", type=float, default=80.0, help="lowest Raman shift of the instrument (1/cm)")
    ap.add_argument("--raman-max", type=float, default=6200.0, help="highest Raman shift of the instrument (1/cm)")
    ap.add_argument("--prefix", default="examples/graphite_ers_dos", help="path prefix of the output files")
    ap.add_argument("--from-csv", default=None, help="replot a table written by an earlier run")
    ap.add_argument("--n-kz", type=int, default=240, help="k_z values in half of the k_z period")
    ap.add_argument("--dk", type=float, default=2.0e-3, help="coarse in-plane mesh spacing (1/A)")
    ap.add_argument("--refine", type=int, default=8,
                    help="mesh subdivision near the Fermi level (hexagonal; four times this for rhombohedral)")
    ap.add_argument("--sigma", type=float, default=0.0, help="Gaussian smoothing of the DOS (eV)")
    ap.add_argument("--slabs", type=int, nargs="*", default=[20, 50, 100],
                    help="layer numbers of the finite slabs (none: skip the slabs); a slab table "
                         "next to --from-csv is reused")
    ap.add_argument("--slab-dk", type=float, default=1.0e-3, help="in-plane mesh spacing for the slabs (1/A)")
    ap.add_argument("--linewidth", type=float, default=500.0,
                    help="ERS line width (1/cm, full width at half maximum on the Raman-shift axis)")
    ap.add_argument("--ers-spectra", nargs="*", default=None, metavar="FILE",
                    help="calculated ERS spectra to compare with (default: examples/ers_calculations/*.dat)")
    ap.add_argument("--page-only", default=None, metavar="PATH",
                    help="also write the report without its document skeleton, for a publisher that adds its own")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")

    if args.from_csv:
        table, info = load_csv(args.from_csv)
    else:
        mlg.use_paper_parameters()
        table, info = compute(n_kz=args.n_kz, dk=args.dk, refine=args.refine, sigma=args.sigma)
        save_csv(args.prefix + ".csv", table, info)
        print(f"table written to {args.prefix}.csv")
    slabs = slab_info = None
    slab_csv = (args.from_csv[:-len(".csv")] if args.from_csv else args.prefix) + "_slabs.csv"
    if args.from_csv and os.path.exists(slab_csv):
        slabs, slab_info = load_slabs_csv(slab_csv)
    elif args.slabs:
        mlg.set_parameters(**info["parameters"])
        slabs, slab_info = compute_slabs(args.slabs, dk=args.slab_dk)
        save_slabs_csv(slab_csv, slabs, slab_info)
        print(f"slab table written to {slab_csv}")
    files = args.ers_spectra if args.ers_spectra is not None else sorted(glob.glob(os.path.join(ERS_DIR, "*.dat")))
    spectra = [load_ers_spectrum(f) for f in files]
    for spectrum in (sp for sp in spectra if sp["key"] is None):
        print(f"skipping {spectrum['path']}: {spectrum['name']} is neither rhombohedral nor hexagonal")
    spectra = [sp for sp in spectra if sp["key"]]
    pairs = pair_info = None
    pair_layers = sorted(set(slab_info["layers"] if slabs is not None else []) | {sp["layers"] for sp in spectra})
    pair_csv = slab_csv[:-len("_slabs.csv")] + "_pairs.csv"
    if args.from_csv and os.path.exists(pair_csv):
        pairs, pair_info = load_pairs_csv(pair_csv)
        if not set(pair_layers) <= set(pair_info["layers"]):
            pairs = None
    if pairs is None and pair_layers:
        mlg.set_parameters(**info["parameters"])
        pairs, pair_info = compute_pair_spectra(pair_layers, dk=args.slab_dk)
        save_pairs_csv(pair_csv, pairs, pair_info)
        print(f"mirror-pair table written to {pair_csv}")
    if pairs is not None and "ab_bulk" not in pairs:
        mlg.set_parameters(**info["parameters"])
        pairs, pair_info = compute_bulk_pair_spectra(pairs, pair_info, info, n_kz=info["n_kz"], dk=info["dk"],
                                                     refine=info["ab"]["refine"])
        save_pairs_csv(pair_csv, pairs, pair_info)
        print(f"bulk mirror-pair spectra added to {pair_csv}")

    report(table, info, args.raman_min, args.raman_max)
    figures = {"comparison": args.prefix + "_comparison.png", "window": args.prefix + "_raman_window.png"}
    plot_comparison(table, info, figures["comparison"])
    plot_raman_window(table, info, figures["window"], args.raman_min, args.raman_max)
    if slabs is not None:
        figures["slabs"] = args.prefix + "_slabs.png"
        plot_slabs(slabs, slab_info, table, info, figures["slabs"], args.raman_min, args.raman_max)
    if spectra:
        figures["ers"] = args.prefix + "_ers_calculation.png"
        plot_ers_calculation(spectra, pairs, figures["ers"], args.raman_min, args.raman_max, args.linewidth)
    if pairs is not None:
        figures["pairs"] = args.prefix + "_raman_axis.png"
        plot_pair_spectra(pairs, table, figures["pairs"], args.raman_min, args.raman_max, args.linewidth,
                          slab_info["layers"] if slabs is not None else pair_info["layers"])
    print("figures written to " + ", ".join(figures.values()))
    arguments = (table, info, args.raman_min, args.raman_max, figures, slabs, slab_info, spectra, pairs, pair_info,
                 args.linewidth)
    write_report(args.prefix + "_report.html", *arguments)
    print(f"report written to {args.prefix}_report.html")
    if args.page_only:
        write_report(args.page_only, *arguments, standalone=False)
        print(f"page without skeleton written to {args.page_only}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
