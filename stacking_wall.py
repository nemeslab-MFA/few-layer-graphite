"""
Stacking domain wall in six-layer graphite: (+5) ABCABC to (+4-) ABCABA
=======================================================================

Local density of states (LDOS) across a single-layer stacking shift of the
topmost graphene layer, the domain wall between hexalayer rhombohedral graphite
(Hägg sign sequence (+5), letters ABCABC) and the stacking-fault structure
((+4-), ABCABA) measured in

    K. Kandrai et al., "One-way electron mirrors from single-layer stacking
    shifts in rhombohedral graphite", arXiv:2610.20905 (2026).

Model
-----
The continuum (k.p) Slonczewski-Weiss-McClure Hamiltonian of
``multilayer_graphene`` is linear in the momentum p measured from the valley
centre, H_S(p) = C_S + p_x X_S + p_y Y_S, for either stacking S.  Across the
wall the Hamiltonian is interpolated linearly between the two stackings,

    H(x, p) = (1 - s(x)) H_RG(p) + s(x) H_SF(p),    0 <= s(x) <= 1,

so that every matrix element that differs between the two structures (the
dimer coupling gamma1 of the top layer, which moves from A5-B6 to B5-A6, the
skew couplings gamma3, gamma4, the next-nearest-layer couplings gamma2,
gamma5 and the on-site energies Delta' of layers 5 and 6) follows the same
profile s(x).  The profile is a smooth step, s = [1 - tanh(x / l)] / 2, with
the stacking fault on the left (x < 0) and the rhombohedral stack on the right.
The wall length l is a parameter; ``fit_wall_profile`` extracts it from a
measured dI/dV line spectrum.

The wall is parallel to the armchair direction (its normal x is the zigzag
direction, the p_x axis of the paper); k_y is conserved.  For every k_y the
one-dimensional problem is solved in a plane-wave basis exp(i q x), q = 2 pi
n / L, on a periodic cell of length L; the x-dependent terms become
convolutions with the Fourier coefficients of s(x), and terms of the form
s(x) p_x are symmetrised, (s p_x + p_x s) / 2.  Periodicity requires a second
wall, which is placed at x = +-L/2, far from the one studied (default L =
480 nm).  The LDOS of every layer is then

    LDOS_n(E, x) = 4 sum_{k_y} (dk_y / 2 pi) sum_s |psi_s(x)|^2_n g(E - E_s),

with g a normalised Gaussian of width ``sigma`` and the factor 4 for spin and
the two valleys (the K- Hamiltonian is the complex conjugate of the K+ one and
gives the same LDOS).  Units: states per eV per A^2 (``*1e16`` for cm^-2).

Experimental profile
--------------------
``load_line_spectrum`` reads an RHK ``.sm4`` line spectrum with ``rhkpy`` and
normalises it as in the notebooks of the paper (lock-in signal divided by the
mean of |I| at the two ends of the bias ramp).  ``mixing_fraction`` writes
every spectrum of the line as a linear combination of the two end-member
spectra far from the wall, dI/dV(V, x) = (1 - s) S_SF(V) + s S_RG(V), and
``fit_wall_profile`` fits s(x) with the same tanh step as the model.
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import eigh

import multilayer_graphene as mlg

__all__ = [
    "SEQ_RG", "SEQ_SF", "kp_blocks", "wall_profile", "fourier_coefficients",
    "wall_hamiltonian", "ldos_wall", "flat_band_energy",
    "load_line_spectrum", "load_map_section", "mixing_fraction", "fit_wall_profile", "tanh_step",
    "save_ldos", "load_ldos",
]

SEQ_RG = "ABCABC"     # (+5): hexalayer rhombohedral graphite
SEQ_SF = "ABCABA"     # (+4-): top layer in the hexagonal (Bernal) position


# =============================================================================
# k.p blocks and the interpolated Hamiltonian
# =============================================================================

def kp_blocks(seq, params=None, xi=+1):
    """
    Decompose the k.p Hamiltonian of ``multilayer_graphene`` for the explicit
    sequence ``seq`` into H(p) = C + p_x X + p_y Y (2N x 2N each, p in 1/A
    measured from K_xi).

    ``_assemble`` is linear in the structure factor u and its conjugate, with
    u = -(3 a_cc / 2)(xi p_x - i p_y) in the k.p model, so the three matrices
    follow from the assembly at u = 0, 1 and i.
    """
    params = mlg.get_parameters() if params is None else {**mlg.get_parameters(), **params}
    a_cc = params["a"] / np.sqrt(3.0)
    c = -1.5 * a_cc
    H0 = mlg._assemble(0.0, seq, params)
    H1 = mlg._assemble(1.0, seq, params)
    Hi = mlg._assemble(1.0j, seq, params)
    X = c * xi * (H1 - H0)
    Y = -c * (Hi - H0)
    return H0, X, Y


def tanh_step(x, x0, ell, lo=0.0, hi=1.0):
    """Smooth step lo -> hi centred at x0 with length ell: lo + (hi - lo)[1 + tanh((x - x0)/ell)]/2."""
    return lo + (hi - lo) * 0.5 * (1.0 + np.tanh((np.asarray(x, dtype=float) - x0) / ell))


def wall_profile(x, ell, length, strip=None):
    """
    Fraction s(x) of the stacking-fault Hamiltonian, periodic with period
    ``length`` (A).  The stacking fault occupies -strip < x < 0 (s = 1), the
    rhombohedral stack the rest (s = 0); the wall studied is at x = 0, the
    return wall at x = -strip (default strip = length / 2, i.e. x = +-L/2).
    Both walls have the tanh length ``ell`` (A).
    """
    x = np.mod(np.asarray(x, dtype=float) + 0.5 * length, length) - 0.5 * length
    strip = 0.5 * length if strip is None else float(strip)
    s = np.zeros_like(x)
    for m in (-1, 0, 1):
        xs = x + m * length
        s += 0.5 * (np.tanh((xs + strip) / ell) - np.tanh(xs / ell))
    return s


def fourier_coefficients(profile, length, n_max, n_grid=2 ** 14):
    """
    Fourier coefficients s_k = (1/L) int s(x) exp(-2 pi i k x / L) dx for
    k = -n_max .. n_max, from samples of ``profile(x)`` on a uniform grid.
    """
    x = -0.5 * length + length * np.arange(n_grid) / n_grid
    s_hat = np.fft.fft(profile(x)) / n_grid
    # the grid starts at -L/2: undo the phase of the shifted origin
    k = np.arange(-n_max, n_max + 1)
    coeff = s_hat[k % n_grid] * np.exp(1j * np.pi * k)
    return coeff


def _mixed_blocks(params=None, xi=+1, layer_potential=None, seq_rg=SEQ_RG, seq_sf=SEQ_SF):
    """Blocks of H_RG and of the difference D = H_SF - H_RG, both as (C, X, Y)."""
    C_rg, X_rg, Y_rg = kp_blocks(seq_rg, params, xi)
    C_sf, X_sf, Y_sf = kp_blocks(seq_sf, params, xi)
    if C_rg.shape != C_sf.shape:
        raise ValueError("both stackings must have the same number of layers.")
    if layer_potential is not None:
        V = np.repeat(np.asarray(layer_potential, dtype=float), 2)
        if V.size != C_rg.shape[0]:
            raise ValueError("layer_potential needs one value per layer.")
        C_rg = C_rg + np.diag(V)
        C_sf = C_sf + np.diag(V)
    return (C_rg, X_rg, Y_rg), (C_sf - C_rg, X_sf - X_rg, Y_sf - Y_rg)


def wall_hamiltonian(ky, s_hat, q, blocks_rg, blocks_d, v_hat=None):
    """
    Plane-wave Hamiltonian at conserved k_y (1/A): basis index i = n_pw * 2N +
    orbital, plane waves q[n] = 2 pi n / L (1/A), Fourier coefficients
    s_hat[k + n_max] of the profile.  ``v_hat`` (N_layers, 2 n_max + 1), the
    Fourier coefficients of position-dependent layer potentials V_l(x) (eV),
    adds the on-site term of every layer.
    """
    C, X, Y = blocks_rg
    DC, DX, DY = blocks_d
    n_pw = q.size
    n_max = (s_hat.size - 1) // 2
    idx = np.arange(n_pw)
    diff = (idx[:, None] - idx[None, :]) + n_max
    S = s_hat[diff]                                                  # s_{n-m}
    SQ = S * (0.5 * (q[:, None] + q[None, :]))                       # symmetrised s p_x
    H = (np.kron(np.eye(n_pw), C + ky * Y)
         + np.kron(np.diag(q), X)
         + np.kron(S, DC + ky * DY)
         + np.kron(SQ, DX))
    if v_hat is not None:
        n_orb = C.shape[0]
        for l in range(v_hat.shape[0]):
            P = np.zeros((n_orb, n_orb))
            P[2 * l, 2 * l] = P[2 * l + 1, 2 * l + 1] = 1.0
            H += np.kron(v_hat[l][diff], P)
    return H


def _ldos_worker(args):
    (ky_list, s_hat, q, blocks_rg, blocks_d, x_out, energy, sigma, length,
     n_layers, dky, v_hat) = args
    n_E, n_x = energy.size, x_out.size
    out = np.zeros((n_layers, n_E, n_x))
    phase = np.exp(1j * np.outer(q, x_out)) / np.sqrt(length)       # (n_pw, n_x)
    e_lo, e_hi = energy.min() - 6.0 * sigma, energy.max() + 6.0 * sigma
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for ky in ky_list:
        H = wall_hamiltonian(ky, s_hat, q, blocks_rg, blocks_d, v_hat)
        w, v = eigh(H, subset_by_value=(e_lo, e_hi), driver="evr")
        if w.size == 0:
            continue
        v = v.reshape(q.size, 2 * n_layers, w.size)                  # (n_pw, orb, state)
        psi = np.einsum("nx,nos->osx", phase, v)                     # (orb, state, n_x)
        weight = (np.abs(psi) ** 2).reshape(n_layers, 2, w.size, n_x).sum(axis=1)
        g = norm * np.exp(-0.5 * ((energy[:, None] - w[None, :]) / sigma) ** 2)   # (n_E, state)
        out += dky * np.einsum("es,lsx->lex", g, weight)
    return out


def ldos_wall(wall_length=40.0, length=4800.0, strip=None, p_cut=0.13,
              ky_max=0.11, dky=0.001, energy=None, sigma=0.005,
              x_out=None, params=None, layer_potential=None, n_proc=None,
              xi=+1, profile=None, seq_rg=SEQ_RG, seq_sf=SEQ_SF,
              layer_potential_profile=None):
    """
    Layer-resolved LDOS(E, x) across the wall.

    wall_length : tanh length l of the wall (A); 10-90 % width = 2.197 l
    length      : period L of the plane-wave cell (A); the return wall sits at +-L/2
    strip       : width of the stacking-fault strip (A), default L/2
    p_cut       : plane-wave cut-off |q| <= p_cut (1/A)
    ky_max, dky : k_y mesh (1/A), symmetric about 0
    energy      : energies (eV), default -0.5 .. 0.5 eV in 2.5 meV steps
    sigma       : Gaussian broadening (eV)
    x_out       : positions (A) where the LDOS is evaluated, default -1200 .. 1200 A
    params      : overrides of the SWMcC parameters (see multilayer_graphene)
    layer_potential : on-site energy of each of the six layers (eV), optional, in the k.p
                  layer order: index 0 is the bottom (substrate) layer, index N-1 the top
                  layer that faces the tip (the bandbending engine numbers them the other
                  way: reverse its arrays with [::-1] before passing them here)
    n_proc      : worker processes (default: all CPUs), one BLAS thread each
    profile     : callable s(x) in place of ``wall_profile``
    seq_rg, seq_sf : the two stackings (same number of layers)
    layer_potential_profile : callable x (A, periodic cell) -> (N_layers, len(x))
        on-site potential energies of the layers (eV), e.g. the far-field
        potentials of a Poisson solver along the wall; must be periodic on the cell;
        layers in the k.p order (bottom first, as ``layer_potential``)

    Returns a dict with 'energy' (eV), 'x' (A), 's' (profile at x), 'ldos'
    of shape (N_layers, n_E, n_x) in states / (eV A^2), spin and valley
    included (bottom layer first: the top-layer LDOS is ldos[-1]), and the
    parameters used.
    """
    energy = np.linspace(-0.5, 0.5, 401) if energy is None else np.asarray(energy, dtype=float)
    x_out = np.arange(-1200.0, 1200.0 + 1e-9, 5.0) if x_out is None else np.asarray(x_out, dtype=float)
    n_max = int(np.floor(p_cut * length / (2.0 * np.pi)))
    q = 2.0 * np.pi * np.arange(-n_max, n_max + 1) / length
    if profile is None:
        profile = lambda x: wall_profile(x, wall_length, length, strip)   # noqa: E731
    s_hat = fourier_coefficients(profile, length, 2 * n_max)
    blocks_rg, blocks_d = _mixed_blocks(params, xi, layer_potential, seq_rg, seq_sf)
    n_layers = blocks_rg[0].shape[0] // 2
    v_hat = None
    if layer_potential_profile is not None:
        v_hat = np.array([fourier_coefficients(lambda x, l=l: np.asarray(layer_potential_profile(x))[l],
                                               length, 2 * n_max) for l in range(n_layers)])

    n_ky = int(round(2.0 * ky_max / dky)) + 1
    ky = np.linspace(-ky_max, ky_max, n_ky)
    dky_eff = ky[1] - ky[0] if n_ky > 1 else 1.0
    weight = dky_eff / (2.0 * np.pi) * 4.0                              # k_y measure, spin, valleys

    n_proc = os.cpu_count() if n_proc is None else int(n_proc)
    n_proc = max(1, min(n_proc, n_ky))
    chunks = [ky[i::n_proc] for i in range(n_proc)]
    args = [(c, s_hat, q, blocks_rg, blocks_d, x_out, energy, sigma, length, n_layers, weight, v_hat)
            for c in chunks]
    if n_proc == 1:
        parts = [_ldos_worker(args[0])]
    else:
        parts = _map_in_processes(_ldos_worker, args, n_proc)
    ldos = np.sum(parts, axis=0)
    return {
        "energy": energy, "x": x_out, "s": profile(x_out), "ldos": ldos,
        "wall_length": wall_length, "length": length, "strip": strip,
        "p_cut": p_cut, "ky_max": ky_max, "dky": dky_eff, "sigma": sigma,
        "n_pw": q.size, "layer_potential": layer_potential,
        "seq_rg": seq_rg, "seq_sf": seq_sf,
        "layer_potential_profile": layer_potential_profile is not None,
        "params": mlg.get_parameters() if params is None else {**mlg.get_parameters(), **params},
    }


BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS")


def _map_in_processes(function, jobs, n_proc):
    """
    ``map`` over freshly spawned processes with one BLAS thread each (a forked
    worker inherits an initialised multi-threaded BLAS and can deadlock).
    """
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(BLAS_THREAD_VARIABLES, "1"))
    try:
        with ProcessPoolExecutor(max_workers=n_proc,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(function, jobs))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def flat_band_energy(energy, ldos_top, window=(-0.12, 0.12)):
    """Energy of the maximum of a top-layer spectrum inside ``window`` (eV)."""
    sel = (energy >= window[0]) & (energy <= window[1])
    return float(energy[sel][np.argmax(ldos_top[sel])])


def save_ldos(path, result):
    """Store the result of ``ldos_wall`` as a compressed .npz (parameters as JSON)."""
    meta = {k: v for k, v in result.items() if k not in ("energy", "x", "s", "ldos")}
    np.savez_compressed(path, energy=result["energy"], x=result["x"], s=result["s"],
                        ldos=np.asarray(result["ldos"], dtype=np.float32),
                        meta=json.dumps(meta, default=_json_default))


def load_ldos(path):
    with np.load(path) as f:
        out = {k: f[k] for k in ("energy", "x", "s", "ldos")}
        out.update(json.loads(str(f["meta"])))
    return out


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not serialisable: {type(obj)}")


# =============================================================================
# Experimental line spectra
# =============================================================================

def load_line_spectrum(path, smooth=(5, 5)):
    """
    Normalised dI/dV of an RHK line spectrum (``rhkpy`` required).

    The lock-in signal, averaged over repetitions and sweep directions, is
    divided by the mean of |I| at the first and last bias of the ramp (the
    normalisation of the paper's notebooks) and smoothed with a running mean
    over ``smooth`` = (bias points, positions).  Returns a dict with 'bias'
    (V, ascending), 'dist' (nm), 'didv' (bias x dist), 'didv_raw' and the
    file attributes.
    """
    import rhkpy
    data = rhkpy.rhkdata(path)
    mean = data.spectra.mean(dim=["repetitions", "biasscandir"])
    i_first = np.abs(mean.current[0, :].data)
    i_last = np.abs(mean.current[-1, :].data)
    lia = (2.0 * mean.lia / (i_first + i_last)).sortby("bias")
    raw = lia.transpose("bias", "dist").values
    if smooth is not None:
        lia = lia.rolling(bias=smooth[0], dist=smooth[1], center=True, min_periods=1).mean()
    out = {
        "bias": lia.bias.values.astype(float),
        "dist": lia.dist.values.astype(float),
        "didv": lia.transpose("bias", "dist").values,
        "didv_raw": raw,
        "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in data.spectra.attrs.items()},
    }
    return out


def load_map_section(path, start_point, end_point, smooth=(5, 3, 3)):
    """
    Normalised dI/dV along a straight section of an RHK grid-spectroscopy map
    (``rhkpy`` required), as in the notebooks of the paper: repetitions and
    sweep directions averaged, lock-in signal divided by the current at the
    first bias, running means over ``smooth`` = (bias points, x pixels,
    y pixels), then ``rhkpy.mapsection`` between ``start_point`` and
    ``end_point`` (map coordinates, nm).  Returns the same dict as
    ``load_line_spectrum`` ('dist' measured from ``start_point``).
    """
    import rhkpy
    data = rhkpy.rhkdata(path)
    mean = data.mrep().msw()
    norm = -mean.spectra.lia / mean.spectra.current[0, :, :]
    if smooth is not None:
        norm = (norm.rolling(bias=smooth[0], center=True, min_periods=1).mean()
                    .rolling(specpos_x=smooth[1], specpos_y=smooth[2], center=True, min_periods=1).mean())
    section = rhkpy.mapsection(norm.to_dataset(name="didv"), start_point=start_point,
                               end_point=end_point).didv.sortby("bias")
    return {
        "bias": section.bias.values.astype(float),
        "dist": section.dist.values.astype(float),
        "didv": section.transpose("bias", "dist").values,
        "didv_raw": section.transpose("bias", "dist").values,
        "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in data.spectra.attrs.items()},
    }


def mixing_fraction(bias, dist, didv, x_sf, x_rg, bias_window=(-0.3, 0.3)):
    """
    Fraction s(x) of the rhombohedral end-member spectrum in every spectrum of
    the line: least-squares solution of
    didv(V, x) = (1 - s) S_SF(V) + s S_RG(V), with S_SF the mean spectrum at
    dist < x_sf and S_RG the mean at dist > x_rg (nm), over ``bias_window``.
    Returns (s, S_SF, S_RG).
    """
    bias = np.asarray(bias, dtype=float)
    dist = np.asarray(dist, dtype=float)
    win = (bias >= bias_window[0]) & (bias <= bias_window[1])
    S_sf = np.nanmean(didv[:, dist < x_sf], axis=1)
    S_rg = np.nanmean(didv[:, dist > x_rg], axis=1)
    D = (S_rg - S_sf)[win]
    s = np.array([np.nansum((didv[win, i] - S_sf[win]) * D) / np.sum(D * D)
                  for i in range(dist.size)])
    return s, S_sf, S_rg


def fit_wall_profile(dist, s, x0_guess=None, ell_guess=5.0):
    """
    Fit s(x) with tanh_step; returns a dict with x0, ell (nm), their standard
    errors, the end values lo, hi and the 10-90 % width 2 atanh(0.8) ell.
    """
    from scipy.optimize import curve_fit
    dist = np.asarray(dist, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    x0_guess = 0.5 * (dist.min() + dist.max()) if x0_guess is None else x0_guess
    p0 = [x0_guess, ell_guess, float(np.nanmin(s)), float(np.nanmax(s))]
    popt, pcov = curve_fit(tanh_step, dist[ok], s[ok], p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return {
        "x0": float(popt[0]), "ell": float(popt[1]), "lo": float(popt[2]), "hi": float(popt[3]),
        "x0_err": float(perr[0]), "ell_err": float(perr[1]),
        "width_10_90": float(2.0 * np.arctanh(0.8) * popt[1]),
    }
