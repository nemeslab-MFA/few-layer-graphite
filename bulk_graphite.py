"""
Bulk graphite: SWMcC bands and density of states of infinite stacks
====================================================================

Bloch Hamiltonian and density of states (DOS) of three-dimensional graphite
with a periodic stacking sequence: hexagonal (Bernal, AB), rhombohedral (ABC)
or any other periodic polytype.

The model is the infinite-stack limit of ``multilayer_graphene``.  The layer
blocks are read from its assembly routine, so the couplings gamma0 ... gamma5,
Delta_prime and all conventions are those of that module, and a hop from
layer n to layer n + m carries the phase exp(i m k_z d) with d = 3.35 A.

For Bernal stacking this is the Slonczewski-Weiss-McClure model.  On the K-H
line, with Gamma = 2 cos(k_z d),

    E_1,2 = 2 Delta' - gamma5 + gamma5 Gamma^2 / 2 +- gamma1 Gamma
    E_3   = -gamma2 + gamma2 Gamma^2 / 2          (doubly degenerate)

which are the usual SWMcC energies shifted rigidly by -gamma2, with
Delta_SWMcC = 2 Delta' + gamma2 - gamma5 (-5 meV for the default parameters).

For rhombohedral stacking every layer is equivalent, the primitive cell holds
one layer (two atoms) and k_z d runs over 2 pi; Delta' is a rigid shift.  With
gamma0 and gamma1 only, the bands are Dirac cones on a spiral of radius
gamma1 / (hbar v) around the K-H line; gamma2, gamma3 and gamma4 make the
energy of the Dirac point depend on k_z (McClure, Carbon 7, 425 (1969)).

Units: energies in eV, wave vectors in 1/Angstrom.  All DOS values include the
spin degeneracy of 2.
"""

import numpy as np

import multilayer_graphene as mlg

__all__ = [
    "D_LAYER", "bulk_period", "bulk_hamiltonian", "bulk_eigenvalues",
    "atomic_density", "calculate_bulk_density_of_states", "calculate_bulk_pair_spectrum",
]

D_LAYER = 3.35                  # interlayer distance (A)

_PERIODS = {"ab": "AB", "aba": "AB", "bernal": "AB", "hexagonal": "AB",
            "abc": "ABC", "rhombohedral": "ABC"}


def bulk_period(stacking_type):
    """
    Return (seq, p): one period of the stacking sequence as letters A/B/C and
    the number of layers p in the primitive cell.

    ``stacking_type`` may be 'ab'/'aba'/'bernal'/'hexagonal', 'abc'/
    'rhombohedral', or an explicit repeat unit such as 'ABCB'.  p is smaller
    than len(seq) when the sequence of stacking steps repeats inside the unit:
    p = 1 for 'ABC', where every step is a forward step.
    """
    key = str(stacking_type).strip().lower()
    seq = _PERIODS.get(key, key.upper())
    if len(seq) < 2 or any(ch not in "ABC" for ch in seq):
        raise ValueError(f"Unknown bulk stacking '{stacking_type}'. Use 'ab', 'abc' or a "
                         "repeat unit of at least two of the letters A, B, C.")
    closed = seq + seq[0]
    if any(closed[n] == closed[n + 1] for n in range(len(seq))):
        raise ValueError(f"Invalid repeat unit '{seq}': adjacent layers must differ, "
                         "including the last and the first.")
    steps = mlg.stacking_steps(closed)
    P = len(seq)
    for p in range(1, P + 1):
        if P % p == 0 and np.array_equal(steps, np.tile(steps[:p], P // p)):
            break
    return seq, p


def atomic_density(a_lat=None):
    """Number of carbon atoms per A^3 (the same for every stacking)."""
    return 2.0 / (mlg.unit_cell_area(a_lat) * D_LAYER)


def _bulk_blocks(seq, p, params):
    """
    Layer blocks of the infinite stack, taken from a finite stack of three
    periods assembled by ``multilayer_graphene``.

    Every matrix element is linear in the intralayer structure factor u and
    its conjugate.  Returns (D, T1, T2), each of shape (3, 2p, 2p) holding the
    coefficients of (1, u, conj(u)): D is the intralayer part, T1 and T2 couple
    each layer to the next and to the next-nearest layer above it.
    """
    P = len(seq)
    h0, h1, hi = (mlg._assemble(u, seq * 3, params) for u in (0.0, 1.0, 1.0j))
    coef = np.array([h0,
                     0.5 * ((h1 - h0) - 1j * (hi - h0)),
                     0.5 * ((h1 - h0) + 1j * (hi - h0))])
    D, T1, T2 = (np.zeros((3, 2 * p, 2 * p), dtype=complex) for _ in range(3))
    for n in range(p):
        L = P + n                           # a layer of the middle period
        rows = slice(2 * n, 2 * n + 2)
        for target, m in ((D, 0), (T1, 1), (T2, 2)):
            j = (n + m) % p
            target[:, rows, 2 * j:2 * j + 2] += coef[:, 2 * L:2 * L + 2, 2 * (L + m):2 * (L + m) + 2]
    return D, T1, T2


def _hamiltonian(u, phase, blocks):
    """H for structure factor u and phase = k_z d (arrays that broadcast against each other)."""
    D, T1, T2 = blocks
    u = np.asarray(u, dtype=complex)[..., None, None]
    e1 = np.exp(1j * np.asarray(phase, dtype=float))[..., None, None]
    w = (1.0, u, np.conj(u))
    H = sum(w[c] * D[c] for c in range(3))
    T = sum(w[c] * (e1 * T1[c] + e1 ** 2 * T2[c]) for c in range(3))
    return H + T + np.conj(np.swapaxes(T, -1, -2))


def _eigenvalues(H):
    """Sorted eigenvalues; closed form for the 2 x 2 rhombohedral Hamiltonian."""
    if H.shape[-1] == 2:
        mean = 0.5 * (H[..., 0, 0].real + H[..., 1, 1].real)
        half = 0.5 * (H[..., 0, 0].real - H[..., 1, 1].real)
        root = np.sqrt(half ** 2 + np.abs(H[..., 0, 1]) ** 2)
        return np.stack([mean - root, mean + root], axis=-1)
    return np.linalg.eigvalsh(H)


def bulk_hamiltonian(kx, ky, kz, stacking_type="ab", params=None):
    """
    Bloch Hamiltonian of bulk graphite for Cartesian wave vectors (1/A).

    kx, ky, kz may be scalars or arrays that broadcast; the result has shape
    broadcast.shape + (2p, 2p) with p layers in the primitive cell (2 for
    Bernal, 1 for rhombohedral).  Basis (A_1, B_1, ..., A_p, B_p); the spectrum
    is periodic in kz with period 2 pi / (p d).
    """
    params = mlg.get_parameters() if params is None else {**mlg.get_parameters(), **params}
    seq, p = bulk_period(stacking_type)
    kx, ky, kz = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (kx, ky, kz)))
    u = mlg.structure_factor(kx, ky, params["a"])
    return _hamiltonian(u, kz * D_LAYER, _bulk_blocks(seq, p, params))


def bulk_eigenvalues(kx, ky, kz, stacking_type="ab", params=None):
    """Sorted band energies (eV) of bulk graphite at Cartesian k (1/A)."""
    return _eigenvalues(bulk_hamiltonian(kx, ky, kz, stacking_type, params))


# =============================================================================
# Density of states
# =============================================================================

def _add_cells(cumulative, corners, area, edges):
    """
    Add the triangles of square cells to the cumulative state count.

    corners: (4, n_cells, n_bands) energies at the cell corners in cyclic
    order; each cell is split into two triangles of the given area.  The lower
    and the upper half of the bands are accumulated in cumulative[0] and [1].
    """
    n_bands = corners.shape[-1]
    for b in range(n_bands):
        for tri in ([0, 1, 2], [0, 2, 3]):
            cumulative[int(b >= n_bands // 2)] += mlg._cumulative_states_triangles(
                corners[tri, :, b].T, area, edges)


def _add_pairs(cumulative, corners, area, edges, fermi):
    """
    Add the triangles of square cells to the cumulative count of mirror pairs:
    the n-th band below and the n-th band above the middle of the spectrum,
    counted at the energy difference of the two.  A triangle counts with the
    fraction of its corners at which the lower band is filled and the upper
    one empty.  corners: (4, n_cells, n_bands) as in ``_add_cells``.
    """
    half = corners.shape[-1] // 2
    for n in range(half):
        lower, upper = corners[:, :, half - 1 - n], corners[:, :, half + n]
        excitation = upper - lower
        allowed = ((lower < fermi) & (upper > fermi)).astype(float)
        for tri in ([0, 1, 2], [0, 2, 3]):
            cumulative += mlg._cumulative_states_triangles(excitation[tri].T, area * allowed[tri].mean(axis=0), edges)


def _mesh_slices(blocks, p, a_lat, k_max, dk, n_kz, refine, centre, window):
    """
    Band energies of the k_z slices on the in-plane mesh around K.  For each of
    n_kz values of k_z in half of its period yields (E, corners, fine,
    subcells): E of shape (n, n, 2p) on the coarse mesh, corners
    (4, n - 1, n - 1, 2p) the energies at the corners of its cells, fine the
    cells in which a band comes within ``window`` of ``centre``, and subcells
    (4, cells, 2p) the corner energies of their subdivision by ``refine`` (None
    if no cell is subdivided).
    """
    n_side = int(np.ceil(2.0 * k_max / dk)) + 1
    t = np.linspace(-k_max, k_max, n_side)
    h = t[1] - t[0]
    K = mlg.K_point(+1, a_lat)
    gx, gy = np.meshgrid(K[0] + t, K[1] + t, indexing="ij")
    u_coarse = mlg.structure_factor(gx, gy, a_lat)
    sub = np.arange(refine + 1) * (h / refine)
    for phase in (np.arange(n_kz) + 0.5) * np.pi / (p * n_kz):
        E = _eigenvalues(_hamiltonian(u_coarse, phase, blocks))
        corners = np.stack([E[:-1, :-1], E[:-1, 1:], E[1:, 1:], E[1:, :-1]])
        fine = subcells = None
        if refine > 1:
            fine = np.any((corners.min(axis=0) < centre + window) & (corners.max(axis=0) > centre - window), axis=-1)
            if fine.any():
                i, j = np.nonzero(fine)
                fx = gx[i, j][:, None, None] + sub[None, :, None]
                fy = gy[i, j][:, None, None] + sub[None, None, :]
                Ef = _eigenvalues(_hamiltonian(mlg.structure_factor(fx, fy, a_lat), phase, blocks))
                subcells = np.stack([Ef[:, :-1, :-1], Ef[:, :-1, 1:], Ef[:, 1:, 1:], Ef[:, 1:, :-1]]).reshape(4, -1, 2 * p)
        yield E, corners, fine, subcells


def _refinement_taper(x, centre, window):
    """1 inside the refined energy window around ``centre``, rolled off to 0 over its outer fifth."""
    return np.cos(0.5 * np.pi * np.clip((np.abs(x - centre) / window - 0.8) / 0.2, 0.0, 1.0)) ** 2


def calculate_bulk_density_of_states(energy_range=(-1.0, 1.0), stacking_type="ab", n_kz=240,
                                     dk=2.0e-3, refine=8, refine_window=0.12,
                                     fermi_guess=None, k_max=None, num_energy_points=2000,
                                     gaussian_sigma=0.0, params=None, verbose=False):
    """
    Density of states of bulk graphite.

    For each of n_kz equally spaced values of k_z the in-plane integral is done
    with the linear-interpolation (triangle) method on a square mesh around K,
    as in ``multilayer_graphene.calculate_density_of_states``.  Mesh cells in
    which a band comes within ``refine_window`` of the Fermi level are
    subdivided ``refine`` times, which resolves the Fermi surface pockets of
    Bernal graphite and the Dirac points of rhombohedral graphite without a
    fine mesh everywhere.  The refined result replaces the coarse one only
    inside that energy window: the interpolation errors of single cells
    (about dk / 8|p| of the cell's share, |p| the distance from the band
    touching point) cancel between neighbours on one mesh but not across the
    border of the refined region, so outside the window the DOS is taken from
    the coarse mesh alone.

    Args:
        energy_range: (emin, emax) window in eV on the energy scale of the model
        stacking_type: 'ab' (Bernal), 'abc' (rhombohedral) or a repeat unit
        n_kz: number of k_z values in half of the k_z period; the other half
              follows from E(kx, ky, kz) = E(kx, -ky, -kz)
        dk: spacing of the coarse in-plane mesh (1/A)
        refine: subdivision of the cells near the Fermi level (1 = none)
        refine_window: half-width (eV) of the energy range that is refined
        fermi_guess: centre of the refined range; found by a coarse run if None
        k_max: half-width of the in-plane window around K (1/A); chosen from
               the energy window if None
        num_energy_points: number of energy bins
        gaussian_sigma: optional Gaussian smoothing (eV)
        params: couplings overriding the module configuration of
                ``multilayer_graphene``

    Returns a dict with 'energy' (bin centres, eV), 'dos_per_atom'
    (states/(eV atom)), 'dos_per_volume' (states/(eV A^3)), 'dos_per_cm3',
    'fermi_energy' (charge neutrality at T = 0, eV), 'carrier_density_cm3'
    (electrons = holes at the Fermi level), 'energy_edges' and 'parameters'.
    """
    params = mlg.get_parameters() if params is None else {**mlg.get_parameters(), **params}
    seq, p = bulk_period(stacking_type)
    blocks = _bulk_blocks(seq, p, params)
    e_min, e_max = sorted(map(float, energy_range))
    if num_energy_points < 2:
        raise ValueError("num_energy_points must be at least 2.")
    edges = np.linspace(e_min, e_max, int(num_energy_points) + 1)
    refine = max(1, int(refine))
    a_lat = params["a"]

    if k_max is None:
        hv = 1.5 * (a_lat / np.sqrt(3.0)) * params["gamma0"]
        k_max = (max(abs(e_min), abs(e_max)) + 2.0 * abs(params["gamma1"]) + 0.15) / hv
    if refine > 1 and fermi_guess is None:
        fermi_guess = calculate_bulk_density_of_states(
            (e_min, e_max), stacking_type, n_kz=max(8, n_kz // 10), dk=2.0 * dk, refine=1,
            k_max=k_max, num_energy_points=num_energy_points, params=params)["fermi_energy"]

    n_side = int(np.ceil(2.0 * k_max / dk)) + 1
    h = 2.0 * k_max / (n_side - 1)
    border = np.ones((n_side, n_side), dtype=bool)
    border[1:-1, 1:-1] = False

    # state counts of the valence and the conduction half of the bands: the
    # coarse mesh, and the change brought by subdividing the cells near E_F
    cumulative = np.zeros((2, edges.size))
    correction = np.zeros((2, edges.size))
    n_border = n_fine = 0
    for E, corners, fine, subcells in _mesh_slices(blocks, p, a_lat, k_max, dk, n_kz, refine, fermi_guess, refine_window):
        e_border = E[border]
        n_border += int(np.count_nonzero((e_border > e_min) & (e_border < e_max)))
        _add_cells(cumulative, corners.reshape(4, -1, 2 * p), 0.5 * h * h, edges)
        if subcells is not None:
            n_fine += int(np.count_nonzero(fine))
            _add_cells(correction, subcells, 0.5 * (h / refine) ** 2, edges)
            _add_cells(correction, corners[:, fine], -0.5 * h * h, edges)
    if n_border:
        print(f"Warning: {n_border} border states of the k window lie inside the energy "
              f"window; increase k_max (currently {k_max:.3f} 1/A).")

    # states per A^2 of a slab of p layers: valley and spin degeneracy, k_z average
    weight = 4.0 / (2.0 * np.pi) ** 2 / n_kz
    cumulative *= weight
    correction *= weight
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    dos_slab = np.diff(cumulative.sum(axis=0)) / bin_width
    if refine > 1:
        # keep the refinement inside its window
        dos_slab += _refinement_taper(centres, fermi_guess, refine_window) * np.diff(correction.sum(axis=0)) / bin_width
        cumulative += correction        # exact state count, used for the Fermi level
    per_atom = mlg.unit_cell_area(a_lat) / (2.0 * p)
    per_volume = 1.0 / (p * D_LAYER)

    # charge neutrality: electrons in the upper half of the bands equal the
    # holes in the lower half, counted inside the k window
    valence_total = p * (2.0 * k_max) ** 2 * weight * n_kz
    net = cumulative.sum(axis=0) - valence_total
    i = int(np.searchsorted(net, 0.0))
    if 0 < i < edges.size:
        fermi = edges[i - 1] - net[i - 1] * (edges[i] - edges[i - 1]) / (net[i] - net[i - 1])
        carriers = np.interp(fermi, edges, cumulative[1]) * per_volume * 1.0e24
    else:
        print("Warning: the charge neutrality point lies outside the energy window.")
        fermi = carriers = np.nan

    dos_slab = mlg._gaussian_smooth(dos_slab, gaussian_sigma, bin_width)
    if verbose:
        print(f"bulk DOS: {seq} ({p} layer(s) per primitive cell), {n_kz} k_z values, "
              f"{n_side}^2 coarse k points, {n_fine / n_kz:.0f} refined cells per k_z, "
              f"E_F = {fermi * 1e3:.2f} meV, n = p = {carriers:.3g} cm^-3")

    return {
        "energy": centres,
        "energy_edges": edges,
        "dos_per_atom": dos_slab * per_atom,
        "dos_per_volume": dos_slab * per_volume,
        "dos_per_cm3": dos_slab * per_volume * 1.0e24,
        "fermi_energy": fermi,
        "carrier_density_cm3": carriers,
        "bin_width": bin_width,
        "parameters": {
            "stacking": seq, "layers_per_cell": p, "energy_range": (e_min, e_max),
            "n_kz": n_kz, "dk": dk, "refine": refine, "refine_window": refine_window,
            "fermi_guess": fermi_guess, "k_max": k_max,
            "num_energy_points": int(num_energy_points), "gaussian_sigma": gaussian_sigma,
            "d_layer": D_LAYER, **params,
        },
    }



def calculate_bulk_pair_spectrum(edges, stacking_type="ab", fermi=0.0, n_kz=240, dk=2.0e-3, refine=8,
                                 refine_window=0.12, k_max=None, params=None, verbose=False):
    """
    Joint density of states of the mirror pairs n- -> n+ of bulk graphite: the
    n-th band below and the n-th band above the middle of the spectrum (two
    pairs for Bernal, one for rhombohedral stacking), at the same wave vector,
    with the occupation of T = 0,

        J(w) = sum_n  int d^3k / (2 pi)^3  theta(E_F - e_n-) theta(e_n+ - E_F)
               delta(hbar w - e_n+(k) + e_n-(k)),

    per atom, spin and valley included.  The integration is that of
    ``calculate_bulk_density_of_states``, applied to the energy difference of
    the pair: triangles on an in-plane mesh for each k_z, cells near the Fermi
    level subdivided, the subdivision used for hbar w below ``refine_window``.

    Args:
        edges: bin edges of the excitation energy hbar w (eV), equally spaced
        stacking_type: 'ab' (Bernal), 'abc' (rhombohedral) or a repeat unit
        fermi: Fermi level on the energy scale of the model (eV), for instance
               'fermi_energy' of ``calculate_bulk_density_of_states``
        n_kz, dk, refine, refine_window, k_max, params: as for the DOS

    Returns a dict with 'excitation' (bin centres, eV), 'pairs_per_atom'
    (pairs/(eV atom)), 'edges' and 'parameters'.
    """
    params = mlg.get_parameters() if params is None else {**mlg.get_parameters(), **params}
    seq, p = bulk_period(stacking_type)
    blocks = _bulk_blocks(seq, p, params)
    edges = np.asarray(edges, dtype=float)
    refine = max(1, int(refine))
    a_lat = params["a"]
    if k_max is None:
        hv = 1.5 * (a_lat / np.sqrt(3.0)) * params["gamma0"]
        k_max = (0.6 * edges[-1] + 2.0 * abs(params["gamma1"]) + 0.15) / hv
    n_side = int(np.ceil(2.0 * k_max / dk)) + 1
    h = 2.0 * k_max / (n_side - 1)
    border = np.ones((n_side, n_side), dtype=bool)
    border[1:-1, 1:-1] = False

    cumulative = np.zeros(edges.size)
    correction = np.zeros(edges.size)
    lowest_on_border = np.inf
    for E, corners, fine, subcells in _mesh_slices(blocks, p, a_lat, k_max, dk, n_kz, refine, fermi, refine_window):
        on_border = E[border]
        lowest_on_border = min(lowest_on_border, (on_border[:, p:] - on_border[:, p - 1::-1]).min())
        _add_pairs(cumulative, corners.reshape(4, -1, 2 * p), 0.5 * h * h, edges, fermi)
        if subcells is not None:
            _add_pairs(correction, subcells, 0.5 * (h / refine) ** 2, edges, fermi)
            _add_pairs(correction, corners[:, fine], -0.5 * h * h, edges, fermi)
    if lowest_on_border < edges[-1]:
        print(f"Warning: pairs on the border of the k window start at {lowest_on_border:.3f} eV, inside the "
              f"range of excitation energies; increase k_max (currently {k_max:.3f} 1/A).")

    centres = 0.5 * (edges[:-1] + edges[1:])
    spectrum = np.diff(cumulative) / np.diff(edges)
    if refine > 1:
        # a pair below refine_window has both of its bands inside the refined cells
        spectrum += _refinement_taper(centres, 0.0, refine_window) * np.diff(correction) / np.diff(edges)
    spectrum *= 4.0 / (2.0 * np.pi) ** 2 / n_kz * mlg.unit_cell_area(a_lat) / (2.0 * p)
    if verbose:
        print(f"bulk pair spectrum: {seq} ({p} layer(s) per primitive cell), {n_kz} k_z values, "
              f"{n_side}^2 coarse k points, E_F = {fermi * 1e3:.2f} meV")
    return {
        "excitation": centres,
        "edges": edges,
        "pairs_per_atom": spectrum,
        "parameters": {
            "stacking": seq, "layers_per_cell": p, "fermi_energy": fermi, "n_kz": n_kz, "dk": dk, "refine": refine,
            "refine_window": refine_window, "k_max": k_max, "d_layer": D_LAYER, **params,
        },
    }
