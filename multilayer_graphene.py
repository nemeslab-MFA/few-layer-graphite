"""
Slonczewski-Weiss-McClure tight-binding model of few-layer graphene
====================================================================

Band structures and densities of states (DOS) for N-layer graphene with an
arbitrary stacking sequence: rhombohedral (ABC...), Bernal (ABAB...) and
mixed polytypes such as ABCB.

The model and all default parameters follow

    A. McEllistrim, A. Garcia-Ruiz, Z. A. H. Goodwin, V. I. Fal'ko,
    "Spectroscopic signatures of tetralayer graphene polytypes",
    arXiv:2302.07374 (2023), Eq. (1).

Their hybrid k.p / tight-binding Hamiltonians are the first-order expansion of
the lattice Hamiltonian built here around the valley centres K_xi.  Both are
available: ``hamiltonian()`` (full tight-binding, periodic in k) and
``kp_hamiltonian()`` (the paper's continuum model), and both share one
assembly routine so that the correspondence is exact by construction.

Geometry (paper's convention)
-----------------------------
* in-plane lattice constant a = 2.46 A, tau = (0, a/sqrt(3))
* inside layer n sublattice B_n sits at tau relative to A_n
* going along the cyclic sequence A -> B -> C -> A ("forward" step) displaces
  the next layer by -tau, so that B_{n+1} sits on top of A_n (dimer A_n-B_{n+1});
  a "backward" step (B -> A, C -> B, A -> C) puts A_{n+1} on top of B_n.

Couplings (eV)
--------------
gamma0  intralayer nearest neighbour (hopping integral -gamma0)
gamma1  vertical dimer coupling between adjacent layers
gamma3  skew coupling between the non-dimer sites of adjacent layers
gamma4  skew coupling between dimer and non-dimer sites of adjacent layers
gamma2  vertical coupling across two layers through a hexagon centre
        (matrix element gamma2/2)
gamma5  vertical coupling across two layers along a dimer chain
        (matrix element gamma5/2)
Delta_prime  on-site energy; every site receives Delta_prime times the number
        of gamma1 bonds attached to it (the paper's Delta'/2 (1 +- sigma_z)
        and Delta' 1 patterns, extended to any stacking)

The paper quotes (v, v3, v4) = (1, 0.1, 0.022) x 10^6 m/s and
(gamma1, gamma2, gamma5, Delta') = (390, -17, 38, 25) meV.  Velocities map to
hoppings through hbar v_i = (3/2) a_cc gamma_i with a_cc = a/sqrt(3).

Note on Eq. (1) of arXiv:2302.07374v1: the layer-3 on-site term of H_ABCB is
printed as Delta'(1 + sigma_z) and the (2,3) block of H_ABAC as V_AB.  The
geometric rule above (and the band-edge labels of the paper's Fig. 1, e.g.
sqrt(2) gamma1 + 3 Delta'/2 for ABCB) require Delta'(1 - sigma_z) and
V_AB^dagger respectively; this module uses the geometric rule.

Units: energies in eV, wave vectors in 1/Angstrom (k = p/hbar), DOS per
eV per unit cell, per eV per A^2 and per eV per cm^2.  All DOS values include
the spin degeneracy of 2.
"""

import csv
import json

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "HBAR_EV_S", "A_LATTICE", "PAPER_PARAMETERS",
    "hopping_from_velocity", "velocity_from_hopping",
    "set_parameters", "get_parameters", "get_info", "use_paper_parameters",
    "stacking_sequence", "stacking_steps", "dimer_bond_counts",
    "structure_factor", "hamiltonian", "kp_hamiltonian", "build_hamiltonian",
    "reciprocal_vectors", "K_point", "get_brillouin_zone", "get_k_path",
    "calculate_bands", "plot_bands", "plot_panel_comparison", "plot_band_density",
    "calculate_density_of_states", "calculate_density_of_states_window",
    "valley_window_limit", "check_valley_window",
    "plot_dos", "load_dos_from_csv",
]

# =============================================================================
# Physical constants and parameter handling
# =============================================================================

HBAR_EV_S = 6.582119569e-16     # reduced Planck constant (eV s)
A_LATTICE = 2.46                # graphene lattice constant (A), as in the paper
_M_PER_S_TO_A_PER_S = 1.0e10


def hopping_from_velocity(v, a=A_LATTICE):
    """Convert a velocity v (m/s) to a hopping energy (eV): gamma = 2 hbar v / (3 a_cc)."""
    a_cc = a / np.sqrt(3.0)
    return 2.0 * HBAR_EV_S * v * _M_PER_S_TO_A_PER_S / (3.0 * a_cc)


def velocity_from_hopping(gamma, a=A_LATTICE):
    """Convert a hopping energy gamma (eV) to a velocity (m/s): v = 3 a_cc gamma / (2 hbar)."""
    a_cc = a / np.sqrt(3.0)
    return 3.0 * a_cc * gamma / (2.0 * HBAR_EV_S * _M_PER_S_TO_A_PER_S)


# Parameters of McEllistrim et al., arXiv:2302.07374 (2023)
PAPER_PARAMETERS = {
    "gamma0": hopping_from_velocity(1.0e6),     # 3.0895 eV  (v  = 1.000e6 m/s)
    "gamma1": 0.390,
    "gamma2": -0.017,
    "gamma3": hopping_from_velocity(0.1e6),     # 0.3090 eV  (v3 = 0.100e6 m/s)
    "gamma4": hopping_from_velocity(0.022e6),   # 0.0680 eV  (v4 = 0.022e6 m/s)
    "gamma5": 0.038,
    "Delta_prime": 0.025,
    "E0": 0.0,                                  # rigid energy offset of all sites
    "a": A_LATTICE,
}

# ---- module level configuration (modified through set_parameters) ----------
gamma0 = PAPER_PARAMETERS["gamma0"]
gamma1 = PAPER_PARAMETERS["gamma1"]
gamma2 = PAPER_PARAMETERS["gamma2"]
gamma3 = PAPER_PARAMETERS["gamma3"]
gamma4 = PAPER_PARAMETERS["gamma4"]
gamma5 = PAPER_PARAMETERS["gamma5"]
Delta_prime = PAPER_PARAMETERS["Delta_prime"]
E0 = PAPER_PARAMETERS["E0"]
a = PAPER_PARAMETERS["a"]

N_layers = 3            # number of layers used when a periodic pattern is requested
stacking = "abc"        # 'abc', 'aba' or an explicit sequence such as 'ABCB'
n_k = 1500              # number of k points along a band-structure path
k_range = 0.6           # half-length (1/A) of the straight 'px'/'py' paths through K

_PARAMETER_KEYS = ("gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "gamma5",
                   "Delta_prime", "E0", "a")
_CONFIG_KEYS = ("N_layers", "stacking", "n_k", "k_range")
_VELOCITY_KEYS = {"v": "gamma0", "v3": "gamma3", "v4": "gamma4"}


def get_parameters():
    """Return the current tight-binding parameters as a dictionary (eV, A)."""
    g = globals()
    return {key: g[key] for key in _PARAMETER_KEYS}


def set_parameters(**kwargs):
    """
    Set model parameters and configuration.

    Accepted keys: gamma0 ... gamma5, Delta_prime, E0, a (eV, A); the velocities
    v, v3, v4 (m/s), which are converted to gamma0, gamma3, gamma4; and the
    configuration keys N_layers, stacking, n_k, k_range.
    """
    g = globals()
    unknown = [key for key in kwargs
               if key not in _VELOCITY_KEYS and key not in _PARAMETER_KEYS and key not in _CONFIG_KEYS]
    if unknown:
        raise KeyError(f"Unknown parameter '{unknown[0]}'. Known keys: "
                       f"{_PARAMETER_KEYS + _CONFIG_KEYS + tuple(_VELOCITY_KEYS)}")
    # couplings and the lattice constant first, so that velocities given in the
    # same call are converted with the new lattice constant
    for key, value in kwargs.items():
        if key not in _VELOCITY_KEYS:
            g[key] = value
    for key, value in kwargs.items():
        if key in _VELOCITY_KEYS:
            g[_VELOCITY_KEYS[key]] = hopping_from_velocity(float(value), g["a"])


def use_paper_parameters():
    """Reset all couplings to the values of McEllistrim et al. (2023)."""
    set_parameters(**PAPER_PARAMETERS)


def get_info():
    """Print the current configuration."""
    p = get_parameters()
    print("=== Few-layer graphene SWMcC tight-binding model ===")
    print(f"stacking: {stacking_sequence()}  ({len(stacking_sequence())} layers)")
    print("couplings (eV):")
    for key in ("gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "gamma5", "Delta_prime", "E0"):
        print(f"  {key:11s} = {p[key]: .5f}")
    print(f"velocities (10^6 m/s): v = {velocity_from_hopping(p['gamma0'], p['a'])/1e6:.4f}, "
          f"v3 = {velocity_from_hopping(p['gamma3'], p['a'])/1e6:.4f}, "
          f"v4 = {velocity_from_hopping(p['gamma4'], p['a'])/1e6:.4f}")
    print(f"lattice constant a = {p['a']} A, n_k = {n_k}, k_range = {k_range} 1/A")


# =============================================================================
# Stacking sequences
# =============================================================================

_LAYER_INDEX = {"A": 0, "B": 1, "C": 2}
_PATTERNS = {"abc": "ABC", "aba": "AB"}                     # pattern, or the trilayer without a layer count
_PATTERN_WORDS = {"rhombohedral": "ABC", "bernal": "AB"}    # always a pattern


def stacking_sequence(N=None, stacking_type=None):
    """
    Return the explicit stacking sequence as a string of letters A/B/C.

    ``stacking_type`` is one of
      * 'rhombohedral' or 'bernal': the periodic pattern of N layers (N defaults
        to the module variable ``N_layers``);
      * 'abc' or 'aba' (any case): the periodic pattern of N layers when a layer
        count is supplied, and otherwise the trilayer itself.  The layer count is
        the argument N, or ``N_layers`` when stacking_type is None and the module
        configuration (``stacking``, ``N_layers``) is used;
      * any other explicit sequence such as 'AB', 'ABCB' or 'ABCAB' (N, if given,
        must equal its length).
    Adjacent letters must differ.  So hamiltonian(kx, ky, stacking_type='ABA')
    is always the Bernal trilayer, whatever ``N_layers`` is, while
    hamiltonian(kx, ky, 8, 'aba') and set_parameters(N_layers=8, stacking='aba')
    give eight Bernal-stacked layers.
    """
    if stacking_type is None:
        stacking_type = stacking
        if N is None and str(stacking_type).strip().lower() in _PATTERNS:
            N = N_layers
    if isinstance(stacking_type, (list, tuple)):
        stacking_type = "".join(stacking_type)
    key = str(stacking_type).strip().lower()
    if key in _PATTERN_WORDS or (key in _PATTERNS and N is not None):
        N = N_layers if N is None else int(N)
        unit = _PATTERNS.get(key) or _PATTERN_WORDS[key]
        seq = (unit * (N // len(unit) + 1))[:N]
    else:
        seq = key.upper()
        if any(ch not in _LAYER_INDEX for ch in seq):
            raise ValueError(f"Unknown stacking '{stacking_type}'. Use 'abc', 'aba' or "
                             "an explicit sequence of the letters A, B, C.")
        if N is not None and int(N) != len(seq):
            raise ValueError(f"N={N} does not match the explicit sequence '{seq}'.")
    if len(seq) < 1:
        raise ValueError("The stacking sequence must contain at least one layer.")
    for n in range(len(seq) - 1):
        if seq[n] == seq[n + 1]:
            raise ValueError(f"Invalid stacking '{seq}': adjacent layers must differ.")
    return seq


def stacking_steps(seq):
    """
    Steps d_n = +1 ('forward': A->B, B->C, C->A) or -1 ('backward') between
    consecutive layers of an explicit sequence.
    """
    s = [_LAYER_INDEX[ch] for ch in seq]
    steps = []
    for n in range(len(s) - 1):
        d = (s[n + 1] - s[n]) % 3
        steps.append(+1 if d == 1 else -1)
    return np.array(steps, dtype=int)


def dimer_bond_counts(seq):
    """
    Number of vertical gamma1 bonds attached to A_n and B_n, shape (N, 2).

    A forward step n->n+1 creates the dimer A_n-B_{n+1}; a backward step the
    dimer B_n-A_{n+1}.
    """
    N = len(seq)
    counts = np.zeros((N, 2), dtype=int)
    for n, d in enumerate(stacking_steps(seq)):
        if d == +1:
            counts[n, 0] += 1
            counts[n + 1, 1] += 1
        else:
            counts[n, 1] += 1
            counts[n + 1, 0] += 1
    return counts


# =============================================================================
# Lattice helpers
# =============================================================================

def _lattice_vectors(a_lat=None):
    a_lat = a if a_lat is None else a_lat
    a1 = np.array([a_lat, 0.0])
    a2 = np.array([a_lat / 2.0, a_lat * np.sqrt(3.0) / 2.0])
    return a1, a2


def reciprocal_vectors(a_lat=None):
    """Reciprocal lattice vectors b1, b2 (1/A) with b_i . a_j = 2 pi delta_ij."""
    a1, a2 = _lattice_vectors(a_lat)
    cross = a1[0] * a2[1] - a1[1] * a2[0]
    b1 = 2.0 * np.pi * np.array([a2[1], -a2[0]]) / cross
    b2 = 2.0 * np.pi * np.array([-a1[1], a1[0]]) / cross
    return b1, b2


def unit_cell_area(a_lat=None):
    """Area of the primitive (two-atom per layer) unit cell (A^2)."""
    a_lat = a if a_lat is None else a_lat
    return np.sqrt(3.0) / 2.0 * a_lat ** 2


def K_point(xi=+1, a_lat=None):
    """Valley centre K_xi = (4 pi / 3a) (xi, 0) in 1/A."""
    a_lat = a if a_lat is None else a_lat
    return np.array([xi * 4.0 * np.pi / (3.0 * a_lat), 0.0])


def _nearest_neighbour_vectors(a_lat=None):
    """The three vectors from an A site to its B neighbours, delta_1 = tau = (0, a_cc)."""
    a_lat = a if a_lat is None else a_lat
    a_cc = a_lat / np.sqrt(3.0)
    return a_cc * np.array([[0.0, 1.0],
                            [-np.sqrt(3.0) / 2.0, -0.5],
                            [np.sqrt(3.0) / 2.0, -0.5]])


def structure_factor(kx, ky, a_lat=None):
    """
    f(k) = sum_j exp(i k . delta_j) over the three A->B bond vectors.

    Near K_xi:  f(K_xi + p) = -(3 a_cc / 2) (xi p_x - i p_y) + O(p^2),
    so that -gamma0 f = hbar v (xi p_x - i p_y) = hbar v pi_xi^*, the H_g of
    the paper.
    """
    d = _nearest_neighbour_vectors(a_lat)
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    phase = kx[..., None] * d[:, 0] + ky[..., None] * d[:, 1]
    return np.exp(1j * phase).sum(axis=-1)


# =============================================================================
# Hamiltonians
# =============================================================================

def _assemble(u, seq, params):
    """
    Assemble the 2N x 2N Hamiltonian for an explicit sequence, given the
    (possibly array-valued) intralayer structure u, which stands for f(k) in the
    tight-binding model and for its linearisation in the k.p model.

    Basis ordering: (A_1, B_1, A_2, B_2, ..., A_N, B_N).
    """
    u = np.asarray(u, dtype=complex)
    uc = np.conj(u)
    N = len(seq)
    g0, g1, g2 = params["gamma0"], params["gamma1"], params["gamma2"]
    g3, g4, g5 = params["gamma3"], params["gamma4"], params["gamma5"]
    dp, e0 = params["Delta_prime"], params["E0"]

    H = np.zeros(u.shape + (2 * N, 2 * N), dtype=complex)
    A = lambda n: 2 * n       # noqa: E731
    B = lambda n: 2 * n + 1   # noqa: E731

    bonds = dimer_bond_counts(seq)
    steps = stacking_steps(seq)

    for n in range(N):
        H[..., A(n), A(n)] = e0 + dp * bonds[n, 0]
        H[..., B(n), B(n)] = e0 + dp * bonds[n, 1]
        H[..., A(n), B(n)] = -g0 * u

    for n, d in enumerate(steps):
        m = n + 1
        if d == +1:     # dimer A_n - B_m  (paper's V_AB)
            H[..., A(n), A(m)] = g4 * uc
            H[..., A(n), B(m)] = g1
            H[..., B(n), A(m)] = g3 * u
            H[..., B(n), B(m)] = g4 * uc
        else:           # dimer B_n - A_m  (paper's V_AB^dagger)
            H[..., A(n), A(m)] = g4 * u
            H[..., A(n), B(m)] = g3 * uc
            H[..., B(n), A(m)] = g1
            H[..., B(n), B(m)] = g4 * u

    for n in range(N - 2):
        m = n + 2
        d1, d2 = steps[n], steps[n + 1]
        if (d1, d2) == (+1, +1):        # W_ABC : B_n above A_{n+2}
            H[..., B(n), A(m)] = 0.5 * g2
        elif (d1, d2) == (-1, -1):      # W_ABC^dagger : A_n above B_{n+2}
            H[..., A(n), B(m)] = 0.5 * g2
        elif (d1, d2) == (+1, -1):      # W_ABA : dimer chain A_n-B_{n+1}-A_{n+2}
            H[..., A(n), A(m)] = 0.5 * g5
            H[..., B(n), B(m)] = 0.5 * g2
        else:                           # W_BAB : dimer chain B_n-A_{n+1}-B_{n+2}
            H[..., A(n), A(m)] = 0.5 * g2
            H[..., B(n), B(m)] = 0.5 * g5

    # Hermitian completion of the strictly upper triangle
    iu = np.triu_indices(2 * N, k=1)
    H[..., iu[1], iu[0]] = np.conj(H[..., iu[0], iu[1]])
    return H


def hamiltonian(kx, ky, N=None, stacking_type=None, params=None):
    """
    Tight-binding Hamiltonian H(k) for Cartesian wave vectors kx, ky (1/A).

    kx and ky may be scalars or arrays of identical shape; the result has
    shape kx.shape + (2N, 2N) and is Hermitian.
    """
    seq = stacking_sequence(N, stacking_type)
    params = get_parameters() if params is None else {**get_parameters(), **params}
    return _assemble(structure_factor(kx, ky, params["a"]), seq, params)


def kp_hamiltonian(px, py, N=None, stacking_type=None, xi=+1, params=None):
    """
    Hybrid k.p Hamiltonian of McEllistrim et al., Eq. (1), generalised to any
    stacking sequence.  px, py are measured from the valley centre K_xi (1/A).

    Obtained from the tight-binding Hamiltonian by replacing the structure
    factor f(K_xi + p) with its linear term -(3 a_cc / 2)(xi p_x - i p_y).
    """
    seq = stacking_sequence(N, stacking_type)
    params = get_parameters() if params is None else {**get_parameters(), **params}
    a_cc = params["a"] / np.sqrt(3.0)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    u = -1.5 * a_cc * (xi * px - 1j * py)
    return _assemble(u, seq, params)


def build_hamiltonian(kx_rec, ky_rec, N=None, stacking_type=None):
    """
    Tight-binding Hamiltonian for fractional reciprocal coordinates
    (k = kx_rec b1 + ky_rec b2).  Kept for backward compatibility; see
    ``hamiltonian`` for Cartesian input.
    """
    b1, b2 = reciprocal_vectors()
    kx_rec = np.asarray(kx_rec, dtype=float)
    ky_rec = np.asarray(ky_rec, dtype=float)
    kx = kx_rec * b1[0] + ky_rec * b2[0]
    ky = kx_rec * b1[1] + ky_rec * b2[1]
    return hamiltonian(kx, ky, N, stacking_type)


def eigenvalues(kx, ky, N=None, stacking_type=None, model="tb", xi=+1):
    """Sorted band energies (eV) at Cartesian k (tb) or p relative to K_xi (kp)."""
    if model == "tb":
        H = hamiltonian(kx, ky, N, stacking_type)
    elif model == "kp":
        H = kp_hamiltonian(kx, ky, N, stacking_type, xi=xi)
    else:
        raise ValueError("model must be 'tb' or 'kp'.")
    return np.linalg.eigvalsh(H)


# =============================================================================
# Brillouin zone and k paths
# =============================================================================

def get_brillouin_zone(plot=True, ax=None, annotate=True, figsize=(4, 4), save_as=None, show=True):
    """
    First Brillouin zone data (and optional sketch): reciprocal vectors, Gamma,
    the six corners (alternating K_+ and K_-) and the six M points, all in 1/A.
    """
    b1, b2 = reciprocal_vectors()
    kK = np.linalg.norm(K_point())
    angles = np.deg2rad(60.0 * np.arange(6))
    corners = kK * np.column_stack([np.cos(angles), np.sin(angles)])
    valleys = np.array([+1, -1, +1, -1, +1, -1])
    M_cart = 0.5 * (corners + np.roll(corners, -1, axis=0))

    fig = None
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        closed = np.vstack([corners, corners[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.0)
        ax.scatter(corners[valleys > 0, 0], corners[valleys > 0, 1], s=18, c="tab:red", label="K+")
        ax.scatter(corners[valleys < 0, 0], corners[valleys < 0, 1], s=18, c="tab:orange", label="K-")
        ax.scatter(M_cart[:, 0], M_cart[:, 1], s=18, c="tab:blue", label="M")
        ax.scatter([0.0], [0.0], s=24, c="tab:green", label="Γ", zorder=3)
        if annotate:
            ax.annotate("Γ", (0.0, 0.0), textcoords="offset points", xytext=(6, 4))
            ax.annotate("K+", corners[0], textcoords="offset points", xytext=(6, 4))
            ax.annotate("M", M_cart[0], textcoords="offset points", xytext=(6, 4))
        ax.set_aspect(1.0)
        ax.set_xlabel("k_x (1/Å)")
        ax.set_ylabel("k_y (1/Å)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        if save_as is not None:
            ax.figure.savefig(save_as, bbox_inches="tight", dpi=300)
        if fig is not None and show:
            plt.show()

    return {
        "b1": b1, "b2": b2, "Gamma": np.zeros(2),
        "K_cart": corners, "K_valley": valleys, "M_cart": M_cart,
        "BZ_vertices": corners, "fig": fig, "ax": ax if plot else None,
    }


def get_k_path(path_type="gkm", n_points=None, half_range=None):
    """
    Build a k path through K_+ with K at coordinate 0.

    path_type:
        'gkm' : Gamma -> K -> M (arc length; Gamma negative, M positive)
        'gkg' : Gamma -> K -> Gamma' (through K along the Gamma-K line)
        'px'  : straight line through K along the Gamma-K direction (p_x axis
                of the paper), from -half_range to +half_range
        'py'  : straight line through K perpendicular to it

    Returns (k_cart, k_mag): Cartesian points of shape (n, 2) in 1/A and the
    1D coordinate along the path (1/A) with K at 0.
    """
    n_points = n_k if n_points is None else int(n_points)
    half_range = k_range if half_range is None else float(half_range)
    K = K_point(+1)
    kind = path_type.lower()

    if kind in ("gkm", "gkg"):
        G = np.zeros(2)
        if kind == "gkm":
            end = 0.5 * (K + np.linalg.norm(K) * np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]))
        else:
            end = 2.0 * K    # Gamma of the neighbouring zone
        L1, L2 = np.linalg.norm(K - G), np.linalg.norm(end - K)
        n1 = max(2, int(round(n_points * L1 / (L1 + L2))))
        n2 = max(2, n_points - n1)
        t1 = np.linspace(0.0, 1.0, n1, endpoint=False)[:, None]
        t2 = np.linspace(0.0, 1.0, n2)[:, None]
        k_cart = np.vstack([G + t1 * (K - G), K + t2 * (end - K)])
        k_mag = np.concatenate([-L1 * (1.0 - t1[:, 0]), L2 * t2[:, 0]])
    elif kind in ("px", "py"):
        t = np.linspace(-half_range, half_range, n_points)
        direction = np.array([1.0, 0.0]) if kind == "px" else np.array([0.0, 1.0])
        k_cart = K + t[:, None] * direction
        k_mag = t
    else:
        raise ValueError(f"Unknown path_type '{path_type}'. Use 'gkm', 'gkg', 'px' or 'py'.")
    return k_cart, k_mag


# =============================================================================
# Band structure
# =============================================================================

def calculate_bands(N=None, stacking_type=None, k_path=None, path_type="gkm"):
    """
    Band energies along a k path.

    Returns (E, k_mag) with E of shape (n_points, 2N), sorted per k point, and
    k_mag the path coordinate (1/A, K at 0).
    """
    if k_path is None:
        k_cart, k_mag = get_k_path(path_type)
    else:
        k_cart, k_mag = k_path
        k_cart = np.asarray(k_cart, dtype=float)
    E = eigenvalues(k_cart[:, 0], k_cart[:, 1], N, stacking_type)
    return E, np.asarray(k_mag)


def _path_axis_labels(ax, k_mag, path_type):
    kind = path_type.lower()
    if kind in ("px", "py"):
        ax.set_xlabel("p_x (1/Å)" if kind == "px" else "p_y (1/Å)")
        return
    ax.set_xlabel("k along Γ → K → M" if kind == "gkm" else "k along Γ → K → Γ")
    lo, hi = ax.get_xlim()
    ticks, labels = [], []
    for pos, lab in ((k_mag.min(), "Γ"), (0.0, "K"), (k_mag.max(), "M" if kind == "gkm" else "Γ")):
        if lo <= pos <= hi:
            ticks.append(pos)
            labels.append(lab)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def plot_bands(E=None, k_mag=None, N=None, stacking_type=None, path_type="gkm",
               xlim=None, ylim=(-0.5, 0.5), figsize=(5, 4), highlight_middle=True,
               save_as=None, ax=None, show=True):
    """Plot the band structure of one stack (computed if E, k_mag are not given)."""
    if E is None or k_mag is None:
        E, k_mag = calculate_bands(N, stacking_type, path_type=path_type)
    seq = stacking_sequence(N, stacking_type)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    nb = E.shape[1]
    mid = {nb // 2 - 1, nb // 2} if highlight_middle else set()
    for b in range(nb):
        ax.plot(k_mag, E[:, b], linewidth=0.9 if b in mid else 0.6,
                color="#d95926" if b in mid else "black")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"{seq} ({len(seq)} layers)")
    if xlim is None:
        span = k_mag.max() - k_mag.min()
        ax.set_xlim(-0.15 * span, 0.15 * span)
    else:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _path_axis_labels(ax, k_mag, path_type)
    if created:
        ax.figure.tight_layout()
        if save_as:
            ax.figure.savefig(save_as, bbox_inches="tight", dpi=300)
            plt.close(ax.figure)
        elif show:
            plt.show()
    return ax


def plot_panel_comparison(N_range=range(1, 9), stacking_type=None, path_type="gkm",
                          xlim=None, ylim=(-0.7, 0.7), figsize=(16, 2),
                          highlight_middle=True, save_as=None, show=True):
    """Side-by-side band structures for several layer numbers of one stacking pattern."""
    N_range = list(N_range)
    k_path = get_k_path(path_type)
    fig, axes = plt.subplots(1, len(N_range), figsize=figsize, squeeze=False)
    for i, N in enumerate(N_range):
        ax = axes[0, i]
        E, k_mag = calculate_bands(N, stacking_type, k_path)
        plot_bands(E, k_mag, N, stacking_type, path_type, xlim, ylim,
                   highlight_middle=highlight_middle, ax=ax, show=False)
        ax.set_title(f"{N} L")
        if i > 0:
            ax.set_ylabel("")
            ax.set_yticks([])
        if i != len(N_range) // 2:
            ax.set_xlabel("")
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, bbox_inches="tight", dpi=300)
        plt.close(fig)
    elif show:
        plt.show()
    return fig


def plot_band_density(band_index, N=None, stacking_type=None, k_max=0.25, n_grid=301,
                      cmap="viridis", vmin=None, vmax=None, figsize=(5, 4),
                      save_as=None, ax=None, show=True):
    """
    Colour map of one band E_b(k) on a square window of half-width k_max (1/A)
    centred on K_+.  Bands are indexed from the bottom (0 = lowest).
    """
    seq = stacking_sequence(N, stacking_type)
    K = K_point(+1)
    t = np.linspace(-k_max, k_max, n_grid)
    kx, ky = np.meshgrid(K[0] + t, K[1] + t, indexing="xy")
    E = eigenvalues(kx, ky, N, stacking_type)
    Z = E[..., int(band_index) % (2 * len(seq))]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(kx - K[0], ky - K[1], Z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("p_x (1/Å)")
    ax.set_ylabel("p_y (1/Å)")
    ax.set_title(f"{seq} | band {int(band_index) % (2 * len(seq))}")
    ax.figure.colorbar(mesh, ax=ax, label="Energy (eV)")
    if created:
        ax.figure.tight_layout()
        if save_as:
            ax.figure.savefig(save_as, bbox_inches="tight", dpi=300)
            plt.close(ax.figure)
        elif show:
            plt.show()
    return ax


# =============================================================================
# Density of states (linear interpolation on triangles)
# =============================================================================

def _square_mesh_triangles(nx, ny):
    """Triangle vertex indices (2 (nx-1)(ny-1), 3) for a grid flattened in C order."""
    i, j = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    p00 = (i * nx + j).ravel()
    p01 = p00 + 1
    p10 = p00 + nx
    p11 = p10 + 1
    return np.vstack([np.column_stack([p00, p01, p11]),
                      np.column_stack([p00, p11, p10])])


def _periodic_mesh_triangles(n):
    """Triangles of an n x n periodic grid in fractional coordinates."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    p00 = (i * n + j).ravel()
    p01 = (i * n + (j + 1) % n).ravel()
    p10 = (((i + 1) % n) * n + j).ravel()
    p11 = (((i + 1) % n) * n + (j + 1) % n).ravel()
    return np.vstack([np.column_stack([p00, p01, p11]),
                      np.column_stack([p00, p11, p10])])


def _cumulative_states_triangles(e_tri, area, edges):
    """
    Integrated k-space measure n(E) = sum_triangles area * fraction of the
    triangle with linear-interpolated energy below E, evaluated at ``edges``.

    e_tri: (T, 3) vertex energies, area: scalar or (T,), edges: (M,) ascending.
    """
    e = np.sort(np.asarray(e_tri, dtype=float), axis=1)
    e1, e2, e3 = e[:, 0], e[:, 1], e[:, 2]
    area = np.broadcast_to(np.asarray(area, dtype=float), e1.shape)
    edges = np.asarray(edges, dtype=float)
    M = edges.size
    n = np.zeros(M)

    # full contribution for every edge above the top vertex energy
    i_full = np.searchsorted(edges, e3, side="right")      # edges[i] >= e3 for i >= i_full... (strict >)
    weights = np.bincount(i_full, weights=area, minlength=M + 1)[:M]
    n += np.cumsum(weights)

    # partial contributions for edges inside [e1, e3)
    i_lo = np.searchsorted(edges, e1, side="left")
    i_hi = i_full - 1                                      # last edge index with edges[i] < e3 ... (<=)
    span = i_hi - i_lo
    d21 = e2 - e1
    d31 = e3 - e1
    d32 = e3 - e2
    den_lo = np.where((d21 > 0) & (d31 > 0), d21 * d31, 1.0)
    den_hi = np.where((d31 > 0) & (d32 > 0), d31 * d32, 1.0)
    for j in range(int(span.max()) + 1 if span.size else 0):
        sel = np.nonzero(span >= j)[0]
        if sel.size == 0:
            break
        idx = i_lo[sel] + j
        E = edges[idx]
        lower = E < e2[sel]
        val = np.where(
            lower,
            area[sel] * (E - e1[sel]) ** 2 / den_lo[sel],
            area[sel] * (1.0 - (e3[sel] - E) ** 2 / den_hi[sel]),
        )
        # a flat triangle (e1 == e3) has an empty open interval; guard anyway
        val = np.where(d31[sel] > 0, val, 0.0)
        n += np.bincount(idx, weights=val, minlength=M)
    return n


def _gaussian_smooth(y, sigma, dx):
    """
    Convolve y (uniform grid of spacing dx) with a normalised Gaussian of
    standard deviation sigma.  The curve is continued by its end values, so the
    ends are not pulled towards zero.
    """
    if sigma is None or sigma <= 0.0:
        return y
    s = sigma / dx
    radius = max(1, int(np.ceil(4.0 * s)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / s) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(y, radius, mode="edge"), kernel, mode="valid")


def valley_window_limit(a_lat=None):
    """
    Largest half-width (1/A) of a square k window around K that does not
    overlap the window around K' (half the K-K' distance along k_y,
    pi / (sqrt(3) a)).  A larger window counts states twice.
    """
    a_lat = a if a_lat is None else a_lat
    return np.pi / (np.sqrt(3.0) * a_lat)


def check_valley_window(k_max, a_lat=None):
    """Raise ValueError if a square window of half-width k_max around K overlaps the one around K'."""
    limit = valley_window_limit(a_lat)
    if k_max > limit:
        raise ValueError(f"k window of half-width {k_max:.3f} 1/A around K overlaps the window around K' "
                         f"(limit {limit:.3f} 1/A): the energy range is too wide for the valley "
                         "integration; use region='bz' or a narrower range.")


def _auto_k_max(energy_range):
    """Radius (1/A) around K that is guaranteed to contain all states inside the energy window."""
    e_max = max(abs(energy_range[0]), abs(energy_range[1]))
    hv = 1.5 * (a / np.sqrt(3.0)) * gamma0
    return (e_max + 2.0 * abs(gamma1) + 0.15) / hv


def _eigenvalues_in_chunks(kx, ky, N, seq, model="tb", xi=+1, max_bytes=2 ** 28):
    """
    ``eigenvalues`` for 1D arrays kx, ky, evaluated in pieces so that the stack
    of Hamiltonians stays below max_bytes (a 100-layer stack takes 640 kB per k
    point).
    """
    step = max(1, int(max_bytes // (16 * (2 * len(seq)) ** 2)))
    return np.concatenate([eigenvalues(kx[i:i + step], ky[i:i + step], N, seq, model=model, xi=xi)
                           for i in range(0, kx.size, step)])


def _valley_window_eigenvalues(t, N, seq, model="tb", xi=+1, max_bytes=2 ** 28):
    """
    Band energies on the square mesh K_xi + (t_j, t_i), shape (n, n, 2N) with
    the first axis along p_y; t must be symmetric about 0.

    Only p_y >= 0 is diagonalised: complex conjugation of H exchanges f and
    f*, so that E(p_x, -p_y) = E(p_x, p_y) for every stack and both models.
    """
    n = t.size
    half = t[n // 2:]
    px, py = np.meshgrid(t, half, indexing="xy")
    K = K_point(xi) if model == "tb" else np.zeros(2)
    E = _eigenvalues_in_chunks(px.ravel() + K[0], py.ravel() + K[1], N, seq, model, xi, max_bytes)
    E = E.reshape(half.size, n, E.shape[-1])
    return np.concatenate([E[:0:-1] if n % 2 else E[::-1], E])


def calculate_density_of_states(energy_range=(-0.8, 0.8), N=None, stacking_type=None,
                                model="tb", region="valley", k_max=None, dk=1.0e-3,
                                n_bz=400, num_energy_points=400, gaussian_sigma=0.0,
                                xi=+1, verbose=False):
    """
    Density of states from a linear-interpolation (triangle) integration.

    Args:
        energy_range: (emin, emax) window in eV
        N, stacking_type: stack (defaults to the module configuration)
        model: 'tb' (tight binding) or 'kp' (paper's k.p Hamiltonian)
        region: 'valley' integrates a square of half-width k_max around K_xi
                (multiplied by the valley degeneracy 2); 'bz' integrates the full
                Brillouin zone on an n_bz x n_bz mesh (tight binding only)
        k_max: half-width of the valley window (1/A); chosen automatically from
               the energy window if None
        dk: mesh spacing in the valley window (1/A)
        num_energy_points: number of energy bins
        gaussian_sigma: optional Gaussian smoothing (eV)
        xi: valley index used with model='kp'

    Returns a dict with 'energy' (bin centres, eV), 'dos_per_unit_cell'
    (states/(eV unit cell)), 'dos_per_area' (states/(eV A^2)), 'dos_per_cm2'
    (states/(eV cm^2)), 'cumulative_per_area' (states/A^2 below each bin edge:
    every state of the k window or zone below that energy, including the ones
    below emin), 'energy_edges' and 'parameters'.  Spin degeneracy 2 is
    included.  With region='valley' the routine checks that no band enters the
    window on the border of the k window and warns otherwise, and refuses a
    tight-binding window that overlaps the one of the other valley
    (k_max > pi/(sqrt(3) a), about 0.74 1/A or |E| of about 3.9 eV); use
    region='bz' there.
    """
    seq = stacking_sequence(N, stacking_type)
    e_min, e_max = sorted(map(float, energy_range))
    if num_energy_points < 2:
        raise ValueError("num_energy_points must be at least 2.")
    edges = np.linspace(e_min, e_max, int(num_energy_points) + 1)
    spin = 2.0

    if region == "valley":
        if k_max is None:
            k_max = _auto_k_max((e_min, e_max))
        if model == "tb":
            check_valley_window(k_max)
        n_side = int(np.ceil(2.0 * k_max / dk)) + 1
        t = np.linspace(-k_max, k_max, n_side)
        h = t[1] - t[0]
        gx, gy = np.meshgrid(t, t, indexing="xy")
        if model not in ("tb", "kp"):
            raise ValueError("model must be 'tb' or 'kp'.")
        E = _valley_window_eigenvalues(t, N, seq, model, xi).reshape(n_side * n_side, -1)
        tri = _square_mesh_triangles(n_side, n_side)
        area = 0.5 * h * h
        degeneracy = 2.0 * spin
        border = np.zeros(gx.shape, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        e_border = E[border.ravel()]
        inside = (e_border > e_min) & (e_border < e_max)
        if np.any(inside):
            print(f"Warning: {int(inside.sum())} border states of the k window lie inside the "
                  f"energy window; increase k_max (currently {k_max:.3f} 1/A).")
    elif region == "bz":
        if model != "tb":
            raise ValueError("region='bz' requires model='tb'.")
        n = int(n_bz)
        b1, b2 = reciprocal_vectors()
        u = np.arange(n) / n
        gu, gv = np.meshgrid(u, u, indexing="ij")
        kx = gu.ravel() * b1[0] + gv.ravel() * b2[0]
        ky = gu.ravel() * b1[1] + gv.ravel() * b2[1]
        E = _eigenvalues_in_chunks(kx, ky, N, seq)
        tri = _periodic_mesh_triangles(n)
        area = abs(b1[0] * b2[1] - b1[1] * b2[0]) / (2.0 * n * n)
        degeneracy = spin
        k_max = None
    else:
        raise ValueError("region must be 'valley' or 'bz'.")

    n_bands = E.shape[1]
    cumulative = np.zeros(edges.size)
    for b in range(n_bands):
        cumulative += _cumulative_states_triangles(E[tri, b], area, edges)
    cumulative *= degeneracy / (2.0 * np.pi) ** 2        # states per A^2 below E
    bin_width = edges[1] - edges[0]
    dos_area = np.diff(cumulative) / bin_width
    dos_area = _gaussian_smooth(dos_area, gaussian_sigma, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])

    if verbose:
        print(f"DOS: {seq}, model={model}, region={region}, {E.shape[0]} k points, "
              f"{tri.shape[0]} triangles, {n_bands} bands")

    return {
        "energy": centres,
        "energy_edges": edges,
        "dos_per_unit_cell": dos_area * unit_cell_area(),
        "dos_per_area": dos_area,
        "dos_per_cm2": dos_area * 1.0e16,
        "cumulative_per_area": cumulative,
        "bin_width": bin_width,
        "total_kpoints": int(E.shape[0]),
        "parameters": {
            "N_layers": len(seq), "stacking": seq, "model": model, "region": region,
            "energy_range": (e_min, e_max), "k_max": k_max, "dk": dk, "n_bz": n_bz,
            "num_energy_points": int(num_energy_points), "gaussian_sigma": gaussian_sigma,
            "xi": xi, **get_parameters(),
        },
    }


def calculate_density_of_states_window(energy_min, energy_max, N=None, stacking_type=None,
                                       model="tb", region="valley", k_max=None, dk=1.0e-3,
                                       n_bz=400):
    """
    Average DOS inside [energy_min, energy_max] (states/(eV unit cell) and
    states/(eV A^2)) and the integrated number of states in the window.
    """
    e_min, e_max = sorted((float(energy_min), float(energy_max)))
    if e_max <= e_min:
        raise ValueError("energy_max must be greater than energy_min.")
    dos = calculate_density_of_states((e_min, e_max), N, stacking_type, model=model,
                                      region=region, k_max=k_max, dk=dk, n_bz=n_bz,
                                      num_energy_points=2)
    states_area = dos["cumulative_per_area"][-1]
    width = e_max - e_min
    return {
        "average_dos_per_unit_cell": states_area * unit_cell_area() / width,
        "average_dos_per_area": states_area / width,
        "integrated_states_per_unit_cell": states_area * unit_cell_area(),
        "integrated_states_per_area": states_area,
        "energy_window": (e_min, e_max),
        "window_width": width,
        "N_layers": dos["parameters"]["N_layers"],
        "stacking": dos["parameters"]["stacking"],
    }


def _save_dos_to_csv(csv_path, dos_data, metadata_extra=None):
    metadata = dict(dos_data["parameters"])
    metadata["bin_width"] = dos_data["bin_width"]
    metadata["total_kpoints"] = dos_data["total_kpoints"]
    if metadata_extra:
        metadata.update(metadata_extra)
    metadata["columns"] = ["energy_eV", "dos_per_unit_cell_states_per_eV",
                           "dos_per_area_states_per_eV_A2", "dos_per_cm2_states_per_eV_cm2"]
    with open(csv_path, "w", newline="") as fh:
        fh.write(f"# metadata: {json.dumps(metadata, sort_keys=True, default=str)}\n")
        writer = csv.writer(fh)
        writer.writerow(metadata["columns"])
        for row in zip(dos_data["energy"], dos_data["dos_per_unit_cell"],
                       dos_data["dos_per_area"], dos_data["dos_per_cm2"]):
            writer.writerow([f"{v:.10g}" for v in row])


def load_dos_from_csv(csv_path):
    """Load a DOS file written by ``plot_dos(csv_path=...)``."""
    with open(csv_path, "r", newline="") as fh:
        first = fh.readline().strip()
        if not first.startswith("# metadata:"):
            raise ValueError("CSV file does not contain a metadata header.")
        metadata = json.loads(first[len("# metadata:"):].strip())
        reader = csv.reader(fh)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader if row])
    data = {name: rows[:, i] for i, name in enumerate(header)}
    return {
        "energy": data["energy_eV"],
        "dos_per_unit_cell": data["dos_per_unit_cell_states_per_eV"],
        "dos_per_area": data["dos_per_area_states_per_eV_A2"],
        "dos_per_cm2": data.get("dos_per_cm2_states_per_eV_cm2"),
        "metadata": metadata,
    }


def plot_dos(energy_range=(-0.8, 0.8), N=None, stacking_type=None, model="tb",
             region="valley", k_max=None, dk=1.0e-3, n_bz=400, num_energy_points=400,
             gaussian_sigma=0.0, units="cm2", ax=None, figsize=(4, 5), label=None,
             show=True, return_data=False, csv_path=None, metadata_extra=None,
             orientation="horizontal"):
    """
    Plot the density of states.  units: 'cm2' (states/(eV cm^2), as in the
    paper), 'A2' or 'cell'.  orientation='horizontal' draws energy on the
    vertical axis like Fig. 1 of the paper; 'vertical' puts energy on x.
    """
    dos = calculate_density_of_states(energy_range, N, stacking_type, model=model,
                                      region=region, k_max=k_max, dk=dk, n_bz=n_bz,
                                      num_energy_points=num_energy_points,
                                      gaussian_sigma=gaussian_sigma)
    key, unit_label = {
        "cm2": ("dos_per_cm2", "DOS (eV⁻¹ cm⁻²)"),
        "A2": ("dos_per_area", "DOS (eV⁻¹ Å⁻²)"),
        "cell": ("dos_per_unit_cell", "DOS (eV⁻¹ per unit cell)"),
    }[units]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    if orientation == "horizontal":
        ax.plot(dos[key], dos["energy"], label=label, linewidth=1.2)
        ax.set_xlabel(unit_label)
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(left=0.0)
    else:
        ax.plot(dos["energy"], dos[key], label=label, linewidth=1.2)
        ax.set_ylabel(unit_label)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylim(bottom=0.0)
    seq = dos["parameters"]["stacking"]
    ax.set_title(f"{seq} ({len(seq)} layers), {model}")
    if label is not None:
        ax.legend()
    if created:
        ax.figure.tight_layout()
        if show:
            plt.show()
    if csv_path is not None:
        _save_dos_to_csv(csv_path, dos, metadata_extra)
    return dos if return_data else None


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    get_info()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, seq in enumerate(["ABAB", "ABCA", "ABCB"]):
        E, kmag = calculate_bands(stacking_type=seq, path_type="px")
        plot_bands(E, kmag, stacking_type=seq, path_type="px", xlim=(-0.2, 0.2),
                   ylim=(-0.8, 0.8), ax=axes[0, col], show=False)
        plot_dos(stacking_type=seq, ax=axes[1, col], show=False)
    fig.tight_layout()
    plt.show()
