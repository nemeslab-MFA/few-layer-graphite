"""
Tests of the SWMcC tight-binding model against analytic limits and against
Fig. 1 of McEllistrim, Garcia-Ruiz, Goodwin and Fal'ko, arXiv:2302.07374 (2023).

Run with:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multilayer_graphene as mlg  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STACKS = ["A", "AB", "ABC", "ABA", "ABCA", "ABAB", "ABCB", "ABAC", "ABCAB", "ABABA", "ABCBA"]
PAPER_STACKS = ["ABAB", "ABCA", "ABCB"]


@pytest.fixture(autouse=True)
def paper_parameters():
    # N_layers is deliberately not 3: explicit trilayers ('ABC', 'ABA') must not depend on it
    mlg.use_paper_parameters()
    mlg.set_parameters(N_layers=5, stacking="abc")
    yield
    mlg.use_paper_parameters()
    mlg.set_parameters(N_layers=3, stacking="abc")


def hbar_v():
    p = mlg.get_parameters()
    return 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma0"]


# ----------------------------------------------------------------------------
# parameters and stacking bookkeeping
# ----------------------------------------------------------------------------

def test_parameters_follow_the_paper():
    p = mlg.get_parameters()
    assert p["gamma1"] == 0.390 and p["gamma2"] == -0.017 and p["gamma5"] == 0.038
    assert p["Delta_prime"] == 0.025
    assert abs(hbar_v() - mlg.HBAR_EV_S * 1.0e6 * 1.0e10) < 1e-12          # hbar v for v = 1e6 m/s
    assert abs(mlg.velocity_from_hopping(p["gamma0"]) - 1.0e6) < 1e-6
    assert abs(mlg.velocity_from_hopping(p["gamma3"]) - 0.1e6) < 1e-6
    assert abs(mlg.velocity_from_hopping(p["gamma4"]) - 0.022e6) < 1e-6
    assert abs(p["gamma0"] - 3.0896) < 1e-3


def test_set_parameters_accepts_velocities():
    mlg.set_parameters(v=1.1e6)
    assert abs(mlg.velocity_from_hopping(mlg.gamma0) - 1.1e6) < 1e-6
    # a velocity is converted with the lattice constant given in the same call, whatever the order
    mlg.set_parameters(v=1.0e6, a=2.50)
    assert mlg.a == 2.50 and abs(mlg.velocity_from_hopping(mlg.gamma0, 2.50) - 1.0e6) < 1e-6
    before = mlg.get_parameters()
    with pytest.raises(KeyError):
        mlg.set_parameters(gamma1=0.5, nonsense=1.0)
    assert mlg.get_parameters() == before                  # nothing is set when a key is unknown


def test_stacking_sequences():
    assert mlg.stacking_sequence(5, "abc") == "ABCAB"
    assert mlg.stacking_sequence(4, "aba") == "ABAB"
    assert mlg.stacking_sequence(None, "abcb") == "ABCB"
    assert mlg.stacking_sequence(None, "AB") == "AB"
    assert mlg.stacking_sequence(1, "abc") == "A"
    with pytest.raises(ValueError):
        mlg.stacking_sequence(None, "AAB")
    with pytest.raises(ValueError):
        mlg.stacking_sequence(3, "ABCB")


@pytest.mark.parametrize("n_layers", [5, 8, 12])
def test_explicit_trilayers_ignore_the_configured_layer_number(n_layers):
    mlg.set_parameters(N_layers=n_layers, stacking="aba")
    # passed without N, 'ABC' and 'ABA' are the trilayers in any case ...
    for name, seq in (("ABC", "ABC"), ("ABA", "ABA"), ("abc", "ABC"), ("aba", "ABA")):
        assert mlg.stacking_sequence(None, name) == seq
        assert mlg.hamiltonian(0.1, 0.2, stacking_type=name).shape == (6, 6)
        assert mlg.kp_hamiltonian(0.01, 0.02, stacking_type=name).shape == (6, 6)
    K = mlg.K_point()
    trilayer = mlg.eigenvalues(K[0] + 0.02, K[1], None, "ABA")
    mlg.set_parameters(N_layers=3)
    assert np.array_equal(trilayer, mlg.eigenvalues(K[0] + 0.02, K[1], None, "ABA"))
    mlg.set_parameters(N_layers=n_layers)
    d = mlg.calculate_density_of_states((-0.2, 0.2), stacking_type="ABC", dk=4e-3, num_energy_points=10)
    assert d["parameters"]["N_layers"] == 3 and d["parameters"]["stacking"] == "ABC"
    # ... and the periodic patterns when a layer count comes with them
    assert mlg.stacking_sequence() == ("AB" * n_layers)[:n_layers]
    assert mlg.stacking_sequence(n_layers, "ABC") == ("ABC" * n_layers)[:n_layers]
    assert mlg.stacking_sequence(None, "rhombohedral") == ("ABC" * n_layers)[:n_layers]
    assert mlg.stacking_sequence(None, "Bernal") == ("AB" * n_layers)[:n_layers]
    assert mlg.hamiltonian(0.1, 0.2).shape == (2 * n_layers, 2 * n_layers)
    E, _ = mlg.calculate_bands(path_type="px")
    assert E.shape[1] == 2 * n_layers


def test_steps_and_dimer_bond_counts():
    assert list(mlg.stacking_steps("ABCB")) == [1, 1, -1]
    assert list(mlg.stacking_steps("ABAC")) == [1, -1, -1]
    assert mlg.dimer_bond_counts("ABAB").tolist() == [[1, 0], [0, 2], [2, 0], [0, 1]]
    assert mlg.dimer_bond_counts("ABCA").tolist() == [[1, 0], [1, 1], [1, 1], [0, 1]]
    assert mlg.dimer_bond_counts("ABCB").tolist() == [[1, 0], [1, 1], [0, 2], [1, 0]]
    assert mlg.dimer_bond_counts("ABAC").tolist() == [[1, 0], [0, 2], [1, 1], [1, 0]]


# ----------------------------------------------------------------------------
# Hamiltonian symmetries and the correspondence with the paper's k.p model
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seq", STACKS)
def test_hermitian_time_reversal_and_zone_periodicity(seq):
    rng = np.random.default_rng(1)
    k = rng.uniform(-2.0, 2.0, size=(5, 2))
    H = mlg.hamiltonian(k[:, 0], k[:, 1], stacking_type=seq)
    assert H.shape == (5, 2 * len(seq), 2 * len(seq))
    assert np.allclose(H, np.conj(np.swapaxes(H, -1, -2)))
    E = np.linalg.eigvalsh(H)
    assert np.allclose(E, np.linalg.eigvalsh(mlg.hamiltonian(-k[:, 0], -k[:, 1], stacking_type=seq)))
    b1, b2 = mlg.reciprocal_vectors()
    G = 2 * b1 - b2
    assert np.allclose(E, np.linalg.eigvalsh(mlg.hamiltonian(k[:, 0] + G[0], k[:, 1] + G[1], stacking_type=seq)))


@pytest.mark.parametrize("seq", STACKS)
@pytest.mark.parametrize("xi", [+1, -1])
def test_tight_binding_reduces_to_kp_near_K(seq, xi):
    K = mlg.K_point(xi)
    ang = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    for pm in (0.002, 0.004):
        px, py = pm * np.cos(ang), pm * np.sin(ang)
        E_tb = np.linalg.eigvalsh(mlg.hamiltonian(K[0] + px, K[1] + py, stacking_type=seq))
        E_kp = np.linalg.eigvalsh(mlg.kp_hamiltonian(px, py, stacking_type=seq, xi=xi))
        assert np.abs(E_tb - E_kp).max() < 2.0e-4            # second order in p a_cc


def _paper_tetralayer_hamiltonians(px, py, xi):
    """Eq. (1a)-(1e) of arXiv:2302.07374 written out block by block.

    Two misprints of the arXiv v1 are corrected: the layer-3 on-site term of
    H_ABCB is Delta'(1 - sigma_z) (site B3 carries two gamma1 bonds) and the
    (2,3) block of H_ABAC is V_AB^dagger (layers 1 and 3 are both 'A')."""
    p = mlg.get_parameters()
    hv = hbar_v()
    hv3 = 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma3"]
    hv4 = 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma4"]
    pi = xi * px + 1j * py
    Hg = np.array([[0.0, hv * np.conj(pi)], [hv * pi, 0.0]])
    V = np.array([[-hv4 * pi, p["gamma1"]], [-hv3 * np.conj(pi), -hv4 * pi]])
    Vd = V.conj().T
    W_ABA = np.diag([p["gamma5"] / 2, p["gamma2"] / 2])
    W_BAB = np.diag([p["gamma2"] / 2, p["gamma5"] / 2])
    W_ABC = np.array([[0.0, 0.0], [p["gamma2"] / 2, 0.0]])
    I2, sz, Z = np.eye(2), np.diag([1.0, -1.0]), np.zeros((2, 2))
    dp = p["Delta_prime"]
    H = {}
    H["ABAB"] = np.block([[Hg + dp / 2 * (I2 + sz), V, W_ABA, Z],
                          [Vd, Hg + dp * (I2 - sz), Vd, W_BAB],
                          [W_ABA.T, V, Hg + dp * (I2 + sz), V],
                          [Z, W_BAB.T, Vd, Hg + dp / 2 * (I2 - sz)]])
    H["ABCA"] = np.block([[Hg + dp / 2 * (I2 + sz), V, W_ABC, Z],
                          [Vd, Hg + dp * I2, V, W_ABC],
                          [W_ABC.T, Vd, Hg + dp * I2, V],
                          [Z, W_ABC.T, Vd, Hg + dp / 2 * (I2 - sz)]])
    H["ABCB"] = np.block([[Hg + dp / 2 * (I2 + sz), V, W_ABC, Z],
                          [Vd, Hg + dp * I2, V, W_ABA],
                          [W_ABC.T, Vd, Hg + dp * (I2 - sz), Vd],
                          [Z, W_ABA.T, V, Hg + dp / 2 * (I2 + sz)]])
    H["ABAC"] = np.block([[Hg + dp / 2 * (I2 + sz), V, W_ABA, Z],
                          [Vd, Hg + dp * (I2 - sz), Vd, W_ABC.T],
                          [W_ABA.T, V, Hg + dp * I2, Vd],
                          [Z, W_ABC, V, Hg + dp / 2 * (I2 + sz)]])
    return H


@pytest.mark.parametrize("xi", [+1, -1])
def test_kp_hamiltonian_equals_paper_equation_1(xi):
    rng = np.random.default_rng(7)
    for _ in range(4):
        px, py = rng.uniform(-0.1, 0.1, size=2)
        H_paper = _paper_tetralayer_hamiltonians(px, py, xi)
        for seq, H in H_paper.items():
            assert np.allclose(H, H.conj().T)
            assert np.allclose(mlg.kp_hamiltonian(px, py, stacking_type=seq, xi=xi), H, atol=1e-12)


def test_tight_binding_at_K_equals_kp_at_p0():
    for seq in STACKS:
        K = mlg.K_point(+1)
        assert np.allclose(mlg.hamiltonian(K[0], K[1], stacking_type=seq),
                           mlg.kp_hamiltonian(0.0, 0.0, stacking_type=seq), atol=1e-12)


# ----------------------------------------------------------------------------
# band edges at the valley centre
# ----------------------------------------------------------------------------

def _E_K(seq):
    K = mlg.K_point(+1)
    return np.linalg.eigvalsh(mlg.hamiltonian(K[0], K[1], stacking_type=seq))


def test_band_edges_bilayer_and_rhombohedral_trilayer():
    p = mlg.get_parameters()
    g1, g2, dp = p["gamma1"], p["gamma2"], p["Delta_prime"]
    assert np.allclose(_E_K("AB"), sorted([dp - g1, 0.0, 0.0, dp + g1]))
    # ABC: two isolated dimers at Delta' +- gamma1, B1-A3 coupled by gamma2/2
    assert np.allclose(_E_K("ABC"), sorted([dp - g1, dp - g1, -abs(g2) / 2, abs(g2) / 2, dp + g1, dp + g1]))


def test_band_edges_tetralayers_from_decoupled_chains():
    p = mlg.get_parameters()
    g1, g2, g5, dp = p["gamma1"], p["gamma2"], p["gamma5"], p["Delta_prime"]

    # ABAB: dimer chain A1-B2-A3-B4 (plus gamma5/2 across) and non-dimer pairs coupled by gamma2/2
    chain = np.array([[dp, g1, g5 / 2, 0.0],
                      [g1, 2 * dp, g1, g5 / 2],
                      [g5 / 2, g1, 2 * dp, g1],
                      [0.0, g5 / 2, g1, dp]])
    expect = np.concatenate([np.linalg.eigvalsh(chain), [-g2 / 2, g2 / 2, -g2 / 2, g2 / 2]])
    assert np.allclose(_E_K("ABAB"), np.sort(expect))
    # paper, Fig. 1(a): steps at +-gamma1/phi and +-phi gamma1 shifted by Delta' (and gamma5)
    phi = (1 + np.sqrt(5)) / 2
    E = _E_K("ABAB")
    assert abs(E[-1] - phi * g1) < 0.07 and abs(E[-2] - g1 / phi) < 0.03

    # ABCB: chain A2-B3-A4 with on-site (Delta', 2Delta', Delta'), analytic
    centre = 1.5 * dp + g5 / 4
    half = np.sqrt(2 * g1 ** 2 + (dp / 2 - g5 / 4) ** 2)
    E = _E_K("ABCB")
    assert abs(E[-1] - (centre + half)) < 1e-9 and abs(E[0] - (centre - half)) < 1e-9
    assert abs(E[-1] - (np.sqrt(2) * g1 + 1.5 * dp)) < 0.012      # label of Fig. 1(c)

    # ABCA: the middle dimer A2-B3 is exactly at Delta' +- gamma1
    E = _E_K("ABCA")
    assert np.any(np.abs(E - (dp + g1)) < 1e-9) and np.any(np.abs(E - (dp - g1)) < 1e-9)


# ----------------------------------------------------------------------------
# density of states: normalisation and analytic limits
# ----------------------------------------------------------------------------

def test_dos_monolayer_is_linear():
    d = mlg.calculate_density_of_states((-0.6, 0.6), stacking_type="A", model="kp",
                                        dk=1.0e-3, num_energy_points=120)
    analytic = 2.0 * np.abs(d["energy"]) / (np.pi * hbar_v() ** 2)   # spin and valley included
    sel = np.abs(d["energy"]) > 0.05
    assert np.abs(d["dos_per_area"][sel] / analytic[sel] - 1.0).max() < 2e-3
    d_tb = mlg.calculate_density_of_states((-0.6, 0.6), stacking_type="A", model="tb",
                                           dk=1.0e-3, num_energy_points=120)
    assert np.abs(d_tb["dos_per_area"][sel] / analytic[sel] - 1.0).max() < 0.02


def test_dos_bilayer_analytic_gamma1_only():
    mlg.set_parameters(gamma3=0.0, gamma4=0.0, gamma2=0.0, gamma5=0.0, Delta_prime=0.0)
    g1 = mlg.gamma1
    d = mlg.calculate_density_of_states((-0.9, 0.9), stacking_type="AB", model="kp",
                                        dk=1.0e-3, num_energy_points=180)
    E = np.abs(d["energy"])
    pref = 2.0 / (np.pi * hbar_v() ** 2)
    analytic = np.where(E < g1, pref * (E + g1 / 2), pref * 2 * E)
    sel = (np.abs(E - g1) > 0.02) & (E > 0.03)
    assert np.abs(d["dos_per_area"][sel] / analytic[sel] - 1.0).max() < 5e-3


@pytest.mark.parametrize("model,n", [("tb", 7), ("tb", 8), ("kp", 7)])
def test_valley_window_eigenvalues_use_the_mirror_symmetry(model, n):
    t = np.linspace(-0.05, 0.05, n)
    px, py = np.meshgrid(t, t, indexing="xy")
    K = mlg.K_point(-1) if model == "tb" else np.zeros(2)
    direct = mlg.eigenvalues(px + K[0], py + K[1], None, "ABCB", model=model, xi=-1)
    folded = mlg._valley_window_eigenvalues(t, None, "ABCB", model, xi=-1, max_bytes=5000)   # 4 k points per chunk
    assert folded.shape == direct.shape
    assert np.abs(folded - direct).max() < 1e-12


def test_dos_full_zone_counts_all_states():
    for seq, n_states in (("A", 4.0), ("AB", 8.0), ("ABCB", 16.0)):
        d = mlg.calculate_density_of_states((-12.0, 12.0), stacking_type=seq, region="bz",
                                            n_bz=120, num_energy_points=240)
        assert abs(d["cumulative_per_area"][-1] * mlg.unit_cell_area() - n_states) < 1e-9


def test_valley_window_must_not_overlap_the_other_valley():
    limit = mlg.valley_window_limit()
    K, Kp = mlg.K_point(+1), mlg.get_brillouin_zone(plot=False)["K_cart"][1]   # neighbouring corner, other valley
    assert abs(limit - 0.5 * abs(Kp[1] - K[1])) < 1e-12
    with pytest.raises(ValueError):
        mlg.calculate_density_of_states((-4.5, 4.5), stacking_type="A")
    with pytest.raises(ValueError):
        mlg.calculate_density_of_states((-0.5, 0.5), stacking_type="A", k_max=1.01 * limit)
    mlg.calculate_density_of_states((-0.5, 0.5), stacking_type="A", k_max=0.99 * limit, dk=0.02,
                                    num_energy_points=10)


def test_gaussian_smoothing_keeps_the_ends():
    y = np.full(200, 2.0) + np.linspace(0.0, 1.0, 200)
    smooth = mlg._gaussian_smooth(y, 0.05, 0.01)                   # kernel radius 20 points
    assert smooth.shape == y.shape
    assert np.allclose(smooth[20:-20], y[20:-20])                  # a straight line is unchanged
    assert abs(smooth[0] - y[0]) < 0.02 and abs(smooth[-1] - y[-1]) < 0.02    # zero padding would halve them
    spike = np.zeros(101)
    spike[50] = 1.0
    assert abs(mlg._gaussian_smooth(spike, 0.05, 0.01).sum() - 1.0) < 1e-12


def test_dos_window_and_csv_roundtrip(tmp_path):
    w = mlg.calculate_density_of_states_window(-0.1, 0.1, stacking_type="ABC", dk=2e-3)
    assert w["integrated_states_per_unit_cell"] > 0.0
    path = tmp_path / "dos.csv"
    d = mlg.plot_dos((-0.3, 0.3), stacking_type="ABC", dk=2e-3, num_energy_points=60,
                     show=False, return_data=True, csv_path=str(path))
    loaded = mlg.load_dos_from_csv(str(path))
    assert np.allclose(loaded["energy"], d["energy"])
    assert np.allclose(loaded["dos_per_cm2"], d["dos_per_cm2"], rtol=1e-8)
    assert loaded["metadata"]["stacking"] == "ABC"


# ----------------------------------------------------------------------------
# comparison with the DOS of McEllistrim et al., Fig. 1
# ----------------------------------------------------------------------------

def read_digitised(path):
    """Read a digitised-curve CSV: '#' comment lines, one header line, four columns."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("energy"):
                continue
            rows.append([float(v) for v in line.split(",")])
    return np.array(rows)


def load_paper_dos(seq, fig=1):
    data = read_digitised(os.path.join(DATA_DIR, f"mcellistrim2023_fig{fig}_{seq}_dos.csv"))
    return {"energy": data[:, 0], "dos": data[:, 1], "dos_min": data[:, 2], "dos_max": data[:, 3]}


def paper_feature_energies(paper, min_width=0.25e14, merge=0.02):
    """Energies of horizontal segments of the plotted curve (steps and van Hove peaks)."""
    width = paper["dos_max"] - paper["dos_min"]
    picked = []
    for i in np.argsort(width)[::-1]:
        if width[i] < min_width:
            break
        if all(abs(paper["energy"][i] - paper["energy"][j]) > merge for j in picked):
            picked.append(i)
    return np.sort(paper["energy"][picked])


def local_shift(E_model, dos_model, E_paper, dos_paper, e0, half=0.05, s_max=0.03, step=0.001):
    """Energy shift that maximises the correlation of the two curves around e0."""
    grid = np.arange(e0 - half, e0 + half + 1e-9, step)
    p = np.interp(grid, E_paper, dos_paper)
    p = p - p.mean()
    best = None
    for s in np.arange(-s_max, s_max + 1e-9, step):
        m = np.interp(grid + s, E_model, dos_model)
        m = m - m.mean()
        c = (m * p).sum() / np.sqrt((m * m).sum() * (p * p).sum() + 1e-30)
        if best is None or c > best[0]:
            best = (c, s)
    return best


FIG7_STACKS = ["ABC", "ABCAB", "ABACA", "ABCAC", "ABABC", "ABCBA", "ABABA"]


@pytest.mark.parametrize("seq,fig", [(s, 1) for s in PAPER_STACKS] + [(s, 7) for s in FIG7_STACKS])
def test_dos_features_match_paper(seq, fig):
    """Every step or van Hove singularity of the published curves (|E| > 30 meV) is
    reproduced at the same energy: the local cross-correlation of model and paper
    is maximal for a shift below 10 meV (Fig. 1, 1.8 meV per pixel) or 12 meV
    (Fig. 7, 7.5 meV per pixel)."""
    paper = load_paper_dos(seq, fig)
    e_lim = np.abs(paper["energy"]).max() + 0.03
    d = mlg.calculate_density_of_states((-e_lim, e_lim), stacking_type=seq, model="kp",
                                        dk=1.0e-3, num_energy_points=int(round(2 * e_lim / 0.001)))
    found = paper_feature_energies(paper, min_width=0.25e14 if len(seq) <= 4 else 0.6e14)
    found = found[np.abs(found) > 0.03]
    if fig == 1:
        assert found.size >= 2
    tol = 0.010 if fig == 1 else 0.012
    for e0 in found:
        corr, shift = local_shift(d["energy"], d["dos_per_cm2"], paper["energy"], paper["dos"], e0)
        assert corr > 0.8 and abs(shift) <= tol, (seq, fig, e0, shift, corr)


@pytest.mark.parametrize("seq", ["ABC", "ABCAB", "ABABA"])
def test_dos_magnitude_matches_paper_fig7(seq):
    """Trilayer and pentalayer curves of Fig. 7 (coarse digitisation, 1 px = 7.5 meV):
    the model is within 2-6 % of the published values away from singularities."""
    paper = load_paper_dos(seq, 7)
    e_lim = np.abs(paper["energy"]).max() + 0.03
    d = mlg.calculate_density_of_states((-e_lim, e_lim), stacking_type=seq, model="kp",
                                        dk=1.5e-3, num_energy_points=400)
    for e in (-0.6, -0.5, 0.5, 0.6):
        mine = np.interp(e, d["energy"], d["dos_per_cm2"])
        theirs = np.interp(e, paper["energy"], paper["dos"])
        assert abs(mine / theirs - 1.0) < 0.08, (seq, e, mine, theirs)


def test_paper_fig7_aba_panel_is_a_factor_two_low():
    """The ABA trilayer panel of Fig. 7 is plotted on a 10^13 scale and lies a factor
    of two below the model (and below the other trilayer panel); the file header says so."""
    paper = load_paper_dos("ABA", 7)
    d = mlg.calculate_density_of_states((-0.78, 0.78), stacking_type="ABA", model="kp",
                                        dk=1.5e-3, num_energy_points=400)
    ratios = [np.interp(e, d["energy"], d["dos_per_cm2"]) / np.interp(e, paper["energy"], paper["dos"])
              for e in (-0.6, -0.5, 0.5, 0.6)]
    assert all(abs(r / 2.0 - 1.0) < 0.08 for r in ratios), ratios


REFERENCE_ENERGIES = (-0.7, -0.5, 0.5, 0.7)


@pytest.mark.parametrize("seq", PAPER_STACKS)
def test_dos_magnitude_matches_paper_fig1(seq):
    """With the paper's printed parameters the k.p DOS lies 1-3 % (electrons) and
    6-9 % (holes) above the digitised curves of Fig. 1; the tight-binding DOS is a
    further 2 % higher (median deviation over the window: 5-8 % for k.p, 7-9 % for
    tight binding).  The hole-side excess is the gamma4 asymmetry that Fig. 1 lacks
    (see test_fig1_lacks_the_gamma4_asymmetry).  The tolerances are set accordingly."""
    paper = load_paper_dos(seq)
    results = {}
    for model in ("kp", "tb"):
        d = mlg.calculate_density_of_states((-0.8, 0.8), stacking_type=seq, model=model,
                                            dk=1.5e-3, num_energy_points=400)
        results[model] = d
        for e in REFERENCE_ENERGIES:
            mine = np.interp(e, d["energy"], d["dos_per_cm2"])
            theirs = np.interp(e, paper["energy"], paper["dos"])
            assert abs(mine / theirs - 1.0) < (0.10 if model == "kp" else 0.12), (seq, model, e, mine, theirs)
        # smooth parts of the curve: median relative deviation over the window
        sel = (np.abs(d["energy"]) > 0.05) & (np.abs(d["energy"]) < 0.75)
        theirs = np.interp(d["energy"][sel], paper["energy"], paper["dos"])
        rel = np.abs(d["dos_per_cm2"][sel] / theirs - 1.0)
        assert np.median(rel) < (0.08 if model == "kp" else 0.10), (seq, model, np.median(rel))
    # tight binding and k.p agree closely inside the window
    for e in REFERENCE_ENERGIES:
        a = np.interp(e, results["tb"]["energy"], results["tb"]["dos_per_cm2"])
        b = np.interp(e, results["kp"]["energy"], results["kp"]["dos_per_cm2"])
        assert abs(a / b - 1.0) < 0.03


def _hole_electron_ratio(energy, dos, e=0.5):
    return np.interp(-e, energy, dos) / np.interp(e, energy, dos)


@pytest.mark.parametrize("seq,fig", [(s, 1) for s in PAPER_STACKS] + [("ABC", 7), ("ABCAB", 7), ("ABABA", 7)])
def test_fig1_lacks_the_gamma4_asymmetry(seq, fig):
    """The hole/electron DOS ratio at +-0.5 eV of the Fig. 1 curves is that of the model
    with gamma4 = 0 (within 0.015); the Fig. 7 curves agree with the printed v4 instead."""
    paper = load_paper_dos(seq, fig)
    measured = _hole_electron_ratio(paper["energy"], paper["dos"])
    ratio = {}
    for name, gamma4 in (("printed", mlg.gamma4), ("none", 0.0)):
        mlg.set_parameters(gamma4=gamma4)
        d = mlg.calculate_density_of_states((-0.8, 0.8), stacking_type=seq, model="kp", dk=1.5e-3,
                                            num_energy_points=400)
        ratio[name] = _hole_electron_ratio(d["energy"], d["dos_per_cm2"])
    mlg.use_paper_parameters()
    assert ratio["printed"] - ratio["none"] > 0.05                  # gamma4 makes the holes heavier
    if fig == 1:
        assert abs(measured - ratio["none"]) < 0.015, (seq, measured, ratio)
    else:
        assert abs(measured - ratio["printed"]) < abs(measured - ratio["none"]), (seq, measured, ratio)
