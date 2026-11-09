"""
Tests of the stacking domain-wall model (stacking_wall.py): the k.p
decomposition, the wall profile and its Fourier series, the plane-wave
Hamiltonian, the LDOS normalisation (graphene in closed form, the uniform
six-layer stacks against the DOS routine of multilayer_graphene) and the
analysis of a synthetic line spectrum.

Run with:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multilayer_graphene as mlg  # noqa: E402
import stacking_wall as sw  # noqa: E402


@pytest.fixture(autouse=True)
def paper_parameters():
    mlg.use_paper_parameters()
    yield
    mlg.use_paper_parameters()


def hbar_v():
    p = mlg.get_parameters()
    return 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma0"]


# ----------------------------------------------------------------------------
# k.p blocks
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seq", [sw.SEQ_RG, sw.SEQ_SF, "AB", "ABCB"])
def test_kp_blocks_reproduce_the_kp_hamiltonian(seq):
    rng = np.random.default_rng(1)
    for xi in (+1, -1):
        C, X, Y = sw.kp_blocks(seq, xi=xi)
        for M in (C, X, Y):
            assert np.allclose(M, M.conj().T)
        for _ in range(4):
            px, py = 0.1 * rng.normal(size=2)
            H = mlg.kp_hamiltonian(px, py, stacking_type=seq, xi=xi)
            assert np.allclose(H, C + px * X + py * Y, atol=1e-12)


def test_difference_blocks_touch_only_the_three_top_layers():
    blocks_rg, blocks_d = sw._mixed_blocks()
    C_rg, X_rg, Y_rg = blocks_rg
    for D in blocks_d:
        assert np.allclose(D[:6, :], 0.0) and np.allclose(D[:, :6], 0.0)
    C_sf, X_sf, Y_sf = sw.kp_blocks(sw.SEQ_SF)
    assert np.allclose(C_rg + blocks_d[0], C_sf)
    # the gamma1 dimer moves from A5-B6 to B5-A6
    g1 = mlg.get_parameters()["gamma1"]
    assert np.isclose(C_rg[8, 11], g1) and np.isclose(C_rg[9, 10], 0.0)
    assert np.isclose(C_sf[9, 10], g1) and np.isclose(C_sf[8, 11], 0.0)


def test_layer_potential_requires_one_value_per_layer():
    with pytest.raises(ValueError):
        sw._mixed_blocks(layer_potential=[0.1, 0.2])
    (C, _, _), _ = sw._mixed_blocks(layer_potential=[0.0, 0.0, 0.0, 0.0, 0.0, 0.3])
    assert np.isclose(C[10, 10] - sw.kp_blocks(sw.SEQ_RG)[0][10, 10], 0.3)


# ----------------------------------------------------------------------------
# profile and Fourier series
# ----------------------------------------------------------------------------

def test_wall_profile_limits_and_periodicity():
    L, ell = 2000.0, 30.0
    x = np.linspace(-L / 2, L / 2, 4001)
    s = sw.wall_profile(x, ell, L)
    assert np.isclose(sw.wall_profile(0.0, ell, L), 0.5)
    assert np.isclose(sw.wall_profile(-L / 2, ell, L), 0.5)
    assert np.isclose(sw.wall_profile(-L / 4, ell, L), 1.0, atol=1e-12)
    assert np.isclose(sw.wall_profile(L / 4, ell, L), 0.0, atol=1e-12)
    assert np.allclose(sw.wall_profile(x + L, ell, L), s)
    assert np.isclose(sw.tanh_step(2.0, 2.0, 1.0, 0.0, 4.0), 2.0)
    strip = 300.0
    s2 = sw.wall_profile(x, ell, L, strip=strip)
    assert np.isclose(sw.wall_profile(-strip / 2, ell, L, strip), 1.0, atol=1e-3)
    assert np.isclose(np.trapezoid(s2, x) / L, strip / L, rtol=1e-3)


def test_fourier_series_reconstructs_the_profile():
    L, ell = 1500.0, 25.0
    n_max = 60
    coeff = sw.fourier_coefficients(lambda x: sw.wall_profile(x, ell, L), L, n_max)
    assert coeff.size == 2 * n_max + 1
    assert np.isclose(coeff[n_max], 0.5)                       # mean of s
    assert np.allclose(coeff[::-1], np.conj(coeff))            # real profile
    x = np.linspace(-L / 2, L / 2, 701)
    k = np.arange(-n_max, n_max + 1)
    rec = (coeff[None, :] * np.exp(2j * np.pi * np.outer(x, k) / L)).sum(axis=1)
    assert np.abs(rec.imag).max() < 1e-12
    assert np.abs(rec.real - sw.wall_profile(x, ell, L)).max() < 1e-4


# ----------------------------------------------------------------------------
# plane-wave Hamiltonian
# ----------------------------------------------------------------------------

def test_wall_hamiltonian_is_hermitian_and_reduces_to_bands_without_a_wall():
    L, p_cut = 900.0, 0.08
    n_max = int(np.floor(p_cut * L / (2 * np.pi)))
    q = 2 * np.pi * np.arange(-n_max, n_max + 1) / L
    blocks_rg, blocks_d = sw._mixed_blocks()
    s_hat = sw.fourier_coefficients(lambda x: sw.wall_profile(x, 20.0, L), L, 2 * n_max)
    H = sw.wall_hamiltonian(0.02, s_hat, q, blocks_rg, blocks_d)
    assert H.shape == (12 * q.size,) * 2
    assert np.allclose(H, H.conj().T)
    # s = 0: block diagonal, eigenvalues are the (+5) k.p bands at (q_n, k_y)
    zero = np.zeros_like(s_hat)
    H0 = sw.wall_hamiltonian(0.02, zero, q, blocks_rg, blocks_d)
    bands = np.sort(np.linalg.eigvalsh(mlg.kp_hamiltonian(q, np.full_like(q, 0.02), stacking_type=sw.SEQ_RG)).ravel())
    assert np.allclose(np.linalg.eigvalsh(H0), bands, atol=1e-10)
    # s = 1: the (+4-) bands
    one = np.zeros_like(s_hat)
    one[2 * n_max] = 1.0
    H1 = sw.wall_hamiltonian(0.02, one, q, blocks_rg, blocks_d)
    bands = np.sort(np.linalg.eigvalsh(mlg.kp_hamiltonian(q, np.full_like(q, 0.02), stacking_type=sw.SEQ_SF)).ravel())
    assert np.allclose(np.linalg.eigvalsh(H1), bands, atol=1e-10)


# ----------------------------------------------------------------------------
# LDOS normalisation
# ----------------------------------------------------------------------------

def test_graphene_ldos_in_closed_form():
    """A single layer without a wall: LDOS = 2|E| / (pi (hbar v)^2) per area, spin and valleys included."""
    E = np.array([-0.3, -0.2, 0.2, 0.3])
    # the plane-wave spacing 2 pi / L must be fine against sigma / (hbar v): L = 4800 A gives 1e-4
    res = sw.ldos_wall(length=4800.0, p_cut=0.08, ky_max=0.08, dky=0.001, energy=E, sigma=0.005,
                       x_out=np.array([-100.0, 0.0, 250.0]), profile=lambda x: np.zeros_like(x),
                       n_proc=1, seq_rg="A", seq_sf="A")
    ldos = res["ldos"][0]
    assert ldos.shape == (E.size, 3)
    assert np.allclose(ldos, ldos[:, :1], rtol=1e-8)
    expected = 2.0 * np.abs(E) / (np.pi * hbar_v() ** 2)
    assert np.allclose(ldos[:, 0], expected, rtol=0.005)


def test_uniform_six_layer_ldos_matches_the_dos_routine():
    E = np.linspace(-0.25, 0.25, 101)
    res = sw.ldos_wall(length=600.0, p_cut=0.1, ky_max=0.1, dky=0.002, energy=E, sigma=0.01,
                       x_out=np.linspace(-200.0, 200.0, 9), profile=lambda x: np.zeros_like(x), n_proc=1)
    total = res["ldos"].sum(axis=0)
    assert np.allclose(total, total[:, :1], rtol=1e-8)        # no wall: x independent
    dos = mlg.calculate_density_of_states((-0.4, 0.4), N=6, stacking_type=sw.SEQ_RG, model="kp", dk=0.002)
    reference = np.interp(E, dos["energy"], dos["dos_per_area"])
    ratio = np.trapezoid(total[:, 0], E) / np.trapezoid(reference, E)
    assert abs(ratio - 1.0) < 0.05


def test_wall_ldos_approaches_the_two_uniform_stacks():
    E = np.linspace(-0.2, 0.2, 41)
    common = dict(length=1200.0, p_cut=0.1, ky_max=0.1, dky=0.0025, energy=E, sigma=0.01, n_proc=1)
    x = np.linspace(-450.0, 450.0, 181)
    res = sw.ldos_wall(wall_length=20.0, x_out=x, **common)
    assert res["ldos"].shape == (6, E.size, x.size)
    assert np.isclose(res["s"][np.argmin(np.abs(x))], 0.5)
    uni_sf = sw.ldos_wall(x_out=np.array([0.0]), profile=lambda t: np.ones_like(t), **common)["ldos"][-1][:, 0]
    uni_rg = sw.ldos_wall(x_out=np.array([0.0]), profile=lambda t: np.zeros_like(t), **common)["ldos"][-1][:, 0]
    top = res["ldos"][-1]
    left = top[:, (x > -400) & (x < -200)].mean(axis=1)      # (+4-) plateau, away from both walls
    right = top[:, (x > 200) & (x < 400)].mean(axis=1)       # (+5) plateau
    # standing waves of the long-wavelength states near E = 0 do not average out over 200 A
    sel = np.abs(E) >= 0.1
    assert np.abs(left[sel] / uni_sf[sel] - 1.0).max() < 0.1
    assert np.abs(right[sel] / uni_rg[sel] - 1.0).max() < 0.1
    # the flat band of the (+5) surface is absent on the (+4-) surface
    i0 = np.argmin(np.abs(E - sw.flat_band_energy(E, uni_rg)))
    assert uni_rg[i0] > 5.0 * uni_sf[i0]
    assert right[i0] > 5.0 * left[i0]


def test_save_and_load_roundtrip(tmp_path):
    E = np.linspace(-0.1, 0.1, 5)
    res = sw.ldos_wall(length=400.0, p_cut=0.05, ky_max=0.02, dky=0.01, energy=E, sigma=0.02,
                       x_out=np.array([-50.0, 50.0]), n_proc=1)
    path = tmp_path / "wall.npz"
    sw.save_ldos(path, res)
    back = sw.load_ldos(path)
    assert np.allclose(back["ldos"], res["ldos"]) and np.allclose(back["x"], res["x"])
    assert back["wall_length"] == res["wall_length"] and back["seq_sf"] == sw.SEQ_SF
    assert np.isclose(back["params"]["gamma1"], 0.39)


# ----------------------------------------------------------------------------
# analysis of a line spectrum
# ----------------------------------------------------------------------------

def test_mixing_fraction_and_fit_recover_a_synthetic_wall():
    bias = np.linspace(-0.5, 0.5, 201)
    dist = np.linspace(0.0, 100.0, 256)
    S_sf = 1.0 + 2.0 * np.abs(bias)
    S_rg = S_sf + 3.0 * np.exp(-0.5 * ((bias - 0.02) / 0.02) ** 2)
    s_true = sw.tanh_step(dist, 42.0, 6.0)
    rng = np.random.default_rng(3)
    didv = (1 - s_true)[None, :] * S_sf[:, None] + s_true[None, :] * S_rg[:, None]
    didv += 0.02 * rng.normal(size=didv.shape)
    s, S_a, S_b = sw.mixing_fraction(bias, dist, didv, x_sf=20.0, x_rg=65.0)
    assert np.abs(s - s_true).max() < 0.05
    fit = sw.fit_wall_profile(dist, s, x0_guess=40.0)
    assert abs(fit["x0"] - 42.0) < 0.3 and abs(fit["ell"] - 6.0) < 0.3
    assert np.isclose(fit["width_10_90"], 2.0 * np.arctanh(0.8) * fit["ell"])
    assert np.isclose(sw.flat_band_energy(bias, S_b), 0.02, atol=0.006)


def test_uniform_layer_potential_profile_equals_layer_potential():
    E = np.linspace(-0.15, 0.15, 13)
    common = dict(length=600.0, p_cut=0.08, ky_max=0.06, dky=0.003, energy=E, sigma=0.01,
                  x_out=np.array([-100.0, 0.0, 120.0]), n_proc=1, wall_length=20.0)
    V = np.array([0.03, 0.01, 0.0, -0.01, -0.02, -0.04])
    a = sw.ldos_wall(layer_potential=V, **common)
    b = sw.ldos_wall(layer_potential_profile=lambda x: np.repeat(V[:, None], np.size(x), axis=1), **common)
    assert b["layer_potential_profile"] is True
    assert np.allclose(a["ldos"], b["ldos"], rtol=1e-8, atol=1e-12)
    # a profile that is not uniform changes the result and keeps the Hamiltonian Hermitian
    c = sw.ldos_wall(layer_potential_profile=lambda x: V[:, None] * np.cos(2 * np.pi * np.asarray(x) / 600.0)[None, :],
                     **common)
    assert not np.allclose(a["ldos"], c["ldos"])


def test_layer_order_is_bottom_first():
    """The k.p arrays are bottom layer first: an on-site energy on the last index moves the flat
    band of the top-layer LDOS ldos[-1], one on index 0 barely does (the bandbending engine
    numbers the layers the other way, so its potentials must be reversed)."""
    E = np.arange(-0.1, 0.1 + 1e-9, 0.0025)
    common = dict(length=600.0, p_cut=0.06, ky_max=0.06, dky=0.004, energy=E, sigma=0.005,
                  x_out=np.array([0.0]), n_proc=1, profile=lambda t: np.zeros_like(t))

    def top_peak(V):
        return sw.flat_band_energy(E, sw.ldos_wall(layer_potential=V, **common)["ldos"][-1][:, 0])

    e0 = top_peak(None)
    e_top = top_peak([0.0, 0.0, 0.0, 0.0, 0.0, -0.05])
    e_bottom = top_peak([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert e_top < e0 - 0.02
    assert abs(e_bottom - e0) < 0.006


def test_lateral_potential_profile_is_periodic_and_reduces_to_the_columns():
    """The k.p on-site profile built from a lateral solution: periodic on the cell, equal to
    the lateral layer potentials (minus mu0) inside the solved box, to the columns of the
    local stack outside it, and continuous at the box ends."""
    pytest.importorskip("bandbending")
    import stacking_wall_lateral as swl
    ell, L, mu0 = 8.0, 300.0, -0.02
    s_grid = np.asarray(swl.S_GRID)
    cols = np.stack([0.03 * s_grid ** 2 * (1 + 0.1 * l) for l in range(6)], axis=1)            # (n_s, 6)
    x_box = np.linspace(-60.0, 60.0, 241)
    s_box = 0.5 * (1.0 - np.tanh(x_box / ell))
    col_box = np.array([np.interp(s_box, s_grid, cols[:, l]) for l in range(6)])
    U_box = col_box + 0.01 * np.exp(-0.5 * (x_box / 5.0) ** 2)[None, :] * np.exp(-0.5 * np.arange(6))[:, None]
    f = swl.potential_profile(x_box, U_box, cols, ell, L, mu0)
    xa = np.linspace(-1500.0, 1500.0, 1201)
    V = f(xa)
    assert V.shape == (6, xa.size)
    assert np.allclose(f(xa + 10.0 * L), V) and np.allclose(f(xa - 10.0 * L), V)
    inside = f(10.0 * x_box)
    assert np.allclose(inside, (U_box - mu0)[::-1], atol=1e-12)                # k.p order: bottom layer first
    far = np.array([-1400.0, -900.0, 900.0, 1400.0])
    s_far = sw.wall_profile(far, 10.0 * ell, 10.0 * L)
    expect = np.array([np.interp(s_far, s_grid, cols[:, l]) for l in range(6)]) - mu0
    assert np.allclose(f(far), expect[::-1], atol=1e-12)
    # continuous across the box ends (the lateral correction vanishes there)
    assert np.max(np.abs(f(np.array([600.0 - 1e-6])) - f(np.array([600.0 + 1e-6])))) < 1e-6
