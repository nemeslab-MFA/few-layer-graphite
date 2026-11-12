"""
Tests of the bulk (infinite-stack) SWMcC model against analytic limits and
against the finite-stack Hamiltonian of multilayer_graphene.

Run with:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multilayer_graphene as mlg  # noqa: E402
import bulk_graphite as bg  # noqa: E402

GAMMA01_ONLY = dict(gamma2=0.0, gamma3=0.0, gamma4=0.0, gamma5=0.0, Delta_prime=0.0)
FAST = dict(n_kz=24, dk=4.0e-3, refine=4, num_energy_points=500)


@pytest.fixture(autouse=True)
def paper_parameters():
    mlg.use_paper_parameters()
    mlg.set_parameters(N_layers=5)          # not 3: nothing here may depend on it
    yield
    mlg.use_paper_parameters()
    mlg.set_parameters(N_layers=3)


def hbar_v():
    p = mlg.get_parameters()
    return 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma0"]


# ----------------------------------------------------------------------------
# stacking bookkeeping and the Bloch Hamiltonian
# ----------------------------------------------------------------------------

def test_bulk_period():
    assert bg.bulk_period("bernal") == ("AB", 2)
    assert bg.bulk_period("aba") == ("AB", 2)
    assert bg.bulk_period("rhombohedral") == ("ABC", 1)
    assert bg.bulk_period("ABCABC") == ("ABCABC", 1)
    assert bg.bulk_period("ABAB") == ("ABAB", 2)
    assert bg.bulk_period("ABCB") == ("ABCB", 4)
    for bad in ("A", "ABCA", "AAB", "ABD"):
        with pytest.raises(ValueError):
            bg.bulk_period(bad)


def test_bernal_KH_line_is_slonczewski_weiss_mcclure():
    p = mlg.get_parameters()
    K = mlg.K_point()
    for phase in (0.0, 0.3, 1.0, np.pi / 2):
        G = 2.0 * np.cos(phase)
        dimer = 2.0 * p["Delta_prime"] - p["gamma5"] + 0.5 * p["gamma5"] * G ** 2
        free = -p["gamma2"] + 0.5 * p["gamma2"] * G ** 2
        expected = np.sort([dimer + p["gamma1"] * G, dimer - p["gamma1"] * G, free, free])
        E = bg.bulk_eigenvalues(K[0], K[1], phase / bg.D_LAYER, "ab")
        assert np.abs(E - expected).max() < 1e-12


@pytest.mark.parametrize("stacking,N", [("ab", 12), ("abc", 12), ("ABCB", 16)])
def test_bulk_bloch_states_solve_the_finite_stack_in_its_interior(stacking, N):
    seq, p = bg.bulk_period(stacking)
    rng = np.random.default_rng(3)
    kx, ky = mlg.K_point() + rng.normal(scale=0.05, size=2)
    kz = rng.uniform(-1.0, 1.0)
    H_bulk = bg.bulk_hamiltonian(kx, ky, kz, stacking)
    assert np.abs(H_bulk - H_bulk.conj().T).max() < 1e-14
    energies, vectors = np.linalg.eigh(H_bulk)
    H_stack = mlg.hamiltonian(kx, ky, stacking_type=(seq * N)[:N])
    for b in range(2 * p):
        psi = np.concatenate([np.exp(1j * kz * bg.D_LAYER * n) * vectors[2 * (n % p):2 * (n % p) + 2, b]
                              for n in range(N)])
        residual = H_stack @ psi - energies[b] * psi
        assert np.abs(residual[4:-4]).max() < 1e-12      # the two outer layers feel the surfaces


@pytest.mark.parametrize("stacking", ["ab", "abc", "ABCB"])
def test_kz_period_and_mirror_symmetry(stacking):
    _, p = bg.bulk_period(stacking)
    K = mlg.K_point()
    px, py, kz = 0.03, -0.02, 0.21
    E = bg.bulk_eigenvalues(K[0] + px, K[1] + py, kz, stacking)
    E_period = bg.bulk_eigenvalues(K[0] + px, K[1] + py, kz + 2.0 * np.pi / (p * bg.D_LAYER), stacking)
    E_mirror = bg.bulk_eigenvalues(K[0] + px, K[1] - py, -kz, stacking)
    assert np.abs(E - E_period).max() < 1e-12
    assert np.abs(E - E_mirror).max() < 1e-12


def test_rhombohedral_dirac_points_lie_on_a_spiral():
    mlg.set_parameters(**GAMMA01_ONLY)
    K = mlg.K_point()
    radius = mlg.gamma1 / hbar_v()
    for phase in (0.0, 1.0, 2.5, 4.0):
        angle = np.linspace(0.0, 2.0 * np.pi, 3601)
        E = bg.bulk_eigenvalues(K[0] + radius * np.cos(angle), K[1] + radius * np.sin(angle),
                                phase / bg.D_LAYER, "abc")
        gap = E[:, 1] - E[:, 0]
        assert gap.min() < 0.03                           # f(k) is not exactly linear at this radius
        assert abs(np.angle(np.exp(1j * (angle[np.argmin(gap)] + phase - np.pi)))) < 0.05


# ----------------------------------------------------------------------------
# density of states
# ----------------------------------------------------------------------------

def test_dos_rhombohedral_gamma0_gamma1_is_that_of_graphene():
    d = bg.calculate_bulk_density_of_states((-0.5, 0.5), "abc", params=GAMMA01_ONLY, **FAST)
    E = d["energy"]
    analytic = mlg.unit_cell_area() * np.abs(E) / (np.pi * hbar_v() ** 2)     # per atom, spin and valley
    sel = np.abs(E) > 0.05
    assert np.abs(d["dos_per_atom"][sel] / analytic[sel] - 1.0).max() < 0.03
    assert abs(d["fermi_energy"]) < 2e-3
    assert np.allclose(d["dos_per_cm3"], d["dos_per_atom"] * bg.atomic_density() * 1e24)


def test_dos_bernal_gamma0_gamma1_is_the_kz_average_of_bilayers():
    d = bg.calculate_bulk_density_of_states((-0.5, 0.5), "ab", params=GAMMA01_ONLY, **FAST)
    E = np.abs(d["energy"])[:, None]
    phase = (np.arange(2000) + 0.5) * np.pi / 4000.0
    g = 2.0 * mlg.gamma1 * np.cos(phase)[None, :]         # bilayer coupling gamma1 Gamma at each k_z
    slab = np.where(E < g, E + 0.5 * g, 2.0 * E).mean(axis=1) * 2.0 / (np.pi * hbar_v() ** 2)
    analytic = slab * mlg.unit_cell_area() / 4.0
    sel = E[:, 0] > 0.02
    assert np.abs(d["dos_per_atom"][sel] / analytic[sel] - 1.0).max() < 0.04
    assert abs(d["fermi_energy"]) < 2e-3


def test_fermi_surfaces_of_the_full_model():
    hexagonal = bg.calculate_bulk_density_of_states((-0.5, 0.5), "ab", **FAST)
    rhombohedral = bg.calculate_bulk_density_of_states((-0.5, 0.5), "abc", **FAST)

    def dos_at_fermi(d):
        return np.interp(d["fermi_energy"], d["energy"], d["dos_per_atom"])

    # Bernal graphite: a semimetal with about 3e18 carriers per cm^3 of each sign
    assert 1.0e18 < hexagonal["carrier_density_cm3"] < 5.0e18
    assert 3.0e-3 < dos_at_fermi(hexagonal) < 8.0e-3
    assert abs(hexagonal["fermi_energy"]) < 0.02
    # rhombohedral graphite: Dirac points at Delta' + 2 gamma1 gamma4 / gamma0, hardly any carriers
    p = mlg.get_parameters()
    dirac = p["Delta_prime"] + 2.0 * p["gamma1"] * p["gamma4"] / p["gamma0"]
    assert abs(rhombohedral["fermi_energy"] - dirac) < 3e-3
    assert rhombohedral["carrier_density_cm3"] < 0.05 * hexagonal["carrier_density_cm3"]
    assert dos_at_fermi(rhombohedral) < 0.25 * dos_at_fermi(hexagonal)


def test_refined_dos_matches_a_uniformly_fine_mesh():
    # one k_z value of rhombohedral graphite: a Dirac cone with its apex near 42 meV
    common = dict(stacking_type="abc", n_kz=1, energy_range=(-0.3, 0.3), num_energy_points=300,
                  k_max=0.16, fermi_guess=0.042)
    refined = bg.calculate_bulk_density_of_states(dk=4.0e-3, refine=8, **common)
    coarse = bg.calculate_bulk_density_of_states(dk=4.0e-3, refine=1, **common)
    fine = bg.calculate_bulk_density_of_states(dk=5.0e-4, refine=1, **common)
    x = np.abs(refined["energy"] - 0.042)
    inside, outside = x < 0.09, x > 0.13
    # subdividing reproduces the fine mesh inside the refinement window, apex included ...
    assert np.abs(refined["dos_per_atom"][inside] - fine["dos_per_atom"][inside]).max() \
        < 2e-3 * fine["dos_per_atom"][inside].max()
    assert np.abs(coarse["dos_per_atom"][x < 0.01] / fine["dos_per_atom"][x < 0.01] - 1.0).max() > 0.2
    # ... and leaves the coarse result untouched outside, without a step at the border
    assert np.array_equal(refined["dos_per_atom"][outside], coarse["dos_per_atom"][outside])
    assert np.abs(refined["dos_per_atom"][x > 0.02] / fine["dos_per_atom"][x > 0.02] - 1.0).max() < 5e-3
