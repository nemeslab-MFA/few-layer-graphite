"""
Tests of the bookkeeping in graphite_ers_background.py: Fermi level of a slab,
Raman shift to energy mapping and the tables written to disk.

Run with:  python -m pytest tests -q
"""
import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multilayer_graphene as mlg  # noqa: E402
import graphite_ers_background as ers  # noqa: E402


@pytest.fixture(autouse=True)
def paper_parameters():
    mlg.use_paper_parameters()
    yield
    mlg.use_paper_parameters()


def test_slab_dos_is_per_atom_and_centred_on_the_neutrality_point():
    # bilayer with gamma0 and gamma1 only: electron-hole symmetric, DOS known in closed form
    params = {**mlg.get_parameters(), "gamma2": 0.0, "gamma3": 0.0, "gamma4": 0.0, "gamma5": 0.0,
              "Delta_prime": 0.0, "E0": 0.03}
    energy = np.arange(-0.3, 0.3005, 1.0e-3)
    dos, fermi = ers._slab_dos((2, "ab", energy, 2.0e-3, 0.2, params))
    assert abs(fermi - 0.03) < 1e-3                       # the rigid offset E0 is found and removed
    hv = 1.5 * (params["a"] / np.sqrt(3.0)) * params["gamma0"]
    analytic = (np.abs(energy) + 0.5 * params["gamma1"]) * 2.0 / (np.pi * hv ** 2) * mlg.unit_cell_area() / 4.0
    sel = np.abs(energy) > 0.03
    assert np.abs(dos[sel] / analytic[sel] - 1.0).max() < 0.03


def test_raman_shift_reads_the_dos_at_half_the_energy():
    energy = np.linspace(-1.0, 1.0, 2001)
    table = {"energy": energy, "ab": 0.5 + np.abs(energy), "abc": 3.0 * np.abs(energy)}
    shift = 0.2 / ers.EV_PER_INVERSE_CM                   # hbar w = 0.2 eV -> |E - E_F| = 0.1 eV
    hexagonal, rhombohedral = ers.dos_at_shift(table, shift)
    assert abs(hexagonal - 0.6) < 1e-9 and abs(rhombohedral - 0.3) < 1e-9
    # the curves cross where 0.5 + E = 3 E, at E = 0.25 eV
    assert abs(ers.crossing_shift(table) * ers.EV_PER_INVERSE_CM - 0.5) < 2.0 * 10.0 * ers.EV_PER_INVERSE_CM
    rows = ers.shift_rows(table, 80.0, 6200.0)
    assert rows[0][0] == 80.0 and rows[-1][0] == 6200.0
    assert abs(ers.window_mean(energy, table["abc"], 80.0, 6200.0)
               - 3.0 * 0.25 * (80.0 + 6200.0) * ers.EV_PER_INVERSE_CM) < 2e-3


def test_tables_survive_a_round_trip(tmp_path):
    energy = np.linspace(-0.1, 0.1, 21)
    table = {"energy": energy, "ab": 1.0 + energy, "abc": np.abs(energy)}
    info = {"parameters": mlg.get_parameters(), "ab": {"fermi_energy_eV": -0.003}}
    ers.save_csv(str(tmp_path / "bulk.csv"), table, info)
    loaded, loaded_info = ers.load_csv(str(tmp_path / "bulk.csv"))
    assert all(np.allclose(loaded[k], table[k]) for k in table)
    assert loaded_info["ab"]["fermi_energy_eV"] == -0.003

    slabs = {"energy": energy, "ab_20": 1.0 + energy, "abc_20": np.abs(energy)}
    slab_info = {"layers": [20], "fermi_energy_eV": {"ab_20": -0.0026, "abc_20": 0.019}}
    ers.save_slabs_csv(str(tmp_path / "slabs.csv"), slabs, slab_info)
    loaded, loaded_info = ers.load_slabs_csv(str(tmp_path / "slabs.csv"))
    assert list(loaded) == list(slabs)
    assert all(np.allclose(loaded[k], slabs[k]) for k in slabs)
    assert loaded_info == slab_info


def test_subband_peaks_of_rhombohedral_films():
    # 2 gamma1 sin q_n; Palinkas et al. (2024) measure 2299 +- 40 1/cm for the first pair of 12 layers
    first = ers.ers_peak(1, 12, mlg.gamma1) / ers.EV_PER_INVERSE_CM
    assert abs(first / 2299.0 - 1.0) < 0.02
    # for n << N it is the 4 gamma1 sin(q_n / 2) of Garcia-Ruiz et al. (2019)
    small = 4.0 * mlg.gamma1 * np.sin(1.5 * np.pi / 201.0)
    assert abs(ers.ers_peak(1, 100, mlg.gamma1) / small - 1.0) < 1e-3
    energy = np.arange(-0.3, 0.3005, 1.0e-3)
    dos = np.exp(-0.5 * ((energy + 0.071) / 0.003) ** 2) + np.exp(-0.5 * ((energy - 0.106) / 0.003) ** 2)
    assert abs(ers.subband_peak_pair(energy, dos, 0.023, 0.179, 0.015) - 0.177) < 1e-9


def test_bulk_curve_is_drawn_at_the_energy_of_the_slab():
    energy = np.linspace(-0.2, 0.2, 401)
    table = {"energy": energy, "abc": np.abs(energy)}
    info = {"abc": {"fermi_energy_eV": 0.042}}
    slabs = {"energy": energy}
    slab_info = {"fermi_energy_eV": {"abc_20": 0.019}}
    bulk = ers.bulk_on_slab_axis(slabs, slab_info, table, info, "abc", 20)
    assert abs(energy[np.argmin(bulk)] - 0.023) < 1e-9      # the bulk Dirac point, 23 meV above the slab's E_F


def test_linewidth_broadening_is_a_normalised_gaussian():
    x = np.arange(0.0, 1.0005, 1.0e-3)
    inner = (x > 0.2) & (x < 0.8)
    assert np.allclose(ers.linewidth_broadening(x, np.full_like(x, 0.7), 0.06), 0.7)
    assert np.abs(ers.linewidth_broadening(x, 2.0 + x, 0.06) - (2.0 + x))[inner].max() < 1e-9
    line = np.where(np.abs(x - 0.5) < 1e-9, 1.0e3, 0.0)                  # unit weight in one bin
    shape = ers.linewidth_broadening(x, line, 0.06)
    assert abs(shape.sum() * 1.0e-3 - 1.0) < 1e-9
    above = x[shape >= 0.5 * shape.max()]
    assert abs((above[-1] - above[0]) - 0.06) < 2.5e-3


def test_peak_contrast_compares_a_peak_with_the_minimum_after_it():
    x = np.linspace(0.0, 1.0, 1001)
    ramp = 1.0 + x
    assert ers.peak_contrast(x, ramp, 0.05, 0.95) == 0.0
    bump = ramp + 0.5 * np.exp(-0.5 * ((x - 0.4) / 0.02) ** 2)
    i = np.argmax(bump[:500])
    j = i + np.argmin(bump[i:700])
    assert abs(ers.peak_contrast(x, bump, 0.05, 0.95) - (1.0 - bump[j] / bump[i])) < 1e-12
    assert ers.peak_contrast(-x[::-1], bump[::-1], 0.05, 0.95) > 0.1      # the far side is away from zero


def test_mirror_pair_spectrum_of_a_bilayer():
    # gamma0 and gamma1 only: the pair 1- -> 1+ has hbar w = 2 e(p), e = sqrt(g1^2/4 + (hv p)^2) - g1/2,
    # so J(w) = (w + g1) / (2 pi (hv)^2) per area below 2 g1, where the second pair starts
    mlg.set_parameters(gamma2=0.0, gamma3=0.0, gamma4=0.0, gamma5=0.0, Delta_prime=0.0, E0=0.02)
    p = mlg.get_parameters()
    hv = 1.5 * (p["a"] / np.sqrt(3.0)) * p["gamma0"]
    edges = np.arange(0.0485, 0.6, 0.003)
    spectrum, fermi = ers.mirror_pair_spectrum("AB", edges, dk=2.0e-3)
    w = 0.5 * (edges[1:] + edges[:-1])
    analytic = (w + p["gamma1"]) / (2.0 * np.pi * hv ** 2) * mlg.unit_cell_area() / 4.0
    assert abs(fermi - 0.02) < 1e-3
    assert np.abs(spectrum / analytic - 1.0).max() < 0.03


def _all_pairs_with_sigma_z(sequence, edges, fermi, dk, k_max):
    """Garcia-Ruiz et al. (2019), Eq. (5), by brute force on the full k window: every pair of bands, weighted."""
    n_layers = len(sequence)
    n_side = int(np.ceil(2.0 * k_max / dk)) + 1
    t = np.linspace(-k_max, k_max, n_side)
    gx, gy = np.meshgrid(t, t, indexing="xy")
    K = mlg.K_point()
    E, V = np.linalg.eigh(mlg.hamiltonian(gx.ravel() + K[0], gy.ravel() + K[1], None, sequence))
    sigma_z = np.tile([1.0, -1.0], n_layers)
    weight = np.abs(np.einsum("kav,a,kac->kvc", V[:, :, :n_layers].conj(), sigma_z, V[:, :, n_layers:])) ** 2
    tri = mlg._square_mesh_triangles(n_side, n_side)
    area = 0.5 * (t[1] - t[0]) ** 2
    cumulative = np.zeros(edges.size)
    for v in range(n_layers):
        for c in range(n_layers):
            lower, upper = E[:, v], E[:, n_layers + c]
            allowed = ((lower < fermi) & (upper > fermi)) * weight[:, v, c]
            cumulative += mlg._cumulative_states_triangles((upper - lower)[tri], area * allowed[tri].mean(axis=1), edges)
    return np.diff(cumulative) / np.diff(edges) * 4.0 / (2.0 * np.pi) ** 2 * mlg.unit_cell_area() / (2.0 * n_layers)


def test_mirror_pairs_carry_the_sigma_z_weight_in_a_rhombohedral_slab():
    edges = np.arange(0.0045, 0.75, 0.003)
    paired, fermi = ers.mirror_pair_spectrum("ABCABC", edges, dk=4.0e-3)
    weighted = _all_pairs_with_sigma_z("ABCABC", edges, fermi, 4.0e-3, 0.2)
    w = 0.5 * (edges[1:] + edges[:-1])
    smooth = [ers.linewidth_broadening(w, y, 0.03) for y in (weighted, paired)]
    assert np.abs(smooth[0] / smooth[1] - 1.0)[(w > 0.02) & (w < 0.7)].max() < 0.03


def test_hexagonal_slab_needs_the_matrix_elements_at_low_energies():
    # the bilayer-like subbands of a Bernal slab overlap near E_F: counting bands from the middle pairs bands of
    # different subbands there, which sigma_z does not connect
    sequence, dk = "ABABABABAB", 2.0e-3
    edges = np.arange(0.0015, 0.3, 0.003)
    w = 0.5 * (edges[1:] + edges[:-1])
    by_index, fermi = ers.mirror_pair_spectrum(sequence, edges, dk=dk)
    corrected, _ = ers.mirror_pair_spectrum(sequence, edges, dk=dk, weighted_below=0.06)
    weighted = _all_pairs_with_sigma_z(sequence, edges, fermi, dk, 0.2)
    low = w < 0.02
    assert by_index[low].sum() > 1.5 * weighted[low].sum()                # index pairing overcounts
    assert abs(corrected[low].sum() / weighted[low].sum() - 1.0) < 0.03   # the weighted sum is used instead
    assert np.array_equal(corrected[w > 0.06], by_index[w > 0.06])
    smooth = [ers.linewidth_broadening(w, y, 0.02) for y in (corrected, weighted)]
    assert np.abs(smooth[0] / smooth[1] - 1.0)[(w > 0.01) & (w < 0.27)].max() < 0.04
    # the small window of the weighted spectrum holds every pair of the energies it is used for
    _, complete_below = ers.sigma_z_pair_spectrum(sequence, edges[:21], fermi, dk=dk)
    assert complete_below > 0.06


def test_bulk_pair_rule_halves_the_dos():
    energy = np.linspace(-1.0, 1.0, 2001)
    table = {"energy": energy, "ab": 0.5 + np.abs(energy), "abc": np.where(energy > 0, 2.0, 4.0) * np.abs(energy)}
    rule = ers.bulk_pair_rule(table, np.array([0.2, 0.4]))
    assert np.allclose(rule["ab"], [0.5 * 0.6, 0.5 * 0.7])                 # D(w / 2) / 2 for mirror bands
    assert np.allclose(rule["abc"], [0.25 * (0.2 + 0.4), 0.25 * (0.4 + 0.8)])  # mean of the two sides otherwise


def test_calculated_ers_spectrum_is_read_and_classified(tmp_path):
    for name, expected in (("ABCABC", "abc"), ("ABABAB", "ab"), ("ABCBAB", None)):
        path = tmp_path / f"{name}.dat"
        path.write_text(f"# Name: {name}\n# NOL: 6\n# Garcia: [1, 1, 1, 1, 1]\n"
                        "# Calculated peaks [meV]: 231, 378.5\n# Measured peaks [meV]: none\n"
                        "# Columns: Raman shift [meV]\tIntensity [arb. units]\n0\t1.5\n3\t0\n6\t2.25\n")
        spectrum = ers.load_ers_spectrum(str(path))
        assert spectrum["name"] == name and spectrum["layers"] == 6 and spectrum["key"] == expected
        assert np.allclose(spectrum["peaks"], [0.231, 0.3785])
        assert np.allclose(spectrum["shift"], [0.0, 0.003, 0.006]) and np.allclose(spectrum["intensity"], [1.5, 0.0, 2.25])
    shift = np.arange(0.003, 1.0, 0.003)
    model = 1.0e-3 * (1.0 + shift)
    assert abs(ers.scale_to_model(shift, 4.0e5 * model, model, 80.0, 6200.0, 500.0) - 2.5e-6) < 1e-12


def _synthetic_results():
    energy = np.arange(-1.0, 1.0005, 1.0e-3)
    table = {"energy": energy, "ab": 5.0e-3 + 1.2e-2 * np.abs(energy), "abc": 3.9e-2 * np.abs(energy)}
    info = {"parameters": mlg.get_parameters(), "d_layer": 3.35, "n_kz": 240, "dk": 2.0e-3, "bin_width": 1.0e-3,
            "gaussian_sigma": 0.0, "atoms_per_cm3": 1.139e23,
            "ab": {"fermi_energy_eV": -0.003, "carrier_density_cm3": 2.0e18, "refine": 8},
            "abc": {"fermi_energy_eV": 0.042, "carrier_density_cm3": 1.2e15, "refine": 32}}
    e = np.arange(-0.5, 0.5005, 1.0e-3)
    slabs = {"energy": e, "ab_20": 5.0e-3 + 1.2e-2 * np.abs(e),
             "abc_20": 3.9e-2 * np.abs(e - 0.023) + 8.0e-3 * np.exp(-0.5 * (e / 0.01) ** 2)}
    slab_info = {"layers": [20], "dk": 1.0e-3, "k_max": 0.18, "bin_width": 1.0e-3,
                 "fermi_energy_eV": {"ab_20": -0.003, "abc_20": 0.019}}
    shift = np.arange(0.003, 1.0, 0.003)
    peaks = [ers.ers_peak(n, 6, mlg.gamma1) for n in (1, 2)]
    ladder = sum(4.0e-3 * np.exp(-0.5 * ((shift - peak) / 0.006) ** 2) for peak in peaks)
    pairs = {"shift": shift}
    for n in (6, 20):
        pairs[f"ab_{n}"] = 2.5e-3 + 6.0e-3 * shift
        pairs[f"abc_{n}"] = np.where(shift > 0.03, 1.0e-2 * shift, 0.0) + (ladder if n == 6 else 0.0)
    pairs["ab_bulk"] = np.where(shift > 0.03, 1.0, shift / 0.03) * 0.98 * ers.bulk_pair_rule(table, shift)["ab"]
    pairs["abc_bulk"] = ers.bulk_pair_rule(table, shift)["abc"]
    pair_info = {"layers": [6, 20], "dk": 1.0e-3, "k_max": 0.2, "step": 0.003,
                 "weighted_below_eV": {"ab": 0.06, "abc": 0.0},
                 "bulk": {"n_kz": 240, "dk": 2.0e-3, "refine": {"ab": 8, "abc": 32}},
                 "fermi_energy_eV": {"ab_6": 0.0, "abc_6": 0.017, "ab_20": 0.0, "abc_20": 0.019}}
    spectra = [{"path": "ABCABC.dat", "name": "ABCABC", "layers": 6, "peaks": peaks, "key": "abc",
                "shift": np.concatenate([[0.0], shift]), "intensity": np.concatenate([[1.0], 4.0e5 * pairs["abc_6"]])}]
    return table, info, slabs, slab_info, spectra, pairs, pair_info


def test_tables_of_pair_spectra_survive_a_round_trip(tmp_path):
    *_, pairs, pair_info = _synthetic_results()
    ers.save_pairs_csv(str(tmp_path / "pairs.csv"), pairs, pair_info)
    loaded, loaded_info = ers.load_pairs_csv(str(tmp_path / "pairs.csv"))
    assert list(loaded) == list(pairs) and loaded_info == pair_info
    assert all(np.allclose(loaded[k], pairs[k]) for k in pairs)


def test_report_is_complete_and_written_in_latex(tmp_path):
    table, info, slabs, slab_info, spectra, pairs, pair_info = _synthetic_results()
    figures = {name: str(tmp_path / f"{name}.png") for name in ("comparison", "window", "slabs", "ers", "pairs")}
    ers.plot_comparison(table, info, figures["comparison"])
    ers.plot_raman_window(table, info, figures["window"], 80.0, 6200.0)
    ers.plot_slabs(slabs, slab_info, table, info, figures["slabs"], 80.0, 6200.0)
    ers.plot_ers_calculation(spectra, pairs, figures["ers"], 80.0, 6200.0, 500.0)
    ers.plot_pair_spectra(pairs, table, figures["pairs"], 80.0, 6200.0, 500.0, slab_info["layers"])
    path = tmp_path / "report.html"
    ers.write_report(str(path), table, info, 80.0, 6200.0, figures, slabs, slab_info, spectra, pairs, pair_info, 500.0)
    page = path.read_text(encoding="utf-8")
    assert "mathjax" in page and page.count("<img ") == 5
    assert page.count("\\(") == page.count("\\)") > 50 and page.count("\\[") == page.count("\\]") >= 5
    assert "<sub>" not in page and "<sup>" not in page
    sections = [page.index(title) for title in ("<h2>Finite slabs", "<h2>Calculated ERS of a 6-layer film",
                                                "<h2>ERS estimate on the Raman-shift axis", "<h2>References")]
    assert sections == sorted(sections)
    # the synthetic calculated spectrum is the synthetic model times a constant, so it is reproduced exactly
    assert "differ by 0.0 % on average" in page
    fragment = tmp_path / "page.html"
    ers.write_report(str(fragment), table, info, 80.0, 6200.0, figures, slabs, slab_info, spectra, pairs, pair_info,
                     500.0, standalone=False)
    assert fragment.read_text(encoding="utf-8").startswith("<title>")
