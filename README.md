# Few-layer graphene: SWMcC tight-binding bands and density of states

`multilayer_graphene.py` computes band structures and densities of states (DOS)
of N-layer graphene with any stacking sequence: rhombohedral (ABC…), Bernal
(ABAB…) and mixed polytypes (ABCB, ABAC, ABCAB, …).  The model is the full
Slonczewski–Weiss–McClure (SWMcC) tight-binding Hamiltonian with the couplings
γ0 … γ5 and the dimer on-site energy Δ'.  Conventions and default parameters follow

> A. McEllistrim, A. Garcia-Ruiz, Z. A. H. Goodwin, V. I. Fal'ko,
> *Spectroscopic signatures of tetralayer graphene polytypes*, arXiv:2302.07374 (2023)

whose hybrid k·p Hamiltonians (their Eq. 1) are the first-order expansion of
this lattice model around the valley centres.  Both models are implemented
(`hamiltonian`, `kp_hamiltonian`) and share one assembly routine, so the
correspondence is exact by construction and is checked by the test suite.

## Model

Basis per layer: (A_n, B_n), B_n displaced by τ = (0, a/√3) from A_n, a = 2.46 Å.
Following the cyclic sequence A→B→C→A ("forward" step) puts B_{n+1} on top of
A_n (dimer A_n–B_{n+1}); a "backward" step (B→A, C→B, A→C) puts A_{n+1} on top
of B_n.  For each adjacent pair the block is the paper's V_AB (forward) or
V_AB† (backward); for next-nearest layers the block is W_ABC, W_ABC†, W_ABA or
W_BAB depending on the two steps in between; each site receives Δ' times the
number of γ1 bonds attached to it.  This reproduces Eq. (1a)–(1d) of the paper
and extends them to any sequence.  (Two misprints of arXiv v1 are corrected:
the layer-3 on-site term of H_ABCB must be Δ'(1 − σ_z) and the (2,3) block of
H_ABAC must be V_AB†; the band-edge labels of the paper's Fig. 1, e.g.
√2γ1 + 3Δ'/2 for ABCB, require this.)

| coupling | meaning | matrix element | default (eV) |
|---|---|---|---|
| γ0 | intralayer nearest neighbour | −γ0 f(k) | 3.0896 (v = 1.0×10⁶ m/s) |
| γ1 | vertical dimer, adjacent layers | γ1 | 0.390 |
| γ3 | skew, non-dimer sites, adjacent layers | γ3 f(k) | 0.3090 (v3 = 0.1×10⁶ m/s) |
| γ4 | skew, dimer–non-dimer, adjacent layers | γ4 f*(k) | 0.0680 (v4 = 0.022×10⁶ m/s) |
| γ2 | vertical, two layers apart, through a hexagon centre | γ2/2 | −0.017 |
| γ5 | vertical, two layers apart, along a dimer chain | γ5/2 | 0.038 |
| Δ' | on-site energy per γ1 bond | Δ' × (bonds) | 0.025 |

f(k) = Σ_j exp(i k·δ_j) over the three A→B bond vectors; near K_ξ,
−γ0 f = ħv (ξ p_x − i p_y), with ħv_i = (3/2) a_cc γ_i.  Velocities can be
given directly: `set_parameters(v=1.0e6, v3=0.1e6, v4=0.022e6)`.

## Usage

```python
import multilayer_graphene as mlg

mlg.set_parameters(N_layers=4, stacking='abc')        # periodic pattern, or
mlg.set_parameters(stacking='ABCB')                   # explicit sequence

E, k = mlg.calculate_bands(path_type='px')            # bands along p_x through K (1/Å)
mlg.plot_bands(E, k, path_type='px', xlim=(-0.2, 0.2), ylim=(-0.8, 0.8))

dos = mlg.calculate_density_of_states((-0.8, 0.8))    # dict: energy, dos_per_cm2, ...
mlg.plot_dos((-0.8, 0.8), units='cm2')                # same layout as Fig. 1 of the paper

H = mlg.hamiltonian(kx, ky, stacking_type='ABCA')     # (…, 2N, 2N) for array kx, ky (1/Å)
H_kp = mlg.kp_hamiltonian(px, py, stacking_type='ABCA', xi=+1)
```

Paths: `'gkm'` (Γ→K→M, default), `'gkg'`, `'px'`, `'py'`; K is always at
coordinate 0.  `plot_panel_comparison(range(1, 9))` draws several layer numbers
side by side, `plot_band_density(b)` maps one band around K.

### Density of states

`calculate_density_of_states` integrates the bands with the linear-interpolation
(triangle) method on a square k-mesh around K (`region='valley'`, spacing `dk`,
valley degeneracy 2 applied) or over the full Brillouin zone (`region='bz'`).
Results are returned per unit cell, per Å² and per cm² (spin degeneracy 2
included).  `model='kp'` evaluates the paper's continuum Hamiltonian instead of
the lattice one.  `plot_dos(..., csv_path=...)` / `load_dos_from_csv` store and
reload curves.  Only p_y ≥ 0 is diagonalised (E(p_x, −p_y) = E(p_x, p_y) for
every stack) and the Hamiltonians are built in pieces of at most 256 MB, so
thick stacks such as 100 layers fit in memory.

## Comparison with McEllistrim et al. (2023)

`python compare_with_mcellistrim.py` overlays the tight-binding and k·p DOS with
the curves digitised from the paper's Fig. 1 (tetralayers, 1.8 meV per pixel)
and Fig. 7 (trilayers and pentalayers, 7.5 meV per pixel); the digitised data
live in `tests/data/`.  With the paper's parameters:

- every band edge and van Hove singularity appears at the published energy:
  local cross-correlation of the curves gives shifts below 5 meV (Fig. 1) and
  7 meV (Fig. 7); the dimer edges at Δ' ± γ1, the ABAB steps at −0.571, −0.226,
  0.256, 0.691 eV and the ABCB steps at −0.505, 0.599 eV are reproduced;
- the absolute DOS of the k·p model is within 1–3 % (electrons) and 6–9 %
  (holes) of Fig. 1 and within 1–6 % of the trilayer and pentalayer curves of
  Fig. 7; the lattice model is a further 2 % higher because its band velocity
  decreases slightly away from K.  The hole-side excess of Fig. 1 is absent in
  Fig. 7, whose curves show the stronger electron–hole asymmetry, so the residual
  reflects the numerics of the published figures rather than the parameters;
- the ABA trilayer panel of Fig. 7 is drawn on a 10¹³ scale and lies a factor
  of two below the model and below every other panel.

## Bulk graphite and the ERS background

`bulk_graphite.py` is the infinite-stack limit of the same model: hexagonal
(Bernal, AB), rhombohedral (ABC) or any periodic polytype, with the layer
blocks taken from `multilayer_graphene.py` and the phase exp(i m k_z d),
d = 3.35 Å, on every hop across m layers.

```python
import bulk_graphite as bg

H = bg.bulk_hamiltonian(kx, ky, kz, 'ab')             # (…, 4, 4); 'abc' gives (…, 2, 2)
dos = bg.calculate_bulk_density_of_states((-1.0, 1.0), 'abc')
dos['energy'], dos['dos_per_atom'], dos['dos_per_cm3'], dos['fermi_energy']
```

For Bernal stacking this is the SWMcC model of graphite, with
Δ = 2Δ' + γ2 − γ5 (−5 meV for the default parameters) and the energy zero
shifted by −γ2.  For rhombohedral stacking the primitive cell holds one layer
and the bands are Dirac cones on a spiral of radius γ1/ħv around the K–H line.
The DOS is integrated slice by slice in k_z with the triangle method; mesh cells
near the Fermi level are subdivided, and the Fermi level follows from charge
neutrality.

`python graphite_ers_background.py --raman-min 80 --raman-max 6200` compares the
two polytypes as an estimate of their electronic Raman scattering (ERS)
continuum.  The ERS of graphitic systems consists of interband excitations
between mirror bands, so the signal at Raman shift ω follows the DOS at
E_F ± ħω/2 (Ponosov et al., PRB 91, 195435 (2015); García-Ruiz et al., Nano
Lett. 19, 6152 (2019)).  The script writes `examples/graphite_ers_dos_comparison.png`
(both DOS curves), `examples/graphite_ers_dos_raman_window.png` (the energies
reached by the spectrometer shaded), the table `examples/graphite_ers_dos.csv`
and the page `examples/graphite_ers_dos_report.html`; `--from-csv` redraws them
for another Raman range without recomputing.  `--slabs 20 50 100` (the default)
adds the DOS of finite slabs of both stackings from `multilayer_graphene.py`,
in `examples/graphite_ers_dos_slabs.png` and `_slabs.csv` and as the last
section of the report; the slabs are diagonalised in parallel processes with
one BLAS thread each.

A Raman shift is the difference between an electron and a hole energy, so the
last two sections of the report work on that difference itself.
`mirror_pair_spectrum` is the joint density of states of the mirror pairs
n⁻ → n⁺ of a slab (Eq. 5 of García-Ruiz et al. with the σ_z matrix element of a
mirror pair set to 1, occupation of T = 0).  It is set against ERS spectra
calculated for whole films, read from `examples/ers_calculations/*.dat`
(`--ers-spectra`), in `examples/graphite_ers_dos_ers_calculation.png`, and drawn
for the thicker slabs and the bulk in `examples/graphite_ers_dos_raman_axis.png`,
as computed and broadened to the measured ERS line width (`--linewidth`, 500 cm⁻¹
at half maximum on the Raman-shift axis); the spectra are kept in `_pairs.csv`.
The bulk spectrum is `bulk_graphite.calculate_bulk_pair_spectrum`, the same
integral over the three-dimensional zone with the pairs taken at equal k_z.  In
hexagonal slabs counting bands from the middle of the spectrum does not find
the mirror partner near the Fermi level, where the bilayer-like subbands
overlap, so their spectra below 60 meV are the sum over all pairs of bands
weighted by |⟨c|σ_z|v⟩|² (`sigma_z_pair_spectrum`).
The mathematics of the report is written in LaTeX and typeset by MathJax, which
the page loads from a CDN; `--page-only PATH` also writes the page without its
document skeleton, for a publisher that adds its own.  With the default parameters:

- hexagonal graphite has 2.0×10¹⁸ cm⁻³ carriers of each sign and a Fermi-level
  DOS of 4.9×10⁻³ states eV⁻¹ atom⁻¹; rhombohedral graphite has the linear DOS
  of graphene, with a band overlap of 2.7 meV and 1.2×10¹⁵ cm⁻³ carriers;
- the rhombohedral DOS is 0.04 of the hexagonal one at 80 cm⁻¹
  (|E − E_F| = 5 meV), 0.40 at 1000 cm⁻¹ and 0.81 at 3000 cm⁻¹; the curves
  cross at 5020 cm⁻¹ and the rhombohedral DOS is 5 % larger at 6200 cm⁻¹;
- hexagonal slabs of 20, 50 and 100 layers stay within 3.5 %, 1.3 % and 0.7 %
  of the bulk DOS between 30 and 384 meV from E_F; rhombohedral slabs split the
  bulk continuum into subbands whose −n and +n peaks are 4γ1 sin[(n + ½)π/(2N + 1)]
  apart (first ERS peak at 1443, 587 and 295 cm⁻¹), and their surface band pins
  E_F about 20 meV below the bulk Dirac energy;
- the ERS spectrum calculated for a 15-layer rhombohedral film is reproduced by
  the mirror-pair spectrum to 1.1 % with one scale factor, with all six peaks on
  the same 3 meV bin; the peaks follow 2γ1 sin[(n + ½)π/(N + ½)];
- broadened to 500 cm⁻¹ the strongest subband peak stands 16 % above the minimum
  after it for 15 layers and 6 % for 20 layers, and no peak is left at 50 and
  100 layers; the hexagonal spectra have no peak at any thickness;
- the bulk pair spectrum of rhombohedral graphite equals D(E_F ± ħω/2)/2 to
  0.3 %; for hexagonal graphite the DOS rule is 1.3 % too high above 500 cm⁻¹
  and misses the fall of the spectrum below about 250 cm⁻¹, where the pockets
  of the semimetal block the pairs (2.8 times too high at 80 cm⁻¹, T = 0).

## Tests

```
python -m pytest tests -q
```

The tests check the parameter conversion, Hermiticity, time reversal and zone
periodicity, the exact reduction of the lattice model to the paper's Eq. (1)
at K, analytic band edges (bilayer, ABC trilayer, ABAB and ABCB dimer chains),
the DOS normalisation (monolayer, bilayer, full-zone state count) and the
agreement with the digitised DOS of the paper.  For the bulk model they check
the SWMcC energies on the K–H line, that bulk Bloch states solve the
finite-stack Hamiltonian in its interior, the DOS in the γ0, γ1 limit
(graphene for ABC, the k_z average of bilayers for AB) and the mesh refinement;
for the ERS script, the Fermi level of a slab, the Raman shift to energy mapping,
the mirror-pair spectrum (bilayer in closed form, σ_z-weighted sum over all pairs
of bands for both stackings), the line-width kernel, the tables written to disk
and the report; for the bulk pair spectrum, the closed forms of the γ0, γ1 limit
and the comparison with the DOS.

## Other scripts

- `example_usage.py`: short tour of the API.
- `make_movie.py [fps] [abc|aba] [max_layers]`: band-structure animation versus
  layer number (needs ffmpeg and/or Pillow).
- `multilayer_graphene_pythtb.py`: an older PythTB-based implementation with
  different parameters; kept for reference only, not maintained.

Requirements: NumPy, Matplotlib (pytest for the tests).
