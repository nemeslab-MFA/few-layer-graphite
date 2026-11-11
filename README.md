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

## Stacking domain wall: (+⁴−) | (+⁵) in six-layer graphite

`stacking_wall.py` models the single-layer stacking shift of the top graphene
layer measured by STM in Kandrai *et al.*, *One-way electron mirrors from
single-layer stacking shifts in rhombohedral graphite*, arXiv:2610.20905
(2026): hexalayer rhombohedral graphite, Hägg sequence (+⁵) = ABCABC, meets the
stacking fault (+⁴−) = ABCABA, whose top layer sits in the hexagonal position.
On the (+⁵) surface the dI/dV shows the flat-band peak; on the (+⁴−) surface
the flat band is buried one layer down and the top layer carries a Dirac-like
surface band instead.

The k·p SWMcC Hamiltonian of `multilayer_graphene` is linear in the momentum p
from the valley centre, H_S(p) = C_S + p_x X_S + p_y Y_S (`kp_blocks`), and
across the wall it is interpolated linearly between the two stackings,

    H(x, p) = (1 − s(x)) H_(+⁵)(p) + s(x) H_(+⁴−)(p),    s(x) = [1 − tanh(x/l)] / 2,

so that the dimer coupling γ1 of the top layer (A5–B6 → B5–A6), its skew
couplings γ3, γ4, the next-nearest-layer couplings γ2, γ5 and the on-site
energies Δ' of layers 5 and 6 all follow the same profile; the stacking fault
is on the left (x < 0).  The wall is parallel to the armchair direction and
k_y is conserved.  For each k_y the x-dependent problem is diagonalised in a
plane-wave basis on a periodic cell (default 480 nm, 229 plane waves per
orbital, |q| ≤ 0.15 Å⁻¹); the x-dependent terms are convolutions with the
Fourier coefficients of s(x), and s(x) p_x is symmetrised.  The second wall
that periodicity requires sits 240 nm away.  `ldos_wall` returns the
layer-resolved LDOS(E, x) in states eV⁻¹ Å⁻² with spin, both valleys and a
Gaussian broadening (5 meV), summed over a k_y mesh of 0.001 Å⁻¹ (the K⁻
Hamiltonian is the complex conjugate of the K⁺ one and gives the same LDOS).

```python
import stacking_wall as sw

res = sw.ldos_wall(wall_length=88.0)         # tanh length l in A; 10-90 % width = 2.2 l
res["energy"], res["x"], res["ldos"][-1]     # eV, A, top-layer LDOS (6, n_E, n_x)[-1]
```

Checks: without a wall the layer-summed LDOS is x-independent and equals the
k·p DOS of `calculate_density_of_states` (1 % between −0.25 and 0.25 eV, 2.6 %
integrated to ±0.38 eV where the k_y window cuts off the trigonal-warping
legs); for a monolayer it reproduces 2|E| / (π ħ²v²) to 10⁻⁴ with the default
cell; far from the walls the top-layer LDOS returns to the two uniform stacks.

### Interpolation length from the measurements

The length l is taken from the dI/dV line spectra.  `load_line_spectrum`
reads an RHK `.sm4` file with `rhkpy` and normalises it as the notebooks of the
paper do (lock-in signal over the mean of |I| at the two ends of the bias
ramp); `mixing_fraction` writes every spectrum of the line as
(1 − s) S_(+⁴−)(V) + s S_(+⁵)(V) with the two end-member spectra averaged far
from the wall (least squares over −0.3 … 0.3 V), and `fit_wall_profile` fits
s(x) with the same tanh step as the model.  For the two line spectra across
single walls of flake MK_ABC_FLG_25 (9 K):

| line spectrum | wall centre x0 | l (tanh) | 10–90 % width | flat band (+⁵) |
|---|---|---|---|---|
| `..._2023_08_03_16_33_49_514.sm4` (125 nm, 256 spectra) | 37.6 ± 0.2 nm | 8.8 ± 0.3 nm | 19.4 nm | +22 mV |
| `..._2023_10_25_17_00_17_940.sm4` (70 nm, 256 spectra) | 39.2 ± 0.1 nm | 5.6 ± 0.1 nm | 12.2 nm | −32 mV |

The flat-band weight within ±25 meV of the peak gives the same lengths to
within 0.5 nm.  These are apparent widths of the surface LDOS, which include
the lateral extent of the states themselves; they are the lengths over which
the Hamiltonian is interpolated here.

```
python stacking_wall_figure.py --sm4 MK_ABC_FLG_25_9K_2023_08_03_16_33_49_514.sm4 --tag 514 --x-sf 22 --x-rg 55
python stacking_wall_figure.py --sm4 MK_ABC_FLG_25_9K_2023_10_25_17_00_17_940.sm4 --tag 940 --x-sf 30 --x-rg 46
```

fits the wall, computes the LDOS with the fitted l (about 10 minutes on four
cores; cached in `examples/stacking_wall_ldos_TAG.npz`), and writes
`examples/stacking_wall_TAG.png`: the measured dI/dV(V, x) next to the
top-layer LDOS(E, x) of the model, both bias derivatives (which bring out the
quasiparticle-interference fringes), the fitted profile and spectra at three
positions; `examples/stacking_wall_layers_TAG.png` shows all six layers as in
Fig. 3 of the paper, and `examples/stacking_wall_TAG.json` the fit and the
parameters; `python stacking_wall_report.py` collects the numbers and figures
into the short page `examples/stacking_wall_report.html`.  The model energy
axis is shifted rigidly so that its (+⁵)
flat-band peak coincides with the measured one (the flake is slightly doped,
differently in the two measurements); no layer-dependent doping is included
(`layer_potential` of `ldos_wall` accepts one).  The raw `.sm4` files are not
part of the repository.

### Tip-induced band bending at the wall

Near the wall the measured flat-band peak moves as it fades: up in spectrum 514
(seen at positive bias, +22 → +55 mV) and down in spectrum 940 (negative bias,
−32 → −88 mV).  `stacking_wall_tibb.py` tests the tip-gating explanation with
the [bandbending](https://github.com/nemeslab-MFA/bandbending) engine
(Hartree + γ0/γ1 tight binding under a hyperbolic tip; `pip install -e` that
repository first).  In the local approximation three laterally uniform stacks
are solved under the same tip on 90 nm SiO₂: (+⁵) far from the wall, the wall
(top dimer shared between the (+⁵) bond with weight 1 − s and the (+⁴−) bond
with weight s) and (+⁴−).  `WallTB` evaluates the wall Hamiltonian, whose bond
loop carries the phase of the structure factor, with a complex Hamiltonian
averaged over that phase (it reproduces the engine's end members exactly and a
brute-force zone sum to 1 %); `WallSimulation` injects it into the engine.  The
doping follows from the measured flat-band energy, the tip contact potential
from the fit of the simulated (+⁵) spectrum (`tunneling_spectrum`, band-bending
term, WKB barrier, lock-in broadening) to the measurement far from the wall;
the wall and the (+⁴−) stack then use the same tip and the same fixed charge
per layer.

```
python stacking_wall_tibb.py --sm4 ..._514.sm4 --tag 514 --x-sf 22 --x-rg 55
python stacking_wall_tibb_figure.py --tags 514 940
```

Results go to `examples/stacking_wall_tibb_TAG.json` (scan, sweeps, spectra)
and `examples/stacking_wall_tibb.png`; the report collects them in its last
section.

`stacking_wall_lateral.py` drops the local approximation: it solves the
three-dimensional Poisson problem of the tip over the wall with the lateral
solver of the bandbending repository (branch `lateral-solver`,
`engine/lateral.py`; needs `pyamg`).  The stack weight s(x) follows the fitted
wall, every layer is a rigid-band charge sheet with the zero-field layer DOS of
the local stack (`WallTB` tables at s = 0, 0.15, …, 1), the fixed charge per
layer makes the (+⁵) stack neutral at the measured flat-band energy, and the
tip (same radius, distance and contact potential) sits at the bias of the
measured wall peak.  For every tip position it stores the layer potentials
along the line with and without the tip and the bending under the apex against
the local approximation; with `--kp` it feeds the lateral potentials to the k·p
wall model and reads the top-layer LDOS under the tip, as the Gaussian-bump
calculation of `stacking_wall_tibb_bump.py` does with the local potentials.

```
python stacking_wall_lateral.py --tag 514
python stacking_wall_lateral.py --tag 514 --kp
python stacking_wall_lateral.py --plot
```

Results go to `examples/stacking_wall_lateral_TAG.json` and
`examples/stacking_wall_lateral.png`.

`stacking_wall_epsfit/` refits the (+⁵) spectrum of line 940 over the
interlayer permittivity ε and the starting doping ΔE_F under the same tip:
`scan.py` for the uniformly doped stack, `polar.py` for the polarized stack
(a fixed charge n_sub in the top 2 nm of the SiO₂, chosen per cell from the
tip-free fit), `analyze.py` for the maps and the summary
(`examples/stacking_wall_epsfit_940.json`, `.png`).
`stacking_wall_polar_series.py` carries the best polarized stack across the
wall in the local approximation and draws the simulated line spectrum next
to the measured one (`examples/stacking_wall_polar_940.json`,
`stacking_wall_polar_map_940.png`, `stacking_wall_polar_spectra_940.png`).

```
python stacking_wall_epsfit/scan.py 0 4     # one of 4 workers; then polar.py the same way
python stacking_wall_epsfit/analyze.py
python stacking_wall_polar_series.py run 0  # one of 4 workers
python stacking_wall_polar_series.py collect --sm4 ..._940.sm4
```

`stacking_wall_cpot_sensitivity.py` solves the k·p wall model with the
tip-induced bump at the wall spectrum and far on the (+⁵) side for tip
work-function offsets of −0.35 to +0.2 eV (the bump of the fitted sweeps at
V + δ), and `stacking_wall_summary_figure.py` draws the figure set of a line
(measured, local approximation and SWMcC + on-site maps, the flat-band peak
along the line, the bend at the wall) for the report.

```
python stacking_wall_cpot_sensitivity.py --tag 940
python stacking_wall_cpot_sensitivity.py --plot
python stacking_wall_summary_figure.py --tag 940 --sm4 ..._940.sm4
```

Layer order: the bandbending engine numbers the layers from the top (index 0
faces the tip), `stacking_wall.ldos_wall` from the bottom (the top-layer LDOS
is `ldos[-1]`); engine potentials are reversed before they go into the k·p
model.

`examples/` holds the outputs of these scripts: figures, result files, the
hybrid notebook and the report `examples/stacking_wall_report.html`. They are
results of the wall study, not tests of the Poisson solver; the solver's
validations (against SEMITIP and published STM measurements) are in the
`examples/` folder of the bandbending repository.

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
and the comparison with the DOS.  For the stacking wall they check the k·p
decomposition, the profile and its Fourier series, the plane-wave Hamiltonian
(Hermitian, the two uniform limits), the LDOS normalisation and the analysis of
a synthetic line spectrum.

## Other scripts

- `example_usage.py`: short tour of the API.
- `make_movie.py [fps] [abc|aba] [max_layers]`: band-structure animation versus
  layer number (needs ffmpeg and/or Pillow).
- `multilayer_graphene_pythtb.py`: an older PythTB-based implementation with
  different parameters; kept for reference only, not maintained.

Requirements: NumPy, SciPy, Matplotlib (pytest for the tests; rhkpy to read
the STM files).
