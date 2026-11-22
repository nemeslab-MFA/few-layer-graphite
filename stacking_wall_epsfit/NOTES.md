# Notes on the (+5) fits of line 940

## Observations to keep in mind

- **Hole-side van Hove peak and the tip (2026-11-09).** In the eps = 2.6 polarized-stack fit (`fixeps.py`; best cell
  Delta E_F = 40 meV, n_sub = -0.849e13 cm^-2, about 1.5e12 holes; `examples/stacking_wall_eps26_940.png`, panel b) the
  spectrum without the tip has a weak hole-side van Hove maximum near -330 mV that disappears once the tip is in the
  solve (the structure near -215 mV survives, at -222 mV). The same disappearance is seen in the experiments: a sign
  that the measured negative-bias spectrum is shaped by the tip's field.
- **Interlayer eps below 2.6; no adsorbate doping (user, 2026-11-22).** Explore eps < 2.6 in future fits (eps 2.6 is
  the current working value; large eps was tried and rejected, `swmcc_eps.py`). The large shifts of the top-layer flat
  band (-32 / +23 mV in 940 / 514) cannot be explained by adsorbates: there are too few of them and they are mostly
  normal alkanes. The (+5) spectrum fixes the layer-potential profile, i.e. fixed charge over eps, so a smaller eps
  needs less charge for the same fit.
