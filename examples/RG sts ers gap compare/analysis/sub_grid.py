"""Tip-free (SiO2 charge only) S7 gaps of N = 3, 4 at the grid n_sub values (map modulations 4 mV / 5.5 mV)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, rgfit as R
R.threads1()
out = {}
for N, mod in ((3, 0.004), (4, 0.0055)):
    for n in (-1.0e13, -0.5e13, 0.0, 0.5e13, 1.0e13):
        y, U = R.free_spectrum(N, n, mod)
        f = R.features(N, R.grid_for(mod), y)
        out[f"N{N}_n{n/1e13:+.2f}"] = dict(N=N, nsub=n, features=f.tolist(), gap=float(f[2] - f[1]), U=U.tolist())
        print(f"N{N} n_sub {n/1e13:+.2f}e13: FB {1e3*f[0]:+.0f} h {1e3*f[1]:+.1f} e {1e3*f[2]:+.1f} gap {1e3*(f[2]-f[1]):.1f}; U-mu0 {np.round(1e3*U).astype(int).tolist()}", flush=True)
json.dump(out, open(os.path.join(R.OUT, "sub_grid.json"), "w"), indent=1)
