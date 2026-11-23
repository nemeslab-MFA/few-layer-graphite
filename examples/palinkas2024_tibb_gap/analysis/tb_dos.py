"""SWMcC (McEllistrim 2023) DOS of N-layer RG at zero field: -1/+1 peak separation of the total DOS and of the
top-layer LDOS (the quantity ERS measures, without excitonic corrections), and the S7 gap of the top-layer LDOS."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import find_peaks
import rgfit as R, s7
R.threads1()
E = np.arange(-0.7, 0.7 + 1e-9, 0.0005)
out = {}
for N in (3, 4, 6):
    U, p = R.tipfree(N, 0.0)
    for sig in (0.003, 0.010):
        _, D = p.tb.ldos(np.full_like(U, p.mu0), E + p.mu0, broadening=sig, resolve="layer")
        D = np.asarray(D); D = D if D.shape[0] == E.size else D.T
        res = {}
        for lab, y in (("total", D.sum(axis=1)), ("top", D[:, 0])):
            pk, _ = find_peaks(y, prominence=0.02 * y.max())
            Ep = E[pk]
            lo = Ep[Ep < -0.12]; hi = Ep[Ep > 0.12]
            res[lab] = (float(lo.max()), float(hi.min()), float(hi.min() - lo.max()))
        h, e, g = s7.gap(E, D[:, 0], *R.WIN.get(N, ((-0.32, -0.14), (0.14, 0.33))))
        out[f"N{N}_sig{sig}"] = dict(peaks=res, s7=(h, e, g))
        print(f"N{N} sigma {1e3*sig:.0f} meV: total-DOS peaks {1e3*res['total'][0]:+.1f} / {1e3*res['total'][1]:+.1f} -> {1e3*res['total'][2]:.1f} meV; "
              f"top-LDOS peaks {1e3*res['top'][0]:+.1f} / {1e3*res['top'][1]:+.1f} -> {1e3*res['top'][2]:.1f}; S7(top LDOS) {1e3*g:.1f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_dos.json"), "w"))
