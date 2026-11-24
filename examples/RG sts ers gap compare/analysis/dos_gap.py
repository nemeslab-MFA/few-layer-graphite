"""ERS-equivalent model gap: separation of the -1 and +1 peaks of the total DOS (all layers, Gaussian 3 meV) of the
tip-free stack, with the layer potentials of the SiO2 charge n_sub (n_sub = 0: the neutral TB stack)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import find_peaks
import rgfit as R

E = np.arange(-0.7, 0.7 + 1e-9, 0.0005)


def dos_gap(N, nsub, sig=0.003):
    U, p = R.tipfree(N, nsub)
    _, D = p.tb.ldos(U, E + p.mu0, broadening=sig, resolve="layer")
    D = np.asarray(D); D = D if D.shape[0] == E.size else D.T
    y = D.sum(axis=1)
    pk, _ = find_peaks(y, prominence=0.02 * y.max())
    Ep = E[pk]
    # the +-1 peaks: the strongest peaks within 0.12..(edge region) on each side, nearest to the gap
    fb = E[np.argmax(np.where(np.abs(E) < 0.1, D[:, 0], 0))]
    lo, hi = Ep[Ep < fb - 0.12], Ep[Ep > fb + 0.12]
    return float(lo.max()), float(hi.min()), float(hi.min() - lo.max())


if __name__ == "__main__":
    R.threads1()
    out = {}
    jobs = [(int(a.split(":")[0]), float(a.split(":")[1]) * 1e13, a.split(":")[2]) for a in sys.argv[1:]]
    p = os.path.join(R.W, "dos_gap.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    for N, n, key in jobs:
        h, e, g = dos_gap(N, n)
        out[key] = dict(N=N, nsub=n, h=h, e=e, gap=g)
        print(f"{key}: N{N} n_sub {n/1e13:+.3f}e13: -1 {1e3*h:+.1f} +1 {1e3*e:+.1f} gap {1e3*g:.1f} meV", flush=True)
    json.dump(out, open(p, "w"), indent=1)
