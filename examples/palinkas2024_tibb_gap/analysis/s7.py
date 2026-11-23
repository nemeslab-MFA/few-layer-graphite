"""Band-edge (-1, +1) positions of an STM / model spectrum by the method of Palinkas et al., Carbon 230, 119608
(2024), Supplementary S7: a linear background with different slopes below and above x0, a sigmoid step and a
Gaussian peak, both centred on x0 (Eq. S1, ``lin_step_gauss_lin`` of the Zenodo notebook "fitting STM gaps").
x0 marks the top of the step (the DOS peak at the band edge); the gap is x0(+1) - x0(-1).

The fit is multi-start (a grid of x0 seeds over the window, the lowest least-squares residual wins) so that the
same code runs unattended on measured and on simulated spectra.
"""
import numpy as np
from scipy.optimize import curve_fit


def lsgl(x, x0, slope_below, slope_above, ampl_gau, width, ampl_step, spread, offset):
    x = np.asarray(x, float)
    y = np.where(x < x0, offset + slope_below * (x - x0), offset + slope_above * (x - x0))
    z = np.clip((x - x0) / spread, -60, 60)
    return y + ampl_step / (np.exp(z) + 1) + ampl_gau * np.exp(-2 * np.log(2) * (x - x0) ** 2 / width ** 2)


def fit_edge(V, y, window, n_seeds=21, margin=0.02, return_params=False):
    """x0 of Eq. S1 fitted to y(V) inside window = (lo, hi) in V.  The spectrum is scaled to its mean in the window
    (the fitted x0 does not depend on the scale).  Returns x0 (V), or (x0, params, rms) with return_params."""
    V, y = np.asarray(V, float), np.asarray(y, float)
    lo, hi = min(window), max(window)
    s = (V >= lo) & (V <= hi) & np.isfinite(y)
    x, yy = V[s], y[s]
    yy = yy / np.mean(np.abs(yy))
    best = None
    for x0 in np.linspace(lo + margin, hi - margin, n_seeds):
        b, a = x < x0, x >= x0
        sb = np.polyfit(x[b], yy[b], 1)[0] if b.sum() > 3 else 0.0
        sa = np.polyfit(x[a], yy[a], 1)[0] if a.sum() > 3 else 0.0
        p0 = [x0, sb, sa, 0.05, 0.03, 0.0, 0.01, float(np.interp(x0, x, yy))]
        lb = [lo + margin / 2, -np.inf, -np.inf, 0.0, 0.004, -np.inf, 0.001, -np.inf]
        ub = [hi - margin / 2, np.inf, np.inf, 2.0, 0.12, np.inf, 0.08, np.inf]
        try:
            p, _ = curve_fit(lsgl, x, yy, p0=p0, bounds=(lb, ub), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        r = float(np.sqrt(np.mean((lsgl(x, *p) - yy) ** 2)))
        if best is None or r < best[2]:
            best = (float(p[0]), p, r)
    if best is None:
        return (np.nan, None, np.nan) if return_params else np.nan
    return best if return_params else best[0]


def gap(V, y, win_h, win_e, **kw):
    """(x0_h, x0_e, gap) in V."""
    h = fit_edge(V, y, win_h, **kw)
    e = fit_edge(V, y, win_e, **kw)
    return h, e, e - h


def flank(V, y, search, side, rel_prom=0.25, smooth=0.008):
    """Steepest point of the first band-edge flank next to the gap: on the hole side (side=-1) the strongest rises
    of dI/dV towards negative bias, on the electron side (+1) towards positive bias; among the rises inside
    search = (lo, hi) with at least rel_prom of the largest one, the one nearest to zero bias."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    V, y = np.asarray(V, float), np.asarray(y, float)
    dv = np.median(np.diff(V))
    ys = gaussian_filter1d(y / np.mean(np.abs(y)), max(smooth / dv, 0.5))
    g = side * np.gradient(ys, V)
    s = np.nonzero((V >= min(search)) & (V <= max(search)))[0]
    pk, _ = find_peaks(g[s])
    if pk.size == 0:
        return float(V[s][np.argmax(g[s])])
    pk = pk[g[s][pk] >= rel_prom * g[s][pk].max()]
    i = pk[np.argmin(np.abs(V[s][pk]))]
    return float(V[s][i])


def gap_auto(V, y, search_h, search_e, half=0.07, **kw):
    """S7 gap with windows of +-half around the first band-edge flank on each side (see flank)."""
    xh, xe = flank(V, y, search_h, -1), flank(V, y, search_e, +1)
    wh, we = (xh - half, xh + half), (xe - half, xe + half)
    h, e = fit_edge(V, y, wh, **kw), fit_edge(V, y, we, **kw)
    return h, e, e - h, wh, we
