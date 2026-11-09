"""Peak positions of a (+5) spectrum of six-layer RG: the flat-band peak, the lower
side-band peak and the two upper side-band edges (steepest rises)."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter

NAMES = ("lower side-band peak", "flat-band peak", "upper side-band edge 1", "upper side-band edge 2")


def features(V, y, smooth=0):
    V, y = np.asarray(V, float), np.asarray(y, float)
    ok = np.isfinite(y)
    V, y = V[ok], y[ok]
    if smooth:
        y = savgol_filter(y, smooth, 3)
    y = y / y.max()
    out = {}
    sel = (V > -0.12) & (V < 0.06)
    out["flat"] = float(V[sel][np.argmax(y[sel])])
    pk, _ = find_peaks(y, prominence=0.01)
    low = [i for i in pk if -0.42 < V[i] < out["flat"] - 0.1]
    out["low_peak"] = float(V[max(low)]) if low else np.nan          # the side-band peak nearest the gap
    d = np.gradient(y, V)
    sel = np.nonzero((V > 0.1) & (V < 0.42))[0]
    rises, prop = find_peaks(d[sel], prominence=0.05 * np.abs(d[sel]).max())
    rises = sel[rises]
    top = sorted(rises, key=lambda i: -d[i])[:2]
    top = sorted(top)
    out["up1"] = float(V[top[0]]) if len(top) > 0 else np.nan
    out["up2"] = float(V[top[1]]) if len(top) > 1 else np.nan
    return np.array([out["low_peak"], out["flat"], out["up1"], out["up2"]])


NAMES5 = ("lower side-band peak", "flat-band peak", "bottom-band onset", "upper side-band edge 1", "upper side-band edge 2")


def features5(V, y, smooth=0):
    """features() plus the onset of the bottom band (steepest rise between the flat-band peak
    + 40 mV and +160 mV), in the order of NAMES5."""
    f4 = features(V, y, smooth)
    V, y = np.asarray(V, float), np.asarray(y, float)
    ok = np.isfinite(y)
    V, y = V[ok], y[ok]
    if smooth:
        y = savgol_filter(y, smooth, 3)
    y = y / y.max()
    d = np.gradient(y, V)
    sel = np.nonzero((V > f4[1] + 0.04) & (V < 0.16))[0]
    pk, _ = find_peaks(d[sel])
    bb = float(V[sel[pk[np.argmax(d[sel][pk])]]]) if pk.size and d[sel][pk].max() > 0 else np.nan
    return np.array([f4[0], f4[1], bb, f4[2], f4[3]])
