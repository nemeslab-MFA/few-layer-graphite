"""Shared pieces of the eps = 2.6 fits: the prominence of the lower side-band peak (a cell counts only
if it is at least half the measured one) and its measured value."""
import json
import os

import numpy as np
from scipy.signal import peak_prominences, savgol_filter

from features import features5

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def low_prominence(V, y, f0):
    """Prominence of the lower side-band peak at f0, in units of the flat-band peak."""
    V = np.asarray(V, dtype=float)
    y = np.asarray(y, dtype=float)
    y = y / np.nanmax(y[(V > -0.12) & (V < 0.06)])
    return float(peak_prominences(y, [int(np.argmin(np.abs(V - f0)))])[0][0])


_st = json.load(open(os.path.join(REPO, "examples", "stacking_wall_tibb_940.json")))
_VM, _YM = np.asarray(_st["measured"]["bias"]), np.asarray(_st["measured"]["rg"], dtype=float)
PROM_M = low_prominence(_VM, savgol_filter(_YM, 9, 3), features5(_VM, _YM, smooth=9)[0])
