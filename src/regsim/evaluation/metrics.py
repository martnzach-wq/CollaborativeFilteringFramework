"""Error and calibration metrics."""

from __future__ import annotations

import numpy as np


def rmse(pred, true, mask) -> float:
    e = (np.asarray(pred) - np.asarray(true))[mask]
    return float(np.sqrt(np.mean(e**2)))


def mae(pred, true, mask) -> float:
    e = (np.asarray(pred) - np.asarray(true))[mask]
    return float(np.mean(np.abs(e)))


def coverage(lo, hi, true, mask) -> float:
    t = np.asarray(true)[mask]
    return float(np.mean((np.asarray(lo)[mask] <= t) & (t <= np.asarray(hi)[mask])))
