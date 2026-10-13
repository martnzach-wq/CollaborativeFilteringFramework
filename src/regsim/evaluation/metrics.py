"""Error and calibration metrics on the model scale."""

from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(pred, true, mask) -> float:
    e = (np.asarray(pred) - np.asarray(true))[mask]
    return float(np.sqrt(np.mean(e**2)))


def mae(pred, true, mask) -> float:
    e = (np.asarray(pred) - np.asarray(true))[mask]
    return float(np.mean(np.abs(e)))


def r2_per_column(pred, true, mask) -> np.ndarray:
    """Coefficient of determination per column over the masked cells (NaN with fewer than two cells)."""
    pred, true = np.asarray(pred), np.asarray(true)
    out = np.full(true.shape[1], np.nan)
    for j in range(true.shape[1]):
        m = mask[:, j]
        if m.sum() < 2:
            continue
        t, p = true[m, j], pred[m, j]
        ss = np.sum((t - t.mean()) ** 2)
        out[j] = 1 - np.sum((t - p) ** 2) / ss if ss > 0 else np.nan
    return out


def coverage(lo, hi, true, mask) -> float:
    t = np.asarray(true)[mask]
    return float(np.mean((np.asarray(lo)[mask] <= t) & (t <= np.asarray(hi)[mask])))


def interval_width(lo, hi, mask) -> float:
    return float(np.mean((np.asarray(hi) - np.asarray(lo))[mask]))


def crps_gaussian(mu, sigma, true, mask) -> float:
    """Mean continuous ranked probability score of Gaussian predictive distributions."""
    m, s, t = np.asarray(mu)[mask], np.asarray(sigma)[mask], np.asarray(true)[mask]
    s = np.maximum(s, 1e-12)
    z = (t - m) / s
    crps = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_samples(samples, true, mask) -> float:
    """Mean CRPS from predictive samples of shape ``(n_samples, n, p)``."""
    x = np.asarray(samples)[:, mask]
    t = np.asarray(true)[mask]
    term1 = np.mean(np.abs(x - t[None]), axis=0)
    xs = np.sort(x, axis=0)
    n = xs.shape[0]
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = np.sum(w * xs, axis=0) / (n * n)
    return float(np.mean(term1 - term2))
