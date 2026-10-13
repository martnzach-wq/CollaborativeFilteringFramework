"""Typed transforms and robust scaling (design §3)."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def forward(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if kind == "identity":
        return x
    if kind == "log":
        return np.log(np.clip(x, EPS, None))
    if kind == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if kind == "logit":
        p = np.clip(x, EPS, 1 - EPS)
        return np.log(p / (1 - p))
    raise ValueError(f"unknown transform {kind}")


def inverse(z: np.ndarray, kind: str) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if kind == "identity":
        return z
    if kind == "log":
        return np.exp(z)
    if kind == "log1p":
        return np.expm1(z)
    if kind == "logit":
        return 1 / (1 + np.exp(-z))
    raise ValueError(f"unknown transform {kind}")


def clr(parts: np.ndarray) -> np.ndarray:
    """Centred log-ratio of compositions (rows sum to 1); zeros replaced by EPS."""
    p = np.clip(np.asarray(parts, dtype=float), EPS, None)
    lp = np.log(p)
    return lp - lp.mean(axis=-1, keepdims=True)


def robust_scale(col: np.ndarray) -> tuple[float, float]:
    """Median and IQR/1.349 of the finite values; falls back to the standard deviation."""
    v = np.asarray(col, dtype=float)
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    s = float((q3 - q1) / 1.349)
    if s <= 0:
        s = float(np.std(v)) or 1.0
    return med, s
