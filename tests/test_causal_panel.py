"""Unit tests for regsim.causal.panel on synthetic panels (no network)."""

import numpy as np
import pandas as pd

from regsim.causal.panel import fit_edge, heterogeneity


def _panel(beta: float, n_geo: int = 120, n_year: int = 15, seed: int = 0, group_beta: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    parents = [f"P{i}" for i in range(8)]
    shocks = {(p, t): rng.normal() for p in parents for t in range(n_year)}
    for g in range(n_geo):
        p = parents[g % len(parents)]
        b = (group_beta or {}).get(g % 2, beta)
        x = y = 0.0
        a = rng.normal()
        for t in range(n_year):
            dx = rng.normal() + 0.5 * shocks[(p, t)]
            x += dx
            y += a * 0.01 + b * dx + 0.3 * shocks[(p, t)] + rng.normal(scale=0.5)
            rows.append({"geo_id": f"G{g}", "parent_id": p, "year": 2000 + t, "x": x, "y": y, "grp": g % 2})
    return pd.DataFrame(rows)


def test_fit_edge_recovers_effect():
    e = fit_edge(_panel(0.4), "x", "y")
    assert e is not None
    assert abs(e.beta_raw - 0.4) < 0.1
    assert e.t > 5


def test_fit_edge_null():
    e = fit_edge(_panel(0.0, seed=1), "x", "y")
    assert abs(e.t) < 3


def test_heterogeneity_detects_groups():
    df = _panel(0.0, group_beta={0: 0.0, 1: 0.6}, seed=2)
    groups = df.drop_duplicates("geo_id").set_index("geo_id")["grp"]
    _, q = heterogeneity(df, "x", "y", groups)
    assert q["p"] < 0.01
