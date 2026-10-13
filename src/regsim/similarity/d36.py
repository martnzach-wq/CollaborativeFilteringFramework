"""KNOWING D3.6 compatibility mode: Four-Tier Weighted Similarity on the D3.6 normalised values.

Within a tier, the distance between a demonstrator d and a target t is a weighted Gower distance

    D_T(d, t) = sum_k w_k delta_k p_k / sum_k w_k delta_k

over the tier's indicators for the climate impact context (CIC), with delta_k = 1 where both values exist and
partial distances p_k: absolute difference of min-max normalised values (continuous), match (binary
categorical), rank difference over the number of levels minus one (hierarchical categorical), and the Köppen distance of
:func:`koppen_distance`. The composite distance is the tier-weighted sum,
the score is (1 - D) x 100 and the transferability classes TRF1-TRF5 follow D3.6 Table 12 (decision D8: classes
only in this mode). Tiers and weights are read from ``configs/d36_scoring.yaml``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = "configs/d36_scoring.yaml"
TIERS = ("T1", "T2", "T3", "T4")


@dataclass
class D36Tables:
    """Wide tables of the D3.6 normalised values (rows: D3.6 region geocodes)."""

    numeric: pd.DataFrame  # normalised continuous values
    text: pd.DataFrame  # categorical codes as strings
    level: pd.Series


def load_tables(
    gpkg_path: str | Path, regions: list[str] | None = None, levels: list[int] | None = None
) -> D36Tables:
    """Read ``knowing.indicator_normalised_values`` into wide tables, optionally for some regions or levels."""
    con = sqlite3.connect(f"file:{Path(gpkg_path).as_posix()}?mode=ro", uri=True)
    where = []
    if regions:
        where.append("region_geocode IN ({})".format(",".join(f"'{r}'" for r in regions)))
    if levels:
        where.append("level IN ({})".format(",".join(str(int(x)) for x in levels)))
    cond = (" WHERE " + " OR ".join(where)) if where else ""
    try:
        raw = pd.read_sql(
            "SELECT DISTINCT region_geocode, level, indicator_id, normalised_value, value_text "
            f'FROM "knowing.indicator_normalised_values"{cond}',
            con,
        )
    finally:
        con.close()
    num = raw.dropna(subset=["normalised_value"]).pivot_table(
        index="region_geocode", columns="indicator_id", values="normalised_value", aggfunc="first"
    )
    txt = raw.dropna(subset=["value_text"]).pivot_table(
        index="region_geocode", columns="indicator_id", values="value_text", aggfunc="first"
    )
    level = raw.drop_duplicates("region_geocode").set_index("region_geocode").level
    return D36Tables(num, txt, level)


class D36Scorer:
    """Score target regions against a demonstrator the D3.6 way.

    Parameters
    ----------
    tables : D36Tables
        Normalised values (see :func:`load_tables`).
    config : str or dict
        Scoring configuration (tiers, weights, categorical rules, TRF thresholds).
    """

    def __init__(self, tables: D36Tables, config: str | dict = DEFAULT_CONFIG):
        self.tables = tables
        self.config = (
            yaml.safe_load(Path(config).read_text(encoding="utf-8")) if isinstance(config, str) else config
        )

    def score(
        self,
        demonstrator: str,
        cic: str,
        targets: list[str] | None = None,
        weight_scheme: str = "default",
        contributions: bool = False,
    ) -> pd.DataFrame:
        """Tier distances and scores, composite distance and score, TRF class and coverage per target.

        With ``contributions=True`` the per-indicator partial distances (D3.6 Layer 4) are added as columns
        ``p:<tier>:<indicator>``.
        """
        tiers = self.config["cics"][cic]
        tw = self.config["tier_weights"][weight_scheme]
        targets = list(targets) if targets is not None else self._all_regions()
        out = pd.DataFrame(index=pd.Index(targets, name="target_geocode"))
        composite = np.zeros(len(targets))
        weight_sum = np.zeros(len(targets))
        n_obs = np.zeros(len(targets))
        n_all = 0
        for tier in TIERS:
            weights = tiers.get(tier, {})
            num = np.zeros(len(targets))
            den = np.zeros(len(targets))
            for k, w in weights.items():
                p = self._partial(demonstrator, targets, k)
                ok = np.isfinite(p)
                num += np.where(ok, w * p, 0.0)
                den += np.where(ok, w, 0.0)
                n_obs += ok
                n_all += 1
                if contributions:
                    out[f"p:{tier}:{k}"] = p
            empty = self.config.get("empty_tier_distance")
            with np.errstate(invalid="ignore", divide="ignore"):
                dist = np.where(den > 0, num / den, np.nan if empty is None else float(empty))
            out[f"tier{tier[1]}_distance"] = dist
            out[f"tier{tier[1]}_score"] = (1 - dist) * 100
            ok = np.isfinite(dist)
            composite += np.where(ok, tw[tier] * dist, 0.0)
            weight_sum += np.where(ok, tw[tier], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["composite_distance"] = np.where(weight_sum > 0, composite / weight_sum, np.nan)
        out["composite_score"] = (1 - out.composite_distance) * 100
        out["transferability_class"] = [self._trf(s) for s in out.composite_score]
        out["coverage_ratio"] = n_obs / max(n_all, 1)
        return out

    # ------------------------------------------------------------------ partial distances
    def _partial(self, demo: str, targets: list[str], k: str) -> np.ndarray:
        rule = self.config.get("categorical", {}).get(k)
        if rule is None:
            col = self.tables.numeric.get(k)
            if col is None or demo not in col.index:
                return np.full(len(targets), np.nan)
            d = col.get(demo, np.nan)
            return np.abs(col.reindex(targets).to_numpy(dtype=float) - d)
        col = self.tables.text.get(k)
        if col is None:
            return np.full(len(targets), np.nan)
        d = col.get(demo)
        t = col.reindex(targets)
        if d is None or (isinstance(d, float) and np.isnan(d)):
            return np.full(len(targets), np.nan)
        out = np.full(len(targets), np.nan)
        ok = t.notna().to_numpy()
        vals = t[ok].astype(str).to_numpy()
        if rule == "binary":
            out[ok] = (vals != str(d)).astype(float)
        elif rule == "hierarchical":
            levels = pd.to_numeric(col.dropna(), errors="coerce")
            span = max(float(levels.max() - levels.min()), 1.0)
            out[ok] = np.abs(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy() - float(d)) / span
        elif rule == "koppen":
            out[ok] = [koppen_distance(str(d), v) for v in vals]
        else:
            raise ValueError(f"unknown categorical rule {rule}")
        return out

    def _trf(self, score: float) -> str | None:
        if not np.isfinite(score):
            return None
        for label, lower in self.config["trf_classes"]:
            if score >= lower:
                return label
        return self.config["trf_classes"][-1][0]

    def _all_regions(self) -> list[str]:
        return sorted(set(self.tables.numeric.index) | set(self.tables.text.index))


def koppen_distance(a: str, b: str) -> float:
    """Partial distance of two Köppen-Geiger codes: 0 identical, 0.25 if two of the three letters agree by position,
    0.6 if only the main group agrees, 1 otherwise."""
    if a == b:
        return 0.0
    matches = sum(x == y for x, y in zip(a.ljust(3), b.ljust(3), strict=True))
    if matches >= 2:
        return 0.25
    if a[:1] == b[:1]:
        return 0.6
    return 1.0


def published_scores(
    gpkg_path: str | Path, demonstrator: str, weight_scheme: str = "default", target_level: int | None = None
) -> pd.DataFrame:
    """The ``knowing.precomputed_scores`` rows of one demonstrator."""
    con = sqlite3.connect(f"file:{Path(gpkg_path).as_posix()}?mode=ro", uri=True)
    cond = f" AND target_level = {int(target_level)}" if target_level is not None else ""
    try:
        return pd.read_sql(
            'SELECT * FROM "knowing.precomputed_scores" WHERE demonstrator_geocode = ? AND weight_scheme = ?'
            + cond,
            con,
            params=[demonstrator, weight_scheme],
        ).set_index("target_geocode")
    finally:
        con.close()
