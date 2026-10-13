"""Observation store: Parquet files queried with DuckDB (design §2.1).

Layout under ``root``::

    observations/<source>.parquet   one file per source, replaced as a whole on rewrite
    geo_unit.parquet                units of all schemes
    geo_edge.parquet                parent-child edges with population and area weights
    indicator.parquet               registry entries (metadata only)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from regsim.schema import GeoEdge, GeoUnit, Lens

OBSERVATION_COLUMNS = {
    "geo_id": "string",
    "indicator_id": "string",
    "period": "string",
    "scenario": "string",
    "value": "float64",
    "component": "string",
    "se": "float64",
    "source": "string",
    "source_version": "string",
    "flags": "string",
}


@dataclass
class EntityMatrix:
    """Entity x indicator matrix on the original scale."""

    X: np.ndarray
    observed: np.ndarray
    geo_ids: list[str]
    indicator_ids: list[str]
    level: np.ndarray
    sources: np.ndarray  # source of each observed cell, "" where missing

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, index=self.geo_ids, columns=self.indicator_ids)


class ObservationStore:
    def __init__(self, root: str = "data/store"):
        self.root = Path(root)

    # ------------------------------------------------------------------ write
    def write(self, observations: pd.DataFrame, table: str = "observations") -> Path:
        """Write observations of one or more sources; existing files of these sources are replaced."""
        missing = [
            c for c in ("geo_id", "indicator_id", "period", "value", "source") if c not in observations
        ]
        if missing:
            raise ValueError(f"observations lack columns {missing}")
        obs = observations.copy()
        for col, dtype in OBSERVATION_COLUMNS.items():
            if col not in obs:
                obs[col] = None
            obs[col] = obs[col].astype(dtype)
        obs["scenario"] = obs["scenario"].fillna("observed")
        obs["flags"] = obs["flags"].fillna("")
        obs["source_version"] = obs["source_version"].fillna("")
        if obs[["geo_id", "indicator_id", "period", "source"]].isna().any().any():
            raise ValueError("geo_id, indicator_id, period and source must not be missing")
        obs = obs[list(OBSERVATION_COLUMNS) + [c for c in obs.columns if c not in OBSERVATION_COLUMNS]]
        folder = self.root / table
        folder.mkdir(parents=True, exist_ok=True)
        last = None
        for source, part in obs.groupby("source", sort=True):
            last = folder / f"{_slug(str(source))}.parquet"
            part.reset_index(drop=True).to_parquet(last, index=False)
        return last or folder

    def write_geo(self, units: list[GeoUnit], edges: list[GeoEdge]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(u) for u in units]).to_parquet(self.root / "geo_unit.parquet", index=False)
        pd.DataFrame(
            [asdict(e) for e in edges], columns=["child", "parent", "weight_pop", "weight_area"]
        ).to_parquet(self.root / "geo_edge.parquet", index=False)

    def write_frame(self, frame: pd.DataFrame, name: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return path

    # ------------------------------------------------------------------ read
    def connect(self) -> duckdb.DuckDBPyConnection:
        """DuckDB connection with one view per table found in the store."""
        con = duckdb.connect()
        obs = self.root / "observations"
        if obs.exists() and any(obs.glob("*.parquet")):
            con.execute(
                f"CREATE VIEW observations AS SELECT * FROM read_parquet('{_posix(obs)}/*.parquet', union_by_name=true)"
            )
        for f in self.root.glob("*.parquet"):
            con.execute(f"CREATE VIEW {f.stem} AS SELECT * FROM read_parquet('{_posix(f)}')")
        return con

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        con = self.connect()
        try:
            return con.execute(sql, params or []).df()
        finally:
            con.close()

    def observations(self) -> pd.DataFrame:
        return self.query("SELECT * FROM observations")

    def units(self) -> pd.DataFrame:
        return self.query("SELECT * FROM geo_unit")

    def matrix(
        self,
        lens: Lens | None = None,
        index_levels: list[int] | None = None,
        period: str | None = None,
        canonical: dict[str, str] | None = None,
        precedence: list[str] | None = None,
        geo_ids: list[str] | None = None,
    ) -> EntityMatrix:
        """Entity x indicator matrix (original scale) with the observed mask and row metadata.

        Parameters
        ----------
        lens : Lens, optional
            Indicator selection (``None`` = all) and scenario.
        index_levels : list of int, optional
            Levels of the rows; ``None`` = all levels in ``geo_unit``.
        period : str, optional
            A window ``"YYYY/YYYY"`` (mean of annual values in the window and of values reported for exactly
            that window) or a single period label.
        canonical : dict, optional
            Indicator id -> model indicator id (``IndicatorRegistry.canonical``); copies of an indicator are
            merged into one column, the first source by ``precedence`` winning per cell.
        precedence : list of str, optional
            Connector names in order of preference; matched against the prefix of ``source``.
        geo_ids : list of str, optional
            Explicit row order; overrides ``index_levels``.
        """
        lens = lens or Lens()
        units = self.units()
        if geo_ids is None:
            if index_levels is not None:
                units = units[units.level.isin(index_levels)]
            units = units.sort_values(["level", "geo_id"])
            geo_ids = units.geo_id.tolist()
        level_of = dict(zip(units.geo_id, units.level, strict=True))
        obs = self.query(
            "SELECT geo_id, indicator_id, period, value, source FROM observations WHERE scenario = ?",
            [lens.scenario],
        )
        obs = obs[obs.geo_id.isin(set(geo_ids))]
        if period is not None:
            obs = obs[_in_period(obs.period, period)]
        if canonical:
            obs = obs.assign(indicator_id=obs.indicator_id.map(lambda i: canonical.get(i, i)))
        indicators = (
            list(lens.indicators) if lens.indicators is not None else sorted(obs.indicator_id.unique())
        )
        obs = obs[obs.indicator_id.isin(set(indicators))]
        cell = obs.groupby(["geo_id", "indicator_id", "source"], as_index=False).value.mean()
        rank = {s: i for i, s in enumerate(precedence or [])}
        cell["rank"] = cell.source.map(lambda s: rank.get(s.split(":")[0], len(rank)))
        cell = cell.sort_values("rank").drop_duplicates(["geo_id", "indicator_id"])
        gi = {g: i for i, g in enumerate(geo_ids)}
        ki = {k: j for j, k in enumerate(indicators)}
        X = np.full((len(geo_ids), len(indicators)), np.nan)
        S = np.full(X.shape, "", dtype=object)
        r = cell.geo_id.map(gi).to_numpy()
        c = cell.indicator_id.map(ki).to_numpy()
        X[r, c] = cell.value.to_numpy()
        S[r, c] = cell.source.to_numpy()
        return EntityMatrix(
            X=X,
            observed=np.isfinite(X),
            geo_ids=list(geo_ids),
            indicator_ids=indicators,
            level=np.array([level_of.get(g, -1) for g in geo_ids]),
            sources=S,
        )


def _in_period(periods: pd.Series, period: str) -> pd.Series:
    if "/" not in period:
        return periods == period
    start, end = (int(x) for x in period.split("/"))
    year = pd.to_numeric(periods.str[:4], errors="coerce")
    annual = periods.str.fullmatch(r"\d{4}")
    return (annual & year.between(start, end)) | (periods == period)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _posix(p: Path) -> str:
    return p.resolve().as_posix()
