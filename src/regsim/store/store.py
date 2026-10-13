"""Observation store: Parquet files queried with DuckDB."""

from __future__ import annotations


class ObservationStore:
    def __init__(self, root: str = "data/store"):
        self.root = root

    def write(self, observations, table: str = "observations") -> None:
        raise NotImplementedError("M1")

    def query(self, sql: str):
        raise NotImplementedError("M1")

    def matrix(self, lens, index_levels, period):
        """Entity x indicator matrix (original scale) with the observed mask and row metadata."""
        raise NotImplementedError("M1")
