"""Connector for curated tables kept in the repository (decision D54).

A table is a CSV file with the columns ``geo_code``, ``indicator_id``, ``period``, ``value``, ``source``,
``source_version`` and ``flags``; it holds values transcribed from reports, such as the coping-type shares of
the KNOWING deliverables. The registry entry names the file as ``dataset``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ("geo_code", "indicator_id", "period", "value", "source", "source_version", "flags")


class TableConnector:
    name = "table"

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    def read(self, dataset: str) -> pd.DataFrame:
        d = pd.read_csv(self.root / dataset, dtype={"period": str, "geo_code": str})
        missing = set(COLUMNS) - set(d.columns)
        if missing:
            raise ValueError(f"{dataset}: missing columns {sorted(missing)}")
        return d[list(COLUMNS)].fillna({"source_version": "", "flags": ""})

    def fetch(self, spec: dict) -> pd.DataFrame:
        """Observations of ``spec["indicator_id"]`` from ``spec["dataset"]`` in the store's long format."""
        d = self.read(spec["dataset"])
        d = d[d.indicator_id == spec["indicator_id"]].copy()
        d["scenario"] = "observed"
        return d.reset_index(drop=True)
