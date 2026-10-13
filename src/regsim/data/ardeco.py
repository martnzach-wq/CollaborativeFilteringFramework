"""JRC ARDECO connector (GraphQL metadata, CSV export per NUTS level).

ARDECO publishes regional series harmonised to one NUTS version, extended backwards and, for recent years,
nowcast. All its observations carry the flag ``ardeco`` and enter the model only where no Eurostat value exists
(decision D5).
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from regsim.data.http import CachedClient, SourceError, as_list

BASE = "https://territorial.ec.europa.eu/ardeco-api-v2/"
GRAPHQL = BASE + "graphql"
EXPORT = BASE + "rest/export/{variable}"
FLAG = "ardeco"


class ArdecoError(SourceError):
    pass


class ArdecoConnector:
    """Fetch ARDECO variables as observations in long format.

    Parameters
    ----------
    cache_dir : str
        Directory for raw responses (``data/raw/ardeco`` by convention).
    version : str
        NUTS version of the export.
    session, sleep, offline, max_retries, timeout
        As for :class:`regsim.data.http.CachedClient`.
    """

    name = "ardeco"

    def __init__(
        self,
        cache_dir: str = "data/raw/ardeco",
        version: str = "2024",
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        offline: bool = False,
        max_retries: int = 4,
        timeout: float = 300.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.client = CachedClient(
            cache_dir,
            session=session,
            sleep=sleep,
            offline=offline,
            max_retries=max_retries,
            timeout=timeout,
            error=ArdecoError,
        )

    def datasets(self, variable: str) -> list[dict[str, str]]:
        """Dimension combinations available for a variable, e.g. ``[{"unit": "NR"}]``."""
        query = (
            f'{{ variable(id: "{variable}") {{ code datasets {{ datasetId dimensions {{ key value }} }} }} }}'
        )
        payload = json.loads(self.client.post_json(GRAPHQL, {"query": query}, kind="meta"))
        var = (payload.get("data") or {}).get("variable")
        if not var:
            raise ArdecoError(f"unknown ARDECO variable {variable}: {payload.get('errors')}")
        return [{d["key"]: d["value"] for d in ds["dimensions"]} for ds in var["datasets"]]

    def validate(self, variable: str, filters: dict[str, Any]) -> None:
        """Raise ``ValueError`` unless the filters select exactly one dataset of the variable."""
        _check(filters)
        combos = self.datasets(variable)
        want = {k: str(v) for k, v in filters.items()}
        if any(set(want) - set(c) for c in combos):
            raise ValueError(f"{variable}: unknown dimensions {sorted(set(want) - set(combos[0]))}")
        match = [c for c in combos if all(c[k] == v for k, v in want.items())]
        if len(match) != 1:
            raise ValueError(f"{variable}: filters {want} select {len(match)} of {len(combos)} datasets")

    def fetch(self, spec: dict) -> pd.DataFrame:
        """Fetch one variable.

        Parameters
        ----------
        spec : dict
            ``dataset`` (ARDECO variable code), ``indicator_id``, ``filters`` (one value per dimension),
            ``levels`` (default 0–3), ``since``, ``until``, ``validate`` (default True).
        """
        variable = spec["dataset"]
        filters = dict(spec.get("filters") or {})
        if spec.get("validate", True):
            self.validate(variable, filters)
        frames = []
        for level in spec.get("levels", (0, 1, 2, 3)):
            params = {"version": self.version, "level_id": level, **filters}
            text = self.client.get(EXPORT.format(variable=variable), params, kind="data", fmt="csv")
            if text.strip():
                frames.append(pd.read_csv(io.StringIO(text), dtype={"TERRITORY_ID": str}))
        if not frames:
            return _empty()
        raw = pd.concat(frames, ignore_index=True)
        year = pd.to_numeric(raw.YEAR, errors="coerce")
        keep = raw.VALUE.notna()
        if spec.get("since") is not None:
            keep &= year >= int(spec["since"])
        if spec.get("until") is not None:
            keep &= year <= int(spec["until"])
        raw = raw[keep]
        return pd.DataFrame(
            {
                "geo_code": raw.TERRITORY_ID.astype(str).values,
                "indicator_id": spec.get("indicator_id", variable),
                "period": raw.YEAR.astype(int).astype(str).values,
                "scenario": "observed",
                "value": raw.VALUE.astype(float).values,
                "component": None,
                "se": np.nan,
                "source": f"ardeco:{variable}",
                "source_version": f"NUTS{self.version}",
                "flags": FLAG,
            }
        )


def _empty() -> pd.DataFrame:
    cols = [
        "geo_code",
        "indicator_id",
        "period",
        "scenario",
        "value",
        "component",
        "se",
        "source",
        "source_version",
        "flags",
    ]
    return pd.DataFrame({c: [] for c in cols})


def _check(filters: dict) -> None:
    for k, v in filters.items():
        if len(as_list(v)) != 1:
            raise ValueError(f"ARDECO filter {k} needs exactly one value")
