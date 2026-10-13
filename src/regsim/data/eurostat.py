"""Eurostat dissemination API connector (JSON-stat 2.0 data, SDMX 2.1 content constraints for validation)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

BASE = "https://ec.europa.eu/eurostat/api/dissemination/"
DATA_URL = BASE + "statistics/1.0/data/{dataset}"
CONSTRAINT_URL = BASE + "sdmx/2.1/contentconstraint/ESTAT/{dataset}"
_DIM_ALIASES = {"time": "time_period"}
_SDMX_NS = {"c": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common"}

OBS_COLUMNS = [
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


class EurostatError(RuntimeError):
    pass


class EurostatConnector:
    """Fetch Eurostat datasets as observations in long format.

    Responses are cached under ``cache_dir`` by request hash; a cached request is never sent again.
    HTTP 413 and asynchronous-processing responses are retried with exponential back-off; if they persist,
    the request is split by period.

    Parameters
    ----------
    cache_dir : str
        Directory for raw responses (``data/raw/eurostat`` by convention).
    max_retries : int
        Number of retries for 413, 429, 5xx and asynchronous responses.
    session : requests.Session, optional
        HTTP session; replaced by a stub in tests.
    sleep : callable
        Sleep function used between retries; replaced in tests.
    offline : bool
        Raise instead of sending a request when the response is not cached.
    """

    name = "eurostat"

    def __init__(
        self,
        cache_dir: str = "data/raw/eurostat",
        max_retries: int = 6,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        offline: bool = False,
        timeout: float = 120.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.sleep = sleep
        self.offline = offline
        self.timeout = timeout

    # ------------------------------------------------------------------ structure and validation
    def structure(self, dataset: str) -> dict[str, list[str]]:
        """Dimensions and the codes actually present in a dataset (SDMX content constraint)."""
        text = self._get(CONSTRAINT_URL.format(dataset=dataset), {}, kind="structure", fmt="xml")
        root = ElementTree.fromstring(text)
        dims: dict[str, list[str]] = {}
        for kv in root.iter(f"{{{_SDMX_NS['c']}}}KeyValue"):
            dims[kv.attrib["id"].lower()] = [v.text or "" for v in kv.findall("c:Value", _SDMX_NS)]
        if not dims:
            raise EurostatError(f"no content constraint for dataset {dataset}")
        return dims

    def validate(self, dataset: str, filters: dict[str, Any]) -> None:
        """Raise ``ValueError`` if a filter names an unknown dimension or a code absent from the dataset."""
        dims = self.structure(dataset)
        problems = []
        for dim, values in filters.items():
            dim = _DIM_ALIASES.get(dim.lower(), dim.lower())
            if dim not in dims:
                problems.append(f"dimension '{dim}' not in {dataset} (available: {sorted(dims)})")
                continue
            missing = [v for v in _as_list(values) if str(v) not in dims[dim]]
            if missing:
                problems.append(f"codes {missing} not available for '{dim}' in {dataset}")
        if problems:
            raise ValueError("; ".join(problems))

    # ------------------------------------------------------------------ data
    def fetch(self, spec: dict) -> pd.DataFrame:
        """Fetch one indicator.

        Parameters
        ----------
        spec : dict
            ``dataset`` (required), ``indicator_id`` (default: dataset), ``filters`` (dimension -> code or
            list of codes), ``since`` and ``until`` (years), ``validate`` (default True).

        Returns
        -------
        pandas.DataFrame
            Observation columns with the raw Eurostat geo code in ``geo_code``; harmonisation to a
            scheme-qualified ``geo_id`` happens in the build step. Dimensions other than ``geo`` and
            ``time`` that are not fixed by a filter are kept as extra columns.
        """
        dataset = spec["dataset"]
        filters = dict(spec.get("filters") or {})
        if spec.get("validate", True):
            self.validate(dataset, filters)
        params: dict[str, Any] = {"format": "JSON", "lang": "EN", **filters}
        if spec.get("since") is not None:
            params["sinceTimePeriod"] = str(spec["since"])
        if spec.get("until") is not None:
            params["untilTimePeriod"] = str(spec["until"])
        try:
            payload = json.loads(self._get(DATA_URL.format(dataset=dataset), params, kind="data"))
            frame = parse_jsonstat(payload)
        except _TooLarge:
            frame = self._fetch_split(dataset, params, spec)
            payload = {}
        return _to_observations(frame, spec, payload.get("updated", ""))

    def _fetch_split(self, dataset: str, params: dict, spec: dict) -> pd.DataFrame:
        since, until = spec.get("since"), spec.get("until")
        if since is None or until is None or int(since) == int(until):
            raise EurostatError(f"{dataset}: request too large and cannot be split further by period")
        frames = []
        for year in range(int(since), int(until) + 1):
            p = {**params, "sinceTimePeriod": str(year), "untilTimePeriod": str(year)}
            frames.append(
                parse_jsonstat(json.loads(self._get(DATA_URL.format(dataset=dataset), p, kind="data")))
            )
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ HTTP with cache and retry
    def _get(self, url: str, params: dict, kind: str, fmt: str = "json") -> str:
        key = _request_key(url, params)
        path = self.cache_dir / kind / f"{key}.{fmt}"
        if path.exists():
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise EurostatError(f"not cached and offline: {url} {params}")
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            r = self.session.get(url, params=_encode(params), timeout=self.timeout)
            if r.status_code == 200 and not _is_async(r.text):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(r.text, encoding="utf-8")
                return r.text
            retryable = r.status_code in (413, 429) or r.status_code >= 500 or _is_async(r.text)
            if not retryable:
                raise EurostatError(f"HTTP {r.status_code} for {r.url}: {r.text[:300]}")
            if attempt < self.max_retries:
                self.sleep(delay)
                delay *= 2
        if r.status_code == 413 or _is_async(r.text):
            raise _TooLarge(url)
        raise EurostatError(f"HTTP {r.status_code} after {self.max_retries} retries: {r.url}")


class _TooLarge(EurostatError):
    pass


def parse_jsonstat(payload: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into one row per non-missing value with a ``flag`` column."""
    ids: list[str] = payload["id"]
    sizes: list[int] = payload["size"]
    cats = []
    for dim in ids:
        index = payload["dimension"][dim]["category"]["index"]
        if isinstance(index, dict):
            codes = [None] * len(index)
            for code, pos in index.items():
                codes[pos] = code
        else:
            codes = list(index)
        cats.append(np.asarray(codes, dtype=object))
    values = payload.get("value", {})
    if isinstance(values, list):
        flat = np.array([i for i, v in enumerate(values) if v is not None], dtype=np.int64)
        vals = np.array([values[i] for i in flat], dtype=float)
    else:
        flat = np.array([int(k) for k in values], dtype=np.int64)
        vals = np.array(list(values.values()), dtype=float)
    status = payload.get("status", {}) or {}
    if isinstance(status, list):
        flags = [status[i] or "" if i < len(status) else "" for i in flat]
    else:
        flags = [status.get(str(i), "") for i in flat]
    coords = np.unravel_index(flat, sizes) if len(flat) else [np.array([], dtype=int)] * len(ids)
    frame = pd.DataFrame({dim: cats[j][coords[j]] for j, dim in enumerate(ids)})
    frame["value"] = vals
    frame["flag"] = flags
    return frame


def _to_observations(frame: pd.DataFrame, spec: dict, updated: str) -> pd.DataFrame:
    dataset = spec["dataset"]
    fixed = {k.lower() for k in (spec.get("filters") or {})}
    extra = [c for c in frame.columns if c not in ("geo", "time", "value", "flag", "freq") and c not in fixed]
    out = pd.DataFrame(
        {
            "geo_code": frame["geo"].astype(str),
            "indicator_id": spec.get("indicator_id", dataset),
            "period": frame["time"].astype(str),
            "scenario": "observed",
            "value": frame["value"].astype(float),
            "component": None,
            "se": np.nan,
            "source": f"eurostat:{dataset}",
            "source_version": updated,
            "flags": frame["flag"].fillna("").astype(str),
        }
    )
    for c in extra:
        out[c] = frame[c].values
    return out.reset_index(drop=True)


def _as_list(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _encode(params: dict) -> list[tuple[str, str]]:
    return [(k, str(x)) for k, v in params.items() for x in _as_list(v)]


def _request_key(url: str, params: dict) -> str:
    canon = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(_encode(params)))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", url.rsplit("/", 1)[-1])[:40]
    return f"{slug}_{hashlib.sha256(canon.encode()).hexdigest()[:16]}"


def _is_async(text: str) -> bool:
    head = text[:2000].lower()
    return "asynchronous" in head and ("warning" in head or "error" in head)
