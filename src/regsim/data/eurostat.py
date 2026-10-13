"""Eurostat dissemination API connector (JSON-stat 2.0; SDMX optional)."""
from __future__ import annotations

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"


class EurostatConnector:
    name = "eurostat"

    def __init__(self, cache_dir: str = "data/raw/eurostat", max_retries: int = 6):
        self.cache_dir = cache_dir
        self.max_retries = max_retries

    def structure(self, dataset: str) -> dict:
        """Dimensions and codes of a dataset, used to validate filters before querying."""
        raise NotImplementedError("M1")

    def fetch(self, spec: dict):
        """spec: {dataset, filters, since, until}. Retries HTTP 413 (asynchronous) with back-off."""
        raise NotImplementedError("M1")
