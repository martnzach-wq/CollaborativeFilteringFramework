"""Connector plug-in interface."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Connector(Protocol):
    name: str

    def fetch(self, spec: dict) -> pd.DataFrame:
        """Return observations in the long format of `regsim.schema.Observation`."""
        ...
