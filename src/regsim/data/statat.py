"""Connector for Statistik Austria open data (data.statistik.gv.at), population by municipality and age.

The OGD table ``OGD_bevstandjbab2002_BevStand_<year>`` holds residents on 1 January of ``<year>`` by sex,
municipality and single year of age (CC BY 4.0). Municipality codes are 5-digit Gemeindekennziffern, equal to
the Austrian LAU codes; Vienna is published by district (901xx–923xx) and is summed to LAU 90001. Code 0
(not classifiable) is dropped.

The tables use the current boundaries of the publication date. Municipalities created by mergers after the LAU
version of the store are mapped back to their predecessors with the ``successors`` table of the source spec:
intensive values (shares) are copied to every predecessor, extensive values (counts) are split by the
predecessors' population weights. Both are flagged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from regsim.data.files import download

SOURCE = "statat"
BASE_URL = "https://data.statistik.gv.at/data/"
COUNTRY = "AT"
VIENNA = "90001"
_DIM = {"year": "C-A10-0", "sex": "C-C11-0", "municipality": "C-GRGEMAKT-0", "age": "C-GALTEJ112-0"}
_COUNT = "F-ISIS-1"


class StatAtConnector:
    """Population counts and age shares per LAU from the Statistik Austria OGD population table.

    Parameters
    ----------
    cache_dir
        Folder for the downloaded CSV files (``data/raw/statat``); files that exist are not downloaded again.
    lau_population
        Population per LAU code of the store's LAU version; weights for splitting counts of a successor
        municipality among its predecessors.
    base_url
        Base URL of the open-data file server.
    """

    name = SOURCE

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/statat",
        lau_population: dict[str, float] | None = None,
        base_url: str = BASE_URL,
    ):
        self.cache_dir = Path(cache_dir)
        self.lau_population = lau_population or {}
        self.base_url = base_url

    # ------------------------------------------------------------------ raw table
    def _file(self, name: str) -> Path:
        return download(self.base_url + name, self.cache_dir / name)

    def table(self, dataset: str, year: int) -> pd.DataFrame:
        """Counts by LAU code and age (both sexes), with Vienna districts summed and code 0 dropped."""
        stem = f"{dataset}_{year}"
        data = pd.read_csv(self._file(f"{stem}.csv"), sep=";", dtype=str)
        ages = pd.read_csv(self._file(f"{stem}_{_DIM['age']}.csv"), sep=";", dtype=str)
        age_of = {c: _age(n) for c, n in zip(ages["code"], ages["name"], strict=True)}
        out = pd.DataFrame(
            {
                "lau_code": data[_DIM["municipality"]].str.split("-", n=1).str[1],
                "age": data[_DIM["age"]].map(age_of),
                "count": pd.to_numeric(data[_COUNT]),
            }
        )
        if out.age.isna().any():
            raise ValueError(
                f"{stem}: unknown age codes {sorted(data[_DIM['age']][out.age.isna()].unique())[:5]}"
            )
        out = out[out.lau_code != "0"]
        out.loc[out.lau_code.str.fullmatch(r"9\d{4}"), "lau_code"] = VIENNA
        return out.groupby(["lau_code", "age"], as_index=False)["count"].sum()

    # ------------------------------------------------------------------ connector interface
    def fetch(self, spec: dict) -> pd.DataFrame:
        """Observations for one indicator.

        ``spec`` keys: ``indicator_id``, ``dataset`` (without year), ``year``, ``measure`` (``population`` or
        ``share_age``), ``min_age`` and optional ``max_age`` for ``share_age``, optional ``successors``
        (new code -> list of predecessor codes).
        """
        year = int(spec["year"])
        t = self.table(spec["dataset"], year)
        total = t.groupby("lau_code")["count"].sum()
        measure = spec["measure"]
        if measure == "population":
            value, extensive = total.astype(float), True
        elif measure == "share_age":
            lo, hi = spec["min_age"], spec.get("max_age", 10_000)
            part = t[(t.age >= lo) & (t.age <= hi)].groupby("lau_code")["count"].sum()
            value, extensive = (part.reindex(total.index, fill_value=0) / total).astype(float), False
        else:
            raise ValueError(f"unknown measure {measure!r}")
        obs = pd.DataFrame({"geo_code": value.index, "value": value.values, "flags": ""})
        obs = self._to_predecessors(obs, spec.get("successors") or {}, extensive)
        return obs.assign(
            country=COUNTRY,
            indicator_id=spec["indicator_id"],
            period=str(year),
            scenario="observed",
            source=SOURCE,
            source_version=f"{spec['dataset']}_{year}",
        )[
            [
                "geo_code",
                "country",
                "indicator_id",
                "period",
                "scenario",
                "value",
                "source",
                "source_version",
                "flags",
            ]
        ]

    def _to_predecessors(self, obs: pd.DataFrame, successors: dict, extensive: bool) -> pd.DataFrame:
        rows = []
        for new, olds in successors.items():
            hit = obs[obs.geo_code == str(new)]
            if hit.empty:
                continue
            v = float(hit.value.iloc[0])
            if extensive:
                w = pd.Series({o: self.lau_population.get(str(o), float("nan")) for o in olds}, dtype=float)
                if w.isna().any() or w.sum() <= 0:
                    raise ValueError(f"successor {new}: no population weights for {list(w.index[w.isna()])}")
                rows += [(str(o), v * w[o] / w.sum(), f"split_from:{new}") for o in olds]
            else:
                rows += [(str(o), v, f"copied_from:{new}") for o in olds]
            obs = obs[obs.geo_code != str(new)]
        if rows:
            obs = pd.concat(
                [obs, pd.DataFrame(rows, columns=["geo_code", "value", "flags"])], ignore_index=True
            )
        return obs.reset_index(drop=True)


def _age(name: str) -> int:
    """Leading integer of an age label ("0 Jahre", "100 Jahre und älter")."""
    m = re.match(r"\s*(\d+)", str(name))
    if not m:
        raise ValueError(f"age label without number: {name!r}")
    return int(m.group(1))
