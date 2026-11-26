"""Coping-type distributions for the SD behaviour model in regions without a survey (decision D54).

Observed shares of the KNOWING Climate Impact Coping Typology (types A–F) come from the curated table
``configs/data/coping_types_knowing.csv``: national samples of six countries (D1.3) and the settlement-type
columns used for the demonstrators (D3.3–D3.5). A country without a survey gets the kernel-weighted
composition of the surveyed countries on country features (`regsim.similarity.transfer.estimate_composition`);
a settlement type (metropolitan, urban, rural) shifts the national composition by the mean log-ratio
difference between the demonstrator columns and their national samples.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from regsim.data.table import TableConnector
from regsim.similarity.transfer import _closure, _clr, estimate_composition

TYPES = ("a", "b", "c", "d", "e", "f")
TABLE = "configs/data/coping_types_knowing.csv"
# country features after the D1.3 reading of the national differences: climate impacts experienced,
# income, age and education
FEATURES = ("cooling_degree_days", "gdp_pps_hab", "median_age", "tertiary_attainment")


def observed(table: str | Path = TABLE) -> tuple[pd.DataFrame, pd.Series]:
    """Shares (geo_code x type) and the settlement type of each row (``national`` for NUTS-0)."""
    d = TableConnector().read(str(table))
    d = d[d.indicator_id.str.startswith("coping_share_")]
    shares = d.pivot_table(index="geo_code", columns="indicator_id", values="value")
    shares.columns = [c.removeprefix("coping_share_") for c in shares.columns]
    shares = shares[list(TYPES)]
    kind = (
        d.groupby("geo_code")
        .flags.first()
        .map(lambda f: f.split(" column")[0] if "column" in f else "national")
    )
    return shares.div(shares.sum(axis=1), axis=0), kind


def country_features(codes: list[str], store: str | Path = "data/store", features=FEATURES) -> pd.DataFrame:
    """Mean 2015–2022 of ``features`` for NUTS-0 ``codes`` from the observation store."""
    geo = ", ".join(f"'NUTS2024:{c}'" for c in codes)
    ind = ", ".join(f"'{f}'" for f in features)
    q = f"""select geo_id, indicator_id, avg(value) v from '{Path(store) / "observations" / "*.parquet"}'
            where geo_id in ({geo}) and indicator_id in ({ind}) and period between '2015' and '2022'
            group by 1, 2"""
    d = duckdb.connect().execute(q).df()
    d["geo"] = d.geo_id.str.split(":").str[-1]
    return d.pivot_table(index="geo", columns="indicator_id", values="v")[list(features)].reindex(codes)


def settlement_offsets(shares: pd.DataFrame, kind: pd.Series) -> dict[str, np.ndarray]:
    """Mean clr difference between each settlement-type column and its national sample."""
    clr = pd.DataFrame(_clr(shares.to_numpy()), index=shares.index, columns=shares.columns)
    diffs: dict[str, list[np.ndarray]] = {}
    for code, k in kind.items():
        if k != "national":
            diffs.setdefault(k, []).append((clr.loc[code] - clr.loc[code[:2]]).to_numpy())
    return {k: np.mean(v, axis=0) for k, v in diffs.items()}


def estimate(
    country: str,
    settlement: str | None = None,
    store: str | Path = "data/store",
    table: str | Path = TABLE,
) -> dict:
    """Coping-type shares A–F for a region in ``country`` (NUTS-0 code) and of ``settlement`` type.

    Surveyed countries keep their national sample; other countries get the estimate from the surveyed ones.
    Returns the shares, the 10–90 % range of the donors (estimated countries only), the donor weights and the
    leave-one-out errors (Aitchison distance) of the kernel and of the plain mean."""
    shares, kind = observed(table)
    national = shares[kind == "national"]
    out: dict = {"country": country, "settlement": settlement}
    if country in national.index:
        base = _clr(national.loc[[country]].to_numpy())[0]
        out.update(source="national sample", lower=None, upper=None, weights=None)
    else:
        donors = list(national.index)
        feats = country_features(donors + [country], store)
        r = estimate_composition(national, feats, [country])
        base = _clr(r.estimate.to_numpy())[0]
        out.update(
            source="estimated from the surveyed countries",
            lower=r.lower.loc[country],
            upper=r.upper.loc[country],
            weights=r.weights.loc[country],
            bandwidth=r.bandwidth,
            loo_error=r.loo_error,
            loo_error_mean=r.loo_error_mean,
        )
    if settlement is not None:
        base = base + settlement_offsets(shares, kind)[settlement]
    out["shares"] = pd.Series(_closure(base), index=list(TYPES))
    return out


def leave_one_out(store: str | Path = "data/store", table: str | Path = TABLE) -> pd.DataFrame:
    """Aitchison error of each surveyed country estimated from the other five (kernel and plain mean), and
    of each demonstrator column from its national sample plus the offset of the other demonstrators of the
    same settlement type (where there is one)."""
    shares, kind = observed(table)
    national = shares[kind == "national"]
    feats = country_features(list(national.index), store)
    rows = []
    for c in national.index:
        donors = national.drop(c)
        r = estimate_composition(donors, feats, [c])
        err = np.linalg.norm(_clr(r.estimate.to_numpy())[0] - _clr(national.loc[[c]].to_numpy())[0])
        mean = np.linalg.norm(_clr(donors.to_numpy()).mean(axis=0) - _clr(national.loc[[c]].to_numpy())[0])
        rows.append({"region": c, "kind": "national", "error": err, "error_plain_mean": mean})
    clr = pd.DataFrame(_clr(shares.to_numpy()), index=shares.index)
    for code, k in kind.items():
        if k == "national":
            continue
        others = [o for o, ko in kind.items() if ko == k and o[:2] != code[:2]]
        if not others:
            continue
        off = np.mean([clr.loc[o] - clr.loc[o[:2]] for o in others], axis=0)
        err = np.linalg.norm(clr.loc[code[:2]] + off - clr.loc[code])
        rows.append(
            {
                "region": code,
                "kind": k,
                "error": err,
                "error_plain_mean": np.linalg.norm(clr.loc[code[:2]] - clr.loc[code]),
            }
        )
    return pd.DataFrame(rows)
