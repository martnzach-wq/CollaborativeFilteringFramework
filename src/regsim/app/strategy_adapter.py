"""City adapter for the strategy dashboard (merge step M5, decision D66).

The dashboard (`regsim.app.strategy_dash`) is city-independent. A city supplies a `StrategyAdapter`: its levers
(multiples m of the strategy level, with unit, bounds, effect class and confidence), scenario presets, targets,
the series to show, a model run, and optionally a response band, a climate band, reference pathways and a
data-gap register. Labels are given per language ({"de": ..., "en": ...}).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Text = dict[str, str]  # language -> text


def tr(text: Text | str | None, lang: str) -> str:
    """Text in ``lang`` (fallback: English, then any language)."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return text.get(lang) or text.get("en") or next(iter(text.values()), "")


@dataclass(frozen=True)
class LeverSpec:
    key: str  # backcast group of the strategy
    sector: str  # key of `StrategyAdapter.sectors`
    label: Text
    lo: float = 0.0
    hi: float = 2.0
    step: float = 0.05
    effect: str = "modelled"  # effect class (D63): modelled, stated effect, exogenous path
    confidence: str = "stated"  # stated, derived, assumed
    level: Text = field(default_factory=dict)  # the strategy level at m = 1, in natural units
    gaps: tuple[str, ...] = ()  # ids of the data-gap register


@dataclass(frozen=True)
class TargetSpec:
    quantity: str  # column of the run results
    year: int
    value: float
    label: Text
    relative_to: int | None = None  # target as change against this year
    bound: str = "max"  # "max": at most, "min": at least
    hard: bool = True  # hard targets decide "met"; the others are reported
    scale: float = 1.0  # display of a level target: value x scale (e.g. 100 for shares)
    unit: str = ""  # unit of a level target as displayed (e.g. "%")


@dataclass(frozen=True)
class SeriesSpec:
    quantity: str
    label: Text
    unit: Text
    scale: float = 1.0  # display = value x scale
    relative_to: int | None = None  # display as % change against this year


@dataclass(frozen=True)
class OptionSpec:
    key: str
    label: Text
    choices: tuple[tuple[str, Text], ...]  # (value, label)
    default: str = ""


class StrategyAdapter:
    """Base class: a city overrides the attributes and `run`; bands, references and gaps are optional."""

    title: Text = {"de": "Strategie-Dashboard", "en": "Strategy dashboard"}
    sectors: dict[str, Text] = {}
    levers: list[LeverSpec] = []
    presets: dict[str, dict] = {}  # name -> {"label": Text, "multipliers": {key: m}, "options": {key: value}}
    targets: list[TargetSpec] = []
    pathways: list[SeriesSpec] = []  # main charts (one per series)
    stacks: dict[str, list[SeriesSpec]] = {}  # stacked composition charts (title key -> parts)
    options: list[OptionSpec] = []
    response_bands: tuple[str, ...] = ()  # option values of the response band, e.g. ("weak", "strong")
    climate_series: list[SeriesSpec] = []  # series with a climate band
    references: dict[str, dict] = {}  # name -> {"label": Text, "results": DataFrame}
    data_gaps: pd.DataFrame | None = None  # columns id, item, current_value, importance, informant

    def defaults(self) -> dict[str, float]:
        return {lv.key: 1.0 for lv in self.levers}

    def run(self, multipliers: dict[str, float], options: dict[str, str]) -> pd.DataFrame:
        """Results (index year, columns quantities) of one scenario; ``options["band"]`` the response band."""
        raise NotImplementedError

    def run_many(self, scenarios: list[tuple[dict, dict]]) -> list[pd.DataFrame]:
        """Several runs (a city can parallelise)."""
        return [self.run(m, o) for m, o in scenarios]

    def run_climate_band(self, multipliers: dict, options: dict) -> pd.DataFrame | None:
        """Long table (member, year, quantity, value) of `climate_series` over the climate members, or None."""
        return None

    def level_text(self, key: str, m: float, lang: str) -> str:
        """The lever at multiple m in natural units (default: the strategy level and the multiple)."""
        lv = next(lv for lv in self.levers if lv.key == key)
        base = tr(lv.level, lang)
        return f"{m:.2f} × {base}" if base else f"{m:.2f} ×"


def display(results: pd.DataFrame, s: SeriesSpec) -> pd.Series:
    """A series as shown: scaled, or as % change against its reference year."""
    v = results[s.quantity].astype(float)
    if s.relative_to is not None:
        return 100.0 * (v / float(v.loc[s.relative_to]) - 1.0)
    return v * s.scale


def target_value(results: pd.DataFrame, t: TargetSpec) -> float:
    v = float(results.loc[t.year, t.quantity])
    if t.relative_to is not None:
        v = v / float(results.loc[t.relative_to, t.quantity]) - 1.0
    return v


def is_met(margin: float, target: TargetSpec) -> bool:
    """Met within the solver tolerance (0.05 pp for changes, 1e-4 relative for levels)."""
    tol = 5e-4 if target.relative_to is not None else 1e-4 * max(1.0, abs(target.value))
    return bool(margin >= -tol)


def margins(results: pd.DataFrame, targets: list[TargetSpec]) -> pd.DataFrame:
    """Margin of each target (>= 0: met), in the target's unit."""
    rows = []
    for t in targets:
        if t.year not in results.index or t.quantity not in results.columns:
            continue
        v = target_value(results, t)
        rows.append(
            {
                "quantity": t.quantity,
                "year": t.year,
                "target": t.value,
                "model": v,
                "hard": t.hard,
                "margin": (t.value - v) if t.bound == "max" else (v - t.value),
            }
        )
    return pd.DataFrame(rows)


def target_status(
    central: pd.DataFrame,
    targets: list[TargetSpec],
    bands: dict[str, pd.DataFrame] | None = None,
    climate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per target: central model value and margin, the margins in each response band and the lowest margin over
    the climate members (where the climate band covers the target's quantity)."""
    out = margins(central, targets)
    if out.empty:
        return out
    for name, res in (bands or {}).items():
        out[f"margin_{name}"] = margins(res, targets).margin.to_numpy()
    if climate is not None and not climate.empty:
        lows = []
        for t in targets:
            sub = climate[climate.quantity == t.quantity]
            if sub.empty or t.year not in central.index:
                lows.append(np.nan)
                continue
            vals = []
            for _, g in sub.groupby("member"):
                s = g.set_index("year").value
                if t.year not in s.index or (t.relative_to is not None and t.relative_to not in s.index):
                    continue
                v = (
                    float(s.loc[t.year]) / float(s.loc[t.relative_to]) - 1
                    if t.relative_to
                    else float(s.loc[t.year])
                )
                vals.append((t.value - v) if t.bound == "max" else (v - t.value))
            lows.append(min(vals) if vals else np.nan)
        out["margin_climate_min"] = lows
    return out


def contributions(
    adapter: StrategyAdapter, multipliers: dict, options: dict, targets: list[TargetSpec]
) -> pd.DataFrame:
    """One-at-a-time contribution of each lever: change of each target value when the lever is set to its
    reference (m = 0, or its lower bound) and the others stay (positive: the lever helps a "max" target)."""
    base = adapter.run(multipliers, options)
    runs = [({**multipliers, lv.key: max(lv.lo, 0.0)}, options) for lv in adapter.levers]
    results = adapter.run_many(runs)
    rows = []
    for lv, res in zip(adapter.levers, results, strict=True):
        for t in targets:
            if t.year not in res.index:
                continue
            d = target_value(res, t) - target_value(base, t)
            rows.append(
                {
                    "lever": lv.key,
                    "quantity": t.quantity,
                    "year": t.year,
                    "contribution": d if t.bound == "max" else -d,
                }
            )
    return pd.DataFrame(rows)
