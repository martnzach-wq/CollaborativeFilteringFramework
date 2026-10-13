"""Indicator registry: YAML files validated by pydantic (design §2.3).

An entry may declare ``same_as`` to mark it as another source's copy of an existing indicator (decision D5).
The store keeps all copies; :meth:`IndicatorRegistry.canonical` maps every copy to the indicator that enters
the model, chosen by ``source_precedence``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from regsim.schema import Aggregation, DType, Indicator, Transform

DEFAULT_PRECEDENCE = ("eurostat", "ardeco", "d36")


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    connector: str
    dataset: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class IndicatorEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    label: str = ""
    dtype: DType
    transform: Transform = Transform.IDENTITY
    aggregation: Aggregation = Aggregation.NONE
    ratio_of: list[str] | None = None
    native_levels: list[int] = Field(default_factory=list)
    polarity: int | None = None
    unit: str = ""
    definition: str = ""
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    note: str = ""
    source: SourceSpec
    same_as: str | None = None

    @field_validator("polarity")
    @classmethod
    def _polarity(cls, v: int | None) -> int | None:
        if v not in (None, 1, -1):
            raise ValueError("polarity must be +1, -1 or None")
        return v

    def to_indicator(self) -> Indicator:
        return Indicator(
            indicator_id=self.indicator_id,
            dtype=self.dtype,
            transform=self.transform,
            aggregation=self.aggregation,
            ratio_of=tuple(self.ratio_of) if self.ratio_of else None,
            native_levels=tuple(str(x) for x in self.native_levels),
            polarity=self.polarity,
            unit=self.unit,
            definition=self.definition or self.label,
            source=self.source.connector + (f":{self.source.dataset}" if self.source.dataset else ""),
            tags=tuple(self.tags),
            categories=tuple(self.categories),
        )


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_version: str
    population_weights: SourceSpec | dict | None = None
    source_precedence: list[str] = Field(default_factory=lambda: list(DEFAULT_PRECEDENCE))
    indicators: list[IndicatorEntry]


class IndicatorRegistry:
    """Indicators from one or more registry files."""

    def __init__(
        self,
        entries: Iterable[IndicatorEntry],
        precedence: Iterable[str] = DEFAULT_PRECEDENCE,
        versions: Iterable[str] = (),
        population_weights: dict | None = None,
    ):
        self.entries: dict[str, IndicatorEntry] = {}
        for e in entries:
            if e.indicator_id in self.entries:
                raise ValueError(f"duplicate indicator_id {e.indicator_id}")
            self.entries[e.indicator_id] = e
        self.precedence = list(precedence)
        self.versions = list(versions)
        self.population_weights = population_weights
        for e in self.entries.values():
            if e.same_as and e.same_as not in self.entries:
                raise ValueError(f"{e.indicator_id}: same_as refers to unknown indicator {e.same_as}")

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> IndicatorRegistry:
        files = [
            RegistryFile.model_validate(yaml.safe_load(Path(p).read_text(encoding="utf-8"))) for p in paths
        ]
        pw = next((f.population_weights for f in files if f.population_weights), None)
        if isinstance(pw, SourceSpec):
            pw = pw.model_dump()
        return cls(
            [e for f in files for e in f.indicators],
            precedence=files[0].source_precedence if files else DEFAULT_PRECEDENCE,
            versions=[f.registry_version for f in files],
            population_weights=pw,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def __getitem__(self, indicator_id: str) -> IndicatorEntry:
        return self.entries[indicator_id]

    def indicator(self, indicator_id: str) -> Indicator:
        return self.entries[indicator_id].to_indicator()

    def groups(self) -> dict[str, list[str]]:
        """Indicator ids grouped by the indicator they copy (the root of the ``same_as`` chain)."""
        out: dict[str, list[str]] = {}
        for e in self.entries.values():
            out.setdefault(self._root(e.indicator_id), []).append(e.indicator_id)
        return out

    def canonical(self) -> dict[str, str]:
        """Map each indicator id to the id whose values enter the model (first source by precedence)."""
        rank = {s: i for i, s in enumerate(self.precedence)}
        out = {}
        for members in self.groups().values():
            best = min(members, key=lambda i: (rank.get(self.entries[i].source.connector, len(rank)), i))
            out.update(dict.fromkeys(members, best))
        return out

    def redundancy(self) -> list[tuple[str, list[str]]]:
        """Groups with more than one source, as ``(model indicator, [other copies])``."""
        canon = self.canonical()
        return [
            (canon[members[0]], sorted(m for m in members if m != canon[members[0]]))
            for members in self.groups().values()
            if len(members) > 1
        ]

    def model_indicators(self) -> list[str]:
        """Indicator ids that enter the model, in registry order."""
        canon = self.canonical()
        return [i for i in self.entries if canon[i] == i]

    def _root(self, indicator_id: str) -> str:
        seen = set()
        while self.entries[indicator_id].same_as:
            if indicator_id in seen:
                raise ValueError(f"same_as cycle at {indicator_id}")
            seen.add(indicator_id)
            indicator_id = self.entries[indicator_id].same_as
        return indicator_id
