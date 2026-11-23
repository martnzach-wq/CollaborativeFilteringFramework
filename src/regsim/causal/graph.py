"""Causal edge registry and graph operations.

The registry is a YAML file (`configs/causal_edges.yaml`) validated by the pydantic models
below. Edges keep all sources; contradicted edges are marked `contested`, never deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Status = Literal["prior", "hypothesis", "supported", "contested", "model_review", "retired"]
Mechanism = Literal["causal", "accounting_identity", "shared_denominator", "shared_survey_base"]
Tier = Literal["E0", "E1", "E2", "E3", "E4"]


class Source(BaseModel):
    type: str  # literature | llm_hypothesis | sd_model | model/deliverable | data_screen | expert
    ref: str
    dois: list[str] = Field(default_factory=list)


class StrengthPrior(BaseModel):
    value: float
    range: list[float] | None = None
    unit: str = "elasticity"


class DataEvidence(BaseModel):
    test: str
    level: str
    between_r: float | None = None
    within_country_r: float | None = None
    within_region_beta_std: float | None = None
    within_region_t: float | None = None
    ci95: list[float | None] | None = None
    status: str
    tier: Tier


class Edge(BaseModel):
    id: str
    cause: str
    effect: str
    sign_prior: Literal["+", "-"] | None = None
    grade: str
    mechanism_type: Mechanism = "causal"
    lag_prior: str | None = None
    horizon: str = "unspecified"  # short_run | long_run | <target year> | unspecified
    strength_prior: StrengthPrior | None = None
    sources: list[Source]
    validity: dict = Field(default_factory=dict)
    data_evidence: DataEvidence | None = None
    data_evidence_regional: list[dict] | None = (
        None  # re-tests with regional series (flags e.g. modelled_effect_data)
    )
    literature_r2: dict | None = (
        None  # strict-measure-checked grade, consensus and by_horizon {short_run, long_run}
    )
    temporal_precedence: dict | None = None  # result: forward | reverse | both | undecided | not tested
    resolution: str | None = None  # how a contradiction was resolved (horizon, ratio definition, population)
    status: Status
    note: str | None = None


class EdgeRegistry(BaseModel):
    schema_version: str
    created: str
    description: str = ""
    status_values: list[str] = Field(default_factory=list)
    mechanism_types: list[str] = Field(default_factory=list)
    edges: list[Edge]

    @classmethod
    def from_yaml(cls, path: str | Path) -> EdgeRegistry:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(exclude_none=True), f, sort_keys=False, allow_unicode=True)

    def parents(
        self, effect: str, statuses: tuple[str, ...] = ("prior", "supported"), causal_only: bool = True
    ) -> list[str]:
        """Causes of `effect` among edges with the given statuses."""
        return [
            e.cause
            for e in self.edges
            if e.effect == effect
            and e.status in statuses
            and (e.mechanism_type == "causal" or not causal_only)
        ]

    def descendants(self, node: str, **kwargs) -> list[str]:
        """All nodes reachable from `node` (breadth-first; cycles allowed)."""
        raise NotImplementedError("M7.1")

    def update_evidence(self, estimates, tier: Tier = "E2", t_crit: float = 1.96) -> None:
        """Write panel estimates into `data_evidence` and update `status`.

        Supported if the sign matches `sign_prior` with |t| > t_crit, contested if the
        sign is opposite with |t| > t_crit, otherwise unchanged.
        """
        raise NotImplementedError("M7.1")


def classify_mechanism(cause: str, effect: str, registry_indicators) -> Mechanism:
    """Classify a data-flagged pair before it enters the registry.

    Uses indicator metadata (numerator, denominator, source survey) from the indicator
    registry: shared denominators (per-capita, % of GDP, shares of final energy), shared
    population stock, shared survey base (LFS), and labour-market identities.
    Requires new optional fields `numerator`, `denominator`, `survey` in `IndicatorEntry`
    (open decision D11).
    """
    raise NotImplementedError("M7.1")
