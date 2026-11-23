"""Causal edge registry and graph operations.

The registry is a YAML file (`configs/causal_edges.yaml`) validated by the pydantic models
below. Edges keep all sources; contradicted edges are marked `contested`, never deleted.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
    model_config = ConfigDict(extra="allow")  # notes such as `new_fields` survive a rewrite

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

    def to_yaml(self, path: str | Path, width: int = 110) -> None:
        """Write the registry. An existing file keeps its key order, explicit nulls and line endings, so
        that an update changes only the edited fields in a diff."""
        path = Path(path)
        new = self.model_dump(exclude_none=True, exclude_unset=True)
        newline = "\n"
        if path.exists():
            raw = path.read_bytes()
            newline = "\r\n" if b"\r\n" in raw else "\n"
            new = _merge(yaml.safe_load(raw.decode("utf-8")), new)
        text = yaml.safe_dump(new, sort_keys=False, allow_unicode=True, width=width)
        path.write_bytes(text.replace("\n", newline).encode("utf-8"))

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

    def children(
        self, cause: str, statuses: tuple[str, ...] = ("prior", "supported"), causal_only: bool = True
    ) -> list[str]:
        """Effects of `cause` among edges with the given statuses."""
        return [
            e.effect
            for e in self.edges
            if e.cause == cause and e.status in statuses and (e.mechanism_type == "causal" or not causal_only)
        ]

    def descendants(
        self, node: str, statuses: tuple[str, ...] = ("prior", "supported"), causal_only: bool = True
    ) -> list[str]:
        """All nodes reachable from `node` (breadth-first; cycles allowed), in order of distance."""
        seen = {node}
        order: list[str] = []
        queue = deque([node])
        while queue:
            for child in self.children(queue.popleft(), statuses, causal_only):
                if child not in seen:
                    seen.add(child)
                    order.append(child)
                    queue.append(child)
        return order

    def edge(self, cause: str, effect: str) -> list[Edge]:
        """Edges from `cause` to `effect` (several when horizons or sources differ)."""
        return [e for e in self.edges if e.cause == cause and e.effect == effect]

    def update_evidence(
        self,
        estimates: Iterable[Mapping[str, Any]] | Any,
        tier: Tier = "E2",
        t_crit: float = 1.96,
        test: str = "within_region_panel",
    ) -> list[str]:
        """Write panel estimates into `data_evidence` and update `status`.

        Supported if the sign matches `sign_prior` with |t| > t_crit, contested if the
        sign is opposite with |t| > t_crit, otherwise unchanged. Edges are never removed;
        retired edges and edges without a prior sign keep their status.

        Parameters
        ----------
        estimates : iterable of mappings or DataFrame
            One row per estimate with `cause`, `effect`, `beta_std`, `t` and optionally
            `edge_id` (restricts the update to that edge), `ci95_std`, `level`.

        Returns
        -------
        list of str
            Ids of edges whose status changed.
        """
        rows = estimates.to_dict("records") if hasattr(estimates, "to_dict") else list(estimates)
        changed = []
        for r in rows:
            targets = (
                [e for e in self.edges if e.id == r["edge_id"]]
                if r.get("edge_id")
                else self.edge(r["cause"], r["effect"])
            )
            beta, t = float(r["beta_std"]), float(r["t"])
            ci = r.get("ci95_std")
            for e in targets:
                if e.status == "retired":
                    continue
                if e.sign_prior is None or abs(t) <= t_crit:
                    verdict = "undecided"
                elif (beta > 0) == (e.sign_prior == "+"):
                    verdict = "supported"
                else:
                    verdict = "contradicted"
                e.data_evidence = DataEvidence(
                    test=test,
                    level=str(r.get("level", "")),
                    within_region_beta_std=beta,
                    within_region_t=t,
                    ci95=[float(x) for x in ci] if ci is not None else None,
                    status=f"{verdict} ({tier})",
                    tier=tier,
                )
                new = {"supported": "supported", "contradicted": "contested"}.get(verdict, e.status)
                if new != e.status:
                    e.status = new
                    changed.append(e.id)
        return changed


def _merge(old: Any, new: Any) -> Any:
    """``new`` in the key order of ``old``; keys of ``old`` that ``new`` lacks keep an explicit null."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = {}
        for k, v in old.items():
            if k in new:
                out[k] = _merge(v, new[k])
            elif v is None:
                out[k] = None
        out.update({k: v for k, v in new.items() if k not in old})
        return out
    if isinstance(old, list) and isinstance(new, list):
        if all(isinstance(x, dict) and "id" in x for x in old + new):
            by_id = {x["id"]: x for x in old}
            return [_merge(by_id[x["id"]], x) if x["id"] in by_id else x for x in new]
        return [_merge(o, n) for o, n in zip(old, new, strict=False)] + new[len(old) :]
    return new


# Quantities made up of others: an indicator whose numerator is a part of another indicator's
# numerator or denominator is linked to it by an accounting identity.
COMPOSITES: dict[str, frozenset[str]] = {
    "labour_force": frozenset({"employed", "unemployed"}),
    "population_change": frozenset({"natural_change", "net_migration"}),
}
# Denominators shared by most per-capita indicators (the population, also by age group); sharing
# them alone is not an artefact.
BROAD_DENOMINATORS = frozenset({"population"})


def _broad(denominator: str) -> bool:
    return denominator in BROAD_DENOMINATORS or denominator.startswith("population_")


def classify_mechanism(cause: str, effect: str, registry_indicators) -> Mechanism:
    """Classify a data-flagged pair before it enters the registry.

    Uses indicator metadata (numerator, denominator, source survey) from the indicator
    registry (decision D11), in this order:

    1. accounting identity: the numerator of one indicator is a part of the numerator or the
       denominator of the other (employment and unemployment rate via the labour force; net
       migration and population growth);
    2. shared denominator: the numerator of one is the denominator of the other (GDP and R&D
       expenditure in % of GDP; population density and any per-capita indicator: shared
       population stock), or both share a denominator other than a population;
    3. shared survey base: both come from the same survey (LFS, EU-SILC);
    4. otherwise causal.

    Parameters
    ----------
    registry_indicators : IndicatorRegistry or mapping
        Indicator id -> entry with optional `numerator`, `denominator`, `survey`.
    """
    entries = getattr(registry_indicators, "entries", registry_indicators)

    def meta(i: str) -> tuple[str | None, str | None, str | None]:
        e = entries.get(i)
        return (None, None, None) if e is None else (e.numerator, e.denominator, e.survey)

    (na, da, sa), (nb, db, sb) = meta(cause), meta(effect)

    def part_of(n: str | None, quantities: tuple[str | None, ...]) -> bool:
        return n is not None and any(n in COMPOSITES.get(q, ()) for q in quantities if q)

    if part_of(na, (nb, db)) or part_of(nb, (na, da)):
        return "accounting_identity"
    if (na and na == db) or (nb and nb == da) or (da and da == db and not _broad(da)):
        return "shared_denominator"
    if sa and sa == sb:
        return "shared_survey_base"
    return "causal"
