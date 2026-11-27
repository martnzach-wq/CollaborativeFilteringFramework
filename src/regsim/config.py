"""Run configuration (pydantic). Every run writes its resolved config and hash next to its outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    indicator_registry: str | list[str]
    geo_scheme: str = "NUTS2024"  # decision D1
    levels: list[int] = Field(default_factory=lambda: [0, 2, 3])
    periods: list[str] = Field(default_factory=lambda: ["2013/2017", "2018/2022"])  # decision D3
    store: str = "data/store"
    raw_dir: str = "data/raw"
    sources: list[str] = Field(
        default_factory=lambda: ["eurostat"]
    )  # decision D5: eurostat, d36, ardeco; table (D54)
    d36_gpkg: str | None = None
    lau: bool = False

    @field_validator("periods")
    @classmethod
    def _windows(cls, v: list[str]) -> list[str]:
        for p in v:
            start, _, end = p.partition("/")
            if not (start.isdigit() and end.isdigit() and int(start) <= int(end)):
                raise ValueError(f"period window must be 'YYYY/YYYY', got {p!r}")
        return v

    @property
    def registries(self) -> list[str]:
        return (
            [self.indicator_registry] if isinstance(self.indicator_registry, str) else self.indicator_registry
        )

    @property
    def nuts_version(self) -> str:
        return self.geo_scheme.removeprefix("NUTS")

    @property
    def span(self) -> tuple[int, int]:
        years = [int(x) for p in self.periods for x in p.split("/")]
        return min(years), max(years)


class ModelConfig(BaseModel):
    estimator: str = "HierTypedMF"
    rank: int = 12
    lam_u: float = 0.03
    lam_v: float = 0.1
    lam_agg: float = 1.0
    coupling: str = "prior_and_constraint"  # broadcast | prior_only | prior_and_constraint
    seed: int = 0


class EvaluationConfig(BaseModel):
    """Masked hold-out benchmark (design §10)."""

    period: str = "2018/2022"
    levels: list[int] = Field(default_factory=lambda: [0, 2, 3])
    registries: list[str] | None = None  # default: data.indicator_registry
    indicators: list[str] | None = None  # default: the registry's model indicators
    merge_copies: bool = True
    require_population: bool = True
    protocol: str = "prototype"
    coarse_level: int = 2
    fine_level: int = 3
    estimators: dict[str, dict] = Field(
        default_factory=lambda: {
            "GlobalMean": {},
            "LevelMean": {},
            "ParentCarryDown": {},
            "KNNGower": {"k": 10},
            "SoftImpute": {"lam": 5.0, "rank_max": 12},
            "IterativePCA": {"rank": 5},
            "MissForest": {"n_estimators": 50, "max_iter": 5},
        }
    )
    n_jobs: int = 1


CAPITAL_REGIONS = [
    "AT13",
    "BE10",
    "BG41",
    "CZ01",
    "DE30",
    "DK01",
    "EE00",
    "EL30",
    "ES30",
    "FI1B",
    "FR10",
    "HR05",
    "HU11",
    "IE06",
    "ITI4",
    "LT01",
    "LV00",
    "NL32",
    "PL91",
    "PT1A",
    "RO32",
    "SE11",
    "SI04",
    "SK01",
]


class ReportConfig(BaseModel):
    """Figures of `regsim report` (docs/visualisation.md)."""

    level: int = 3
    reference: str = "AT130"  # V4 similarity map and V6 difference profile
    pair_with: str | None = None  # V6 partner; default: nearest region of the reference
    similarity_level: int = 2  # V2
    trajectory_level: int = 2  # V7
    trajectory_regions: list[str] = Field(default_factory=lambda: list(CAPITAL_REGIONS))
    observed_share: float = 0.5  # lens b of V4: indicators observed for at least this share of `level`
    typology_k: int | None = None
    typology_boot: int = 20
    n_sample_distortion: int = 400
    benchmark: str | None = None  # benchmark.csv of `regsim evaluate`; None runs a reduced benchmark
    html: bool = True  # also write interactive plotly versions
    dpi: int = 150
    seed: int = 0


class CausalConfig(BaseModel):
    """Causal layer (M7): edge registry, reference periods and evidence store (decision D10)."""

    edges: str = "configs/causal_edges.yaml"
    reference_periods: str = "configs/indicator_reference_periods.yaml"
    evidence: str | None = None  # Parquet evidence store; default <data.store>/causal_evidence.parquet
    level: int = 2  # level of the regional panels; national-only indicators use level 0
    min_years: int = 5  # years an entity needs in a series to enter a region mean
    transforms: dict[str, str] = Field(default_factory=dict)  # per-indicator override of the registry
    macro_regions: dict[str, list[str]] = Field(default_factory=dict)  # group -> country codes (D14 fallback)
    groups: Literal["typology", "macro"] = "typology"  # region groups for partial pooling (decision D14)
    estimator: Literal["fe", "gmm"] = "fe"  # decision D13
    bias_correction: Literal["jackknife", "none"] = "jackknife"  # decision D13, for estimator "fe"
    effect_links: list[str] = Field(  # links of the effect similarity (M7.6), "cause->effect"
        default_factory=lambda: [
            "gdp_pps_hab->life_expectancy",
            "gdp_pps_hab->unemployment_rate",
            "gdp_pps_hab->cars_per_1000",
            "hh_income_pps->cars_per_1000",
            "old_age_dependency->gdp_pps_hab",
            "unemployment_rate->poverty_exclusion_risk",
            "employment_rate->fertility_rate",
        ]
    )
    effect_groups: Literal["typology", "macro"] = "macro"  # region-type moderators of the effect model
    equivalence_delta: float = 0.1  # decision D19: practical equivalence in SD of the standardised effect
    equivalence_delta_by_link: dict[str, float] = Field(default_factory=dict)  # per-link override (D19)
    llm: str | None = None  # literature module (D15): "module:callable", a function prompt -> answer text
    llm_verifier: str | None = None  # stronger model for verification and the strict measure check
    literature_max_results: int = 25
    sd_model_path: str | None = None  # checkout of the KNOWING SD model (else env REGSIM_SD_MODEL)
    sd_cities: list[str] = Field(default_factory=lambda: ["TALLINN", "GRANOLLERS", "NAPLES"])
    sd_demonstrators: list[str] = Field(default_factory=lambda: ["EE00", "ES51", "ITF3"])  # NUTS codes


class RunConfig(BaseModel):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    report: ReportConfig = ReportConfig()
    causal: CausalConfig | None = None
    output_dir: str = "runs"

    def config_hash(self) -> str:
        dump = self.model_dump()
        if dump.get("causal") is None:  # configs without a causal section keep their earlier hash
            dump.pop("causal", None)
        return hashlib.sha256(json.dumps(dump, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def evidence_path(self) -> Path:
        """Parquet evidence store of the causal layer (decision D10)."""
        c = self.causal or CausalConfig()
        return Path(c.evidence) if c.evidence else Path(self.data.store) / "causal_evidence.parquet"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def write_resolved(self, folder: str | Path, name: str = "config") -> Path:
        """Write the resolved config and its hash to ``folder``; return the config path."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.resolved.yaml"
        path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8")
        (folder / f"{name}.hash").write_text(self.config_hash() + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------------------------- rules (design §8)
class Rule(BaseModel):
    """One declarative gate: a condition on demonstrator ``d`` and target ``t`` with its severities."""

    id: str
    name: str = ""
    scope: list[str] = Field(default_factory=lambda: ["all"])
    when: str
    severity_table10: Literal["hard", "soft"] = "soft"
    severity_published: Literal["hard", "soft"] = "soft"
    published: bool = True

    @field_validator("when")
    @classmethod
    def _parses(cls, v: str) -> str:
        from regsim.rules.engine import check_expression

        check_expression(v)
        return v


class KoppenAdjacency(BaseModel):
    observed_zones: list[str] = Field(default_factory=list)
    adjacent: list[tuple[str, str]] = Field(default_factory=list)


class RuleSet(BaseModel):
    aliases: dict[str, str] = Field(default_factory=dict)
    koppen_adjacency: KoppenAdjacency = KoppenAdjacency()
    rules: list[Rule]

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuleSet:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---- local strategies as inputs of the SD model (decision D57) ------------------------------------------


class StrategySource(BaseModel):
    document: str
    section: str = ""
    note: str = ""


class StrategyExposure(BaseModel):
    kind: str  # e.g. "charged_km_per_car_trip", "zone_trip_share"
    value: float
    source: StrategySource | None = None


class StrategyMeasure(BaseModel):
    """A measure of the catalogue (`regsim.causal.strategy.CATALOGUE`) with its intensity over time."""

    measure: str
    unit: str
    intensity: dict[int, float | None]
    exposure: StrategyExposure | None = None
    source: StrategySource
    confidence: Literal["stated", "derived", "assumed"] = "stated"
    # "stated": the intensity is an input from the strategy; "solved": ``intensity`` is the shape of a lever
    # bundle whose common scale the model finds against the outcome targets of ``solve_group`` (decision D61)
    role: Literal["stated", "solved"] = "stated"
    solve_group: str | None = None

    @model_validator(mode="after")
    def _solve_group(self) -> StrategyMeasure:
        if (self.role == "solved") != (self.solve_group is not None):
            raise ValueError(f"{self.measure}: solve_group is required for role 'solved' and only for it")
        return self

    @field_validator("measure")
    @classmethod
    def _known(cls, v: str) -> str:
        from regsim.causal.strategy import CATALOGUE

        if v not in CATALOGUE:
            raise ValueError(f"unknown measure {v!r}; see regsim.causal.strategy.CATALOGUE")
        return v


class StrategyPathway(BaseModel):
    """Trajectory of a bundled sector that replaces a MAED-City row (no modelled response)."""

    row: str  # label of the CETS row, e.g. "EN_finEN_hh_space heating_fossil fuel"
    unit: Literal["GWh"] = "GWh"
    values: dict[int, float | None]
    source: StrategySource


class StrategyOutcome(BaseModel):
    """A target stated as a result; compared with the model, never an input."""

    quantity: str  # a column of the SD runner output, e.g. "car_km_share", "emissions_total"
    values: dict[int, float | None]
    relative_to: int | None = None  # targets as change against this year (e.g. -0.5 for minus 50 %)
    source: StrategySource


class StrategySolve(BaseModel):
    """Outcome targets that set the common scale of the solved measures of one group, one scale per year in
    ``years`` (linear in between, 0 in the base year)."""

    group: str
    targets: list[str]  # outcome quantities of the strategy
    years: list[int]
    bounds: tuple[float, float] = (0.0, 5.0)


class StrategyFile(BaseModel):
    region: str
    name: str
    base_year: int = 2019
    measures: list[StrategyMeasure] = Field(default_factory=list)
    pathways: list[StrategyPathway] = Field(default_factory=list)
    outcomes: list[StrategyOutcome] = Field(default_factory=list)
    solve: list[StrategySolve] = Field(default_factory=list)

    @model_validator(mode="after")
    def _groups(self) -> StrategyFile:
        groups = {g.group for g in self.solve}
        used = {m.solve_group for m in self.measures if m.role == "solved"}
        if used - groups:
            raise ValueError(f"solve groups without targets: {sorted(used - groups)}")
        quantities = {o.quantity for o in self.outcomes}
        for g in self.solve:
            if set(g.targets) - quantities:
                raise ValueError(
                    f"group {g.group}: targets not among the outcomes: {sorted(set(g.targets) - quantities)}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategyFile:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
