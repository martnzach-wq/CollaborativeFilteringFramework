"""Core data types (design §2). Values are stored on the original scale."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DType(str, Enum):
    CONTINUOUS = "continuous"
    COUNT = "count"
    SHARE = "share"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    COMPOSITIONAL = "compositional"


class Transform(str, Enum):
    IDENTITY = "identity"
    LOG = "log"
    LOG1P = "log1p"
    LOGIT = "logit"
    CLR = "clr"
    RANK = "rank"
    YEOJOHNSON = "yeojohnson"


class Aggregation(str, Enum):
    SUM = "sum"                # extensive quantities (population, emissions)
    WMEAN_POP = "wmean_pop"    # intensive, population-weighted (rates per inhabitant)
    WMEAN_AREA = "wmean_area"  # intensive, area-weighted (climate means, land shares)
    RATIO = "ratio"            # numerator/denominator given in `ratio_of`
    NONE = "none"              # not aggregable (typologies, medians); parent value is a covariate only


@dataclass(frozen=True)
class Indicator:
    indicator_id: str
    dtype: DType
    transform: Transform = Transform.IDENTITY
    aggregation: Aggregation = Aggregation.NONE
    ratio_of: tuple[str, str] | None = None
    native_levels: tuple[str, ...] = ()
    polarity: int | None = None          # display only (LEVITATE TargetVal); never used in fitting
    unit: str = ""
    definition: str = ""
    source: str = ""
    tags: tuple[str, ...] = ()           # search and lens construction only; never used in models
    categories: tuple[str, ...] = ()     # for ordinal/categorical/compositional


@dataclass(frozen=True)
class GeoUnit:
    geo_id: str                          # scheme-qualified, e.g. "NUTS2021:AT130"
    scheme: str
    level: int
    name: str = ""
    country: str = ""
    area_km2: float | None = None
    population: float | None = None
    valid_from: str | None = None
    valid_to: str | None = None


@dataclass(frozen=True)
class GeoEdge:
    child: str
    parent: str
    weight_pop: float = 1.0
    weight_area: float = 1.0


@dataclass(frozen=True)
class Observation:
    geo_id: str
    indicator_id: str
    period: str
    value: float
    scenario: str = "observed"
    component: str | None = None
    se: float | None = None
    source: str = ""
    source_version: str = ""
    flags: str = ""


@dataclass(frozen=True)
class Lens:
    """Selection under which similarity and typology are computed (design §5)."""
    indicators: tuple[str, ...] | None = None   # None = all
    levels: tuple[int, ...] | None = None
    period: str | None = None
    scenario: str = "observed"
    name: str = "all"
    meta: dict = field(default_factory=dict, hash=False, compare=False)
