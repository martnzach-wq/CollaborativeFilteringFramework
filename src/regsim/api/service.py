"""REST service (FastAPI) mirroring the CLI, for the KNOWING DSS or other clients.

Run with ``uvicorn regsim.api.service:app`` (configuration from ``REGSIM_CONFIG``, default
``configs/example.yaml``) or build an app around a fitted model with :func:`create_app`. Geo ids may be given
with or without scheme prefix (``AT130`` or ``NUTS2024:AT130``). The response models below are the contract.
"""

from __future__ import annotations

import os
from functools import cached_property

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import regsim
from regsim.api.runtime import Fitted, resolve_id, short_id
from regsim.config import RunConfig


# ---------------------------------------------------------------------------------------------------- contract
class Health(BaseModel):
    version: str
    config_hash: str | None
    entities: int
    indicators: int


class Entity(BaseModel):
    geo_id: str
    code: str
    level: int
    observed_share: float
    domain: str


class IndicatorInfo(BaseModel):
    indicator_id: str
    label: str
    dtype: str
    unit: str
    observed_share: float


class Value(BaseModel):
    indicator: str
    value: float | None
    lower: float | None
    upper: float | None
    observed: bool


class Prediction(BaseModel):
    geo_id: str
    domain: str
    alpha: float
    values: list[Value]


class Neighbour(BaseModel):
    geo_id: str
    level: int
    distance: float
    score: float | None = None
    observed_share: float
    distance_lo: float | None = None
    distance_hi: float | None = None
    top_contributors: str | None = None


class Similar(BaseModel):
    geo_id: str
    lens: list[str]
    neighbours: list[Neighbour]


class NewEntityRequest(BaseModel):
    values: dict[str, float] = Field(..., description="indicator id -> value on the original scale")
    parent: str | None = None
    alpha: float = 0.1


class VisionRequest(BaseModel):
    base: str
    targets: dict[str, float]
    alpha: float = 0.1


class VisionValue(BaseModel):
    indicator: str
    present: float | None
    present_observed: bool
    vision: float | None
    lower: float | None
    upper: float | None
    target: bool
    change: float | None


class Vision(BaseModel):
    base: str
    values: list[VisionValue]
    analogues: list[Neighbour]


class Candidate(BaseModel):
    indicator: str
    variance_reduction: float
    share: float | None


class LevelValue(BaseModel):
    geo_id: str
    value: float | None
    lower: float | None
    upper: float | None
    observed: bool
    parent: str
    parent_observed: float | None


class TypologyRequest(BaseModel):
    level: int = 3
    algorithm: str = "gmm"
    k: int | None = None
    indicators: list[str] | None = None
    n_boot: int = 20
    seed: int = 0


class Cluster(BaseModel):
    cluster: int
    size: int
    label: str
    dispersion: float
    stability: float | None


class Typology(BaseModel):
    config_hash: str
    k: int
    stability: float
    clusters: list[Cluster]
    assignments: dict[str, int]


def _num(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _records(df, model):
    return [
        model(**{k: (_num(v) if isinstance(v, float) else v) for k, v in r.items()})
        for r in df.to_dict("records")
    ]


# ---------------------------------------------------------------------------------------------------- app
class _State:
    def __init__(self, cfg: RunConfig | None, fitted: Fitted | None):
        self.cfg, self._fitted = cfg, fitted

    @property
    def fitted(self) -> Fitted:
        if self._fitted is None:
            from regsim.api.runtime import fit

            self._fitted = fit(self.cfg)
        return self._fitted

    @cached_property
    def domain(self):
        from regsim.uncertainty.domain import applicability

        return applicability(self.fitted.model, self.fitted.data).domain

    @cached_property
    def space(self):
        from regsim.similarity.entities import s_profile

        return s_profile(self.fitted.model, self.fitted.data)


def create_app(cfg: RunConfig | None = None, fitted: Fitted | None = None) -> FastAPI:
    """FastAPI app for a run configuration (model fitted or loaded on first request) or a fitted model."""
    st = _State(cfg, fitted)
    app = FastAPI(title="regsim", version=regsim.__version__, description=__doc__)

    def gid(code: str) -> str:
        try:
            return resolve_id(st.fitted.data, code)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    def col(code: str) -> str:
        if code not in st.fitted.data.column_ids:
            raise HTTPException(404, f"unknown indicator {code!r}")
        return code

    @app.get("/health", response_model=Health)
    def health():
        d = st.fitted.data
        return Health(
            version=regsim.__version__,
            config_hash=cfg.config_hash() if cfg else None,
            entities=len(d.geo_ids),
            indicators=len(d.column_ids),
        )

    @app.get("/entities", response_model=list[Entity])
    def entities(level: int | None = None):
        d = st.fitted.data
        lv = d.levels()
        return [
            Entity(
                geo_id=g,
                code=short_id(g),
                level=int(lv[i]),
                observed_share=float(d.mask[i].mean()),
                domain=str(st.domain.iloc[i]),
            )
            for i, g in enumerate(d.geo_ids)
            if level is None or lv[i] == level
        ]

    @app.get("/indicators", response_model=list[IndicatorInfo])
    def indicators():
        d = st.fitted.data
        return [
            IndicatorInfo(
                indicator_id=c,
                label=getattr(m, "definition", "") or c,
                dtype=str(m.dtype),
                unit=getattr(m, "unit", "") or "",
                observed_share=float(d.mask[:, j].mean()),
            )
            for j, (c, m) in enumerate(zip(d.column_ids, d.column_meta, strict=True))
        ]

    @app.get("/predict/{geo_id}", response_model=Prediction)
    def predict(geo_id: str, indicator: str | None = None, alpha: float = Query(0.1, gt=0, lt=1)):
        from regsim.predict.tasks import fill_gaps

        g = gid(geo_id)
        out = fill_gaps(st.fitted.model, g, col(indicator) if indicator else None, alpha=alpha)
        return Prediction(
            geo_id=g,
            domain=str(st.domain[g]),
            alpha=alpha,
            values=_records(out.drop(columns="geo_id"), Value),
        )

    @app.get("/similar/{geo_id}", response_model=Similar)
    def similar(geo_id: str, k: int = Query(10, ge=1, le=200), indicators: str | None = None):
        from regsim.schema import Lens
        from regsim.similarity.entities import nearest, s_profile

        g = gid(geo_id)
        if indicators:
            lens = Lens(indicators=tuple(col(x.strip()) for x in indicators.split(",")))
            space = s_profile(st.fitted.model, st.fitted.data, lens)
        else:
            space = st.space
        out = nearest(space, g, k=k)
        return Similar(geo_id=g, lens=space.lens.names, neighbours=_records(out, Neighbour))

    @app.post("/entity", response_model=list[Value])
    def new_entity(req: NewEntityRequest):
        from regsim.predict.tasks import new_entity as task

        vals = {col(k): v for k, v in req.values.items()}
        out = task(st.fitted.model, vals, parent=gid(req.parent) if req.parent else None, alpha=req.alpha)
        return _records(out.drop(columns="geo_id"), Value)

    @app.post("/vision", response_model=Vision)
    def vision(req: VisionRequest, k: int = 5):
        from regsim.predict.tasks import analogues, vision_entity

        g = gid(req.base)
        out = vision_entity(st.fitted.model, g, {col(c): v for c, v in req.targets.items()}, alpha=req.alpha)
        profile = dict(zip(out.indicator, out.vision, strict=True))
        level = int(st.fitted.data.levels()[st.fitted.data.geo_ids.index(g)])
        an = analogues(st.fitted.model, profile, k=k, level=level)
        return Vision(base=g, values=_records(out, VisionValue), analogues=_records(an, Neighbour))

    @app.get("/next-indicator/{geo_id}", response_model=list[Candidate])
    def next_indicator(geo_id: str, targets: str = Query(..., description="comma-separated indicator ids")):
        from regsim.predict.tasks import next_indicator as task

        return _records(
            task(st.fitted.base, gid(geo_id), [col(t.strip()) for t in targets.split(",")]), Candidate
        )

    @app.get("/level-transfer/{indicator}", response_model=list[LevelValue])
    def level_transfer(indicator: str, level: int = 3, alpha: float = 0.1):
        from regsim.predict.tasks import level_transfer as task

        out = task(st.fitted.model, col(indicator), level, alpha=alpha)
        return _records(out.drop(columns="indicator"), LevelValue)

    @app.post("/typology", response_model=Typology)
    def typology(req: TypologyRequest):
        from regsim.schema import Lens
        from regsim.typology.run import TypologyRun, run_typology

        lens = Lens(
            indicators=tuple(col(c) for c in req.indicators) if req.indicators else None, levels=(req.level,)
        )
        try:
            run = TypologyRun(lens=lens, algorithm=req.algorithm, k=req.k, n_boot=req.n_boot, seed=req.seed)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        ty = run_typology(st.fitted.base, st.fitted.data, run)
        clusters = _records(ty.cluster_table.reset_index(), Cluster)
        return Typology(
            config_hash=ty.config_hash,
            k=ty.k,
            stability=float(ty.stability),
            clusters=clusters,
            assignments={g: int(c) for g, c in zip(ty.geo_ids, ty.labels, strict=True)},
        )

    return app


def _default_app() -> FastAPI:
    return create_app(RunConfig.from_yaml(os.environ.get("REGSIM_CONFIG", "configs/example.yaml")))


def __getattr__(name):  # `uvicorn regsim.api.service:app` builds the app lazily
    if name == "app":
        return _default_app()
    raise AttributeError(name)
