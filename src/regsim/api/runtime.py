"""Shared runtime of the CLI, the REST service and the Dash app: data preparation and fitted models, cached on
disk per configuration hash and period window."""

from __future__ import annotations

import functools
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

from regsim.config import RunConfig
from regsim.models.base import ModelData


@dataclass
class Fitted:
    """A calibrated model with its data and the folder it is stored in."""

    model: object  # Conformalised(HierTypedMF)
    data: ModelData
    folder: Path

    @property
    def base(self):
        return getattr(self.model, "estimator_", self.model)


def model_factory(cfg: RunConfig):
    from regsim.models.hier_mf import HierTypedMF

    mc = cfg.model
    return functools.partial(
        HierTypedMF,
        rank=mc.rank,
        lam_u=mc.lam_u,
        lam_v=mc.lam_v,
        lam_agg=mc.lam_agg,
        coupling=mc.coupling,
        seed=mc.seed,
    )


def prepare_data(cfg: RunConfig, period: str | list[str] | None = None) -> ModelData:
    from regsim.prep.matrix import prepare
    from regsim.registry import IndicatorRegistry
    from regsim.store.store import ObservationStore

    registry = IndicatorRegistry.from_yaml(*cfg.data.registries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare(
            ObservationStore(cfg.data.store), registry, period or cfg.data.periods[-1], levels=cfg.data.levels
        )


def fit(cfg: RunConfig, period: str | None = None, refresh: bool = False) -> Fitted:
    """Calibrated ``HierTypedMF`` (split-conformal, M5) for ``period`` (default: the last window); loaded from
    ``<output_dir>/fit-<hash>-<period>/model.pkl`` when present."""
    from regsim.uncertainty.conformal import Conformalised

    period = period or cfg.data.periods[-1]
    folder = Path(cfg.output_dir) / f"fit-{cfg.config_hash()}-{period.replace('/', '_')}"
    path = folder / "model.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as f:
            model, data = pickle.load(f)
        return Fitted(model, data, folder)
    data = prepare_data(cfg, period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Conformalised(model_factory(cfg), seed=cfg.model.seed).fit(data)
    folder.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(folder)
    with path.open("wb") as f:
        pickle.dump((model, data), f)
    return Fitted(model, data, folder)


def resolve_id(data: ModelData, code: str) -> str:
    """Geo id of ``code`` in ``data``: the id itself or the id with a scheme prefix (``AT130`` ->
    ``NUTS2024:AT130``)."""
    if code in data.geo_ids:
        return code
    hits = [g for g in data.geo_ids if g.split(":", 1)[-1] == code]
    if len(hits) != 1:
        raise KeyError(f"unknown geo id {code!r}")
    return hits[0]


def short_id(geo_id: str) -> str:
    """Geo id without scheme prefix and period suffix."""
    return geo_id.split(":", 1)[-1].split("@")[0]
