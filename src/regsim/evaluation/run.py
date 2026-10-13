"""`regsim evaluate`: benchmark the configured estimators and write a model card."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from regsim.config import RunConfig
from regsim.evaluation.benchmark import model_card, prototype_splits, run_benchmark
from regsim.models.baselines import BASELINES
from regsim.models.hier_mf import HierTypedMF
from regsim.models.tuning import Tuned
from regsim.prep.matrix import prepare
from regsim.registry import IndicatorRegistry
from regsim.store.store import ObservationStore

ESTIMATORS = {**BASELINES, "HierTypedMF": HierTypedMF, "Tuned": Tuned}


def estimator_factories(spec: dict[str, dict]) -> dict:
    """Factories from ``{label: params}``; ``params["class"]`` names the estimator (default: the label).

    ``class: Tuned`` takes ``estimator`` (class name), ``grid`` and optional ``base``, ``n_folds``, ``frac``.
    """
    out = {}
    for label, params in spec.items():
        params = dict(params or {})
        cls_name = params.pop("class", label)
        if cls_name not in ESTIMATORS:
            raise ValueError(f"unknown estimator {cls_name}; available: {sorted(ESTIMATORS)}")
        if cls_name == "Tuned":
            params["estimator"] = ESTIMATORS[params.get("estimator", "HierTypedMF")]
        out[label] = partial(ESTIMATORS[cls_name], **params)
    return out


def evaluate(config: RunConfig) -> tuple[pd.DataFrame, Path]:
    """Run the benchmark of ``config.evaluation``; outputs go to ``<output_dir>/evaluate-<hash>/``."""
    ev = config.evaluation
    registry = IndicatorRegistry.from_yaml(*(ev.registries or config.data.registries))
    data = prepare(
        ObservationStore(config.data.store),
        registry,
        ev.period,
        levels=ev.levels,
        indicators=ev.indicators,
        require_population=ev.require_population,
        merge_copies=ev.merge_copies,
    )
    if ev.protocol != "prototype":
        raise ValueError(f"unknown protocol {ev.protocol}")
    splits = prototype_splits(data, ev.coarse_level, ev.fine_level)
    results = run_benchmark(data, estimator_factories(ev.estimators), splits, n_jobs=ev.n_jobs)
    folder = Path(config.output_dir) / f"evaluate-{config.config_hash()}"
    config.write_resolved(folder, "evaluate")
    meta = {
        "period": ev.period,
        "levels": ev.levels,
        "rows": data.shape[0],
        "columns": data.shape[1],
        "observed cells": int(data.mask.sum()),
        "registries": ", ".join(ev.registries or config.data.registries),
        "config hash": config.config_hash(),
    }
    return results, model_card(results, folder, meta)
