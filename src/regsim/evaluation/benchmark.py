"""Masked hold-out benchmark and model card (design §10)."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from regsim.evaluation import masking
from regsim.evaluation.metrics import crps_gaussian, mae, rmse
from regsim.models.base import ModelData

Z90 = 1.6448536269514722


def prototype_splits(
    data: ModelData,
    coarse_level: int = 2,
    fine_level: int = 3,
    m1_seeds=(0, 1, 2),
    m2_seeds=(100, 101),
    m3_seeds=(200, 201),
) -> list[masking.Split]:
    """Splits of the feasibility benchmark (``reference/prototype/bench.py``).

    M1: 10 % random cells, three seeds. M2: cold start of 15 % of the ``coarse_level`` entities with at least
    15 observed cells, keeping 1, 3, 6 or 10 cells. M3: ten country blocks of ``coarse_level``-native columns
    at ``coarse_level``. M4: every ``fine_level``-native column hidden at ``fine_level``.
    """
    mask, level = data.mask, data.levels()
    native = [[int(x) for x in ind.native_levels] for ind in data.column_meta]
    splits = [masking.random_cells(mask, 0.1, s) for s in m1_seeds]
    for s in m2_seeds:
        splits += masking.entity_cold_start(mask, level, coarse_level, 0.15, (1, 3, 6, 10), 15, s)
    coarse_cols = [j for j, n in enumerate(native) if n and n[0] == coarse_level]
    for s in m3_seeds:
        splits.append(masking.country_blocks(mask, level, data.country, coarse_cols, coarse_level, 10, 4, s))
    for j, n in enumerate(native):
        if n and n[0] == fine_level:
            splits.append(masking.level_transfer(mask, level, j, fine_level, data.column_ids[j]))
    return splits


def evaluate_split(name: str, factory: Callable[[], object], data: ModelData, split: masking.Split) -> dict:
    """Fit on the training cells of ``split`` and score the test cells."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = factory().fit(data.with_mask(split.train(data.mask)))
    P = model.predict()
    test = split.test
    row = {
        "scheme": split.scheme,
        "seed": split.seed,
        "k": split.k,
        "indicator": split.column,
        "method": name,
        "n_cells": int(test.sum()),
        "rmse": rmse(P, data.Z, test),
        "mae": mae(P, data.Z, test),
        "seconds": time.perf_counter() - t0,
    }
    sigma = None
    if hasattr(model, "predict_std"):
        sigma = model.predict_std()
    elif getattr(model, "scale_", None) is not None:
        sigma = np.broadcast_to(model.scale_, P.shape)
    if sigma is not None and np.isfinite(sigma[test]).all():
        row["crps"] = crps_gaussian(P, sigma, data.Z, test)
        inside = np.abs(P - data.Z)[test] <= Z90 * sigma[test]
        row["coverage90"] = float(inside.mean())
        row["width90"] = float(2 * Z90 * sigma[test].mean())
    return row


def run_benchmark(
    data: ModelData,
    estimators: dict[str, Callable[[], object]],
    splits: list[masking.Split],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Evaluate every estimator on every split; parallel over (split, estimator) with joblib if ``n_jobs != 1``.

    Splits without test cells are skipped. Falls back to sequential execution where process pools are
    unavailable.
    """
    tasks = [(name, f, s) for s in splits if s.test.any() for name, f in estimators.items()]
    rows = None
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(evaluate_split)(name, f, data, s) for name, f, s in tasks
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            warnings.warn(f"parallel execution failed ({exc}); running sequentially", stacklevel=2)
    if rows is None:
        rows = [evaluate_split(name, f, data, s) for name, f, s in tasks]
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of RMSE per scheme and method (runs averaged as in the prototype)."""
    g = results.groupby(["scheme", "method"]).rmse
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(), "count": g.count()}).reset_index()
    out["rank"] = out.groupby("scheme")["mean"].rank(method="min").astype("Int64")
    return out.sort_values(["scheme", "rank"]).reset_index(drop=True)


def cold_start_table(results: pd.DataFrame) -> pd.DataFrame:
    m2 = results[results.scheme.str.startswith("M2")]
    if m2.empty:
        return pd.DataFrame()
    m2 = m2.assign(k=m2.k.astype(int))
    return m2.pivot_table(index="method", columns="k", values="rmse", aggfunc="mean")


def model_card(results: pd.DataFrame, folder: str | Path, meta: dict | None = None) -> Path:
    """Write ``benchmark.csv``, ``summary.csv`` and ``model_card.md`` to ``folder``; return the Markdown path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results.to_csv(folder / "benchmark.csv", index=False)
    summary = summarise(results)
    summary.to_csv(folder / "summary.csv", index=False)
    lines = ["# Model card: masked hold-out benchmark", ""]
    for k, v in (meta or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "RMSE on the robust-scaled transformed scale, mean (standard deviation) over runs.", ""]
    table = summary.assign(
        cell=summary["mean"].map("{:.3f}".format) + " (" + summary["std"].fillna(0).map("{:.3f}".format) + ")"
    )
    wide = table.pivot(index="method", columns="scheme", values="cell").fillna("–")
    lines += _markdown(wide)
    if "coverage90" in results:
        cal = results.groupby("method")[["coverage90", "width90", "crps"]].mean().round(3)
        lines += ["", "Calibration of the Gaussian predictive distributions (all schemes):", ""]
        lines += _markdown(cal)
    cs = cold_start_table(results)
    if not cs.empty:
        lines += ["", "Cold start: RMSE by number of known indicators k.", ""]
        lines += _markdown(cs.round(3))
    path = folder / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _markdown(df: pd.DataFrame) -> list[str]:
    df = df.reset_index()
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return [head, sep, *body]


def level_transfer_benchmark(
    data: ModelData,
    columns: list[str],
    estimators: dict[str, Callable[[], object]],
    fine_level: int,
    reference: str = "ParentCarryDown",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Scheme M4 for selected columns: each column is hidden in turn at ``fine_level`` (e.g. LAU) and predicted
    from coarser values and the other columns. Adds the skill 1 - MSE / MSE(``reference``) per column, i.e. the
    share of the within-parent variation recovered beyond carrying the parent value down."""
    level = data.levels()
    splits = [
        masking.level_transfer(data.mask, level, data.column_ids.index(c), fine_level, c)
        for c in columns
        if c in data.column_ids
    ]
    res = run_benchmark(data, estimators, splits, n_jobs=n_jobs)
    if reference in set(res.method):
        ref = res[res.method == reference].set_index("indicator").rmse
        res["skill"] = 1 - (res.rmse / res.indicator.map(ref)) ** 2
    return res


def _rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(pred)) ** 2)))


def graph_forecast_benchmark(
    panel: pd.DataFrame,
    indicators: list[str],
    registry,
    holdout: int = 2,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    statuses: tuple[str, ...] = ("prior", "supported", "contested", "hypothesis"),
) -> pd.DataFrame:
    """Hold-out forecast of annual changes with and without the causal graph (M7.3 baseline).

    For each indicator the standardised annual change in each of the last ``holdout`` years is forecast
    from the changes of the previous year, trained on earlier years only (pooled over entities):

    - persistence (no change), region trend (mean training change of the entity);
    - own AR(1); unconstrained VAR (ridge on the lagged changes of all indicators);
    - graph-constrained VAR (ridge on the own lag and the lagged changes of the indicator's causes in
      the edge ``registry``).

    Missing lagged changes enter as 0 (the standardised mean). The ridge penalty is chosen by
    cross-validation on the training years.

    Returns
    -------
    DataFrame
        One row per indicator and hold-out year: RMSE of each model divided by the training standard
        deviation of the indicator's annual change, and the number of causes used.
    """
    from sklearn.linear_model import RidgeCV

    p = panel.sort_values(["geo_id", "year"])
    cols = [c for c in indicators if c in p.columns]
    d = p.groupby("geo_id")[cols].diff()
    d[["geo_id", "year"]] = p[["geo_id", "year"]]
    years = np.sort(d.year.unique())
    test_years = years[-holdout:]
    train = d[d.year < test_years[0]]
    sd = train[cols].std()
    z = d.copy()
    z[cols] = d[cols] / sd
    lag = z.groupby("geo_id")[cols].shift(1).fillna(0.0).add_suffix("_l1")
    trend = z[z.year < test_years[0]].groupby("geo_id")[cols].mean()
    rows = []
    for k in cols:
        parents = [c for c in dict.fromkeys(registry.parents(k, statuses=statuses)) if c in cols and c != k]
        sets = {
            "own AR(1)": [k],
            "unconstrained VAR (ridge)": cols,
            "graph-constrained VAR": [k, *parents],
        }
        ok_train = (z.year < test_years[0]) & z[k].notna()
        fitted = {}
        for name, feats in sets.items():
            X = lag.loc[ok_train, [f"{c}_l1" for c in feats]].to_numpy()
            fitted[name] = (feats, RidgeCV(alphas=alphas).fit(X, z.loc[ok_train, k].to_numpy()))
        for y in test_years:
            ok = (z.year == y) & z[k].notna()
            if ok.sum() < 10:
                continue
            truth = z.loc[ok, k].to_numpy()
            row = {"indicator": k, "year": int(y), "n": int(ok.sum()), "n_parents": len(parents)}
            row["persistence (no change)"] = _rmse(truth, np.zeros_like(truth))
            row["region trend"] = _rmse(truth, z.loc[ok, "geo_id"].map(trend[k]).fillna(0.0).to_numpy())
            for name, (feats, m) in fitted.items():
                row[name] = _rmse(truth, m.predict(lag.loc[ok, [f"{c}_l1" for c in feats]].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)
