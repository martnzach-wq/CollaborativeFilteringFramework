"""Causal queries P8-P10.

Every result row carries `horizon` (short_run, long_run) and `tier` (E0-E4), and lists
the edges and sources used. Effects are propagated along registry edges with status
`prior` or `supported` and mechanism `causal`; contested edges are reported, not used.

Effect source per edge, in order of preference (decisions D12, D12a):
0. short run: the entity's own effect from the effect-similarity model (M7.6) where the link passed
   the validation gate;
1. region-type-specific panel estimate (short run) or its long-run transform;
2. pooled panel estimate;
3. literature elasticity (long run): used for the long run when no panel long-run estimate exists;
4. sign only (direction without magnitude).

Effects are combined as relative (log) changes. A panel coefficient on the transformed scales of
cause and effect is turned into a local elasticity at the entity's current values: with
g(v) = d transform(v) / d log(v) (1 for log, v for identity, v / (1 + v) for log1p), the
elasticity is beta * g(cause) / g(effect). Changes along several paths add up (first order);
interval half-widths combine the standard errors of the edges along each path.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy import stats

TIER_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}


def _g(value: float, transform: str) -> float:
    """d transform(v) / d log v at the value ``v``."""
    if transform == "log":
        return 1.0
    if transform == "log1p":
        return value / (1.0 + value)
    if transform == "logit":
        return 1.0 / (1.0 - value)
    return value  # identity


def current_values(model, geo_id: str) -> dict[str, float]:
    """Completed values of ``geo_id`` on the original scale (observed where available)."""
    from regsim.predict.tasks import fill_gaps

    t = fill_gaps(getattr(model, "model", model), geo_id)  # a `Fitted` wrapper or the model itself
    return dict(zip(t.indicator, t.value, strict=True))


def edge_effect(
    edge,
    effects: pd.DataFrame | None,
    group: str | None,
    horizon: str,
    values: dict,
    geo_id: str | None = None,
    effect_fits: dict | None = None,
) -> dict:
    """Local elasticity of one edge for an entity of ``group`` by the source order of decisions D12 and
    D12a: for the short run, the entity's own effect from the effect-similarity model (``effect_fits``,
    link id -> `VaryingFit`) where the link passed the validation gate; then the group and pooled panel
    estimates, the literature (long run) and the sign.

    Returns a dict with ``elasticity``, ``se`` (NaN when unknown), ``source`` and ``tier``.
    """
    lr = horizon == "long_run"
    fit = (effect_fits or {}).get(f"{edge.cause}->{edge.effect}")
    if not lr and fit is not None and fit.heterogeneous and geo_id in fit.moderators.index:
        mu, cov = fit.effect_of([geo_id])
        t = (
            effects[(effects.cause == edge.cause) & (effects.effect == edge.effect)]
            if effects is not None
            else None
        )
        tc, te = (t.t_cause.iloc[0], t.t_effect.iloc[0]) if t is not None and len(t) else ("log", "log")
        gc, ge = _g(values.get(edge.cause, np.nan), tc), _g(values.get(edge.effect, np.nan), te)
        if np.isfinite(gc) and np.isfinite(ge) and ge != 0:
            factor = gc / ge
            return {
                "elasticity": float(mu[0]) * factor,
                "se": float(np.sqrt(max(cov[0, 0], 0.0))) * abs(factor),
                "source": "effect_model",
                "tier": "E2",
            }
    if effects is not None and len(effects):
        rows = effects[(effects.cause == edge.cause) & (effects.effect == edge.effect)]
        for source, sel in (
            ("panel_group", rows[rows.group == group]),
            ("panel_pooled", rows[rows.group == "all"]),
        ):
            if group is None and source == "panel_group":
                continue
            if len(sel) == 0:
                continue
            r = sel.iloc[0]
            beta = r.long_run_raw if lr else r.beta_pooled
            se = r.se_long_run_raw if lr else r.se_pooled
            if not np.isfinite(beta):
                continue
            vc, ve = values.get(edge.cause, np.nan), values.get(edge.effect, np.nan)
            gc, ge = _g(vc, r.t_cause), _g(ve, r.t_effect)
            if not (np.isfinite(gc) and np.isfinite(ge)) or ge == 0:
                continue
            factor = gc / ge
            return {"elasticity": beta * factor, "se": abs(se * factor), "source": source, "tier": "E2"}
    if lr and edge.strength_prior is not None and "elasticity" in edge.strength_prior.unit:
        lo, hi = (edge.strength_prior.range or [np.nan, np.nan])[:2]
        se = (hi - lo) / 3.29 if np.isfinite(lo) and np.isfinite(hi) and hi > lo else np.nan
        tier = "E3" if edge.grade == "A" else "E1"
        return {"elasticity": edge.strength_prior.value, "se": se, "source": "literature", "tier": tier}
    sign = {"+": 1.0, "-": -1.0}.get(edge.sign_prior or "", np.nan)
    tier = (
        "E1"
        if edge.data_evidence is not None and str(edge.data_evidence.status).startswith("consistent")
        else "E0"
    )
    return {"elasticity": np.nan, "sign": sign, "se": np.nan, "source": "sign_only", "tier": tier}


def what_if(
    model,
    registry,
    geo_id: str,
    shocks: dict[str, float],
    horizon: str = "short_run",
    alpha: float = 0.1,
    effects: pd.DataFrame | None = None,
    groups: pd.Series | None = None,
    values: dict[str, float] | None = None,
    effect_fits: dict | None = None,
) -> pd.DataFrame:
    """P8: consequences of relative changes in `shocks` for the descendants of the shocked nodes.

    Parameters
    ----------
    model : Fitted or Estimator
        Source of the entity's current values (``values`` overrides).
    registry : EdgeRegistry
    geo_id : str
    shocks : dict
        Indicator id -> relative change (0.1 = +10 %). Shocked indicators are held at their shocked
        values; edges into them are not followed.
    horizon : str
        ``"short_run"`` or ``"long_run"``.
    effects : DataFrame, optional
        Panel estimates from :func:`regsim.causal.effects.estimate_effects`.
    groups : Series, optional
        Entity id -> group of the panel estimates (decision D14).
    effect_fits : dict, optional
        Varying-coefficient fits of the effect-similarity model (M7.6, `regsim causal effects`); their
        entity-specific effects come first for the short run (decision D12a).

    Returns
    -------
    DataFrame
        One row per descendant: ``indicator``, ``value_now`` and ``change_abs`` (original scale; the
        meaningful change for rates near zero such as net migration), ``change`` (relative), ``lower``,
        ``upper``, ``sign`` (for sign-only paths), ``horizon``, ``tier`` (weakest edge along the paths), ``edge_ids``,
        ``sources``; contested edges leaving a reached node are listed in ``contested``.

    M7 example: +10 % GDP in EE00 -> cars per 1,000 +0.6 % (pooled short run), +1.1 %
    (eastern regions), +1.2 % (eastern long run), +3.0 % (cross-section), +4.6 %
    (literature).
    """
    values = values if values is not None else (current_values(model, geo_id) if model is not None else {})
    group = groups.get(geo_id) if groups is not None else None
    usable = [
        e for e in registry.edges if e.status in ("prior", "supported") and e.mechanism_type == "causal"
    ]
    contested = [e for e in registry.edges if e.status == "contested" and e.mechanism_type == "causal"]
    out_edges: dict[str, list] = {}
    for e in usable:
        if e.effect not in shocks and e.effect != "*":
            out_edges.setdefault(e.cause, []).append(e)
    # depth-first search from the shocked nodes; edges back to a node on the current path close a
    # cycle and are left out (first-order propagation on the remaining acyclic graph)
    on_path, done, kept, cyclic = set(), set(), [], []

    def visit(node: str) -> None:
        on_path.add(node)
        for e in out_edges.get(node, []):
            if e.effect in on_path:
                cyclic.append(e.id)
                continue
            kept.append(e)
            if e.effect not in done:
                visit(e.effect)
        on_path.discard(node)
        done.add(node)

    for k in shocks:
        visit(k)
    # topological order of the kept edges (Kahn)
    incoming: dict[str, list] = {}
    for e in kept:
        incoming.setdefault(e.effect, []).append(e)
    indeg = {n: len({e.cause for e in incoming.get(n, [])}) for n in done}
    ready = deque(sorted(n for n in done if indeg[n] == 0))
    topo: list[str] = []
    children: dict[str, set] = {}
    for e in kept:
        children.setdefault(e.cause, set()).add(e.effect)
    while ready:
        n = ready.popleft()
        topo.append(n)
        for c in sorted(children.get(n, ())):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    # log change, variance, sign, weakest tier, edges and sources per node
    state: dict[str, dict] = {
        k: {
            "dlog": float(np.log1p(v)),
            "var": 0.0,
            "sign": np.sign(v),
            "tier": "E4",
            "edges": [],
            "sources": [],
        }
        for k, v in shocks.items()
    }
    order = [n for n in topo if n not in shocks]
    for node in order:
        dst = {
            "dlog": 0.0,
            "var": 0.0,
            "sign": 0.0,
            "tier": "E4",
            "edges": [],
            "sources": [],
            "unknown": False,
        }
        for e in incoming.get(node, []):
            src = state[e.cause]
            eff = edge_effect(e, effects, group, horizon, values, geo_id, effect_fits)
            if np.isfinite(eff["elasticity"]) and np.isfinite(src["dlog"]):
                dst["dlog"] += eff["elasticity"] * src["dlog"]
                se = eff["se"] if np.isfinite(eff["se"]) else 0.0
                dst["var"] += (src["dlog"] * se) ** 2 + eff["elasticity"] ** 2 * src["var"]
            else:  # sign only along this path
                dst["unknown"] = True
                dst["sign"] = dst["sign"] or (eff.get("sign", np.nan) * src["sign"])
            dst["tier"] = min([eff["tier"], src["tier"], dst["tier"]], key=TIER_RANK.get)
            dst["edges"] = sorted({*dst["edges"], *src["edges"], e.id})
            dst["sources"] = sorted({*dst["sources"], eff["source"]})
        if dst["unknown"] and dst["dlog"] == 0.0:
            dst["dlog"] = np.nan  # no path with a magnitude: direction only
        elif dst["dlog"] != 0.0:
            dst["sign"] = float(np.sign(dst["dlog"]))
        state[node] = dst
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for k in order:
        s = state[k]
        d, sd = s["dlog"], float(np.sqrt(s["var"]))
        now = values.get(k, np.nan)
        rows.append(
            {
                "indicator": k,
                "value_now": now,
                "change_abs": float(now * np.expm1(d)) if np.isfinite(d) else np.nan,
                "change": float(np.expm1(d)) if np.isfinite(d) else np.nan,
                "lower": float(np.expm1(d - z * sd)) if np.isfinite(d) else np.nan,
                "upper": float(np.expm1(d + z * sd)) if np.isfinite(d) else np.nan,
                "sign": s["sign"],
                "horizon": horizon,
                "tier": s["tier"],
                "edge_ids": ",".join(s["edges"]),
                "sources": ",".join(s["sources"]),
                "contested": ",".join(e.id for e in contested if e.cause == k or e.cause in shocks),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(geo_id=geo_id, group=group, shocks=dict(shocks), cyclic_edges_left_out=cyclic)
    return out


def explain_difference(
    model, registry, geo_a: str, geo_b: str, target: str, level: int | None = None, data_model=None
) -> pd.DataFrame:
    """P9: contributions of the target's parents to the difference between two entities,
    plus residual. A residual larger than the observed difference is reported as
    'missing causes'.

    The contributions come from a between-entity regression of the target on its causal parents
    (completed profiles of all entities of the level, model scale): contribution_j = b_j (x_aj - x_bj).
    """
    from regsim.predict.tasks import _base, _data

    model = getattr(model, "model", model)
    data = _data(model)
    base = _base(model)
    Z = base.predict()
    ga, gb = data.geo_ids.index(geo_a), data.geo_ids.index(geo_b)
    level = int(data.levels()[ga]) if level is None else level
    rows = np.flatnonzero(data.levels() == level)
    parents = [p for p in dict.fromkeys(registry.parents(target)) if p in data.column_ids]
    j_t = data.column_ids.index(target)
    J = [data.column_ids.index(p) for p in parents]
    X = np.column_stack([np.ones(len(rows)), Z[np.ix_(rows, J)]]) if J else np.ones((len(rows), 1))
    coef, *_ = np.linalg.lstsq(X, Z[rows, j_t], rcond=None)
    diff = float(Z[ga, j_t] - Z[gb, j_t])
    out = [
        {
            "term": p,
            "coefficient": float(coef[i + 1]),
            "difference_in_term": float(Z[ga, J[i]] - Z[gb, J[i]]),
            "contribution": float(coef[i + 1] * (Z[ga, J[i]] - Z[gb, J[i]])),
        }
        for i, p in enumerate(parents)
    ]
    explained = sum(r["contribution"] for r in out)
    residual = diff - explained
    out.append(
        {
            "term": "residual",
            "coefficient": np.nan,
            "difference_in_term": np.nan,
            "contribution": residual,
        }
    )
    frame = pd.DataFrame(out)
    frame.attrs.update(
        target=target,
        difference=diff,
        missing_causes=bool(abs(residual) > abs(diff)),
        scale="model scale (transformed and robust-scaled)",
        tier="E0",
    )
    return frame


def vision_check(
    model,
    registry,
    geo_id: str,
    targets: dict[str, float],
    year: int,
    levers: list[str] | None = None,
    effects: pd.DataFrame | None = None,
    groups: pd.Series | None = None,
    horizon: str = "long_run",
    values: dict[str, float] | None = None,
    max_change: float = 1.0,
    effect_fits: dict | None = None,
) -> pd.DataFrame:
    """P10: required change in levers for each target given expected driver changes;
    flags conflicts and mismatched target variables (e.g. lever elasticity on car-km
    used for a car-ownership target). Extends `predict.tasks.vision_entity`.

    Parameters
    ----------
    targets : dict
        Indicator id -> target value in ``year`` (original scale).
    levers : list of str, optional
        Candidate levers (default: all causal parents of each target).

    Returns
    -------
    DataFrame
        One row per target and lever: ``required_change`` of the lever (relative) to reach the target
        through the direct edge, ``elasticity``, ``source``, ``tier``, ``horizon``; ``feasible`` is false when
        the required relative change exceeds ``max_change`` (default +/-100 %); ``conflict`` marks levers
        that two targets need to move in opposite directions; ``mismatch`` marks targets that no edge of
        the registry reaches directly (the lever acts on a different variable).
    """
    values = values if values is not None else current_values(model, geo_id)
    group = groups.get(geo_id) if groups is not None else None
    rows = []
    for target, goal in targets.items():
        now = values.get(target, np.nan)
        needed = float(np.log(goal / now)) if now and np.isfinite(now) and goal > 0 else np.nan
        cands = [
            e
            for e in registry.edges
            if e.effect == target
            and e.status in ("prior", "supported")
            and e.mechanism_type == "causal"
            and (levers is None or e.cause in levers)
        ]
        if not cands:
            rows.append(
                {
                    "target": target,
                    "target_value": goal,
                    "current": now,
                    "year": year,
                    "lever": None,
                    "mismatch": True,
                    "horizon": horizon,
                    "tier": "E0",
                }
            )
            continue
        for e in cands:
            eff = edge_effect(e, effects, group, horizon, values, geo_id, effect_fits)
            el = eff["elasticity"]
            req = (
                float(np.expm1(needed / el))
                if np.isfinite(el) and el != 0 and np.isfinite(needed)
                else np.nan
            )
            rows.append(
                {
                    "target": target,
                    "target_value": goal,
                    "current": now,
                    "year": year,
                    "needed_change": float(np.expm1(needed)) if np.isfinite(needed) else np.nan,
                    "lever": e.cause,
                    "edge_id": e.id,
                    "elasticity": el,
                    "required_change": req,
                    "source": eff["source"],
                    "tier": eff["tier"],
                    "horizon": horizon,
                    "mismatch": False,
                }
            )
    out = pd.DataFrame(rows)
    if "required_change" in out:
        # a lever that has to more than double, or fall to zero, is outside the range the
        # elasticity describes
        out["feasible"] = out.required_change.between(-max_change, max_change)
        signs = (
            out.dropna(subset=["required_change"])
            .groupby("lever")
            .required_change.apply(lambda s: (s > 0).any() and (s < 0).any())
        )
        out["conflict"] = out.lever.map(signs).fillna(False).astype(bool)
    return out
