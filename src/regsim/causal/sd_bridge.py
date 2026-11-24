"""Bridge between regsim and system dynamics models (option C4).

Functions
---------
- `parse_vensim_edges`: structural edges (cause -> effect) from a Vensim `.mdl` file.
  M7 extracted 109 edges from Submodel_Mobility_v10.mdl and 69 from
  Submodel_EnergySupply_v4.mdl.
- `perturbation_elasticities`: run a Python SD model with one input perturbed at a time
  and return output elasticities at a target year (M7: KNOWING model, TALLINN, 2050).
- `map_to_registry`: map SD variables to regsim indicators via a mapping table with a
  `relation` column (same, proxy, inverse_proxy). Inverse proxies flip the sign.
- `compare`: compare SD elasticities with panel and literature priors; flag sign
  conflicts and differences above a factor threshold as `model_review`.
- `export_inputs`: regsim estimates (imputed values, intervals) for SD model inputs of a
  follower region.
- `transfer_parameters`: transfer calibrated parameters from a pioneer to a follower,
  weighted by `similarity.causal_lens` on the parameter's causes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("\\", " ").replace('"', "")).strip().lower()


def _equations(text: str) -> list[tuple[str, str]]:
    """(left-hand side, right-hand side) of each equation of a Vensim model; the sketch section, data
    equations (``:=``, e.g. comparison data) and subscript range definitions are left out."""
    text = text.split("\\\\\\---///")[0]
    text = text.replace("{UTF-8}", "")
    text = re.sub(r"\\\s*\n\s*", " ", text)  # line continuations
    out = []
    for block in text.split("|"):
        eq = block.split("~")[0].strip()
        if not eq or eq.startswith("*") or "=" not in eq:
            continue
        m = re.match(r"(.+?)(:=|==|=)(.*)", eq, flags=re.S)
        if m is None:
            continue
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        if op == ":=" or (":" in lhs and "[" not in lhs.split(":")[0]):  # data equation, subscript range
            continue
        out.append((lhs, rhs))
    return out


def parse_vensim_edges(mdl_path: str | Path) -> pd.DataFrame:
    """Return columns model, cause, effect, effect_is_stock, cause_is_constant.

    Structural edges between model variables: a variable that occurs on the right-hand side of another
    variable's equation is a cause of it. Subscripts are removed (edges between variables, not
    elements), names are lower case with single spaces, and self-references are dropped. A stock is a
    variable defined by INTEG; a constant is a variable with one equation that is a single number
    (element-wise definitions, such as base attractiveness per mode, are calibrated data).
    """
    path = Path(mdl_path)
    eqs = _equations(path.read_text(encoding="utf-8", errors="replace"))
    lhs_name = [_norm(re.sub(r"\[.*?\]", "", lhs).split("(")[0]) for lhs, _ in eqs]
    names = set(lhs_name)
    stocks = {n for n, (_, rhs) in zip(lhs_name, eqs, strict=True) if "INTEG" in rhs.upper()}
    count = pd.Series(lhs_name).value_counts()
    number = r"\s*[+\-]?(\d+\.?\d*|\.\d+)([eE][+\-]?\d+)?\s*"
    constants = {  # one equation, a single number (element-wise definitions are calibrated data)
        n for n, (_, rhs) in zip(lhs_name, eqs, strict=True) if count[n] == 1 and re.fullmatch(number, rhs)
    }
    by_len = sorted(names, key=len, reverse=True)
    rows = set()
    for n, (_, rhs) in zip(lhs_name, eqs, strict=True):
        body = _norm(re.sub(r"\[.*?\]", " ", rhs))
        found = set()
        for cand in by_len:  # longest names first, so that names inside longer names are not matched
            pattern = r"(?:(?<=^)|(?<=[^a-z0-9_]))" + re.escape(cand) + r"(?![a-z0-9_])"
            if re.search(pattern, body):
                found.add(cand)
                body = re.sub(pattern, " ", body)
        for c in found - {n}:
            rows.add((c, n))
    out = pd.DataFrame(sorted(rows), columns=["cause", "effect"])
    out.insert(0, "model", path.name)
    out["effect_is_stock"] = out.effect.isin(stocks)
    out["cause_is_constant"] = out.cause.isin(constants)
    return out


def perturbation_elasticities(
    run_model: Callable[[dict], pd.DataFrame],
    baseline: dict,
    inputs: list[str],
    outputs: list[str] | None = None,
    rel_step: float = 0.1,
    year: int = 2050,
    method: str = "forward",
) -> pd.DataFrame:
    """Elasticities of model outputs with respect to inputs at ``year``.

    ``run_model(scales)`` returns outputs by year; ``scales`` maps input names to factors (``baseline``
    holds the factors of the reference run, usually empty). ``method`` "forward" gives the relative change
    of the output for a +``rel_step`` change of the input divided by ``rel_step`` (as in the M7 runs);
    "central" the central difference of logarithms.

    Returns
    -------
    DataFrame
        ``driver``, ``output``, ``elasticity``.
    """
    base = run_model(dict(baseline))
    outputs = outputs or list(base.columns)
    rows = []
    for name in inputs:
        up = run_model({**baseline, name: baseline.get(name, 1.0) * (1 + rel_step)})
        if method == "central":
            down = run_model({**baseline, name: baseline.get(name, 1.0) * (1 - rel_step)})
        for o in outputs:
            y0 = float(base.loc[year, o])
            if method == "central":
                yu, yd = float(up.loc[year, o]), float(down.loc[year, o])
                e = np.log(yu / yd) / np.log((1 + rel_step) / (1 - rel_step)) if yu > 0 and yd > 0 else np.nan
            else:
                e = (float(up.loc[year, o]) / y0 - 1) / rel_step if y0 != 0 else 0.0
            rows.append({"driver": name, "output": o, "elasticity": float(e)})
    return pd.DataFrame(rows)


def lever_effects(
    run_model: Callable[..., pd.DataFrame], levers: list[str], year: int = 2050
) -> pd.DataFrame:
    """Effect of each policy lever: relative change of every output with the lever against the lever held
    at its starting value (``run_model(scales, removed=[lever])``). Lever effects are city-specific
    intervention effects (M7.5 intervention table), not indicator-graph edges."""
    full = run_model({})
    rows = []
    for lever in levers:
        without = run_model({}, removed=[lever])
        for o in full.columns:
            w = float(without.loc[year, o])
            rows.append(
                {"lever": lever, "output": o, "effect": float(full.loc[year, o]) / w - 1 if w else np.nan}
            )
    return pd.DataFrame(rows)


def load_mapping(path: str | Path = "configs/sd_mapping.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """SD variable -> regsim indicator mapping (with ``relation``) and expected responses."""
    import yaml

    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(d["variables"]), pd.DataFrame(d.get("expectations", []))


def map_to_registry(sd_edges: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Map SD drivers and outputs (or structural edges ``cause``/``effect``) to regsim indicators.

    Rows whose both ends are mapped get ``cause_regsim``, ``effect_regsim`` and ``relation`` (the weaker of
    the two: same < proxy < inverse_proxy); an inverse proxy at one end flips ``sign_factor`` to -1.
    """
    m = mapping.set_index("sd")
    cause_col = "driver" if "driver" in sd_edges else "cause"
    effect_col = "output" if "output" in sd_edges else "effect"
    out = sd_edges.copy()
    out["cause_regsim"] = out[cause_col].map(m.regsim)
    out["effect_regsim"] = out[effect_col].map(m.regsim)
    rel_c, rel_e = out[cause_col].map(m.relation), out[effect_col].map(m.relation)
    order = {"same": 0, "proxy": 1, "inverse_proxy": 2}
    out["relation"] = [
        max(a, b, key=order.get) if isinstance(a, str) and isinstance(b, str) else None
        for a, b in zip(rel_c, rel_e, strict=True)
    ]
    out["sign_factor"] = [
        -1.0 if (a == "inverse_proxy") != (b == "inverse_proxy") else 1.0
        for a, b in zip(rel_c, rel_e, strict=True)
    ]
    return out[out.cause_regsim.notna() & out.effect_regsim.notna()].reset_index(drop=True)


def compare(
    sd_elasticities: pd.DataFrame,
    registry=None,
    factor: float = 3.0,
    expectations: pd.DataFrame | None = None,
    tolerance: float = 0.05,
    zero: float = 0.01,
) -> pd.DataFrame:
    """Compare SD elasticities with expected responses and registry priors; flag ``model_review`` items.

    Parameters
    ----------
    sd_elasticities : DataFrame
        ``city``, ``driver``, ``output``, ``elasticity`` and optionally ``variant`` (protocol runs: "as
        configured", "calibration off") and ``year``. The driver "population" (all population inputs
        scaled together) is the consistent counterpart of "population (internal)".
    registry : EdgeRegistry, optional
        Edges with a strength prior from literature or data add a magnitude check (ratio beyond ``factor``).
    expectations : DataFrame
        ``driver``, ``output``, ``sign`` (+, - or 0) and optional ``value`` (`load_mapping`).
    tolerance : float
        Largest |elasticity| of a per-capita output under population scaling that counts as zero.
    zero : float
        Largest |elasticity| that counts as no response.

    Returns
    -------
    DataFrame
        One row per elasticity of the configured runs with ``expected``, ``status`` (consistent,
        model_review, no expectation), ``reason`` and ``diagnosis``: the protocol variant that lowers the
        conflict (a sign conflict becoming a missing response counts), with the remaining issue, else
        "unexplained".
    """
    e = sd_elasticities.copy()
    if "variant" not in e:
        e["variant"] = "as configured"
    if "year" not in e:
        e["year"] = 0
    exp = (
        expectations if expectations is not None else pd.DataFrame(columns=["driver", "output", "sign"])
    ).copy()
    if "value" not in exp:
        exp["value"] = np.nan
    if registry is not None:  # magnitude priors from non-SD sources
        prior = {
            (ed.cause, ed.effect): ed.strength_prior.value
            for ed in registry.edges
            if ed.strength_prior is not None and not any(src.type == "sd_model" for src in ed.sources)
        }
        exp["value"] = [
            v if pd.notna(v) else prior.get((d, o), np.nan)
            for d, o, v in zip(exp.driver, exp.output, exp.value, strict=True)
        ]

    def conflict(v: float, sign, value) -> tuple[int, str | None]:
        """Severity (3 sign conflict, 2 size, 1 no response, 0 none) and description."""
        if sign == "0":
            return (
                (2, f"|{v:.2f}| > {tolerance} for a per-capita output") if abs(v) > tolerance else (0, None)
            )
        if abs(v) <= zero:
            return 1, f"no response ({v:+.3f}) where {sign} is expected"
        if np.sign(v) != (1 if sign == "+" else -1):
            return 3, f"sign {v:+.2f} against expected {sign}"
        if pd.notna(value) and not (1 / factor <= abs(v / value) <= factor):
            return 2, f"{v:+.2f} differs from {value:+.2f} by more than a factor {factor:g}"
        return 0, None

    configured = e[e.variant == "as configured"].merge(exp, on=["driver", "output"], how="left")
    rows = []
    for r in configured.to_dict("records"):
        if pd.isna(r.get("sign")):
            rows.append(
                {**r, "expected": None, "status": "no expectation", "reason": None, "diagnosis": None}
            )
            continue
        level, why = conflict(r["elasticity"], r["sign"], r["value"])
        diagnosis = None
        if level:
            same = (e.city == r["city"]) & (e.output == r["output"]) & (e.year == r["year"])
            alt = e[same & (e.driver == r["driver"]) & (e.variant != "as configured")].assign(
                label=lambda t: t.variant
            )
            if r["driver"] == "population (internal)":
                pop = e[same & (e.driver == "population") & (e.variant == "as configured")]
                alt = pd.concat([alt, pop.assign(label="population scaled consistently")])
            better = []
            for a in alt.to_dict("records"):
                lv, rest = conflict(a["elasticity"], r["sign"], r["value"])
                if lv < level:
                    better.append(
                        f"{a['label']} ({a['elasticity']:+.3f})" + (f"; remaining: {rest}" if rest else "")
                    )
            diagnosis = better[0] if better else "unexplained"
        rows.append(
            {
                **r,
                "expected": r["sign"],
                "status": "model_review" if level else "consistent",
                "reason": why,
                "diagnosis": diagnosis,
            }
        )
    return pd.DataFrame(rows).drop(columns=["sign"], errors="ignore")


def export_inputs(model, geo_id: str, inputs: list[str], alpha: float = 0.1) -> pd.DataFrame:
    """regsim estimates for SD model inputs of class O of a follower region: completed value (observed
    where available) with ``1 - alpha`` interval on the original scale, per regsim indicator in
    ``inputs`` (mapped from SD inputs with `load_mapping`)."""
    from regsim.predict.tasks import fill_gaps

    t = fill_gaps(getattr(model, "model", model), geo_id, alpha=alpha)
    t = t[t.indicator.isin(inputs)]
    missing = sorted(set(inputs) - set(t.indicator))
    out = t[["indicator", "value", "lower", "upper", "observed"]].reset_index(drop=True)
    out.attrs.update(
        geo_id=geo_id, missing=missing, imputed_share=float((~out.observed).mean()) if len(out) else np.nan
    )
    return out


def transfer_parameters(
    params: pd.DataFrame,
    pioneer: str,
    follower: str,
    registry=None,
    model=None,
    observed: dict | None = None,
    behavioural: dict[str, Callable[[], tuple[float, float, float]]] | None = None,
    recalibrate: Callable[[list[str]], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Set parameters for a follower region by class (configs/sd_parameter_classes.yaml):
    O from data or regsim imputation with interval; B from the literature function;
    R recalibrated to follower targets (never copied); S identical to the donor variant.
    Returns one row per parameter with value, interval, class and source.

    Parameters
    ----------
    params : DataFrame
        The pioneer's parameters: ``parameter``, ``class``, ``value``.
    observed : dict, optional
        Class O values of the follower: parameter -> value or (value, lower, upper), from data or
        `export_inputs`.
    behavioural : dict, optional
        Class B: parameter -> function returning (value, lower, upper) for the follower (literature).
    recalibrate : callable, optional
        Class R: called with the list of class-R parameters (e.g. a wrapper of `recalibrate_residuals`);
        returns ``parameter``, ``value``, ``lower``, ``upper``. Without it class R stays unset.
    """
    observed, behavioural = observed or {}, behavioural or {}
    rows = []
    for r in params.to_dict("records"):
        name, cls = r["parameter"], r["class"]
        row = {"parameter": name, "class": cls, "value": np.nan, "lower": np.nan, "upper": np.nan}
        if cls == "O":
            v = observed.get(name)
            if v is None:
                row["source"] = f"missing for {follower}: take from data or regsim imputation"
            else:
                lo_hi = v if isinstance(v, tuple) else (v, v, v)
                row.update(value=lo_hi[0], lower=lo_hi[1], upper=lo_hi[2], source="data or regsim imputation")
        elif cls == "B":
            f = behavioural.get(name)
            if f is None:
                row["source"] = "literature function missing"
            else:
                v, lo, hi = f()
                row.update(value=v, lower=lo, upper=hi, source="literature function")
        elif cls == "R":
            row["source"] = "recalibrated to follower targets"  # never copied from the pioneer
        elif cls == "S":
            row.update(
                value=r["value"], lower=r["value"], upper=r["value"], source=f"model variant of {pioneer}"
            )
        else:  # P: scenario input
            row["source"] = "scenario"
        rows.append(row)
    out = pd.DataFrame(rows)
    residual = list(out.parameter[out["class"] == "R"])
    if residual and recalibrate is not None:
        cal = recalibrate(residual).set_index("parameter")
        for col in ("value", "lower", "upper"):
            out.loc[out["class"] == "R", col] = out.parameter[out["class"] == "R"].map(cal[col]).to_numpy()
    out.attrs.update(pioneer=pioneer, follower=follower)
    return out


# ---- parameter-transfer protocol (docs/causal/sd_parameter_transfer.md) --------------


def load_parameter_classes(path: str | Path = "configs/sd_parameter_classes.yaml") -> dict:
    """Classes O (observable), B (behavioural, literature), R (residual calibration),
    S (structural switch), P (policy lever) with their parameter paths.

    Returns the YAML content with an added ``by_parameter`` map (parameter path, e.g.
    ``mobility.value_of_time_base``, -> class); entries are reduced to their first token, so that
    descriptions in the file (``mobility.ev_pi_trajectory (adoption curve, ...)``) map to the field.
    """
    import yaml

    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    by = {}
    for cls, spec in d["classes"].items():
        for entry in spec["parameters"]:
            key = str(entry).split()[0]
            if "." in key:
                if key in by and by[key] != cls:
                    raise ValueError(f"{key} is in classes {by[key]} and {cls}")
                by[key] = cls
    d["by_parameter"] = by
    return d


def standardise(values: pd.DataFrame) -> pd.DataFrame:
    """Z-scores of a region x indicator matrix (all regions with complete values)."""
    v = values.dropna()
    return (v - v.mean()) / v.std()


def _distance(Z: pd.DataFrame, to: pd.DataFrame) -> pd.DataFrame:
    """Root mean square difference of z-scores between every region of ``Z`` and every row of ``to``."""
    d = ((Z.to_numpy()[:, None, :] - to.to_numpy()[None, :, :]) ** 2).mean(axis=2) ** 0.5
    return pd.DataFrame(d, index=Z.index, columns=to.index)


DOMAIN_CLASSES = ((0.5, "close"), (0.75, "moderate"), (1.0, "distant"), (np.inf, "extrapolation"))


def domain_class(distance: float) -> str:
    return next(lab for lim, lab in DOMAIN_CLASSES if distance < lim)


def domain_check(model, geo_id: str | None, demonstrators: list[str], lens: list[str]) -> dict | pd.DataFrame:
    """Distance to the nearest demonstrator in the standardised lens, outside-range flags per
    dimension, and domain class (close < 0.5, moderate < 0.75, distant < 1.0, extrapolation).
    M7 reference: 82 % of NUTS-2 regions are outside the range of EE00/ES51/ITF3 on at least
    one dimension; median nearest distance 0.87.

    ``model`` is a region x indicator matrix of the lens (e.g. region means on the transformed scale);
    distances are root mean square differences of z-scores over the lens. With ``geo_id`` None the
    table of all regions is returned.
    """
    values = model[lens]
    Z = standardise(values)
    demo = [d for d in demonstrators if d in Z.index]
    if not demo:
        raise ValueError("no demonstrator has complete lens values")
    D = _distance(Z, Z.loc[demo])
    lo, hi = values.loc[demo].min(), values.loc[demo].max()
    v = values.loc[Z.index]
    out = pd.DataFrame(
        {
            "nearest_demonstrator": D.idxmin(axis=1),
            "distance_nearest": D.min(axis=1),
            **{f"outside_range_{c}": (v[c] < lo[c]) | (v[c] > hi[c]) for c in lens},
        }
    )
    out["outside_any"] = out[[f"outside_range_{c}" for c in lens]].any(axis=1)
    out["domain_class"] = out.distance_nearest.map(domain_class)
    if geo_id is None:
        return out
    r = out.loc[geo_id]
    return {
        "geo_id": geo_id,
        "nearest_demonstrator": r.nearest_demonstrator,
        "distance": float(r.distance_nearest),
        "outside_range": [c for c in lens if r[f"outside_range_{c}"]],
        "domain_class": r.domain_class,
    }


def recalibrate_residuals(
    run_model: Callable[[dict], pd.DataFrame],
    city_config,
    targets: pd.DataFrame,
    residual_params: list[str],
    n_draws: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Calibrate class-R parameters to target mode split and trip lengths. When targets are
    regsim imputations, repeat over draws from their predictive distribution and return the
    parameter distribution.

    Parameters
    ----------
    run_model : callable
        ``run_model(values)`` with parameter values (dict) returns the calibration quantities (Series or
        one-row DataFrame indexed by quantity name).
    city_config : dict
        Parameter values of the target region; the class-R entries are only the start of the search
        (e.g. the donor's values), the others stay fixed.
    targets : DataFrame
        ``quantity``, ``value`` and optionally ``sd`` (predictive standard deviation of an imputed target;
        0 or missing for observed targets).

    Returns
    -------
    DataFrame
        ``parameter``, ``value`` (median over draws), ``lower``, ``upper`` (5 % and 95 %), ``draws``,
        ``loss`` (median squared error at the optimum).
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    t = targets.set_index("quantity")
    sd = t["sd"].fillna(0.0) if "sd" in t else pd.Series(0.0, index=t.index)
    imputed = bool((sd > 0).any())
    draws = n_draws if imputed else 1
    x0 = np.array([float(city_config[p]) for p in residual_params])
    sols, losses = [], []
    for k in range(draws):
        goal = t["value"] + (rng.normal(size=len(t)) * sd if imputed and k else 0.0)

        def loss(x, goal=goal):
            out = run_model({**city_config, **dict(zip(residual_params, x, strict=True))})
            out = out.iloc[0] if isinstance(out, pd.DataFrame) else out
            return float(((out.reindex(goal.index) - goal) ** 2).sum())

        res = minimize(
            loss, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000}
        )
        sols.append(res.x)
        losses.append(res.fun)
    S = np.array(sols)
    return pd.DataFrame(
        {
            "parameter": residual_params,
            "value": np.median(S, axis=0),
            "lower": np.percentile(S, 5, axis=0),
            "upper": np.percentile(S, 95, axis=0),
            "draws": draws,
            "loss": float(np.median(losses)),
        }
    )


def cross_calibrate(
    cities: dict, run_model: Callable[[str, dict], pd.Series], observed: dict, levers: Callable | None = None
) -> pd.DataFrame:
    """Leave-one-city-out transfer test: calibrate on donor, transfer by protocol to target,
    compare with target observations and target lever responses. Reports error by parameter
    class and Spearman agreement of lever rankings for every directed pair.

    Parameters
    ----------
    cities : dict
        City -> DataFrame of its calibrated parameters (``parameter``, ``class``, ``value``).
    run_model : callable
        ``run_model(target, values)`` returns the target's calibration quantities with the parameter
        values given (others as in the target's own configuration).
    observed : dict
        City -> Series of observed calibration quantities (mode shares, trip lengths).
    levers : callable, optional
        ``levers(city, values)`` returns lever effects (Series by lever) for the Spearman agreement.

    Returns
    -------
    DataFrame
        One row per directed pair and class: root mean square error against the target's observations
        when that class is copied from the donor (other classes the target's own), and, with ``levers``,
        the Spearman correlation of lever effects between donor and target.
    """
    from scipy import stats

    rows = []
    for donor, dp in cities.items():
        for target, tp in cities.items():
            if donor == target:
                continue
            obs = observed[target]
            own = run_model(target, {})
            base_err = float(np.sqrt(((own.reindex(obs.index) - obs) ** 2).mean()))
            rho = np.nan
            if levers is not None:
                a, b = levers(donor, {}), levers(target, {})
                common = a.index.intersection(b.index)
                rho = (
                    float(stats.spearmanr(a[common].abs(), b[common].abs()).statistic)
                    if len(common) > 2
                    else np.nan
                )
            for cls in sorted(set(dp["class"])):
                copied = dict(
                    zip(dp.parameter[dp["class"] == cls], dp.value[dp["class"] == cls], strict=True)
                )
                copied = {k: v for k, v in copied.items() if k in set(tp.parameter)}
                out = run_model(target, copied)
                err = float(np.sqrt(((out.reindex(obs.index) - obs) ** 2).mean()))
                rows.append(
                    {
                        "donor": donor,
                        "target": target,
                        "class": cls,
                        "n_parameters": len(copied),
                        "rmse": err,
                        "rmse_own": base_err,
                        "lever_spearman": rho,
                    }
                )
    return pd.DataFrame(rows)


def lever_ranking(lever_effects: pd.DataFrame, output: str = "car_km_cap") -> pd.DataFrame:
    """Rank levers by effect per city and output; agreement across cities (M7 reference:
    road pricing > cycling attractiveness > parking adaptation on car-km in Tallinn and
    Granollers, while magnitudes differ up to a factor of four).

    Levers are ranked by the size of their effect (largest first) among the levers that all cities
    configure. ``attrs['spearman']`` holds the rank correlation for each pair of cities and
    ``attrs['max_ratio']`` the largest ratio of effect sizes between cities.
    """
    from itertools import combinations

    from scipy import stats

    t = lever_effects[lever_effects.output == output]
    common = set.intersection(*(set(g.lever) for _, g in t.groupby("city")))
    t = t[t.lever.isin(common)].copy()
    t["size"] = t.effect.abs()
    t["rank"] = t.groupby("city")["size"].rank(ascending=False, method="min").astype(int)
    t = t.sort_values(["city", "rank"]).reset_index(drop=True)
    wide = t.pivot(index="lever", columns="city", values="size")
    rho = {
        f"{a}-{b}": float(stats.spearmanr(wide[a], wide[b]).statistic)
        for a, b in combinations(wide.columns, 2)
    }
    t.attrs.update(
        spearman=rho, max_ratio=float((wide.max(axis=1) / wide.min(axis=1)).replace(np.inf, np.nan).max())
    )
    return t


def report_rules(domain_class: str, residuals_from_observed: bool) -> dict:
    """Which outputs may be reported: rankings and directions always; absolute levels only for
    close/moderate regions or when class-R parameters were recalibrated to observed targets."""
    return {
        "rankings": True,
        "directions": True,
        "levels": domain_class in ("close", "moderate") or residuals_from_observed,
    }


def next_demonstrator(
    lens_matrix: pd.DataFrame,
    demonstrators: list[str],
    candidates: list[str] | None = None,
    within: float = 0.75,
) -> pd.DataFrame:
    """Rank candidate regions by the reduction in median distance to the nearest demonstrator
    (M7 reference: a central or western European regional city region such as DED2, DE94,
    FRF1 or NL13 raises the share of regions within 0.75 from 0.29 to about 0.58).

    Returns one row per candidate with ``median_nearest`` and ``share_within`` (share of regions within
    ``within`` of a demonstrator) after adding it, sorted by the median; ``attrs`` hold the current values.
    """
    Z = standardise(lens_matrix)
    demo = [d for d in demonstrators if d in Z.index]
    current = _distance(Z, Z.loc[demo]).min(axis=1)
    cands = [c for c in (candidates or Z.index) if c in Z.index and c not in demo]
    to_cand = _distance(Z, Z.loc[cands])
    rows = []
    for c in cands:
        d = np.minimum(current, to_cand[c])
        rows.append(
            {"candidate": c, "median_nearest": float(d.median()), "share_within": float((d < within).mean())}
        )
    out = pd.DataFrame(rows).sort_values(["median_nearest", "candidate"]).reset_index(drop=True)
    out.attrs.update(median_nearest=float(current.median()), share_within=float((current < within).mean()))
    return out
