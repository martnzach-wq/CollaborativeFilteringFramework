"""Literature module: hypothesise, retrieve, extract, verify and grade causal edges.

Pipeline used in the M7 analysis (literature_hypotheses.md):

1. `propose_edges`: an LLM proposes candidate edges between registry indicators that lack
   prior evidence (status `hypothesis`, grade D until supported).
2. `search`: OpenAlex (API key required) and Crossref queries per candidate edge; abstracts
   reconstructed from the inverted index.
3. `extract`: first-pass LLM extraction of claims (direction, sign, design, level, region,
   claim strength, quantity, elasticity).
4. `verify`: second-pass extraction with a stronger model; claims where sign or design
   disagree are dropped or sent to manual check (pass agreement in M7: sign 83.5 %,
   design 76.9 %).
5. `measure_check`: strict measure check (mandatory third pass, round 2): does the study measure
   the cause and the effect themselves or their standard proxy, on a population representative of
   regions? It kept 84 of 158 claims and removed the three wrong cases of the hand-check.
6. `grade`: weighted evidence score and grade A-D (weights below), overall and per horizon.

LLM calls go through an injected callable so that tests run without network access.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

DESIGN_WEIGHT = {
    "experiment": 3.0,
    "quasi_experiment": 2.5,
    "meta_analysis": 2.5,
    "panel_fe": 2.0,
    "time_series": 1.5,
    "cross_section": 1.0,
    "simulation": 0.75,
    "descriptive": 0.5,
}
LEVEL_WEIGHT = {"regional": 1.0, "national": 1.0, "multi_country": 1.0, "city": 0.8, "individual": 0.7}
EUROPE_WEIGHT = {True: 1.0, False: 0.6}
CLAIM_WEIGHT = {"causal": 1.0, "associational_with_mechanism": 0.7, "associational": 0.3}
QUANTITY_WEIGHT = {"exact": 1.0, "proxy": 0.8, "loose": 0.4}

# Labels used by the extraction prompts (M7 analysis) -> keys of the weight tables
DESIGN_LABELS = {
    "experiment/RCT": "experiment",
    "quasi-experiment (DiD, RDD, IV, natural experiment)": "quasi_experiment",
    "meta-analysis / systematic review": "meta_analysis",
    "panel with fixed effects / GMM": "panel_fe",
    "time-series (VAR, Granger, cointegration)": "time_series",
    "cross-section regression": "cross_section",
    "simulation / model": "simulation",
    "descriptive / correlation": "descriptive",
}
LEVEL_LABELS = {
    "regional (NUTS-2/3 or similar)": "regional",
    "national": "national",
    "multi-country panel": "multi_country",
    "city": "city",
    "individual/household": "individual",
    "not stated": "individual",
}
CLAIM_LABELS = {
    "causal claim": "causal",
    "association only": "associational_with_mechanism",
    "association claim": "associational_with_mechanism",
    "authors disclaim causality": "associational",
}
QUANTITY_LABELS = {"same": "exact", "close proxy": "proxy", "distant proxy": "loose"}
MEASURED_OK = {"same quantity", "acceptable proxy"}  # strict measure check

GRADE_RULES = {
    "A": {
        "min_weight": 4.0,
        "min_consistency": 0.75,
        "requires": ">=1 quasi-experiment/meta-analysis or >=3 panel studies",
    },
    "B": {"min_weight": 2.0, "min_consistency": 0.70},
    "C": {"min_weight": 0.0, "min_consistency": 0.60},
}

LLMCall = Callable[[str], str]


OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"
# Mechanism and horizon terms for targeted queries of contested or data-flagged edges
MECHANISM_TERMS = {
    ("gdp_pps_hab", "life_expectancy"): ["procyclical mortality business cycle", "income health long run"],
    ("gdp_pps_hab", "unemployment_rate"): ["Okun's law regional", "output unemployment elasticity"],
    ("employment_rate", "fertility_rate"): [
        "procyclical fertility business cycle",
        "female employment fertility",
    ],
    ("gdp_pps_hab", "renewables_share"): ["economic growth renewable energy share"],
}
HORIZON_TERMS = {"short_run": "short run business cycle", "long_run": "long run"}


def parse_json(text: str):
    """JSON from an LLM answer (with or without a fenced code block)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(t)


def _ask(llm: LLMCall, prompt: str, log: list | None, step: str):
    answer = llm(prompt)
    if log is not None:
        log.append(
            {"step": step, "model": getattr(llm, "model", "unknown"), "prompt": prompt, "answer": answer}
        )
    return parse_json(answer)


def propose_edges(
    indicators: pd.DataFrame, known_edges: pd.DataFrame, llm: LLMCall, log: list | None = None
) -> pd.DataFrame:
    """Propose candidate edges (cause, effect, expected sign, mechanism, typical lag, horizon).

    ``indicators`` has ``indicator_id`` and ``label``; ``known_edges`` has ``cause`` and ``effect``. Proposals
    between unknown indicators or duplicating known edges are dropped. New edges start as ``hypothesis``
    with grade D.
    """
    ids = set(indicators.indicator_id)
    known = set(zip(known_edges.cause, known_edges.effect, strict=True))
    prompt = (
        "Propose direct causal relations between regional indicators that are not in the known list. "
        "Exclude definitional relations (shared numerators, denominators or survey bases).\n"
        f"Indicators: {json.dumps(dict(zip(indicators.indicator_id, indicators.label, strict=True)))}\n"
        f"Known: {json.dumps(sorted(f'{a}->{b}' for a, b in known))}\n"
        'Answer as a JSON list of objects with keys "cause", "effect", "sign" (+ or -), "mechanism", '
        '"typical_lag", "horizon" (short_run, long_run or both), "query" (search terms).'
    )
    rows = [
        r
        for r in _ask(llm, prompt, log, "propose")
        if r.get("cause") in ids and r.get("effect") in ids and (r["cause"], r["effect"]) not in known
    ]
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(["cause", "effect"]).assign(status="hypothesis", grade="D")
    return out


def _abstract(inverted: dict | None) -> str:
    """Abstract text from an OpenAlex inverted index."""
    if not inverted:
        return ""
    pos = sorted((i, w) for w, idx in inverted.items() for i in idx)
    return " ".join(w for _, w in pos)


def search(
    edge: pd.Series,
    api_key: str | None = None,
    max_results: int = 25,
    client=None,
    queries: list[str] | None = None,
    since: int = 1995,
) -> pd.DataFrame:
    """Retrieve candidate studies for one edge from OpenAlex and Crossref.

    Parameters
    ----------
    edge : Series
        ``cause``, ``effect`` and ``query`` (search terms) or ``queries``.
    api_key : str, optional
        OpenAlex API key (sent as ``api_key``).
    client : CachedClient, optional
        HTTP client with a response cache; a client with ``offline=True`` on a cache of recorded
        responses runs without network access (tests).

    Returns
    -------
    DataFrame
        One row per work: ``source``, ``query``, ``openalex``, ``doi``, ``title``, ``year``, ``cited``,
        ``venue``, ``abstract``; works without abstract are dropped, duplicates by DOI removed.
    """
    from regsim.data.http import CachedClient

    client = client or CachedClient("data/raw/literature")
    if queries is None and isinstance(edge.get("queries"), list):
        queries = edge["queries"]
    qs = queries or (
        [edge["query"]] if "query" in edge and edge["query"] else [f"{edge['cause']} {edge['effect']}"]
    )
    rows = []
    for q in qs:
        params = {
            "search": q,
            "filter": f"has_abstract:true,from_publication_date:{since}-01-01",
            "per-page": max_results,
        }
        if api_key:
            params["api_key"] = api_key
        for w in json.loads(client.get(OPENALEX, params, "openalex")).get("results", []):
            rows.append(
                {
                    "source": "openalex",
                    "query": q,
                    "openalex": w.get("id"),
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "title": w.get("title") or w.get("display_name"),
                    "year": w.get("publication_year"),
                    "cited": w.get("cited_by_count"),
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "abstract": _abstract(w.get("abstract_inverted_index")),
                }
            )
        cr = json.loads(
            client.get(CROSSREF, {"query": q, "rows": max_results, "filter": "has-abstract:true"}, "crossref")
        )
        for w in cr.get("message", {}).get("items", []):
            rows.append(
                {
                    "source": "crossref",
                    "query": q,
                    "openalex": None,
                    "doi": w.get("DOI", ""),
                    "title": (w.get("title") or [""])[0],
                    "year": ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0],
                    "cited": w.get("is-referenced-by-count"),
                    "venue": (w.get("container-title") or [None])[0],
                    "abstract": re.sub(r"<[^>]+>", " ", w.get("abstract", "")).strip(),
                }
            )
    out = pd.DataFrame(
        rows, columns=["source", "query", "openalex", "doi", "title", "year", "cited", "venue", "abstract"]
    )
    out = out[out.abstract.str.len() > 0]
    key = out.doi.str.lower().where(out.doi != "", out.title.str.lower())
    return out.loc[~key.duplicated()].reset_index(drop=True)


EXTRACT_FIELDS = (
    "relevant", "direction_matches", "sign", "horizon", "magnitude", "elasticity", "spatial_level", "geography",
    "identification", "europe_relevant", "measure_note",
)  # fmt: skip
VERIFY_FIELDS = (
    "v_direction", "v_sign", "v_horizon", "v_identification", "v_claim_strength", "v_quantity_match", "v_elasticity",
    "v_reason",
)  # fmt: skip
MEASURE_FIELDS = ("m_cause_measured", "m_effect_measured", "m_unit_level_ok", "m_note")


def _edge_text(edge: pd.Series) -> str:
    return (
        f"cause: {edge.get('cause_label', edge['cause'])}; effect: {edge.get('effect_label', edge['effect'])}"
    )


def extract(abstracts: pd.DataFrame, edge: pd.Series, llm: LLMCall, log: list | None = None) -> pd.DataFrame:
    """First-pass claim extraction, including the horizon of the claim (short run, long run)."""
    rows = []
    for r in abstracts.itertuples():
        prompt = (
            f"Edge: {_edge_text(edge)}\nStudy: {r.title} ({r.year}, doi {r.doi})\nAbstract: {r.abstract}\n"
            f"Answer as JSON with keys {list(EXTRACT_FIELDS)}. sign is +, -, mixed, none or not reported; "
            "identification one of "
            + json.dumps(sorted(DESIGN_LABELS))
            + "; spatial_level one of "
            + json.dumps(sorted(LEVEL_LABELS))
            + "; horizon short run, long run, both or not stated."
        )
        ans = _ask(llm, prompt, log, "extract")
        rows.append({**r._asdict(), **{k: ans.get(k) for k in EXTRACT_FIELDS}})
    out = pd.DataFrame(rows).drop(columns="Index", errors="ignore")
    return out[out.relevant.fillna(False).astype(bool)].reset_index(drop=True) if len(out) else out


def verify(
    claims: pd.DataFrame, llm: LLMCall, edge: pd.Series | None = None, log: list | None = None
) -> pd.DataFrame:
    """Second-pass verification; adds `agree_sign`, `agree_design`."""
    rows = []
    for r in claims.itertuples():
        prompt = (
            (f"Edge: {_edge_text(edge)}\n" if edge is not None else "")
            + f"Study: {r.title} (doi {r.doi})\nAbstract: {r.abstract}\nFirst-pass extraction: "
            + json.dumps({k: getattr(r, k) for k in EXTRACT_FIELDS if hasattr(r, k)}, default=str)
            + f"\nCheck it. Translate the sign to the stated cause and effect. Answer as JSON with keys {list(VERIFY_FIELDS)}; "
            "v_claim_strength one of "
            + json.dumps(sorted(CLAIM_LABELS))
            + "; v_quantity_match one of "
            + json.dumps(sorted(QUANTITY_LABELS))
            + "."
        )
        ans = _ask(llm, prompt, log, "verify")
        rows.append({**r._asdict(), **{k: ans.get(k) for k in VERIFY_FIELDS}})
    out = pd.DataFrame(rows).drop(columns="Index", errors="ignore")
    if len(out):
        out["agree_sign"] = out.sign == out.v_sign
        out["agree_design"] = out.identification == out.v_identification
    return out


def measure_check(
    claims: pd.DataFrame, edge: pd.Series, llm: LLMCall, log: list | None = None
) -> pd.DataFrame:
    """Strict measure check (mandatory third pass): does the study measure the cause and the effect, or
    their standard proxy, on a population representative of regions? Adds the ``m_*`` answers and
    ``measure_ok`` (`measure_ok` rule)."""
    rows = []
    for r in claims.itertuples():
        prompt = (
            f"Edge: {_edge_text(edge)}\nStudy: {r.title} (doi {r.doi})\nAbstract: {r.abstract}\n"
            "Is the cause measured as itself, by an acceptable standard proxy, or as a different quantity? The "
            "same for the effect. Is the unit of analysis representative of regions (not individuals in a "
            f"programme)? Answer as JSON with keys {list(MEASURE_FIELDS)}; m_cause_measured and m_effect_measured "
            'one of "same quantity", "acceptable proxy", "different quantity"; m_unit_level_ok true or false.'
        )
        ans = _ask(llm, prompt, log, "measure_check")
        rows.append({**r._asdict(), **{k: ans.get(k) for k in MEASURE_FIELDS}})
    out = pd.DataFrame(rows).drop(columns="Index", errors="ignore")
    if len(out):
        out["measure_ok"] = [
            measure_ok(a, b, c)
            for a, b, c in zip(out.m_cause_measured, out.m_effect_measured, out.m_unit_level_ok, strict=True)
        ]
    return out


def weighted_claims(verified: pd.DataFrame, edge_id: str) -> pd.DataFrame:
    """Verified claims in the form `grade` expects (``edge``, ``sign``, ``design``, ``horizon``, ``weight``,
    ``measure_ok``), using the second-pass labels."""
    v = verified
    return pd.DataFrame(
        {
            "edge": edge_id,
            "doi": v.doi,
            "sign": v.v_sign,
            "design": v.v_identification,
            "horizon": v.v_horizon.fillna("not stated"),
            "weight": [
                claim_weight(
                    r.v_identification,
                    r.spatial_level,
                    r.europe_relevant,
                    r.v_claim_strength,
                    r.v_quantity_match,
                )
                for r in v.itertuples()
            ],
            "measure_ok": v.measure_ok if "measure_ok" in v else True,
        }
    )


def combine_rounds(previous: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Claims of a targeted round together with the earlier claims of the same edges. Earlier claims with a
    mixed result are not carried over: the targeted round re-examines them by horizon."""
    keep = previous[previous.edge.isin(set(new.edge)) & (previous.sign != "mixed")]
    return pd.concat([keep, new], ignore_index=True)


def run_edge(
    edge: pd.Series,
    llm: LLMCall,
    verifier: LLMCall | None = None,
    client=None,
    api_key: str | None = None,
    max_results: int = 25,
) -> dict:
    """search -> extract -> verify -> strict measure check -> grade for one edge.

    ``verifier`` (default ``llm``) runs the second and third pass, usually a stronger model. Returns the
    tables of each step, the graded edge and the log of prompts and answers with model ids (stored with
    the run so that grades can be reproduced).
    """
    log: list = []
    verifier = verifier or llm
    hits = search(edge, api_key, max_results, client)
    claims = extract(hits, edge, llm, log)
    checked = verify(claims, verifier, edge, log)
    checked = measure_check(checked, edge, verifier, log) if len(checked) else checked
    edge_id = edge.get("id", f"{edge['cause']}->{edge['effect']}")
    weighted = weighted_claims(checked, edge_id) if len(checked) else pd.DataFrame()
    graded = (
        grade(weighted) if len(weighted) else pd.DataFrame([{"edge": edge_id, "grade": "D", "n_studies": 0}])
    )
    return {"hits": hits, "claims": checked, "weighted": weighted, "grade": graded, "log": pd.DataFrame(log)}


def targeted_queries(flagged: pd.DataFrame, per_edge: int = 3) -> list[str]:
    """Build queries for edges flagged by the panel module (contested or newly discovered),
    including horizon and mechanism terms (e.g. 'procyclical mortality', 'Okun').

    ``flagged`` has ``cause``, ``effect`` and optionally ``cause_label``, ``effect_label`` and ``status``;
    contested edges get one short-run and one long-run query.
    """
    out = []
    for r in flagged.to_dict("records"):
        a, b = r.get("cause_label", r["cause"]), r.get("effect_label", r["effect"])
        qs = [f"{a} {b} regional panel"]
        qs += [f"{a} {b} {t}" for t in MECHANISM_TERMS.get((r["cause"], r["effect"]), [])]
        if r.get("status") == "contested":
            qs += [f"{a} {b} {HORIZON_TERMS['short_run']}", f"{a} {b} {HORIZON_TERMS['long_run']}"]
        out += list(dict.fromkeys(qs))[: per_edge + (2 if r.get("status") == "contested" else 0)]
    return out


def _key(value, table: dict, labels: dict) -> str:
    v = labels.get(value, value)
    if v not in table:
        raise ValueError(f"unknown label {value!r}; expected one of {sorted(labels)}")
    return v


def claim_weight(design: str, level: str, europe: bool, claim: str, quantity: str) -> float:
    """Weight of one claim: design x spatial level x Europe relevance x claim strength x quantity match.
    Labels may be the keys of the weight tables or the labels of the extraction prompts."""
    return (
        DESIGN_WEIGHT[_key(design, DESIGN_WEIGHT, DESIGN_LABELS)]
        * LEVEL_WEIGHT[_key(level, LEVEL_WEIGHT, LEVEL_LABELS)]
        * EUROPE_WEIGHT[bool(europe)]
        * CLAIM_WEIGHT[_key(claim, CLAIM_WEIGHT, CLAIM_LABELS)]
        * QUANTITY_WEIGHT[_key(quantity, QUANTITY_WEIGHT, QUANTITY_LABELS)]
    )


def measure_ok(cause_measured: str, effect_measured: str, unit_level_ok: bool) -> bool:
    """Strict measure check: cause and effect are measured as themselves or by their standard proxy, on
    a population representative of regions."""
    return cause_measured in MEASURED_OK and effect_measured in MEASURED_OK and bool(unit_level_ok)


def _aggregate(g: pd.DataFrame) -> dict:
    """Weight, smoothed share of positive weight, consensus sign, consistency and grade of one group."""
    counted = g[g.sign != "not reported"]
    w = counted.weight
    pos, neg = w[counted.sign == "+"].sum(), w[counted.sign == "-"].sum()
    total = float(w.sum())
    signed = counted[counted.sign.isin(["+", "-"])]
    design = signed.design.map(lambda d: DESIGN_LABELS.get(d, d))
    n_qm = int(design.isin(["quasi_experiment", "meta_analysis"]).sum())
    n_panel = int((design == "panel_fe").sum())
    consensus = "+" if pos > neg else "-" if neg > pos else "?"
    consistency = max(pos, neg) / total if total > 0 else np.nan
    if len(signed) == 0 or not consistency >= GRADE_RULES["C"]["min_consistency"]:
        g_ = "D"
    elif (
        total >= GRADE_RULES["A"]["min_weight"]
        and consistency >= GRADE_RULES["A"]["min_consistency"]
        and (n_qm >= 1 or n_panel >= 3)
    ):
        g_ = "A"
    elif total >= GRADE_RULES["B"]["min_weight"] and consistency >= GRADE_RULES["B"]["min_consistency"]:
        g_ = "B"
    else:
        g_ = "C"
    return {
        "n_studies": len(g),
        "n_signed": len(signed),
        "weight": round(total, 4),
        "p_positive": (pos + 1) / (pos + neg + 2),
        "consensus": consensus,
        "consistency": consistency,
        "n_quasi_or_meta": n_qm,
        "n_panel": n_panel,
        "grade": g_,
    }


def grade(claims: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Aggregate verified claims per edge into weight, consensus sign, consistency and grade.

    Parameters
    ----------
    claims : DataFrame
        One row per claim with ``edge``, ``sign`` (``+``, ``-``, ``mixed``, ``none``, ``not reported``),
        ``design`` and ``weight`` (from `claim_weight`), optionally ``horizon`` (short run, long run,
        both, not stated) and ``measure_ok``.
    strict : bool
        Use only claims that pass the strict measure check (where ``measure_ok`` is given).

    Grades (GRADE_RULES): A needs weight >= 4, consistency >= 0.75 and at least one quasi-experiment or
    meta-analysis or three panel studies; B weight >= 2 and consistency >= 0.7; C consistency >= 0.6;
    D no signed study or conflicting evidence. Consistency is the weight of the consensus sign over the
    weight of all claims with a reported result (mixed and null results count against it). Grades are
    also given for short-run and long-run claims separately (``both`` counts for each).
    """
    c = claims
    if strict and "measure_ok" in c:
        c = c[c.measure_ok.fillna(True).astype(bool)]
    rows = []
    for edge, g in c.groupby("edge", sort=True):
        row = {"edge": edge, **_aggregate(g)}
        if "horizon" in g:
            for tag, hz in (("sr", "short run"), ("lr", "long run")):
                sub = g[g.horizon.isin([hz, "both"])]
                agg = _aggregate(sub) if len(sub) else {"n_studies": 0, "consensus": "?", "grade": "D"}
                row.update(
                    {
                        f"{tag}_n": agg["n_studies"],
                        f"{tag}_consensus": agg["consensus"],
                        f"{tag}_grade": agg["grade"],
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)
