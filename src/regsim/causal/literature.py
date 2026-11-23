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
5. `grade`: weighted evidence score and grade A-D (weights below).

LLM calls go through an injected callable so that tests run without network access.
"""

from __future__ import annotations

from collections.abc import Callable

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
EUROPE_WEIGHT = {True: 1.0, False: 0.6}
CLAIM_WEIGHT = {"causal": 1.0, "associational_with_mechanism": 0.7, "associational": 0.3}
QUANTITY_WEIGHT = {"exact": 1.0, "proxy": 0.8, "loose": 0.4}

GRADE_RULES = {
    "A": {"min_weight": 4.0, "min_consistency": 0.75, "requires": ">=1 quasi-experiment/meta-analysis or >=3 panel studies"},
    "B": {"min_weight": 2.0, "min_consistency": 0.70},
    "C": {"min_weight": 0.0, "min_consistency": 0.60},
}

LLMCall = Callable[[str], str]


def propose_edges(indicators: pd.DataFrame, known_edges: pd.DataFrame, llm: LLMCall) -> pd.DataFrame:
    """Propose candidate edges (cause, effect, expected sign, mechanism, typical lag, horizon)."""
    raise NotImplementedError("M7.2")


def search(edge: pd.Series, api_key: str, max_results: int = 25) -> pd.DataFrame:
    """Retrieve candidate studies for one edge from OpenAlex and Crossref."""
    raise NotImplementedError("M7.2")


def extract(abstracts: pd.DataFrame, edge: pd.Series, llm: LLMCall) -> pd.DataFrame:
    """First-pass claim extraction, including the horizon of the claim (short run, long run)."""
    raise NotImplementedError("M7.2")


def verify(claims: pd.DataFrame, llm: LLMCall) -> pd.DataFrame:
    """Second-pass verification; adds `agree_sign`, `agree_design`."""
    raise NotImplementedError("M7.2")


def grade(claims: pd.DataFrame) -> pd.DataFrame:
    """Aggregate verified claims per edge into weight, consensus sign, consistency and grade."""
    raise NotImplementedError("M7.2")


def targeted_queries(flagged: pd.DataFrame) -> list[str]:
    """Build queries for edges flagged by the panel module (contested or newly discovered),
    including horizon and mechanism terms (e.g. 'procyclical mortality', 'Okun')."""
    raise NotImplementedError("M7.2")
