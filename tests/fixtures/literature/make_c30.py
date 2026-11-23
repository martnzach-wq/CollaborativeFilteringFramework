"""Build the offline fixture of the literature pipeline for C30 (household income -> cars per 1,000).

Constructed, not recorded live: the OpenAlex response holds the six round-1 studies of C30 from
docs/causal/causal_edges_literature.csv (ids, DOIs, titles, years, venues; the abstract is the title,
since the M7 claim table keeps no abstracts), and c30_answers.json holds the round-1 extraction and
verification results of each study as LLM answers. Run from the repository root:

    python tests/fixtures/literature/make_c30.py
"""

import json
from pathlib import Path

import pandas as pd

from regsim.causal.literature import CROSSREF, OPENALEX
from regsim.data.http import request_key

HERE = Path(__file__).parent
QUERY = "household income car ownership elasticity panel"  # docs/causal/causal_candidates_llm.csv
MAX_RESULTS = 25


def main() -> None:
    c = pd.read_csv("docs/causal/causal_edges_literature.csv")
    c = c[c.cand_id == "C30"]
    works = [
        {
            "id": r.openalex,
            "doi": f"https://doi.org/{r.doi}" if isinstance(r.doi, str) else None,
            "title": r.title,
            "publication_year": int(r.year),
            "cited_by_count": int(r.cited),
            "primary_location": {"source": {"display_name": r.venue}},
            "abstract_inverted_index": {w: [i] for i, w in enumerate(r.title.split())},
        }
        for r in c.itertuples()
    ]
    oa = {
        "search": QUERY,
        "filter": "has_abstract:true,from_publication_date:1995-01-01",
        "per-page": MAX_RESULTS,
    }
    cr = {"query": QUERY, "rows": MAX_RESULTS, "filter": "has-abstract:true"}
    for kind, url, params, body in (
        ("openalex", OPENALEX, oa, {"results": works}),
        ("crossref", CROSSREF, cr, {"message": {"items": []}}),
    ):
        folder = HERE / "http" / kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{request_key(url, params)}.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    answers = {}
    for r in c.itertuples():
        answers[r.title] = {
            "extract": {
                "relevant": True,
                "direction_matches": True,
                "sign": r.sign_pass1,
                "horizon": "not stated",
                "magnitude": r.magnitude if isinstance(r.magnitude, str) else "not reported",
                "elasticity": None if pd.isna(r.elasticity) else float(r.elasticity),
                "spatial_level": r.spatial_level,
                "geography": r.geography,
                "identification": r.identification,
                "europe_relevant": bool(r.europe_relevant),
                "measure_note": "",
            },
            "verify": {
                "v_direction": "cause to effect",
                "v_sign": r.sign_final,
                "v_horizon": "not stated",
                "v_identification": r.identification,
                "v_claim_strength": r.claim_strength,
                "v_quantity_match": r.quantity_match,
                "v_elasticity": None if pd.isna(r.elasticity) else float(r.elasticity),
                "v_reason": r.verification_note if isinstance(r.verification_note, str) else "",
            },
            "measure_check": {
                "m_cause_measured": "acceptable proxy" if r.quantity_match != "same" else "same quantity",
                "m_effect_measured": "same quantity",
                "m_unit_level_ok": True,
                "m_note": "round-1 claim; the strict check was introduced in round 2",
            },
        }
    (HERE / "c30_answers.json").write_text(
        json.dumps(answers, indent=1, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
