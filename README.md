# regsim

Similarity and prediction for indicators observed on geo-entities at several levels (NUTS-0 to LAU), based on a hierarchical, typed low-rank model. Start with `HANDOVER.md`, then `docs/design.md`.

| Document | Content |
|---|---|
| `HANDOVER.md` | Scope, milestones, acceptance criteria, open decisions |
| `CLAUDE.md` | Repository conventions |
| `docs/review.md` | Critical review of LEVITATE D4.2 (collaborative filtering) and KNOWING D3.6 (transferability framework) |
| `docs/related_work.md`, `docs/references.csv` | Related work with verified references |
| `docs/design.md` | Framework specification |
| `docs/prototype.md` | Feasibility benchmark on Eurostat data |
| `docs/visualisation.md` | Visualisation catalogue with rendered examples |
| `docs/examples/walkthrough.md` | Worked example (Vienna, Estonia): similarity, typology, vision, what-if, effect similarity, events, SD bridge |
| `docs/data_sources.md` | Data sources, known pitfalls, D3.6 database caveats |
| `docs/decisions.md`, `docs/progress.md`, `docs/limitations.md` | Decisions D1–D9, milestone results, limitations |

## Installation

```
pip install -e ".[dev]"          # core, tests, REST service (FastAPI, uvicorn) and Dash app
pytest                           # unit tests; acceptance tests need a built store
```

Extras: `api` (FastAPI, uvicorn), `app` (Dash, decision D7), `viz` (plotly, UMAP), `geo` (geopandas).

## Usage

```
regsim build configs/example.yaml              # Eurostat, ARDECO and D3.6 into data/store
regsim fit configs/example.yaml                # calibrated HierTypedMF, cached in runs/fit-<hash>-<period>
regsim predict configs/example.yaml AT130      # completed values with 90 % intervals, applicability domain
regsim similar configs/example.yaml AT130      # nearest regions (S-profile); --mode d36 for D3.6 scoring
regsim typology configs/example.yaml --level 2 # typology with stability, labels and configuration hash
regsim evaluate configs/benchmark_m3.yaml      # masked hold-out benchmark and model card
regsim report configs/example.yaml             # figures of docs/visualisation.md (PNG and HTML)
regsim serve configs/example.yaml              # REST service (OpenAPI at /docs)
regsim app configs/example.yaml                # explorer with linked map and embedding
regsim build configs/causal.yaml                # annual NUTS-0/2 panels 2008-2024 for the causal layer (M7)
regsim causal check --config configs/causal.yaml  # C1: sign checks of the edge registry (E1 evidence)
regsim causal fit --config configs/causal.yaml    # C2: panel effects of all edges, direction tests
regsim causal what-if EE00 --shock gdp_pps_hab=0.1 --config configs/causal.yaml  # P8 with horizon and tier
regsim causal effects --config configs/causal.yaml # effect similarity: gate and region effects (M7.6)
regsim causal compare EE00 ES51 --config configs/causal.yaml  # P(practical equivalence) per link
regsim app configs/example.yaml --effects configs/causal.yaml # explorer with the Effects tab
regsim causal literature E018 --config configs/causal.yaml  # literature pipeline (needs causal.llm, OPENALEX_API_KEY)
regsim causal sd --config configs/causal.yaml      # SD bridge (needs REGSIM_SD_MODEL: checkout of the KNOWING SD model)
regsim causal events --config configs/causal.yaml  # intervention effects for configs/events.yaml
```

The LAU extension uses `configs/lau.yaml` (NUTS-3 → LAU with the D3.6 indicators, store `data/store_lau`).

Licence: Apache-2.0 (`LICENSE`).
