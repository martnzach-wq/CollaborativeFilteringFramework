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
| `docs/data_sources.md` | Data sources, known pitfalls, D3.6 database caveats |
| `docs/decisions.md`, `docs/progress.md`, `docs/limitations.md` | Decisions D1–D9, milestone results, limitations |

## Installation

```
pip install -e ".[dev]"          # core, tests, REST service and Dash app
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
```

The LAU extension uses `configs/lau.yaml` (NUTS-3 → LAU with the D3.6 indicators, store `data/store_lau`).

Licence: Apache-2.0 (`LICENSE`).
