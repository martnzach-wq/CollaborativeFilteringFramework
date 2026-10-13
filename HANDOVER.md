# HANDOVER: regsim — similarity and prediction for multi-level regional indicators

This file is the brief for implementing `regsim`. Read it together with `CLAUDE.md` (conventions) and `docs/design.md` (specification). Background: `docs/review.md` (critical review of LEVITATE D4.2 and KNOWING D3.6), `docs/related_work.md`, `docs/prototype.md` (feasibility benchmark), `docs/visualisation.md`.

## 1 Goal

A Python package that, for any set of indicators observed on geo-entities at any level (NUTS-0 to LAU, later other schemes):

1. fits a hierarchical, typed low-rank model to the sparse entity × indicator (× period) data;
2. predicts unknown indicator values with intervals (recommender tasks P1–P7);
3. computes similarity of geo-entities and of indicators, including transfer-specific similarity;
4. derives typologies on demand (lenses, clustering, stability, labels), with no static indicator classification in the model;
5. produces the views V1–V8 and layered reports;
6. exposes all of this via a Python API, a CLI and a REST service (for the KNOWING DSS or other clients).

Europe first; geo-schemes, connectors and estimators are plug-ins.

## 2 What exists

| Item | Location | Status |
|---|---|---|
| Specification | `docs/design.md` | Agreed basis; sections referenced as §n below |
| Prototype estimators | `reference/prototype/proto_regsim.py` | Reference only; numpy, squared loss |
| Prototype data build | `reference/prototype/build_matrix.py` | Reads Eurostat JSON-stat; 25 indicators |
| Prototype benchmark | `reference/prototype/bench.py`, `prototype_benchmark.csv` | Target numbers for milestone M2 |
| Package skeleton | `src/regsim/` | Interfaces and docstrings, `NotImplementedError` bodies |
| Tests | `tests/` | Schema tests run; acceptance tests marked `skip` until their milestone |
| Configs | `configs/` | Example run config, indicator registry v1, D3.6 gate rules |

Do not copy the prototype code into `src/` unchanged. It shows the maths and gives reference numbers.

## 3 Milestones and acceptance criteria

Each milestone ends with green tests, an updated `CHANGELOG.md` and a short note in `docs/progress.md`.

### M0 Scaffold
- Package layout as in `src/regsim/`, `pyproject.toml`, pytest, ruff, pre-commit, GitHub Actions (Linux and Windows).
- `regsim.synthetic`: generator of hierarchical low-rank data with known factors, three levels, mixed types, configurable missingness (MCAR, block-MNAR, level-missing).
- **Accept**: `pytest` green on both operating systems; generator reproducible by seed.

### M1 Data layer
- Geo registry for NUTS 2021 and 2024 (levels 0–3) and LAU from GISCO, with population and area weights, correspondence tables between NUTS versions (§2.2).
- Indicator registry from YAML (`configs/indicators_eurostat_v1.yaml`, §2.3).
- Eurostat connector: JSON-stat and SDMX, retry for asynchronous 413 responses, filter validation against the dataset structure (`docs/data_sources.md` lists known pitfalls).
- Observation store: Parquet + DuckDB (§2.1); adapter that reads the KNOWING D3.6 GeoPackage into the same format.
- **Accept**: `regsim build --config configs/example.yaml` reproduces the prototype matrix (1,929 units, 18,863 observed cells for 2018–2022) within ±2 % after code harmonisation; the D3.6 import round-trips the row count of `indicator_values`.

### M2 Baselines and evaluation harness
- Estimator interface (§4.5); baselines `GlobalMean`, `LevelMean`, `ParentCarryDown`, `KNNGower`, `IterativePCA`, `SoftImpute`, `MissForest`.
- Masking schemes M1–M5 and metrics (§10); model card output (Markdown + CSV).
- **Accept**: rerun of the prototype benchmark gives the same method ranking as `docs/prototype.md` in schemes M1, M2 and M4, with baseline RMSE within ±0.05 of the prototype values.

### M3 Core model `HierTypedMF`
- Typed losses (squared/Huber, ordinal, categorical), hierarchical prior with fractional parents, aggregation constraints on the original scale according to the `aggregation` metadata, side information, optional spatial Laplacian, Laplace posterior, fold-in for new entities and new indicators (§4, §7).
- Hyperparameters by nested masked cross-validation.
- **Accept**: on synthetic data, principal angles between true and estimated factor subspaces < 15°; on Eurostat data, RMSE ≤ prototype `MF, prior + aggregation` in M1–M4; coherence error of aggregable indicators (children vs parent) < 1 % of the parent value; fold-in prediction within 0.02 RMSE of a full refit.

### M4 Similarity
- S-profile, S-latent, S-gower, S-transfer for entities; I-model, I-observed, I-partial, I-predictive for indicators; lenses; percentile scores; per-indicator contribution vectors; intervals from posterior samples (§5).
- D3.6 compatibility mode: Gower with D3.6 normalisation, tier weights and TRF classes.
- **Accept**: D3.6 mode reproduces the urban T2 tier score of the published `precomputed_scores` for the Tallinn demonstrator, and the composite within the deviation documented in `docs/data_sources.md`; neighbour-prediction skill of S-profile ≥ S-gower on M2.

### M5 Typology, uncertainty, applicability domain, rules
- `TypologyRun` with GMM, Ward, HDBSCAN, spatially constrained Ward; consensus stability; automatic labels with dispersion; Hungarian tracking across periods and levels (§6).
- Split-conformal calibration with Mondrian groups (level, number of observed indicators); applicability-domain flags; rule engine reading `configs/rules_d36.yaml` (§8).
- **Accept**: empirical coverage of nominal 90 % intervals between 0.87 and 0.93 on M1 and M2 hold-outs at each level; identical typology for identical configuration hash; all D3.6 gates G001–G010 expressible in the rule file.

### M6 Interfaces, views, extensions
- Views V1–V8 as functions returning matplotlib figures (static) and plotly figures (interactive); linked map–embedding view.
- CLI (`regsim build | fit | evaluate | similar | typology | predict | report`), FastAPI service with endpoints mirroring the CLI.
- Geo-time entities (two or more period windows, random-walk penalty).
- LAU extension: NUTS-3 → LAU level transfer using the D3.6 GeoPackage indicators.
- **Accept**: `regsim report` regenerates all figures of `docs/visualisation.md` from a config; REST service passes contract tests; LAU M4 benchmark reported.

## 4 Decisions the implementation must not take alone

Ask the user before deciding any of these, and record the answer in `docs/decisions.md`.

| ID | Question | Prototype default |
|---|---|---|
| D1 | Target NUTS version (2021 or 2024) and LAU year for harmonisation | Codes as published, no harmonisation |
| D2 | Repository location, package name and licence | Name `regsim`; licence open |
| D3 | Period windows and whether time enters v1 | One window 2018–2022 |
| D4 | Default lens; whether polarity is shown | All indicators; polarity off |
| D5 | Sources beyond Eurostat in v1 (D3.6 GeoPackage, ARDECO, WDI, AlphaEarth embeddings) | Eurostat only |
| D6 | PyTorch as a dependency, or numpy/scipy only | numpy |
| D7 | Interactive front end (Dash, Streamlit, or DSS only via REST) | none |
| D8 | Whether TRF classes are reported for regsim similarities or only in D3.6 mode | D3.6 mode only |
| D9 | Severity of D3.6 gates in regsim mode | soft |

## 5 Known pitfalls

- Eurostat API: large queries return HTTP 413 with an asynchronous-processing message; retry later or split by period. Dimension names differ between datasets (e.g. `nama_10r_3gva` has `unit`, not `currency`; `hlth_rs_bdsrg2` has no `facility`); `sdg_11_50` is not disseminated. Validate filters against the dataset structure before querying.
- NUTS versions: in the prototype, Czech and some UK NUTS-3 codes in the data did not match the NUTS 2021 boundaries. Harmonise via correspondence tables (D1).
- The aggregation constraint must be applied on the original scale (population-weighted mean of rates, sum of counts); the prototype applied it on the transformed scale.
- With all indicators in the lens, NUTS-3 typologies follow national borders, because most indicators are inherited from parents (`docs/visualisation.md`, V4). Show the share of observed values per entity wherever typologies or similarities are reported.
- Windows: `multiprocessing` pools may be unavailable in sandboxed environments; use `joblib` (loky backend) or process-level parallelism with a sequential fallback.
