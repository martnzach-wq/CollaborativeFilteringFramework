# Changelog

## 0.9.3
- Calibration targets for the SD model from the KNOWING deliverables D1.3, D3.3, D3.4a/b and D3.5 (`configs/sd_calibration_targets.csv`, `docs/causal/demonstrator_calibration_targets.md`): ABM light trips and passenger-km by mode 2019–2050, single-measure effects, Coping Types, CDD, energy supply and microclimate coverage.
- `sd_knowing.runner` uses the centroid method for passenger-km by default (D22).
- Decisions D22–D25.

## 0.9.2
- Worked example `examples/walkthrough.py` with guide `docs/examples/walkthrough.md` (Vienna: similarity, gap filling, typology, biplot, vision; Estonia: C1 check, what-if, effect similarity, vehicle-tax event, SD levers and domain).
- Visualisation: legends for categorical maps and for the biplot colours, indicator labels in the biplot, integer year ticks and an interval marker for short counterfactuals in `event_plot`.

## 0.9.1
- `what_if` and `vision_check`: for the short run, the entity-specific effect of the effect-similarity model comes first for links that pass the validation gate (decision D12a); the typology-class estimate depended on the number of classes chosen at each refit.

## 0.9.0 (M7 complete)
- M7.5 intervention effects: event table, counterfactual matrix completion with entity and year effects, analogues on the causal lens or pre-period path, anticipation and treatment phases, placebo intervals and p-values; `regsim causal events`; Estonian vehicle tax as first event.
- SD review issue posted in the SD repository (martnzach-wq/KNOWING-integrated-SD-model#1).
- All M7 acceptance tests implemented (M7.1–M7.6).

## 0.8.5 (M7.4)
- SD bridge: Vensim parser, perturbation protocol (consistent input groups, calibration on/off), comparison with expected responses and diagnosis, lever effects and ranking, parameter classes, domain check, next demonstrator, residual recalibration, class-based parameter transfer, cross-calibration (decisions D16, D20, D21).
- Adapter for the KNOWING SD model (`causal.sd_knowing`); `regsim causal sd`.
- Regional degree days (`nrg_chddr2_a`) for the causal store.

## 0.8.4 (M7.2)
- Literature module: propose, search (OpenAlex, Crossref), extract, verify, strict measure check, grade overall and per horizon; LLM as a configurable callable (decision D15); `regsim causal literature`.
- Grading reproduces the round-1 and round-2 grades of the M7 analysis; offline fixture for C30.
- Decision D14a: macro-regions as moderators of the effect-similarity model.

## 0.8.3 (M7.6)
- Effect similarity: varying-coefficient panel model with leave-one-country-out validation gate, region effects as derived indicators (scenario `effect`), `s_effect` with practical-equivalence threshold 0.1 SD per link (decision D19).
- `regsim causal effects`, `regsim causal compare`; Explorer tab "Effects" (`regsim app --effects`).
- Effect views: pairwise comparison, effect space, effect map with value-suppressing palette, neighbour overlap, profile strip, transfer matrix.
- Registry: E2 panel evidence and temporal precedence of the causal store (16 status changes, see `docs/progress.md`).

## 0.8.2 (M7.3)
- Panel estimation: half-panel jackknife against the Nickell bias (default) and Arellano–Bond GMM (decision D13); partial pooling of edge effects over typology classes or macro-regions (D14); `estimate_effects`.
- `direction_test` with reference-period alignment, lagged Wald tests both ways and event-study profile; `screen` labels artefacts with `classify_mechanism`.
- Causal queries `what_if`, `explain_difference`, `vision_check` (D12); `s_causal` and causal-lens `analogues`; `graph_forecast_benchmark`.
- CLI `regsim causal fit`, `regsim causal what-if`, `regsim causal direction`.
- `configs/indicators_causal_v1.yaml` (NUTS-2 demographic series back to 2008, S80/S20); `causal.transforms`, `causal.macro_regions`, `causal.groups`, `causal.estimator`, `causal.bias_correction` in the config.

## 0.8.1 (M7.1)
- Causal edge registry: `descendants`, `update_evidence`, format-preserving `to_yaml`; `classify_mechanism` from the new indicator fields `numerator`, `denominator`, `survey` (decisions D10, D11).
- C1 consistency checks (`regsim.causal.check`, `regsim causal check --config configs/causal.yaml`) on annual NUTS-0/2 panels 2008–2024 (`configs/causal.yaml`); evidence store in Parquet; E1 items for 10 edges in `configs/causal_edges.yaml`.
- `RunConfig.causal` section; configs without it keep their hash.

## 0.8.0
- Statistik Austria connector (`regsim.data.statat`, source `statat`): residents per LAU on 1 January and age shares from the OGD table `OGD_bevstandjbab2002_BevStand_<year>` (CC BY 4.0). Vienna districts are summed to LAU 90001; municipalities created after LAU 2024 are mapped to their predecessors through `successors` in the registry (shares copied, counts split by LAU-list population, both flagged).
- Registry `configs/indicators_statat_v1.yaml`: `statat_population`, `statat_share_age_65plus` (1 January 2026).
- Build: source `statat` (needs `data.lau: true`); `configs/lau_at.yaml` builds an Austrian LAU store (`data/store_lau_at`); `configs/lau.yaml` includes the new source.
- `regsim export-lau CONFIG OUT --country AT`: GeoPackage (layer `lau`) of one country's LAUs with chosen indicators and KNOWING D3.6 geometries, plus `<OUT>.provenance.json` (periods, source versions, store build hash). Used by heatimpact.

## 0.7.5
- Explorer: compare any two regions. A "Compare with" field (searchable by name) sets the comparison region of the difference profile; by default it is the nearest region, and a click on a table row sets it as well. Clicks on the map or in the embedding alternate between the reference (✕) and the comparison region (◆); a highlighted line above the plots says what the next click does. The difference profile states the overall similarity of the pair (percentile score, distance, rank among all regions of the level); the comparison region is marked on the map, in the embedding and in the table.

## 0.7.4
- Explorer selection fixed: a lasso or box selection on the map or in the embedding no longer disappears when the figures are redrawn (empty selection events are ignored; zoom and pan persist), a click adds the region to the selection and makes it the reference, the selection bar shows the reference and the selected regions, selected regions are outlined in red on the map, and a change of level or lens keeps the selected regions still shown.

## 0.7.3
- Explorer: readable short indicator labels (full name, code and source on hover) in arrows, difference profile, neighbour table and type labels; indicators grouped by source ([Eurostat], [KNOWING D3.6]) in the lens field, buttons to set the lens to all indicators of one source, arrows coloured by source, source counts in the summary.

## 0.7.2
- Explorer: embedding (with indicator directions) and trajectories share one coordinate system, a PCA of both period windows for the current lens and level; indicator arrows switchable in both; trajectories follow the lens.
- Explorer usability: region names with codes everywhere (hover, table, labels, selection list), readable indicator labels in the lens field, reference region searchable by name, list of selected regions with a clear button, neighbour row selects the partner of the difference profile, short descriptions under plots and controls, loading indicators.

## 0.7.1
- Explorer: indicator directions (V1 biplot) and trajectories tab (V7), each switchable by checkbox; arrow labels at the arrow tips in all plotly views.
- `python -m regsim`; uvicorn in the dev extra; clear messages for missing extras and an unbuilt store.

## 0.7.0 (M6)
- Prediction tasks P1–P7; views V1–V8 and diagnostic views (matplotlib and plotly); `regsim report`, `fit`, `predict`, `serve`, `app`.
- REST service (FastAPI) with pydantic response models; Dash explorer with linked map and embedding (decision D7).
- LAU extension (`configs/lau.yaml`) and `level_transfer_benchmark`.
- Fixes: D3.6 baseline values stored as observed; D3.6 electricity demand aggregated as area-weighted mean; "days per year" indicators log-transformed; LAU rows kept by their LAU-list population; model cache keyed by registry content.

## 0.6.0 (M5)
- Rule engine with `configs/rules_d36.yaml` (G001–G010, Table 10 and published severities, derived Köppen adjacency); gates in the D3.6 mode and in `regsim similar --mode d36`; per-tier coverage in `D36Scorer`; `load_values`, `published_gates`, `apply_gates`.
- `Conformalised` (Mondrian split-conformal intervals), `applicability` (applicability domain).
- Typology runs (GMM, Ward, HDBSCAN, spatial Ward) with stability, labels, tracking and configuration hash; `regsim typology`.
- Decision D9: gates are soft in regsim mode.

## 0.5.0 (M4)
- D3.6 compatibility mode with inferred scoring rules (`configs/d36_scoring.yaml`); entity similarity S-profile, S-latent, S-gower with percentile scores, contributions and intervals; transfer-specific weights; indicator similarity I-model, I-observed, I-partial, I-predictive; neighbour-prediction skill; `regsim similar`.
- `HierTypedMF.predict_dist(noise=...)`.

## 0.4.0 (M3)
- `HierTypedMF` with typed losses, hierarchical prior, aggregation constraints, side information, Laplacian and random-walk penalties, Laplace posterior, fold-in, MinT-type reconciliation; `Tuned` nested CV; geo-time `prepare`; `configs/benchmark_m3.yaml`.
- `ModelData.parent` holds membership fractions; ratio indicators aggregate with their denominator's weights.

## 0.3.0 (M2)
- Model matrix preparation, seven baselines, masking schemes M1–M4, metrics, benchmark runner, model card, `regsim evaluate`, `configs/benchmark_prototype.yaml`.
- M2 acceptance: cold-start tolerance ±0.10 (see `docs/decisions.md`).

## 0.2.1
- ARDECO connector and fallback registry (GDP per inhabitant, employment rate, population); shared cached HTTP client.

## 0.2.0 (M1)
- Eurostat connector, NUTS/LAU 2024 geo registry with NUTS 2021 → 2024 correspondence, indicator registry with source links, Parquet/DuckDB observation store, D3.6 GeoPackage adapter, `regsim build`.
- Registry: `fertility_rate` filtered to `TOTFERRT`; D3.6 registry for the 41 populated indicators.
- M1 acceptance criterion adapted to NUTS 2024 (see `docs/decisions.md`).

## 0.1.0 (M0)
- Synthetic generator `regsim.synthetic.generate_hierarchical` with known factors, mixed types, fractional parents, period random walk and missingness schemes MCAR, block-MNAR and level-missing.
- Decisions D1 (NUTS/LAU 2024), D2 (Apache-2.0), D3 (two period windows) and D6 (PyTorch core) applied.
- Ruff, pre-commit and GitHub Actions (Linux, Windows).

## 0.0.1
- Hand-over skeleton, specification and prototype reference.
