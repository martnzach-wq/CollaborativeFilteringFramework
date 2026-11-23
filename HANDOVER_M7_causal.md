# Hand-over addendum M7: causal layer (version 3)

This addendum extends `HANDOVER.md` (M0–M6, v0.8.0). It adds:
- a causal edge registry;
- a literature module;
- within-region panel estimation with a direction test;
- an SD-model bridge;
- intervention-effect estimation;
- the causal queries P8–P10;
- causal-lens similarity and effect similarity.

Background and evidence:
- `docs/causal/causal_extension_report.md`
- `docs/causal/causal_extension_addendum.md` (version-2 results: effect similarity, literature round 2, SD review, regional data, temporal precedence)
- the files listed in §6.

Conventions in `CLAUDE.md` apply unchanged (British English, `src/` layout, pydantic configs, no hard-coded geo-schemes, `fig.savefig`, acceptance tests per milestone).

## Changes in version 3

- SD parameter-transfer protocol in M7.4: parameter classes (`configs/sd_parameter_classes.yaml`), domain check against the demonstrators, recalibration of residual constants, cross-calibration test, reporting rules (rankings always, levels only within the domain or after recalibration to observed targets), ranking of candidate demonstrators. Evidence and protocol: `docs/causal/sd_parameter_transfer.md`.
- New open decisions D20 and D21.

## Changes from version 1 (in version 2)

- **Registry schema 0.2, 91 edges.** New fields:
  - `literature_r2`: grade after the strict measure check, with sign and grade per horizon;
  - `temporal_precedence`: forward / reverse / both / undecided / not tested;
  - `data_evidence_regional`: re-tests with EDGAR and Eurostat NUTS-2 series, flag `modelled_effect_data`;
  - `resolution`: how a contradiction was explained.
- **New config** `configs/indicator_reference_periods.yaml`, used to align series before timing tests.
- **New milestone M7.6, effect similarity.** New module `similarity/effect.py`.
- **Literature module.** Third mandatory pass, the strict measure check. Grades are stored per horizon.
- **SD bridge.** Perturbation protocol revised after the Tallinn review: consistent input groups, constants and series, runs with and without the calibration.
- **Panel module.** `direction_test` added: lagged tests both ways, event-study profile, pre-trend.

## 1 Contents of the addendum

| Path | Content | State |
|---|---|---|
| `configs/causal_edges.yaml` | 91 edges, schema 0.2 | data |
| `configs/indicator_reference_periods.yaml` | reference period per indicator | data |
| `src/regsim/causal/graph.py` | `EdgeRegistry` (pydantic, schema 0.2), `parents`; stubs `descendants`, `update_evidence`, `classify_mechanism` | partial, validates the registry |
| `src/regsim/causal/panel.py` | `fit_edge`, `screen`, `heterogeneity`; stub `direction_test` | reference implementation, unit-tested |
| `src/regsim/causal/literature.py` | grading constants; stubs for propose/search/extract/verify/measure-check/grade | stub |
| `src/regsim/causal/sd_bridge.py` | stubs for Vensim parsing, perturbation elasticities, comparison, parameter transfer protocol (`domain_check`, `recalibrate_residuals`, `cross_calibrate`, `lever_ranking`, `next_demonstrator`); `report_rules` implemented | stub with specification |
| `configs/sd_parameter_classes.yaml` | all KNOWING `CityConfig` parameters in classes O, B, R, S, P | data |
| `src/regsim/causal/events.py` | stubs for counterfactual intervention effects | stub |
| `src/regsim/causal/queries.py` | stubs `what_if` (P8), `explain_difference` (P9), `vision_check` (P10) | stub |
| `src/regsim/similarity/causal_lens.py` | `lens_for`; stub `s_causal` | partial |
| `src/regsim/similarity/effect.py` | stubs `fit_varying`, `validate_gate`, `region_effects`, `s_effect` | stub with specification |
| `tests/unit/test_causal_panel.py` | 3 tests on synthetic panels | passing |
| `tests/acceptance/test_m7.py` | acceptance criteria M7.1–M7.6 | skipped |
| `docs/causal/` | reports, design, test notes, CSV results, figures | docs |

## 2 Milestones

### M7.1 Registry and consistency checks

- Load and validate `configs/causal_edges.yaml`; implement `descendants` and `update_evidence`.
- Extend `IndicatorEntry` with optional `numerator`, `denominator`, `survey` (D11) and implement `classify_mechanism`. Known cases to cover:
  - shared denominators (R&D/GDP, renewables share);
  - shared population stock;
  - shared survey base (LFS; EU-SILC for S80/S20 and poverty risk).
- C1: sign checks of observed between-region and within-country associations against `sign_prior`; report per edge and per grade.
- CLI: `regsim causal check --config ...` writes a table and updates `data_evidence` with tier E1.
- Acceptance:
  - the registry validates;
  - contested edges are retained;
  - agreement rates fall within the ranges in `test_m7.py`.

### M7.2 Literature module

- Six-step pipeline: propose → search → extract → verify → strict measure check → grade.
  - LLM calls go through an injected callable.
  - Tests use recorded OpenAlex/Crossref responses.
- **Strict measure check (mandatory).** Does the study measure the cause and the effect, or their standard proxy, on a representative population? In round 2 it kept 84 of 158 claims and removed all three wrong cases of a 10-claim hand-check.
- **Grades per horizon.** Store sign and grade for short run and long run separately, and keep both when they differ (GDP → life expectancy: short run −, long run +).
- `targeted_queries` for contested edges and data-flagged pairs, including horizon-specific queries.
- Acceptance:
  - the offline pipeline reproduces fixture grades;
  - the grading rules are enforced;
  - the measure check removes the three hand-check failures (all-cause mortality for road deaths, disability insurance for life expectancy, re-training programme for tertiary attainment).

### M7.3 Panel module, direction test, causal queries and causal-lens similarity

- Integrate `panel.py` with `regsim.store`:
  - long panels from the observation store;
  - transforms from the indicator registry;
  - parent-by-year groups from `GeoScheme`.
- Partial pooling of edge effects by region type, and Nickell-bias handling for short panels (D13).
- `direction_test`:
  - align series with `indicator_reference_periods.yaml`;
  - run lagged tests both ways and an event-study profile with pre-trend;
  - write the `temporal_precedence` item.
- `what_if`, `explain_difference` and `vision_check`, with horizon and tier on every row.
- `s_causal` is used by `predict.tasks.analogues` when a target is given.
- Graph-constrained forecast as an optional baseline in `evaluation.benchmark`.
- Acceptance:
  - GDP → cars: β_std 0.06–0.14, I² ≥ 0.6;
  - the EE00 what-if range;
  - Jaccard ≤ 0.3;
  - the graph-constrained VAR beats the unconstrained one;
  - `direction_test` returns `undecided` for GDP ↔ cars (same-year response in both directions) and `forward` for life expectancy → old-age dependency.

### M7.4 SD bridge

- Vensim `.mdl` parser.
- Perturbation runner for the Python KNOWING model (`martnzach-wq/KNOWING-integrated-SD-model`; `CityConfig` TALLINN and GRANOLLERS; NAPLES once its input workbook is available).
- **Perturbation protocol:**
  - scale consistent input groups (all population inputs together);
  - perturb constants and series alike;
  - where a calibration exists (Tallinn `apply_cdd_baseattr_compensation`), run with and without it and report both.
- Mapping table SD variable ↔ regsim indicator with `relation` (same, proxy, inverse_proxy).
- `compare` writes `model_review` items with the diagnosed cause.
- Lever effects go to the intervention table (M7.5), not to the indicator graph.
- `export_inputs` and `transfer_parameters` use `s_causal` and `s_effect`.
- **Parameter transfer to non-demonstrator regions** (`docs/causal/sd_parameter_transfer.md`):
  - classes from `configs/sd_parameter_classes.yaml`: O observable (data or regsim imputation with interval), B behavioural (literature function, e.g. value of time from income), R residual calibration constants (never copied; recalibrated to target mode split and trip lengths, over draws when targets are imputed), S structural switches (one variant for donor and target), P policy levers (scenario inputs);
  - `domain_check` against the demonstrators in the six-indicator lens, with domain class reported on every output;
  - `report_rules`: lever rankings and directions always; absolute levels only for close or moderate regions or after recalibration to observed targets;
  - `cross_calibrate`: leave-one-city-out transfer test, error by parameter class and ranking agreement;
  - `next_demonstrator`: candidate regions ranked by coverage gain.
- Acceptance:
  - edge counts per model;
  - the Granollers fuel-cost elasticity on car-km is −0.28 ± 0.05;
  - the Tallinn CDD response is 0 with calibration and negative on active km without it;
  - consistent population scaling gives |elasticity| < 0.05 for per-capita outputs;
  - every `CityConfig.mobility` field has a parameter class; class-R values are never copied;
  - domain-check and lever-ranking reference values in `test_m7.py`.

### M7.5 Intervention effects

- Event table schema and loader (`configs/events.yaml`).
- Counterfactual by masking post-treatment cells and imputing with the fitted estimator on lens or effect analogues; placebo distribution; pre-treatment window ending at the announcement year.
- First case once 2025–2026 data are published: the Estonian motor vehicle tax (in force 1 January 2025) on cars per 1,000.
- Acceptance: recovery of a synthetic effect; announcement-year handling.

### M7.6 Effect similarity

- `fit_varying`: varying-coefficient panel model with standardised moderators (GDP level, level of the effect indicator, density, region type) and a ridge penalty chosen by leave-one-country-out validation.
- `validate_gate`: a link is heterogeneous only if moderators cut the hold-out error by ≥ 10 % against one pooled effect.
- `region_effects`: derived indicators `effect:<cause>-><effect>` in the store (type `effect`, interval, horizon), not used as model inputs.
- `s_effect`: effect distance and probability of practical equivalence.
- App views:
  - pairwise effect comparison;
  - effect space with ellipses;
  - effect maps with a value-suppressing palette;
  - neighbour comparison;
  - effect profile strip;
  - transfer matrix.
- Acceptance:
  - the gate passes GDP → life expectancy, GDP → unemployment and GDP → cars;
  - it fails household income → cars, old-age dependency → GDP, unemployment → poverty and employment → fertility;
  - Estonia vs Catalonia equivalence probability for GDP → life expectancy is below 0.3.

## 3 Order and effort

M7.1 → M7.3 (panel, `s_causal`, `what_if`, `direction_test`) → M7.6 → M7.2 → M7.4 → M7.5.

M7.1, M7.3 and M7.6 use existing data and deliver the transferability functions (`s_causal`, `s_effect`) early.

## 4 Open decisions (ask the user, record in `docs/decisions.md`)

| ID | Question | Options | Recommendation |
|---|---|---|---|
| D10 | Registry format and location | YAML in repo; YAML plus Parquet evidence store; database table | YAML for edges, Parquet for claims and estimates |
| D11 | Indicator metadata for mechanism classification | add `numerator`/`denominator`/`survey` to indicator registry; separate mapping file | add to indicator registry |
| D12 | Effect source order in `what_if` | as in `queries.py` docstring; literature first for long run | as in docstring, long run from literature when no panel long-run estimate |
| D13 | Panel estimator for short panels | FE with lagged dependent variable (current); bias-corrected FE; Arellano–Bond GMM | FE with bias correction; GMM as option |
| D14 | Region groups for partial pooling and moderators | macro-regions from config; typology classes from `regsim.typology`; GDP terciles | typology classes, macro-regions as fallback |
| D15 | LLM provider and model for the literature module | configurable callable; fixed provider | configurable callable |
| D16 | Treatment of SD review items | registry only; issue in the SD repository; both | both (five items in `sd_model_review.md`) |
| D17 | Use of modelled regional emissions (EDGAR NUTS-2) as test data | accept as E2 with flag; accept only after checking the downscaling proxies; exclude | accept with flag after checking the proxies |
| D18 | Weight of temporal precedence in the evidence score | not used; one weighted item; decisive for dated events only | one weighted item; decisive only in event studies with flat pre-trend |
| D19 | Practical-equivalence threshold δ for effect similarity | 0.05 SD; 0.1 SD; user-set per link | 0.1 SD default, user-set per link |
| D20 | Source of between-region spread for hierarchical priors on SD parameters | literature and meta-analyses; other calibrated models; effect-similarity results | literature first, other calibrated models where available; never the demonstrators alone |
| D21 | Calibration targets for class-R recalibration in regions without observed mode split | regsim-imputed mode split with intervals; national travel survey shares; no levels reported | regsim imputation with intervals, levels reported only within the domain |

## 5 Data requirements

- Annual NUTS-2 panels 2008–2024 for the 16 indicators used in M7 (Eurostat datasets in `docs/causal/causal_test_notes.md`); NUTS-2 population back to 2000.
- Added in version 2:
  - EDGAR NUTS-2 GHG, total and by sector (1990–2024);
  - Eurostat `nrg_chddr2_a` / `nrg_chddr2_m` degree days at NUTS-2, annual and monthly;
  - Eurostat `ilc_di11_r` S80/S20 at NUTS-2.
- Still missing at NUTS-2:
  - renewables share and household electricity (national or DSO sources for demonstrator countries);
  - car-km and active-mode km;
  - monthly series for timing tests.
- Event list for interventions (D3.6 Annex A interventions with dates and locations).
- Observed mode split and trip-length distribution for target regions (national travel surveys, city mobility surveys) as calibration targets for class-R parameters.

## 6 Reference results

- `docs/causal/causal_panel_results.csv` (46 edges), `causal_panel_screening.csv` (240 pairs), `causal_forecast_skill.csv`, `causal_edge_heterogeneity.csv`
- `docs/causal/causal_lagged_direction.csv`, `causal_lagged_direction_prior.csv`, `causal_event_direction.csv`: temporal precedence
- `docs/causal/region_effects.csv`, `effect_similarity_validation*.csv`: effect similarity
- `docs/causal/causal_edges_literature_r2.csv`, `lit_r2_claims_graded.csv`, `literature_r2_handcheck.csv`: literature round 2
- `docs/causal/sd_city_elasticities.csv`, `sd_anomaly_tests.csv`, `sd_model_review.md`: SD review
- `docs/causal/regional_retests.csv`, `regional_data_gaps.md`: regional data
- `docs/causal/sd_parameter_transfer.md`, `sd_calibration_parameters.csv`, `demonstrator_coverage.csv`, `next_demonstrator_candidates.csv`: parameter transfer
- `docs/causal/causal_edges_prior.csv`, `causal_candidates_llm.csv`, `causal_edges_literature.csv`, `sd_dependency_edges.csv`, `sd_tallinn_elasticities.csv`: version-1 inputs
