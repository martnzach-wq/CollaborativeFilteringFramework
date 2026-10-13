# Progress

## M5 Typology, uncertainty, applicability domain, rules (2026-10-13)

**Done**
- Rule engine (`regsim.rules.engine`, `configs/rules_d36.yaml` validated by `regsim.config.RuleSet`): gates G001–G010 as conditions on demonstrator `d` and target `t` in a restricted expression language (parsed and checked with `ast`, evaluated vectorised over all targets; a comparison with a missing value is false). Each rule carries the severity of D3.6 Table 10 and the one stored in GeoPackage v2.5 (only G001 hard). Regsim mode treats all gates as soft flags (decision D9); D3.6 mode marks pairs failing a hard gate as ineligible and drops their TRF class. The `koppen_adjacency` table of D3.6 is not in the export; its adjacent pairs were derived from the published G005 firings. `regsim similar --mode d36` lists eligible pairs with their fired gates.
- Split-conformal calibration (`regsim.uncertainty.conformal.Conformalised`): calibration cells are random cells plus cold-start entities (k = 1, 3, 6 or 10 cells kept) at every level; normalised scores |z − ẑ|/σ pooled over three calibration splits; quantiles per Mondrian group (level × observed-count bin ≤3, 4–7, 8–12, ≥13) with fallbacks; predictions from the first calibration model.
- Applicability domain (`regsim.uncertainty.domain.applicability`): observed count, k-NN distance in the latent space to well-observed entities and RMS standardised residual, as percentiles within the level; `in_domain`, `edge` (< 5 observed or > 95th percentile), `outside` (< 2 observed or > 99th percentile).
- Typology runs (`regsim.typology.run`): GMM (diagonal), Ward, HDBSCAN and contiguity-constrained Ward on whitened completed profiles or the latent projection; consensus over bootstrap resamples and posterior samples; k by bootstrap stability (ARI ≥ 0.7) and BIC; labels from standardised centroid deviations with dispersion; Hungarian tracking with transition matrices; configuration hash over run, model parameters and data. `regsim typology` writes the resolved config, assignments, cluster table and selection table.

**Results**
- Gates: the published flags of all nine evaluated gates and `eligible` are reproduced for all five demonstrators (502,130 pairs, 100 %). G005 agrees by construction (its adjacency table was derived from the same flags); the other gates are independent checks.
- Coverage of nominal 90 % intervals (Eurostat 2018/2022, prototype protocol):

| Hold-out | Level | Laplace | Conformal |
|---|---|---|---|
| M1 random cells (seeds 0–2) | NUTS 0 | 0.948 | 0.926 |
| | NUTS 2 | 0.860 | 0.909 |
| | NUTS 3 | 0.937 | 0.891 |
| M2 cold start (seeds 100, 101) | NUTS 2 | 0.916 | 0.899 |

  Over more seeds (M1 0–5, M2 100–107) the conformal coverage is 0.919 / 0.914 / 0.900 (M1) and 0.913 (M2, ensemble variant 0.912). Calibrating once without pooling gave 0.885–0.934 between seeds; refitting on all cells after calibration over-covered cold start (0.936).
- Applicability domain on M2 cold start: RMSE 0.291 (`in_domain`), 0.297 (`edge`), 0.425 (`outside`); standardised RMS error 0.88 / 0.85 / 1.18, so `outside` flags entities whose intervals are too narrow before calibration.
- Typology of the NUTS-2 regions (`configs/example.yaml`, GMM): bootstrap stability 0.62 for k = 3 and 0.36–0.44 for k = 4–12, while BIC decreases up to k ≈ 7–12. The regions form a continuum rather than well-separated types; typologies are reported with their stability for this reason.

**Tests**: 106 unit tests pass; M0–M5 acceptance pass.

**Open**
- Decision D7 (front end, M6).
- The D3.6 Köppen adjacency for zone pairs not covered by the five demonstrators (asked of the D3.6 authors together with the categorical rules).

## M4 Similarity (2026-10-13)

**Done**
- D3.6 compatibility mode (`regsim.similarity.d36`, `configs/d36_scoring.yaml`): per-tier weighted Gower distance on the D3.6 normalised values, composite with the weight schemes `default`, `equal` and `domain_primacy`, TRF1–TRF5 classes (decision D8: classes only in this mode), per-indicator partial distances (Layer 4). The rules not specified in the deliverable were inferred from `precomputed_scores`: indicators linked to all four CICs carry weight 4 (join fan-out), CIC-specific ones weight 1; T1 categorical weights climate 1, terrain 4, others 1, `degurba` and `urban_type` 0; Köppen distance 0 / 0.25 / 0.6 / 1; an empty tier has distance 1.
- Entity similarity (`regsim.similarity.entities`): S-profile (whitened completed profiles with the model-implied covariance), S-latent, S-gower with coverage ratio, percentile scores among all pairs at the same level, Mahalanobis contributions per indicator, 90 % distance intervals from posterior samples; lens selection (decision D4: all model indicators, no polarity).
- Transfer-specific weights (`regsim.similarity.transfer`): non-negative indicator weights minimising the leave-one-out error of kernel-weighted neighbour predictions, pulled towards equal weights, with bootstrap intervals.
- Indicator similarity (`regsim.similarity.indicators`): I-model, I-observed (Spearman with Fisher intervals), I-partial (graphical lasso), I-predictive.
- Neighbour-prediction skill (`regsim.evaluation.neighbours`) and `regsim similar` (modes `profile` and `d36`).

**Results**
- D3.6 mode against `precomputed_scores` (all CICs, NUTS-3 targets): T2, T3 and T4 exact; composite median absolute deviation 0.1–0.9 points, 95th percentile below 1 point, maximum 13 points; TRF class agreement 96–99 %. Tallinn (urban): T2 exact for 1,523 of 1,524 targets (the remaining one has an empty T2, NaN in the table and distance 1 in the published composite), composite median 0.17, p95 0.96, maximum 6.7 points; TRF agreement 98.7 %.
- Neighbour skill on M2 cold start (NUTS 2, seeds 100 and 101, k = 10): RMSE S-profile 0.617, S-gower 0.909, coverage 1.0 for both.
- I-model and I-observed agree (correlation of the pairwise values 0.92).

**Tests**: 91 unit tests pass; M0–M4 acceptance pass.

**Open**
- Decisions D7 (M6) and D9 (M5).

## M3 Core model HierTypedMF (2026-10-13)

**Done**
- `HierTypedMF` (PyTorch, float64, full-batch L-BFGS with preconditioned entity factors): typed losses (squared or Huber; cumulative logit for ordinal; multinomial for categorical columns), hierarchical prior with membership-weighted (fractional) parents, aggregation constraints on the original scale per `aggregation` metadata (ratios weighted by their denominator, sums only for fully covered parents), side information, optional spatial Laplacian (GISCO adjacency) and random-walk penalty for geo-time entities; couplings `none`, `broadcast` (LEVITATE), `prior_only`, `prior_and_constraint`.
- Block-diagonal Laplace posterior per entity (`predict_std`, `predict_dist`); fold-in of entities (with children, parents' aggregation constraints and siblings when the entity is a row of the fitted data) and of indicators.
- Coherence by a variance-weighted MinT-type projection on the model scale (observed cells fixed, with a measurement variance so that inconsistencies between published values are not pushed onto single predicted cells).
- `Tuned`: hyperparameters by masked cross-validation within the training cells (nested in the benchmark); `configs/benchmark_m3.yaml`.
- `prepare` builds several period windows as geo-time entities (decision D3), side information and adjacency; `ModelData.parent` now holds membership fractions.

**Benchmark (Eurostat 2018/2022, prototype protocol, `runs/evaluate-d8409fe4239a`)**

| Method | M1 random cells | M2 cold start | M3 country blocks | M4 level transfer |
|---|---|---|---|---|
| HierTypedMF (tuned) | **0.360** | **0.324** | 0.625 | **0.392** |
| MF, hierarchical prior | 0.361 | 0.325 | 0.606 | 0.411 |
| MF, broadcast (LEVITATE) | 0.441 | 0.417 | **0.499** | 0.507 |
| ParentCarryDown | 0.729 | 0.807 | 0.556 | 0.615 |
| MissForest | 0.590 | 0.860 | 0.737 | 0.724 |
| Prototype "MF, prior + aggregation" | 0.425 | 0.507 | 0.723 | 0.410 |

Other results: synthetic recovery (generator defaults, MCAR 30 %) 0.6° (indicator factors) and 6.8° (entity factors); with level-missingness (28 % observed) 8° and 16°. Fold-in vs full refit: NUTS-3 0.438 vs 0.437, NUTS-2 0.380 vs 0.407. Coherence after reconciliation: 95 % of parent groups with predicted children within 0.8 %.

**Tests**: 82 unit tests pass; M0–M3 acceptance pass (M3: subspace recovery, RMSE below the prototype in M1–M4 with the tuned model, coherence within 1 % for 95 % of groups, NUTS-3 fold-in within 0.02 of a full refit; 15 minutes on 8 cores).

**Calibration**: nominal 90 % Laplace intervals cover 92 % of held-out cells in M1 (NUTS-0 96 %, NUTS-2 88 %, NUTS-3 94 %).

**Findings**
- The hierarchical prior gives most of the gain over plain MF; the aggregation constraint adds to it in level transfer (M4: 0.411 → 0.392).
- Country blocks (M3): when all regions of a country are hidden for an indicator, copying the national value (0.556) or broadcasting it into the matrix (0.499) beats the model (0.625), whose predicted regional variation adds error. Candidate remedies: a scheme-matched inner CV for `lam_u`/`lam_agg`, or shrinking towards the parent value when a block has no regional information.
- The tuned model chose the prototype's hyperparameters (rank 12, λ_u = 0.03, λ_agg = 1) in this benchmark; tuning costs about 17 fits per split.

**Open**
- Conformal recalibration by level and number of observed indicators (M5).
- Country-block weakness (see findings).
- Decisions D4, D7, D8, D9.


## M2 Baselines and evaluation harness (2026-10-13)

**Done**
- `regsim.prep.matrix.prepare`: store → model matrix for a period window (registry transforms, robust scaling on the native level, `same_as` merge by precedence, static indicators, hierarchy, population-weighted aggregation matrix, countries).
- Baselines behind the `Estimator` interface: `GlobalMean`, `LevelMean`, `ParentCarryDown` (fractional parents), `KNNGower` (D3.6-type, same level, pairwise deletion), `SoftImpute`, `IterativePCA` (regularised, missMDA-type), `MissForest` (scikit-learn `IterativeImputer` with random forests, out-of-bag in-sample predictions).
- Masking schemes M1–M4 (`Split` objects), metrics (RMSE, MAE, R² per column, coverage, interval width, Gaussian and sample CRPS), benchmark runner with joblib (loky) and sequential fallback, model card (Markdown + CSV), prototype protocol preset.
- `regsim evaluate configs/benchmark_prototype.yaml` writes `runs/evaluate-<hash>/` with model card, results, resolved config and hash.

**Benchmark (Eurostat 2018/2022, NUTS-0/2/3, 1,669 × 25, prototype protocol)**

| Method | M1 random cells | M2 cold start | M3 country blocks | M4 level transfer |
|---|---|---|---|---|
| ParentCarryDown | 0.729 | **0.807** | **0.556** | **0.615** |
| MissForest | **0.590** | 0.860 | 0.737 | 0.724 |
| SoftImpute | 0.657 | 0.926 | 0.870 | 0.846 |
| KNNGower | 0.670 | 0.909 | 0.854 | 1.256 |
| IterativePCA | 0.693 | 0.951 | 0.763 | 0.656 |
| LevelMean | 1.223 | 1.209 | 1.158 | 1.256 |

Prototype MF reference values (target for M3): 0.425 / 0.507 / 0.723 / 0.410.

**Tests**: 69 unit tests pass; M0, M1 and M2 acceptance pass (M2: identical ranking of the prototype baselines in M1, M2, M4; RMSE within ±0.05 in M1 and M4 and within ±0.10 in cold start, as agreed).

**Open**
- Hyperparameters of the baselines are the prototype's (SoftImpute λ = 5, rank ≤ 12; IterativePCA rank 5); nested masked cross-validation follows with M3.
- Decisions D4, D7, D8, D9.


## M1 Data layer (2026-10-13)

**Done**
- Eurostat connector: JSON-stat 2.0 with flags, filter validation against SDMX content constraints, response cache, retry for 413/429/5xx and asynchronous responses, split by period.
- Geo registry: NUTS 2024 levels 0–3 from GISCO (area from 1:1 million geometry), LAU 2024 from the Eurostat LAU–NUTS list, `HierarchyGraph` with population and area weights, fractional membership and aggregation matrices.
- NUTS 2021 → 2024 correspondence (1:1 for 1,575 codes; LAU population overlap for FI and PT; polygon overlap for DE, LV, NL) and `recode()` by aggregation rule. Most Eurostat regional tables are already republished in NUTS 2024 back to 2013; recoding affects 0–3 % of rows per indicator.
- Indicator registry (pydantic) with `same_as` links and source precedence (D5) and a `temporal` field for time-invariant indicators; observation store (Parquet per source, DuckDB views, entity × indicator matrix per period window).
- D3.6 adapter: 41 of 79 indicators are populated in GeoPackage v2.5, each for all 100,426 regions; none of the Eurostat copies is populated, `population_density` is linked to Eurostat `pop_density`. Registry generated as `configs/indicators_d36_v2.5.yaml`; precomputed scores stored for M4.
- `regsim build configs/example.yaml`: 1,683 units (39 / 299 / 1,345), 236,795 observations at NUTS-0/2/3 for 2013–2022 plus D3.6 descriptors.
- Registry fix: `fertility_rate` needed `indic_de: TOTFERRT`.

**Tests**: 47 unit tests pass, M0 and M1 acceptance pass (M1: observed cells 17,002 vs 17,025 after removing the UK, −0.14 %; units with population 1,669; D3.6 round trip of 4,117,466 rows).

**Added after M1: ARDECO fallback (D5)**
- `ArdecoConnector`: GraphQL metadata (dimension combinations per variable) and CSV export per NUTS level; shared HTTP client with cache and retries (`regsim.data.http`), also used by the Eurostat connector.
- `configs/indicators_ardeco_v1.yaml`: `SUVGDP` (GDP per inhabitant, PPS) and `RPECNP` (employment rate 20–64) linked to their Eurostat counterparts with `same_as`; `SNPTN` as population fallback. Agreement with Eurostat 2018–2022: GDP median ratio 1.000 (Spearman 0.999), employment rate 0.991 (0.985), population identical.
- Effect in the 2018/2022 matrix: ARDECO fills 1,357 NUTS-3 cells (1,323 employment rate, 34 GDP) and a few NUTS-0/2 cells where Eurostat has none; Eurostat values always take precedence. The population fallback adds no unit (ARDECO also lacks BA, UA, XK).

**Open**
- LAU level in the build (`lau: true`) is implemented but not yet exercised on the full data (98,395 D3.6 LAUs, 98.9 % match LAU 2024).
- Decisions D4, D7, D8, D9.

## M0 Scaffold (2026-10-13)

**Done**
- Hand-over imported unchanged as the first commit.
- Decisions D1, D2, D3 and D6 taken and recorded in `docs/decisions.md`; `pyproject.toml`, `configs/example.yaml` and `regsim.config` updated (NUTS 2024, two period windows, Apache-2.0, PyTorch as core dependency).
- `regsim.synthetic.generate_hierarchical`: nested levels with optional fractional parents, hierarchical factor prior, random walk over periods, four indicator types (continuous, share, count, ordinal) on the original scale, coarse values recomputed as exact aggregates of their children (population-weighted mean or sum), missingness schemes `none`, `mcar`, `block_mnar`, `level`. Returns `SyntheticData` with true factors, aggregation matrices and a long-format export.
- Ruff configuration (lint rules and formatter), skeleton reformatted; pre-commit hooks; GitHub Actions on Linux and Windows with Python 3.12 and 3.13 (CPU build of PyTorch).

**Tests**: 19 passed, 6 skipped (acceptance tests M1–M6). M0 acceptance test passes locally on Linux; Windows is covered by CI.

**Open**
- Decisions D4, D5, D7, D8, D9 (needed from M1/M4 on; D5 before M1).
- Categorical indicators are not yet generated by the synthetic generator (needed for the typed losses in M3).
