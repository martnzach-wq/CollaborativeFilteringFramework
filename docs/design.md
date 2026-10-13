# Framework design: similarity and prediction for multi-level regional indicators

Working title of the software: **regsim** (regional indicator similarity and estimation). The design builds on the review (`review.md`) and the related work (`related_work.md`). Requirements R1–R10 are referred to throughout.

## 0 Requirements

| ID | Requirement |
|---|---|
| R1 | Any indicator, of any data type, observed at any level from NUTS-0 to LAU, in any period |
| R2 | Similarity of indicators, derived from data, including across levels |
| R3 | Similarity of geo-entities at each level and across levels |
| R4 | No static classification of indicators in the model; typologies of indicators and geo-entities are derived on demand |
| R5 | Prediction of unknown values from known indicators of the same entity and from similar entities (recommender tasks) |
| R6 | Uncertainty for every prediction and similarity; explicit domain of applicability |
| R7 | Transparent decomposition of every similarity score (D3.6 reporting layers) |
| R8 | Visualisation based on dimensional reduction, with stated distortion |
| R9 | Europe first; geo-schemes, data connectors and models as plug-ins for extension beyond Europe |
| R10 | Validation against baselines under realistic missingness |

## 1 Overview

```
 connectors ──► observation store ──► typed transforms ──► latent model ──► completed profiles + factors + uncertainty
 (Eurostat,     (long format,         (log, logit, CLR,     (hierarchical,        │
  GISCO, D3.6,   geo registry,         ordinal, robust       typed, side-info      ├─► similarity (entities, indicators)
  WDI, ARDECO,   indicator registry)   scaling)              matrix factorisation) ├─► typologies (lenses, clustering)
  embeddings)                                                                      ├─► prediction tasks (recommender)
                                                                                   ├─► applicability domain
                                                                                   └─► visualisation, reports, API
```

`figures/fig_architecture.png` shows the same structure with the plug-in interfaces.

## 2 Data model

### 2.1 Observations (long format)

One row per observed value:

| Field | Type | Note |
|---|---|---|
| `geo_id` | str | Scheme-qualified, e.g. `NUTS2021:AT130`, `LAU2021:AT:90001`, `GADM41:KEN.30_1` |
| `indicator_id` | str | Registry key |
| `period` | str | ISO year (`2019`) or window (`2015/2019`) |
| `scenario` | str | `observed` by default; `ssp245`, … for projections (as in D3.6) |
| `value` | float | Numeric; categories as integer codes; compositions as one row per component (`component` field) |
| `component` | str, nullable | For compositional indicators |
| `se` | float, nullable | Reported standard error, if any |
| `source`, `source_version`, `flags` | str | Provenance and Eurostat flags (`b`, `e`, `p`, …) |

The store is Parquet files queried with DuckDB (spatial extension for geometries). An adapter reads the D3.6 GeoPackage and PostGIS database into this format.

### 2.2 Geo registry and geo-schemes

- `geo_unit`: `geo_id`, `scheme`, `level` (integer depth in the scheme), `name`, `country`, `area_km2`, `population`, `geometry` (EPSG:3035 for Europe, EPSG:4326 globally), `valid_from`, `valid_to`.
- `geo_edge`: parent–child edges with `weight_pop` and `weight_area` (shares of the child in the parent). Edges may cross schemes (LAU → NUTS-3, GADM-2 → GADM-1), and a child may have several parents with fractional weights (for non-nested units such as functional urban areas).
- `geo_correspondence`: mapping between versions (NUTS 2016 ↔ 2021 ↔ 2024) with population weights.
- `GeoScheme` is a plug-in: NUTS and LAU (Eurostat GISCO) in v1; GADM, national schemes or custom grids later. Nothing in the model refers to NUTS directly; it only sees a directed acyclic graph of units with weights.

### 2.3 Indicator registry

Metadata per indicator, extending D3.6 Annex B:

| Field | Values | Used by |
|---|---|---|
| `dtype` | `continuous`, `count`, `share`, `ordinal`, `categorical`, `compositional` | transforms, loss |
| `transform` | `identity`, `log1p`, `logit`, `clr`, `rank`, `yeojohnson`; default by rule (skewness, bounds) | transforms |
| `aggregation` | `sum`, `wmean_pop`, `wmean_area`, `ratio(num, den)`, `none` | multi-level coupling |
| `native_levels` | list | evaluation, reporting |
| `polarity` | `+1`, `−1`, `None` | display only (LEVITATE TargetVal); never used in fitting |
| `unit`, `definition`, `source` | text | documentation |
| `tags` | free list (e.g. "heat", "transport") | search and lens construction only; not used in the model |

Thematic tags exist for navigation. They do not enter the fit, the similarity or the typology unless a user selects them to define a lens (Section 5.1). This keeps R4.

## 3 Preprocessing

1. **Harmonise** geo codes to one target version per scheme via the correspondence table (open decision D1).
2. **Transform** by `dtype`/`transform`: `log1p` for right-skewed positives, `logit` for shares in (0, 1), CLR for compositions (each CLR component becomes a separate model column, with a sum-to-zero note in metadata), ordinal codes to rank scores.
3. **Scale** each transformed column by a robust location and scale (median, scaled IQR) estimated on the native level of the indicator. Min–max scaling is not used, since a few extreme regions would set the range (review K3).
4. **Deduplicate** near-identical indicators only on user request; redundancy is reported (Section 4.4) instead of removed silently.

## 4 Latent model

### 4.1 Base model (typed matrix factorisation)

For entity g and transformed indicator column k:

ẑ(g,k) = μ_k + b_g + u_g · v_k

with factors u_g, v_k ∈ ℝ^r. The loss sums a type-specific term over observed cells (generalised low-rank model, Udell et al. 2016): squared or Huber loss for continuous columns, ordinal hinge or cumulative-logit loss for ordinal columns, multinomial loss for categorical columns. Rank r and penalties are chosen by masked cross-validation (Section 8), not set by judgement (review L1).

### 4.2 Hierarchy and multi-level coupling

The central change against LEVITATE (review L2). Two mechanisms:

**(a) Hierarchical prior on entity factors.**

u_g = u_parent(g) + A·x_g + δ_g,  δ_g ~ N(0, σ²_level(g) I)

Entities borrow strength from their parent (shrinkage, as in small-area estimation), and from side information x_g (area, population density, coordinates, DEGURBA, aggregated satellite embeddings such as AlphaEarth, Section 6.3). Fractional parents use the weighted mean of parent factors. A graph-Laplacian penalty on spatially adjacent units (same level) is optional.

**(b) Aggregation constraints from coarse observations.**

For an indicator with aggregation rule agg_k and an observed value at parent p:

L_agg = Σ_(p,k observed) w_k · ( z(p,k) − T_k( agg_k{ T_k⁻¹(ẑ(c,k)) : c ∈ children(p) } ) )²

A national or NUTS-2 value is thus a constraint on the aggregate of the finer-level predictions, not a value copied to every child. For `aggregation = none` (e.g. typologies, Köppen class), the parent value enters only as a covariate in x_g.

Three modes are implemented to make the effect testable: `broadcast` (LEVITATE behaviour, for comparison), `prior_only` (a), `prior_and_constraint` (a + b, default). Coherent outputs across levels can be enforced after fitting by reconciliation (Wickramasuriya et al., 2019).

### 4.3 Time

- **v1, geo-time entities.** Rows are (g, period window), as in LEVITATE, with u_(g,t) = u_g + τ_(g,t) and a random-walk penalty on τ over consecutive windows. This gives trajectories and "velocity vectors" (LEVITATE p. 62) with uncertainty.
- **v2, tensor model.** Entity × indicator × period with smooth temporal factors (Bahadori et al., 2014); scenario projections (SSPs) as additional slices, as foreseen in the D3.6 `scenario` column.

### 4.4 Model outputs

- Completed profile Ẑ (all entities × all indicators) with predictive standard deviation per cell.
- Entity factors U (with posterior covariance per entity), indicator factors V, biases.
- Residuals of observed cells (for outlier and data-error detection; e.g. implausible Köppen codes, review K5).

### 4.5 Estimators

All estimators implement the same interface (`fit`, `predict`, `predict_dist`, `fold_in`, `embed_entities`, `embed_indicators`):

| Estimator | Role |
|---|---|
| `GlobalMean`, `ParentCarryDown`, `KNNGower` | Baselines; `KNNGower` reproduces the D3.6 metric as a predictor |
| `IterativePCA` (missMDA-type), `SoftImpute` | Statistical baselines |
| `MissForest` / chained trees | Strong non-linear baseline (no latent space) |
| `HierTypedMF` (ALS or PyTorch, MAP + Laplace posterior) | Core model, v1 |
| `BayesHierMF` (variational or MCMC) | Full posterior, v2 |

## 5 Similarity

All similarities are computed for a **lens** Λ = (indicator set K, level set, period, scenario). A lens defaults to all indicators; it can be chosen by the user, taken from a data-derived indicator cluster (Section 6.2), or learned for a transfer target (Section 5.3).

### 5.1 Geo-entity similarity

| Measure | Definition | Property |
|---|---|---|
| **S-profile** (default) | Distance between completed profiles on lens K after decorrelation: d(g,h) = ‖W_K (ẑ_g − ẑ_h)‖ with W_K the whitening matrix of the model-implied covariance on K | Uses all indicators for every pair (no pairwise deletion, K4); removes double counting of correlated indicators (K2); comparable across pairs |
| **S-latent** | ‖(u_g − u_h) V_Kᵀ‖, equivalent to S-profile without residual terms | Fast, defined for entities with few observations |
| **S-gower** | D3.6 Gower distance on observed values with coverage ratio | Transparency and continuity with D3.6 |
| **S-transfer(k)** | Metric learned so that k-nearest neighbours predict target quantity k by leave-one-out (Section 5.3) | Answers "similar for what?" |

Every distance is reported with an interval obtained by propagating posterior samples of ẑ (or U), and with the per-indicator contribution vector (D3.6 Layer 4). Scores on 0–100 are derived from the empirical distribution of distances at the same level (percentile rank), not from 1 − D (review K3, K8). The D3.6 TRF classes can be reproduced on S-gower for comparison.

Cross-level similarity (e.g. a LAU compared with a NUTS-3 region) uses the same completed profiles, restricted to indicators whose transform is scale-free (shares, densities, per-capita values); extensive indicators are excluded automatically through `aggregation = sum`.

### 5.2 Indicator similarity

| Measure | Definition | Property |
|---|---|---|
| **I-model** (default) | Model-implied correlation across entities of a chosen level or subset S: corr(k,l) = v_kᵀ Σ_U,S v_l / √(v_kᵀ Σ_U,S v_k · v_lᵀ Σ_U,S v_l) | Replaces cosine of item vectors (L5); can differ between levels, regions or periods, which tests the stationarity assumption (L6) |
| **I-observed** | Pairwise-complete Spearman correlation with n and confidence interval | Raw evidence; shown with its n |
| **I-partial** | Partial correlation from the graphical lasso on the model-implied covariance | Separates direct from indirect association |
| **I-predictive** | Reduction in held-out error for k when l is revealed | Measures redundancy and information value; feeds Section 7.5 |

Polarity (`+1`/`−1`) can be applied for display, so that positive association reads as synergy (LEVITATE convention).

### 5.3 Transfer-specific similarity

For a target quantity k (an indicator, or an external quantity such as a calibrated SD parameter known for a few demonstrators), learn non-negative weights w over lens indicators by minimising the leave-one-out error of a kernel-weighted neighbour prediction of k. The result is a ranked list of indicators that matter for the transfer of k, and a similarity that is specific to k. With few labelled entities (four demonstrators), the weights are regularised towards the default lens and reported with bootstrap intervals; the method is then a diagnostic, not a fitted metric.

## 6 Dynamic typologies

### 6.1 Typology run

A typology is the output of a run, not a fixed attribute:

`TypologyRun(lens, level(s), representation, algorithm, k or min_cluster_size, spatial_constraint, seed, model_version)`

with a configuration hash, so that runs are reproducible and comparable.

- **Representation**: whitened completed profile on K (default) or latent projection U V_Kᵀ.
- **Algorithms**: Gaussian mixture (soft membership, BIC), Ward hierarchical clustering (nested typology, dendrogram), HDBSCAN (outliers stay unassigned), spatially constrained Ward (ClustGeo-type, optional).
- **Stability**: consensus matrix over bootstrap resamples of entities and over posterior samples of the model; adjusted Rand index between runs; number of clusters chosen by stability and BIC jointly.
- **Labelling**: per cluster, the indicators with the largest standardised centroid deviation (±), formatted as a short template label; optional language-model text from that table, flagged as generated.
- **Tracking**: clusters aligned across periods, levels or runs by Hungarian matching on centroid distance; transitions reported as matrices (Sankey diagrams in Section 9).

### 6.2 Indicator groups and co-clustering

Indicators are clustered on I-model or I-partial (hierarchical clustering with stability), and entities and indicators are co-clustered on the completed, whitened matrix (spectral co-clustering). An indicator cluster can be used as a lens. This replaces the fixed LEVITATE dimensions and the fixed D3.6 tiers by groups that follow from data, and it shows which indicator groups distinguish which entity types.

### 6.3 Demonstrator-centred typology

For KNOWING-type use, the typology can be computed under the lens learned for a transfer target (Section 5.3), giving "regions that resemble demonstrator X with respect to quantity k" as a cluster or neighbourhood.

## 7 Prediction tasks (recommender functions)

| Task | Input | Method | Output |
|---|---|---|---|
| P1 Fill gaps | Known entity, missing indicator | ẑ(g,k) | Value, interval, contributing neighbours |
| P2 New entity (cold start) | A few indicator values, optional side info and parent | Fold-in: ridge solution for u_new with prior u_parent + A·x_new | Completed profile, interval, applicability flag |
| P3 New indicator | Values for some entities, metadata | Fold-in of v_new with entity factors fixed | Completed column |
| P4 Vision entity | Target values for a subset of indicators at a future period | Fictive geo-time entity (LEVITATE p. 47), optionally inequality targets (v2) | Completed vision profile; position relative to present trajectory |
| P5 Analogue retrieval | Entity or vision | Nearest real geo-time entities under a lens | "Who already resembles this?" as donor list for transfer |
| P6 Next indicator to measure | Entity, target indicator set | Expected reduction of posterior variance of targets for each candidate indicator (closed form with the Laplace posterior) | Ranked list |
| P7 Level transfer | Indicator known only at coarse level | Constrained disaggregation via Section 4.2 (b) | Fine-level estimates consistent with the coarse total |

P4 and P5 link the framework to backcasting: the vision profile shows which unspecified indicators would move with the targets, and its analogues are candidate donor regions whose pathways may be studied.

## 8 Uncertainty, applicability domain, rules

- **Intervals**: posterior (Laplace or variational) intervals, recalibrated by split-conformal prediction on masked cells, with Mondrian calibration by level and by number of observed indicators per entity.
- **Applicability domain** per entity: (i) number and posterior leverage of observed indicators, (ii) distance of u_g to the training support (k-NN distance percentile in latent space), (iii) residual of the observed cells. Output: `in_domain`, `edge`, `outside`.
- **Rules**: the D3.6 gates are re-implemented as a declarative rule file (YAML, condition + severity). The default severity is soft (flag). A hard rule masks the pair in rankings but never changes a distance.

## 9 Visualisation

Detailed in `visualisation.md`. Principles: every 2D display states its explained variance or trustworthiness; quantitative statements use the full space; maps and embeddings are linked.

## 10 Evaluation protocol

Masked hold-out on the observation store, repeated over seeds, with four masking schemes that mimic real missingness:

| Scheme | What is hidden | Question answered |
|---|---|---|
| M1 random cells | 10 % of observed cells | Interpolation skill |
| M2 entity cold start | All but k ∈ {1, 2, 3, 5, 10} indicators of 10 % of entities | P2 quality versus amount of known data |
| M3 indicator blocks by country | One indicator for all entities of one country | MNAR robustness |
| M4 level transfer | All fine-level values of an indicator, coarse values kept | P7 and the value of aggregation constraints |
| M5 temporal | Last period window | Forecast skill (v2) |

Metrics: RMSE and MAE on the scaled transformed scale, R² per indicator, rank correlation of neighbour lists against the full-data model, interval coverage and width, continuous ranked probability score. Similarity is evaluated indirectly: a similarity is better if its nearest neighbours predict held-out indicators better (neighbour-prediction skill). D3.6 compatibility: Spearman correlation of S-gower rankings with the published `precomputed_scores`.

## 11 Extension beyond Europe

- Add geo-schemes (GADM, national) as plug-ins with their own hierarchy edges; cross-scheme edges to NUTS are not required.
- Side information that is globally available (satellite embeddings, gridded population, global DEGURBA, ERA5 climate) carries the cold-start predictions where statistics are sparse.
- Indicators with European definitions only (e.g. Urban Audit perception surveys) remain valid columns; their absence elsewhere is ordinary missingness, and the applicability domain flags extrapolation.

## 12 Implementation stack

Python ≥ 3.12; numpy, scipy, pandas (or polars), scikit-learn (GMM, HDBSCAN, spectral co-clustering, graphical lasso), PyTorch optional for the typed model, DuckDB + Parquet storage, geopandas/shapely for geometry, matplotlib for publication figures, plotly for interactive views, Typer CLI, FastAPI for a DSS-facing service. Tests with pytest; synthetic data generator with known low-rank hierarchical structure for unit and recovery tests.
