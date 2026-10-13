# Critical review of the LEVITATE collaborative-filtering approach and the KNOWING transferability framework

Sources: LEVITATE D4.2 *Definition of desirable scenarios* (final), and KNOWING D3.6 *Transferability framework* (final, PU). Page numbers refer to the printed page numbers of each deliverable.

## 1 Scope of the comparison

The two documents address related but different questions.

- **LEVITATE D4.2** asks how indicators relate to each other (supporting or conflicting goals), how geo-entities cluster and move over time in indicator space, and what the full indicator profile of a partially specified *vision* would be. It is an exploratory, data-driven analysis of a very sparse entity × indicator × year matrix (pp. 9–10, 43–47).
- **KNOWING D3.6** asks which European regions are similar enough to one of four demonstrators to make a pathway transferable. It is a screening and ranking tool built on an expert-defined indicator set, tiers and weights (pp. 16, 34–49).

The framework to be designed needs both: the latent, sparsity-tolerant view of LEVITATE (prediction, indicator similarity, dynamics) and the transparency, mixed-type handling and reporting layers of D3.6.

## 2 LEVITATE D4.2

### 2.1 Method in brief

Two approaches were applied to Eurostat and World Development Indicators data (128 indicators in the CF model, 46 LEVITATE indicators in the statistical model):

1. **PCA-based iterative imputation** (missMDA; Josse and Husson) on a NUTS-3 × (indicator, year) matrix with 767 columns and 45.4 % missing cells, followed by a second imputation on a (NUTS-3, year) × indicator matrix with 87,056 rows and 71 % missing cells (pp. 43–44, 51–52).
2. **Collaborative filtering by matrix factorisation** (fast.ai `EmbeddingDotBias`): users = geo-entity × year, items = indicators, ratings = indicator values; 20 latent factors; three-year moving averages on a two-year grid; country values copied down to all cities and regions of that country (pp. 44–47).

Results are shown as 2D PCA projections of the 20-dimensional embedding (indicators and geo-entities in one plot), cosine distances between indicator vectors, regional clusters, decade trajectories of Vienna (AT13) and *fictive entities* AT13_2030/AT13_2050 for the vision (pp. 56–60).

### 2.2 Strengths

- **Joint embedding of entities and indicators.** One latent space carries both similarity questions, and the item factors give an indicator similarity that does not depend on pairwise-complete overlaps (p. 45).
- **No explicit imputation step.** The loss is evaluated on observed cells only, which is the appropriate treatment of a matrix with more than 70 % missing cells.
- **Polarity alignment (TargetVal).** Multiplying "lower is better" indicators by −1 makes positive association read as goal synergy and negative association as conflict (pp. 37–38, 48). This is a useful presentation device for policy questions.
- **Geo-time entities.** Treating (entity, year) as a user allows comparisons such as "city A in 2015 resembles city B in 2005" and yields trajectories in latent space (p. 45). The *velocity vector* versus *direction to vision* idea (p. 62) is a direct link to backcasting.
- **Fictive entity for visions.** A vision with a few target values is inserted as an additional user, and the model completes the rest of the profile (p. 47). This is conditional completion and generalises directly to the cold-start case of a new region with few known indicators.
- **Plausibility probes.** Dummy indicators and test entities were inserted to check the model (p. 46).
- **Limits stated openly.** The deliverable notes that 2D projections can mislead and that the imputation and factorisation have an uncertain effect on results (p. 61).

### 2.3 Weaknesses

| # | Issue | Consequence |
|---|---|---|
| L1 | **No out-of-sample validation.** Neither approach reports a hold-out error, a comparison with simple baselines (column mean, parent-level value, k-nearest neighbours) or a choice of rank by cross-validation; 20 factors for 128 indicators were set by judgement (p. 46). | Predictive skill, and hence the meaning of the embedding, is unknown. |
| L2 | **Country values copied to cities and regions** (p. 46). | Creates many identical cells, inflates similarity within countries, leaks information between levels, and treats a national mean as an observation of each city. The latent space partly encodes "same country" instead of structural similarity. |
| L3 | **Normalisation and output link not documented.** `EmbeddingDotBias` passes the prediction through a sigmoid scaled to a rating range; the scaling of indicators with very different distributions (counts, rates, shares, survey percentages) is not described. | Results depend on undocumented preprocessing; heavy-tailed indicators dominate the loss. |
| L4 | **Redundant indicators.** Survey items such as "strongly agree" and "strongly disagree" for the same question, or several income measures, enter as separate items (pp. 38–43). | Built-in correlations dominate some latent factors; they also reappear as "findings" (e.g. Income_n cluster, p. 56). |
| L5 | **Cosine similarity of item vectors read as correlation** (p. 56). | Item biases are excluded and factor scales are not identified (the factorisation is invariant under invertible transforms of the latent space, with regularisation selecting one solution). Cosine values are a heuristic, not an estimate of correlation. |
| L6 | **Stationarity assumption.** Indicator relations are assumed constant over time, across levels and across regions of indicator space (p. 45). | Relations that change over decades or differ between cities and countries are averaged. |
| L7 | **Missing-not-at-random data.** Urban Audit coverage, survey years and national reporting are selective. | The model learns from a biased sample of cells; errors are larger exactly where data are missing. |
| L8 | **No uncertainty.** Predicted vision profiles and positions in latent space are point estimates. | Distances between a city and its vision, or between two cities, cannot be judged against noise. The authors note fluctuations with the number of available values (p. 58). |
| L9 | **Compounded imputation** in the PCA route (two successive imputations, the second on 71 % missing cells) (p. 52). | Imputed values feed into further imputation; variance is understated. |
| L10 | **Typology only by external labels.** Clusters are shown by colouring entities with predefined European regions (pp. 51, 58); no clustering algorithm or stability analysis is applied. | Clusters are confirmed visually, not derived. |

### 2.4 Assessment

The LEVITATE approach provides the right *model class* for sparse, multi-source indicator data and the right *questions* (indicator similarity, entity similarity, dynamics, conditional completion). What is missing is the statistical scaffolding: validation against baselines, explicit handling of geographic levels, documented preprocessing, uncertainty, and derived (not imposed) typologies.

## 3 KNOWING D3.6

### 3.1 Method in brief

More than 60 indicators (Table 1, pp. 17–29) are assigned to four tiers: T1 environmental (40 %), T2 Climate-Impact-Context specific (25 %), T3 implementation feasibility (15 %), T4 contextual factors (20 %) (pp. 34–39). For a demonstrator–target pair:

1. Continuous variables are min–max normalised across Europe, with a log pre-transform when Fisher skewness > 2 (pp. 44–45, Annex B).
2. Gower partial distances are computed per indicator: Manhattan on [0, 1], binary or hierarchical match for categories, normalised rank difference for ordinal variables, Aitchison distance for compositions (Table 11, p. 46).
3. Intra-tier Gower distance with uniform weights and pairwise exclusion of missing values; composite distance = 0.40·D_T1 + 0.25·D_T2 + 0.15·D_T3 + 0.20·D_T4; score = 100·(1 − D); classes TRF1–TRF5 at 20-point steps (pp. 46–47).
4. Ten eligibility gates (seven hard, three soft) on coastal, mountain, DEGURBA, floodplain, Köppen and coverage conditions (Table 10, pp. 41–42).
5. Five reporting layers, from composite score to per-indicator partial distances (pp. 46–49), implemented as SQL functions in PostgreSQL/PostGIS and published as a GeoPackage on Zenodo (doi:10.5281/zenodo.20376109) (pp. 50–53).

### 3.2 Strengths

- **Mixed data types handled explicitly.** Continuous, ordinal, categorical and compositional indicators each have a defined partial distance (Table 11). The CLR/Aitchison treatment of compositions is correct and often overlooked.
- **Transparency by layered reporting.** Composite, tier, thematic and per-indicator layers make each score traceable (pp. 46–49).
- **Distance instead of correlation.** The argument that correlation-based profile similarity ignores absolute level differences is valid for the transfer question (p. 44).
- **Domain knowledge encoded as rules.** Gates express structural incompatibilities that a statistical distance does not see (pp. 39–42).
- **Multi-scale awareness.** The deliverable discusses NUTS versus LAU trade-offs and the modifiable areal unit problem (MAUP), and plans sensitivity tests across levels (pp. 30–33).
- **Reproducible infrastructure.** Indicator metadata, normalisation, distance and scoring are database objects with a scenario column, ready for future-climate indicators (pp. 50–53).
- **Clear statement of what quantitative similarity cannot capture** (governance, *Eigenlogik*) and an explicit future-work agenda: archetype clustering, sensitivity of weights, temporal dynamics, validation (pp. 61–63).

### 3.3 Weaknesses

| # | Issue | Consequence |
|---|---|---|
| K1 | **Weights, tiers and indicator assignment are fixed by expert judgement** and not validated (pp. 34, 62). | Rankings depend on choices that cannot be tested inside the framework. This is the static classification of indicators that the new framework should avoid. |
| K2 | **Correlated indicators within tiers** (mean/max/min elevation and slope; heat days, tropical nights, desert days; three wind and three radiation statistics) with uniform intra-tier weights. | Groups of redundant indicators carry implicit extra weight; effective weights differ from the nominal ones. |
| K3 | **Min–max scaling over all European regions.** | A few extreme regions set the range; most differences shrink to small values. Combined with averaging over many indicators, distances concentrate and scores cluster in the upper classes (Vienna–Tallinn urban 92.7, yet only rank 82 of 1,524 NUTS-3 regions; Vienna–Tallinn similarity on the heat indicators alone 95.9; values from an earlier check of the D3.6 database export). |
| K4 | **Pairwise deletion of missing values.** | Each pair is compared on a different indicator subset, so scores are not comparable across pairs; dropping a dissimilar indicator raises similarity. The coverage gate (G010) flags but does not correct this. |
| K5 | **Hard gates create discontinuities and propagate data errors.** | A single misclassified category (e.g. implausible majority Köppen codes for Tallinn, Naples and Granollers found in the database export) blocks or admits a pair entirely. |
| K6 | **Similarity is demonstrator-centred and CIC-specific.** | No all-pairs similarity, no indicator similarity, no typology, and no prediction of missing values. Each demonstrator was scored for one CIC only in the export. |
| K7 | **Similarity is not linked to transferability of effects.** | Closeness on input indicators does not imply that model parameters or intervention effects transfer. In the Vienna transfer study, mobility damping parameters were not interpolable via D3.6 similarity, and ERA5-Land heat indicators miss the urban heat island. The deliverable acknowledges the lack of validation cases (pp. 62–63). |
| K8 | **Linear score rescaling and fixed class limits.** | TRF classes at 20-point steps have no empirical meaning; with concentrated distances the lower classes are almost never used. |
| K9 | **Multi-level treatment is implicit.** | LAU, NUTS-3 and NUTS-2 units coexist in one table, but there is no model of how values at different levels relate (aggregation, inheritance, uncertainty from scale change). |
| K10 | **Categorical partial distances not fully specified** (Köppen adjacency, hierarchical DEGURBA credit). | Independent recomputation reproduced the urban T2 score exactly but deviated by 3–9 points for T1 and the composite. |

### 3.4 Assessment

D3.6 is strong on *measurement and reporting* (types, compositions, layers, infrastructure) and weak on *statistical learning* (no use of the joint structure of indicators, no handling of sparsity beyond deletion, no validation). It addresses one question, ranking regions against a reference, and leaves the questions of indicator similarity, typology and prediction to future work, which it names explicitly (archetype clustering, sensitivity, temporal dynamics, validation; pp. 61–63).

## 4 Side-by-side comparison

| Aspect | LEVITATE D4.2 (CF) | KNOWING D3.6 (Gower tiers) | Requirement for the new framework |
|---|---|---|---|
| Core question | Structure of indicator space, visions | Ranking of regions vs. demonstrator | Both, plus prediction of unknown values |
| Similarity of indicators | Latent cosine, correlations | Not addressed | Required, from data, level-aware |
| Similarity of geo-entities | Latent distance (implicit) | Weighted Gower | Required, several definitions, task-specific |
| Indicator grouping | Dimensions/goals (fixed), used for interpretation | Tiers/CICs (fixed), used in the score | Dynamic, derived (no static classes in the model) |
| Missing data | Fitted on observed cells only | Pairwise deletion | Model-based, with uncertainty |
| Geographic levels | Country values copied down | Mixed units, MAUP discussed | Explicit hierarchy, aggregation constraints |
| Data types | Numeric only, undocumented scaling | Continuous, ordinal, categorical, compositional | Typed transforms as in D3.6 |
| Time | Geo-time entities, decade averages | Static, scenario column prepared | Geo-time entities first, tensor later |
| Validation | Plausibility probes only | None | Masked hold-out with realistic missingness |
| Uncertainty | None | Coverage flag | Predictive intervals, applicability domain |
| Domain rules | None | Hard/soft gates | Optional rule layer, soft by default |
| Visualisation | 2D PCA of embedding, trajectories | Maps, layered tables | Biplot, heatmaps, maps, distortion diagnostics |
| Implementation | fast.ai notebook, R missMDA | PostgreSQL/PostGIS functions, GeoServer | Python package, file-based store, PostGIS adapter |

## 5 Conclusions for the design

1. Keep the **matrix-factorisation core** of LEVITATE, but fit it under a documented, typed preprocessing (from D3.6 Annex B) and validate it against baselines with masked hold-out.
2. Replace country-value copying by an **explicit hierarchical model**: an entity's latent vector is its parent's vector plus a deviation, and coarse-level observations constrain the aggregate of finer-level predictions.
3. Derive **indicator similarity** and **geo-entity similarity** from the fitted model and its residual structure, not from fixed categories; report observed-data Gower distance as a transparent companion measure for continuity with D3.6.
4. Make similarity **task-specific**: the relevant neighbours of a region depend on which indicator (or which model parameter) is to be transferred. Learn this from predictive performance rather than setting tier weights.
5. Derive **typologies dynamically** by clustering in latent space under user-chosen lenses, with stability analysis and automatic labelling, as D3.6 lists for future work.
6. Attach **uncertainty** to every prediction and similarity, and replace hard gates by a soft applicability-domain check, keeping user-defined rules as an optional layer.
7. Treat visualisation as an output with stated **distortion** (Shepard diagram, trustworthiness), as both deliverables warn against over-reading 2D projections.
