# Limitations

## Data layer (M1)

- **NUTS version correspondence.** Only NUTS 2021 → 2024 is supported (Eurostat change table). Values that Eurostat still publishes under older codes (NUTS 2016 and 2013, e.g. `BE221`, `HR04`, `FR2`, `NO01`) are dropped and listed in `build_report.json`; they concern earlier years of series that Eurostat republished under current codes.
- **Correspondence weights.** For changed regions the weights are LAU population overlaps where the LAU codes of both years cover at least 95 % of every changed region (Finland, Portugal); otherwise polygon overlaps of the 1:1 million boundaries weighted by 2021 population (Germany, Latvia, the Netherlands), which assumes uniform density within each 2021 region.
- **Aggregation rules in recoding.** `wmean_area` and `ratio` indicators are recoded with population weights; `none` indicators are recoded only where the mapping is 1:1.
- **United Kingdom.** NUTS 2024 has no UK regions, so UK data are dropped (decision D1).
- **Population.** Units without Eurostat population (BA, UA, XK) use area shares as edge weights.
- **LAU.** Only the LAU year matching the NUTS version is linked; LAU 2021 → 2024 changes are not mapped.
- **ARDECO.** NUTS-3 employment rates are experimental ARDECO estimates and enter the model as observations flagged `ardeco`. ARDECO has no GVA counterpart in the Eurostat registry, so GVA is not used. Restrict an entry with `source.levels` to exclude levels.

## Baselines and evaluation (M2)

- **Baseline intervals.** Baseline predictive distributions are Gaussian with the in-sample residual scale per column; they are too narrow for the low-rank and neighbour methods (90 % nominal coverage: 46–77 %). Calibrated intervals follow with split-conformal calibration (M5).
- **Cold start (scheme M2).** RMSE depends strongly on which entities are drawn (standard deviation 0.07–0.10 between seed pairs); single runs are not comparable to ±0.05.
- **MissForest.** About 30 s per fit on the Eurostat matrix (50 trees, 5 rounds); in-sample predictions at observed cells are out-of-bag predictions.

## Core model (M3)

- **Laplace posterior.** Block-diagonal per entity over (u, b) with V, mu and the hyperparameters fixed; uncertainty in the indicator factors and the aggregation constraints are not propagated. Ordinal and categorical columns get predictive probabilities but no Laplace term.
- **Fold-in.** Uses continuous columns only; aggregation and children terms need the entity to be a row of the fitted data.
- **Reconciliation.** Linearised projection (three Gauss-Newton steps); observed cells are never changed, so groups whose published values are inconsistent stay inconsistent.
- **Tuning.** Inner cross-validation uses random cells for all schemes; it does not mimic cold start, country blocks or level transfer.
- **Runtime.** About 6-15 s per fit on the Eurostat matrix (1,669 × 25) with 4 threads; tuning with 8 combinations and 2 folds about 9 minutes per split.

## Similarity (M4)

- **D3.6 scoring rules.** Indicator weights, categorical partial distances and the empty-tier rule are inferred from `precomputed_scores`, not taken from the deliverable. T1 deviates in some regions (median 0.4, maximum 17 points for Tallinn) because the treatment of missing climate values and of `degurba` and `urban_type` is not fully determined by the published scores; T2–T4 are exact.
- **Empty tiers.** The published table leaves the score of an empty tier NaN while its composite uses distance 1; `D36Scorer` reports distance 1 (score 0) for the tier as well.
- **S-profile.** Distances use completed profiles; for entities with few observed cells they reflect the model more than the data, which the observed share and the intervals make visible.
- **Transfer weights.** With fewer than about 30 labelled entities the weights are a diagnostic; the bootstrap intervals are wide.
