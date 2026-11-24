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

## Typology, uncertainty and rules (M5)

- **Köppen adjacency.** Derived from the published G005 flags, which cover pairs with the zones BSk, Cfa, Cwc and Dfa; other pairs fall back to "same main group = adjacent". The published G005 flags agree by construction.
- **Gate severities.** D3.6 Table 10 lists six hard gates, the GeoPackage v2.5 export stores only G001 as hard; the D3.6 mode follows the export by default (`severity="table10"` for the table). G004 needs floodplain indicators that are not in the export and never fires; G010 is not evaluated in the export.
- **Conformal intervals.** Calibration assumes that held-out cells resemble the calibration cells (random cells or cold-start entities); other hold-out patterns (country blocks, level transfer) are not calibrated. Predictions come from a model fitted without the calibration cells; the mean of the calibration models predicts better (M1 RMSE 0.337 vs 0.377 at NUTS 3) but over-covers (0.943 at NUTS 0). At NUTS 0, with about 80 hold-out cells per seed, the coverage of a single split varies between 0.85 and 0.97.
- **Applicability domain.** Percentile thresholds are relative to each level, so a level always has about 5 % of its well-observed entities at `edge` or beyond.
- **Typologies.** NUTS-2 regions show low cluster stability (ARI ≤ 0.62); the number of clusters is then chosen by stability alone. Consensus matrices are dense (n × n) and limit runs to a few thousand entities.

## Interfaces and LAU extension (M6)

- **LAU scale.** One HierTypedMF fit on the LAU matrix (100,130 rows) takes about 11 minutes and 2.6 GB; the LAU benchmark covers six indicators. Calibration, typologies with bootstrap consensus (dense n × n matrices) and KNN baselines are not run at LAU scale.
- **LAU indicators.** Below NUTS-3 only the D3.6 indicators and the LAU-list population are observed; Eurostat indicators at LAU are predictions from the hierarchy.
- **Maps.** Polygons are simplified to 1 km and drawn without holes; overseas regions lie outside the default extent.
- **Explorer.** Typology runs in the app use 10 bootstrap resamples to stay interactive; the app holds the model in memory for a single user.
- **Report figures.** The images in `docs/visualisation.md` are those of the prototype; `regsim report` writes the current ones to the run folder.

## Causal layer: consistency checks (C1)
- C1 correlates region means; agreement with a prior sign is evidence tier E1 (an association in the expected direction), not a causal effect. Between-region correlations mix within- and between-country differences; four edges contradict their prior between regions but agree within countries.
- `classify_mechanism` knows two composite quantities (labour force, population change); other accounting links (GDP and household income in the national accounts) are not yet described by the metadata.

## Causal layer: panel effects and queries (M7.3)
- Panel effects describe year-to-year co-movement within regions net of country×year shocks; country×year effects also remove national policies. Long-run effects are the autoregressive transform of these short-run effects unless a literature elasticity is used.
- `what_if` propagates first-order log changes along the graph; edges that close a cycle are left out, and interactions between causes are not modelled.
- Group effects depend on the grouping: Estonia belongs to "North" among the macro-regions of the M7 analysis (no GDP effect on car ownership) and to a typology class with a small positive effect.
- `explain_difference` uses a between-region regression on the causal parents; collinear parents (GDP and household income) give coefficients of opposite sign.



## Causal layer: effect similarity (M7.6)
- Region effects are short-run within-region effects from annual data, predicted from four moderators and region groups; three of seven links pass the validation gate. A pooled link means the data do not support a transferable difference, not that the effects are equal.
- The gate depends on the region groups used as moderators: with typology classes instead of macro-regions, GDP → unemployment does not pass.

## Causal layer: literature module (M7.2)
- Grades rest on abstracts; magnitudes are rarely reported there. The strict measure check removes proxies that the first passes accept, and with it fewer edges reach grades A and B.
- The C30 fixture is built from the claim table of the M7 analysis, with titles in place of abstracts; it tests the pipeline mechanics and the grading, not the extraction prompts.

## Causal layer: SD bridge (M7.4)
- Elasticities are model properties at 2050 for one-sided +10 % steps; Tallinn's level-input elasticities depend on its GC-neutral calibration and are not transferable.
- The expected responses in `configs/sd_mapping.yaml` are signs from the review and the literature; the comparison does not test magnitudes where no literature value exists.
- Parameter transfer and cross-calibration are implemented as protocol functions; their application to the KNOWING cities needs observed mode shares and trip lengths as calibration targets.

## Causal layer: intervention effects (M7.5)
- Effects are estimated for one treated region at a time against untreated analogues; spill-overs to the analogues and other interventions in the same years are not modelled.
- Intervals and p-values come from in-space placebos on the analogues; with few untreated regions of similar profile they are wide, and per-year intervals are less reliable than those of the averages.

