# Progress
## SD destination-weighted mode change and coping sensitivity (2026-11-25)

- Mode change weighted by destination share (D43); cycling factor now identified (Tallinn 1.85, Naples 10.7, South Westphalia 0.75); Tallinn coping-type sensitivity 2050 (D44, off by default).
- Coping types start from their own mode split (D45); Tallinn profile strength and value-of-time profile adjusted so that type differences stay stable (D46).

## SD active modes and EV fleet (2026-11-25)

- Perceived-time levers for cycling, traffic calming and the 15-minute city (D41), held with their lever in single-measure runs; Tallinn EV 18 / 45 / 80 % linked to the sales-ban compliance by coping type (D42); joint recalibration of Tallinn, Naples and South Westphalia (car share of pkm 2050 optimised 0.22 / 0.25 / 0.49).
- Tests: `tests/test_sd_knowing.py` and `tests/test_sd_bridge.py` pass (10).
- Open: cycling factor at its lower bound, since the push measures already raise the bicycle share to the ABM light pathway level.

## SD checks against IESopt and MAED-City (2026-11-25)

- Energy supply fixed (district-heating gap, electrolyser demand) and checked against IESopt 2050; fuel-specific CO2 factors and other passenger transport fuels (D37); direct CO2 within 0–8 % of MAED-City for Naples and South Westphalia up to 2035 and for Tallinn from 2030.
- Tests: 181 unit tests pass (1 skipped); 43 acceptance tests pass (1 skipped) with the SD model's `main` checkout, to which the M7 criteria refer.

## SD class-R recalibration (2026-11-25)

- OD traffic counted with 0.5 in the energy balance (D31), MAED industry convention (D33), class-R calibration of Tallinn, Naples and South Westphalia against the ABM light trajectories (D32); SD branch `claude/demonstrator-calibration`.
- Open: OD-zone populations; Naples and South Westphalia car energy above MAED-City by 70 % and 22 %.

## SD microclimate and demonstrator inputs (2026-11-25)

- SD repository (branch `claude/demonstrator-calibration`): microclimate submodel calibrated on the Tallinn PALM-4U table and coupled through the CDD; CDD from MAED-City; Naples demand and supply from D3.4a; city transport energy factors; trip-length classes for Naples and South Westphalia. Decisions D26–D30.
- Open: OD-zone populations (commuter statistics not reachable from this environment), class-R recalibration of base attractiveness and damping to the ABM reference trajectories (BAU drift in Tallinn and South Westphalia), boundary between OD traffic and resident energy.

## SD calibration inputs from the demonstrator deliverables (2026-11-25)

- Calibration targets extracted from D1.3, D3.3, D3.4a/b and D3.5: `configs/sd_calibration_targets.csv`, `docs/causal/demonstrator_calibration_targets.md`. Decisions D22–D25.
- SD repository, branch `claude/demonstrator-calibration`: South Westphalia as fourth demonstrator (MAED-City and IESopt inputs reconstructed from D3.5), Naples policy paths from D3.4a, regional Coping Types; fixes for the double-counted manufacturing total and the zero wind/hydro generation.
- Open: class-R recalibration of all four demonstrators to the 2019 targets; city-specific transport energy factors; populations of the OD zones; MAED-City and IESopt inputs for Naples from D3.4a.

## M7 follow-up: effect source of `what_if` (2026-11-24)

- After the store rebuild of M7.4 the model was refitted (registry content changed) and the typology chose 10 instead of 8 classes; EE00 moved into a class of 57 regions, and `what_if` (+10 % GDP, cars per 1,000, short run) fell from +0.32 % to +0.11 %, below the acceptance range.
- Decision D12a: for the short run, `edge_effect` takes the entity's own effect from the effect-similarity model where the link passed the validation gate (M7.6), before the typology-class and pooled estimates. `what_if`, `vision_check` and `regsim causal what-if` use the fits of `regsim causal effects`.
- EE00: +0.37 % (90 % interval +0.15 to +0.60 %) from the effect model; pooled panel estimate +0.42 %, typology class +0.11 %. All 26 M7 acceptance tests pass.

## M7.5 Intervention effects (2026-11-24)

**Done**
- `causal.events`: event schema and loader (`configs/events.yaml`, pydantic: announcement not after start), `mc_nnm` (matrix completion with entity and year effects fitted on the observed cells and soft-impute of the residuals; Athey et al. 2021), `choose_analogues` (causal lens where given, else the pre-period path), `effect` (post cells from the announcement year masked and imputed from untreated analogues; effect = observed − counterfactual per year; anticipation and treatment phases; averages with intervals and two-sided p-values from in-space placebos). Any imputation callable can replace `mc_nnm`.
- CLI `regsim causal events --config configs/causal.yaml`; results in `runs/causal-events-<hash>` and the evidence store (tier E3).
- `configs/events.yaml`: Estonian motor vehicle tax (EE00, announced 2024, in force 2025, outcome cars per 1,000).

**Results**
- Simulation (120 regions, 18 years, low-rank panel, effect 0.5 SD, 30 runs): 90 % interval of the average effect covers the true effect in 87 % of runs; p < 0.1 in 93 % of runs with the effect and in 10 % without it. The first version of the imputation was biased by about +0.9 (effects alternated with the low-rank update); fitting the two-way effects first removed the bias (RMSE 0.29 against 0.65 for two-way effects alone).
- Estonia, cars per 1,000: treatment effect pending (no Eurostat data after 2024). Anticipation year 2024: −4.2 % against the counterfactual (90 % interval −11.6 % to −0.4 %, p 0.08, 200 placebos). A plausible reading is deregistration of unused cars before the annual tax (the indicator is the stock on 31 December); purchases brought forward would have raised it.

**Tests**: 5 unit tests (`tests/test_events.py`); M7.5 acceptance (synthetic effect and announcement-year handling). All 26 M7 acceptance tests are implemented.

**Open**
- Events from the D3.6 Annex A interventions (dates and places) and Eurostat data for 2025–2026 for the Estonian case.

## M7.4 SD bridge (2026-11-24)

**Done**
- Decisions D16 (review items to the registry and an issue in the SD repository), D20 (between-region spread for SD parameter priors from the literature first) and D21 (class-R recalibration to regsim-imputed mode split with intervals), recorded in `docs/decisions.md`.
- `causal.sd_bridge`: `parse_vensim_edges` (variables of left-hand sides found in right-hand sides; subscripts, data equations and range definitions left out; stocks and single-number constants flagged), `perturbation_elasticities` (forward or central), `lever_effects`, `load_mapping` / `map_to_registry` (relation same, proxy, inverse proxy), `compare` (expected responses in `configs/sd_mapping.yaml` and registry priors; `model_review` with the diagnosis taken from the protocol variants), `lever_ranking`, `load_parameter_classes`, `domain_check`, `next_demonstrator`, `export_inputs`, `recalibrate_residuals` (Nelder–Mead, over draws of imputed targets), `transfer_parameters` (by class; class R recalibrated, never copied), `cross_calibrate`, `report_rules`.
- `causal.sd_knowing`: adapter for the KNOWING model (separate checkout, `causal.sd_model_path` or `REGSIM_SD_MODEL`): input groups scaled consistently (constants and series alike), levers held at their 2019 value for removal, runs with and without the Tallinn calibration, `perturbation_protocol`, `lever_table`.
- CLI `regsim causal sd --config configs/causal.yaml` (5 s): protocol, comparison, levers, ranking, domain check, next demonstrator; results in `runs/causal-sd-<hash>` and the evidence store.
- Data: regional heating and cooling degree days (`nrg_chddr2_a`) in `configs/indicators_causal_v1.yaml`; the registry entries use the national table only.
- Review issue posted in the SD repository: martnzach-wq/KNOWING-integrated-SD-model#1 (text in `docs/causal/sd_review_issue.md`).

**Results**
- Vensim parser: 109 edges (mobility) and 69 (energy supply), identical to `sd_dependency_edges.csv` including the stock and constant flags.
- Perturbation elasticities reproduce `sd_city_elasticities.csv`: 223 of 224 BAU values within 0.005 (one at 0.012), with the mode-specific trip lengths of the M7 runs and car share including car passengers; Granollers fuel and EV cost differ by design (constants perturbed as well: fuel cost → car-km −0.28). Lever effects reproduce with the lever held at its 2019 value (for factor levers this is 1, not 0).
- `compare` flags Tallinn fuel cost → active km (−0.34, resolved with the calibration off: +0.09), CDD and parking cost (no response, resolved with the calibration off), population (internal) → emissions (−0.47, sign resolved by consistent population scaling; the missing response remains: energy population not linked) and value of time (no response, unexplained: hard-coded constant).
- Lever ranking on car-km 2050: road pricing > cycling > parking in Tallinn, Granollers and Naples; Spearman Tallinn–Granollers 0.94, sizes differ up to a factor of 4.3.
- Domain check (EE00, ES51, ITF3; lens of six indicators, means 2008–2024 on the transformed scale): 231 regions, median distance 0.868 (reference 0.868), 81 % outside the range on at least one dimension (82 %), distances correlate 0.997 with `demonstrator_coverage.csv`. Next demonstrator: DED2, NL13, DE94, FRF1 (share within 0.75 from 0.26 to about 0.57–0.60).
- NAPLES runs now that its workbook is in the repository.

**Tests**: 9 unit tests (`tests/test_sd_bridge.py`); M7.4 acceptance (8 tests) pass with the SD checkout and are skipped without it.

**Open**
- `cross_calibrate` and `recalibrate_residuals` with the KNOWING model need observed mode shares and trip lengths of the cities as calibration targets.
- Next: M7.5 (intervention effects).

## M7.2 Literature module (2026-11-23)

**Done**
- Decision D15 (configurable LLM callable) and D14a (macro-regions as effect-model moderators), recorded in `docs/decisions.md`.
- `causal.literature`: `propose_edges`, `search` (OpenAlex and Crossref through `CachedClient`; responses cached by request, offline mode for recorded responses), `extract`, `verify` (second pass, `agree_sign`, `agree_design`), `measure_check` (strict measure check, mandatory third pass), `weighted_claims`, `grade` (overall and per horizon), `combine_rounds`, `targeted_queries` (mechanism and horizon terms for contested and data-flagged edges), `run_edge`. Every LLM call goes through an injected callable (prompt → text); prompts, answers and model ids are logged with the run.
- Claim weights: design × spatial level × Europe relevance × claim strength × quantity match, with the label tables of the extraction prompts (`claim_weight`); the spatial-level factor (city 0.8, individual or not stated 0.7) was missing from the stub.
- CLI `regsim causal literature E018 ... --config CFG` with `causal.llm` / `causal.llm_verifier` (`"module:callable"`) and the OpenAlex key from `OPENALEX_API_KEY`; grades, claims and prompts go to `runs/causal-literature-<hash>` and the evidence store.
- Fixture `tests/fixtures/literature/` for C30 (household income → cars), built by `make_c30.py` from the claim table of the M7 analysis (study ids, titles, years and the recorded extraction and verification results); not a live recording.

**Results**
- Claim weights reproduce the recorded weights of all 121 round-1 and 158 round-2 claims.
- `grade` reproduces the grades of all 45 round-1 edges (C30: A, 6 panel studies, consistency 1.0) and the overall, short-run and long-run grades of all 38 round-2 targets (round-1 claims with mixed results are not carried into the targeted round). GDP → life expectancy: short run − (C), long run + (B).
- The strict measure check rule (cause and effect measured as themselves or an acceptable proxy, unit representative of regions) reproduces the 84 of 158 kept claims and removes the three hand-check failures; GDP → unemployment (Okun) stays grade A.
- The offline pipeline (stub LLM with the recorded answers) gives C30 grade A.

**Tests**: 9 unit tests (`tests/test_literature.py`); M7.2 acceptance (3 tests).

**Open**
- Live use needs an LLM callable and access to api.openalex.org and api.crossref.org; both hosts are allowed in the cloud environment, a live recording of fixtures has not been made.
- Next: M7.4 (SD bridge); decisions D16, D20, D21.

## M7.6 Effect similarity (2026-11-23)

**Done**
- Decision D19 (practical equivalence at 0.1 SD, per link in `causal.equivalence_delta_by_link`), recorded in `docs/decisions.md`.
- `similarity.effect`: `fit_varying` (cumulative effect linear in standardised moderators: mean log GDP, mean level of the effect indicator, mean log density, region-group dummies; ridge penalty relative to the diagonal of X'X on the moderator terms, `alpha=inf` = one pooled effect; cluster-robust covariance), `validate_gate` (leave-one-country-out prediction of each country's own estimate, precision-weighted, relative to a pooled effect; heterogeneous if the best penalty gives ≤ 0.9), `region_effects`, `s_effect` (P(|β_A − β_B| < δ) from the joint covariance, effect distance), `effect_distances`, `effect_observations`.
- Region effects are written to the observation store as derived indicators `effect:<cause>-><effect>` with scenario `effect`; models read the scenario `observed` only, so the effects never become model inputs.
- CLI `regsim causal effects --config configs/causal.yaml` (gate, fits, region effects; about 8 minutes for 7 links) and `regsim causal compare A B --config ...`; config `causal.effect_links`, `causal.effect_groups`, `causal.equivalence_delta`.
- Views (`viz.effect_views`): pairwise comparison, effect space with 90 % ellipses, effect map with a value-suppressing palette (colour fades to grey with the interval width, in four steps), neighbour overlap, effect profile strip, transfer matrix; plotly versions of the comparison and the map.
- Explorer: tab "Effects" (`regsim app CONFIG --effects configs/causal.yaml`) with the comparison of the reference and the comparison region and the effect map of a chosen link.

**Results** (296 NUTS-2 regions, 2008–2024; moderators with the macro-regions of the M7 analysis)

| Link | Countries held out | Penalty | Relative hold-out error (unshrunk) | Gate | M7 analysis |
|---|---|---|---|---|---|
| GDP → life expectancy | 23 | 0.1 | 0.35 (0.38) | heterogeneous | 0.37 |
| GDP → unemployment | 23 | 0.3 | 0.53 (0.72) | heterogeneous | 0.73 |
| GDP → cars per 1,000 | 21 | 3 | 0.74 (2.25) | heterogeneous | 0.77 |
| household income → cars | 20 | ∞ | 1.00 (2.79) | pooled | (4.28) |
| old-age dependency → GDP | 23 | 3 | 0.99 (1.31) | pooled | (1.41) |
| unemployment → poverty risk | 18 | 1 | 0.98 (1.15) | pooled | (1.11) |
| employment rate → fertility | 24 | 1 | 0.95 (1.13) | pooled | (1.03) |

- Moderators of GDP → life expectancy (change of the standardised effect): North +0.16, South −0.12, East −0.02 against West, density +0.06 (M7 analysis: North +0.10, South −0.07, less negative in dense regions).
- Estonia (EE00) vs Catalonia (ES51), P(practical equivalence): GDP → cars 1.00, GDP → unemployment 0.19, GDP → life expectancy 0.05 (M7 analysis 1.00, 0.33, 0.14); the pooled links are reported as "no transferable difference". Effect distance 0.19.
- With typology classes as region-group moderators GDP → unemployment fails the gate (relative error 1.00); the other six decisions are the same. Decision D14a: the effect model uses the macro-regions (`causal.effect_groups: macro`).

**Tests**: 5 unit tests (`tests/test_effect_similarity.py`), app test of the Effects tab; M7.6 acceptance (2 tests).

**Open**
- Next: M7.2 (literature module), decision D15 taken.

## M7.3 Panel module, direction test, causal queries, causal-lens similarity (2026-11-23)

**Done**
- Decisions D12 (effect sources in `what_if`), D13 (fixed effects with bias correction, Arellano–Bond GMM as option) and D14 (typology classes for partial pooling, macro-regions as fallback), recorded in `docs/decisions.md`.
- `fit_edge`: half-panel jackknife (default) and one-step Arellano–Bond difference GMM with collapsed instruments; the plain within estimator stays available (`bias_correction="none"`, the M7 reference values). On simulated panels (T = 15, ρ = 0.5) the plain estimator gives ρ = 0.39 and a long-run effect of 0.70 (true 0.8); jackknife 0.51 / 0.83, GMM 0.50 / 0.80 with twice the spread of the jackknife.
- `causal.groups`: macro-regions from `causal.macro_regions`, typology classes, `pooled_effects` (DerSimonian–Laird τ², empirical-Bayes shrinkage of group effects towards the pooled effect).
- `causal.effects.estimate_effects`: pooled and group effects of all edges; `causal.run` and CLI `regsim causal fit` (effects and direction tests into `runs/causal-fit-<hash>` and the evidence store; `--write-registry` writes E2 items and `temporal_precedence`), `regsim causal what-if`, `regsim causal direction`.
- `direction_test`: lagged within-region Wald tests both ways (Bonferroni over the tests of a batch), event-study profile with pre-trend test; series aligned by `reference_offsets` (EU-SILC income measures to t−1; 1 January stocks keep their year, as in the M7 analysis). `screen` labels pairs with `classify_mechanism` and marks those for the literature module.
- Queries (`causal.queries`): `what_if` (first-order propagation of log changes in topological order, cycles left out and reported, shocked indicators held fixed; local elasticities from coefficients on transformed scales; source order D12; interval, horizon, weakest tier and edges per row; absolute change in original units), `explain_difference` (between-entity regression on the causal parents, residual and `missing_causes` flag), `vision_check` (required lever change per target, `feasible`, `conflict`, `mismatch`).
- `similarity.causal_lens.s_causal` (nearest entities on the causes of a target, Jaccard overlap with general neighbours); `predict.tasks.analogues(target=, registry=)` uses it.
- `evaluation.benchmark.graph_forecast_benchmark`: one-year forecasts of standardised annual changes, hold-out of the last two years.
- Data: `configs/indicators_causal_v1.yaml` (NUTS-2 copies back to 2008, S80/S20); `annual_panel` merges `same_as` copies and applies `causal.transforms` (cars per 1,000 on the log scale, see `docs/data_sources.md`).

**Results** (NUTS-2, 2008–2024)
- GDP → cars per 1,000: β_std 0.099, t 3.67 with the M7 estimator (reference 0.099, t 3.69); 0.066 (t 2.45, ρ 0.20 instead of 0.10) with the jackknife. The 32 NUTS-2 edges of `causal_panel_results.csv` reproduce with r = 0.998.
- Heterogeneity by macro-region (GDP → cars): East 0.22, South 0.08, North −0.04, West −0.04; I² 0.80, p 0.002 (reference I² 80 %, p 0.002).
- Direction tests: GDP ↔ cars undecided (p 0.72 forward, 0.0034 reverse; reference 0.73 / 0.0035); life expectancy → old-age dependency forward (p 0.00026, flat pre-trend). Over the 54 observed registry pairs (Bonferroni over 108 tests): 5 forward, 1 reverse, 1 both, 47 undecided.
- `what_if` +10 % GDP in EE00 (short run, typology class of Estonia): cars per 1,000 +0.3 % (90 % interval −0.06 to +0.7 %), unemployment −0.38 points, employment rate +0.5 points; tier E2 on all magnitudes.
- `explain_difference` EE00 − ES51 for cars per 1,000: the residual (1.13 on the model scale) exceeds the observed difference (0.10), so the parents in the graph do not explain it (`missing_causes`).
- `s_causal` EE00, cars per 1,000, k = 10: lens GDP, household income, density, cars; Jaccard with the general neighbours 0.11.
- Forecasts (median RMSE relative to the training SD of annual change, 42 indicator-years): no change 1.09, region trend 1.06, own AR(1) 0.91, ridge VAR 0.99, graph-constrained VAR 0.95 (reference 1.07, 1.04, 0.92, 1.02, 0.94). The graph constraint beats the unconstrained VAR in 67 % of indicator-years.
- Screen of 21 NUTS-2 indicators (420 pairs, Bonferroni): 34 pairs flagged, 10 of them accounting identities, shared denominators or shared survey bases.
- C1 on the rebuilt store: between-region agreement 40/53, within-country 39/44.

**Tests**: 11 unit tests (`tests/test_causal_queries.py`) on synthetic panels; M7.1 and M7.3 acceptance (12 tests) pass; the M7.2, M7.4–M7.6 tests stay skipped.

**Open**
- Registry updated with `regsim causal fit --write-registry` (2026-11-23): E2 items (jackknife estimates; between-region and within-country correlations kept) and `temporal_precedence` for the 54 observed edges. Status changes (16): supported E011, E014, E045, E046, E051, E060 (GDP → unemployment, Okun), E061, E066, E075; contested E002 (tertiary attainment → GDP), E039, E047, E056, E057, E072, E076. Temporal precedence at Bonferroni over 108 tests: 5 forward, 1 reverse, 1 both, 47 undecided; six edges that the M7 analysis listed as forward or reverse are now undecided.
- `classify_mechanism` does not describe age-structure composition (e.g. old-age dependency and median age) or the national-accounts link between GDP and household income; such pairs pass the screen as causal.
- Next: M7.6 (effect similarity), decision D19 first.

## M7.1 Causal registry and consistency checks (2026-11-23)

**Done**
- Decisions D10 (edges in YAML, claims and estimates in a Parquet evidence store) and D11 (`numerator`, `denominator`, `survey` in `IndicatorEntry`), recorded in `docs/decisions.md`.
- `EdgeRegistry`: `children`, `descendants` (breadth-first, cycles allowed), `edge`, `update_evidence` (supported / contested by sign and |t|, edges never removed, retired edges and edges without a prior sign keep their status). `to_yaml` keeps the key order, explicit nulls and line endings of an existing file: a rewrite without changes is byte-identical, an update changes only the edited fields.
- `classify_mechanism`: accounting identity (numerator of one indicator is a part of the numerator or denominator of the other: employment/unemployment via the labour force, net migration/population growth), shared denominator (GDP and R&D in % of GDP; population density and per-capita indicators: shared population stock; any shared denominator other than a population), shared survey base (LFS, EU-SILC, ICT household survey). Metadata added to 21 indicators of `configs/indicators_eurostat_v1.yaml`.
- `regsim.causal.data`: annual panels from the observation store (country from the hierarchy graph, registry transforms) and region means.
- C1 (`regsim.causal.check`): between-region Spearman correlation of region means (NUTS-2 where both indicators cover ≥ 30 regions, else national) and within-country correlation (country means removed), labelled against `sign_prior`; agreement rates over distinct signed cause–effect pairs, overall and per grade. Results go to the evidence store (`<store>/causal_evidence.parquet`, tier E1); `--write-registry` writes E1 items into edges without data evidence.
- `configs/causal.yaml`: annual NUTS-0/2 store 2008–2024 (`data/store_causal`, Eurostat; 39 countries, 299 NUTS-2 regions, 93,484 observations). CLI `regsim causal check --config configs/causal.yaml`.

**Results**
- Between-region sign agreement 40 of 53 signed pairs (0.75; M7 analysis 27/38, 0.71), within-country 40 of 44 (0.91; M7 analysis 23/25). By grade (between / within country): A 4/5, 3/3; B 4/8, 7/8; C 21/24, 19/21; D 3/6, 4/5.
- Agreement with the correlations of the M7 analysis stored in the registry (44 edges): r = 0.996 between regions, 0.999 within countries; the two sign differences are correlations of ±0.03.
- 14 edges contradict their prior between regions. Four of them (hospital beds → life expectancy, density → cars, old-age dependency → GDP, median age → GDP) agree within countries: the between-region sign comes from differences between countries. GDP → road deaths (B, +) is negative at both levels (−0.65, −0.30).
- 27 edges are not observed: 10 graph nodes are SD-model variables, EDGAR sector emissions or indicators without a registry entry (household electricity, S80/S20, fuel cost, car-km, active-mode km).
- E1 items written for 10 edges without earlier data evidence.

**Tests**: 8 unit tests (`tests/test_causal_graph.py`); M7.1 acceptance (5 tests in `tests/acceptance/test_m7.py`) passes; other M7 tests stay skipped.

**Open**
- National-only indicators of the M7 analysis (household electricity, S80/S20 at NUTS-2 from `ilc_di11_r`, EDGAR sectors) are not yet in the registry.
- Next: M7.3 (panel module with the store, `direction_test`, `what_if`, `s_causal`); decisions D12, D13, D14 are needed first.

## LAU demographic indicators and export for heatimpact (2026-10-29)
- `regsim.data.statat.StatAtConnector`: OGD population table by municipality and single year of age; 2,116 municipality codes in the 2026 table (23 Vienna districts, code 0 with 0 persons, Fürstenfeld 62280 created 2025 from LAU 2024 62252 Söchau and 62267 Fürstenfeld). After mapping, the 2,093 Austrian LAU 2024 codes are covered.
- Build step for source `statat`, `configs/lau_at.yaml`, `regsim export-lau`.
- Tests: `tests/test_statat.py` (connector on recorded-format tables, build step, export to GeoPackage).

## M6 Interfaces, views, extensions (2026-10-14)

**Done**
- Decision D7 (front end): Dash, recorded in `docs/decisions.md`.
- Prediction tasks P1–P7 (`regsim.predict.tasks`) on the original scale with intervals and observed flags: gap filling, new entity (fold-in), new indicator, vision entity (base entity as prior, targets as observations), analogues, next indicator to measure (Sherman–Morrison update of the Laplace posterior), level transfer.
- Views V1–V8 and further views (`regsim.viz.views`): matplotlib and plotly versions; 2D views state explained variance or trustworthiness; maps from the GISCO geometry without geopandas. `regsim report CONFIG` renders all figures of `docs/visualisation.md` (PNG and HTML, statistics in `stats.json`; settings in the config's `report` section).
- CLI: `fit` (calibrated model cached per configuration and registry content), `predict`, `report`, `serve`, `app`; `similar` and `predict` accept geo ids with or without scheme prefix.
- REST service (`regsim.api.service`, FastAPI): health, entities, indicators, predict, similar, entity, vision, next-indicator, level-transfer, typology; pydantic response models are the contract (OpenAPI at `/docs`).
- Dash explorer (`regsim.app.dash_app`): lens, level, number of types, colouring and embedding (PCA or UMAP) as controls; lasso selection on the map highlights the regions in the embedding and vice versa; clicking a region makes it the reference and shows its nearest regions and the difference profile (V6). Checkbox "show indicator directions" turns the PCA embedding into the V1 biplot (arrows for the 12 strongest or the lens indicators); tab "Trajectories" shows V7 for the capital regions or the current selection, optionally relative to the mean drift, from a model fitted on both windows (`runtime.fit_panel`, cached). Embedding with directions and trajectories share one PCA basis (both windows, current lens and level), so axes and arrows are identical in both plots; region names, a searchable reference, the list of selected regions and short descriptions make the page self-explanatory.
- Geo-time entities (two windows with random-walk penalty, since M3) are used by V7.
- LAU extension: `configs/lau.yaml` (store `data/store_lau`): 98,461 LAUs with the D3.6 indicators enter the model below NUTS-3 (100,130 rows, one fit about 11 minutes); `level_transfer_benchmark` for scheme M4 at a fine level.

**Fixes found in this milestone**
- D3.6 values were stored under the scenario `baseline` and dropped by the model matrix (scenario `observed`); they now enter as observed values. This affected runs with `configs/example.yaml` (CLI demonstrations, the NUTS-2 typology statement of M5); the acceptance benchmarks of M2–M5 use the Eurostat registry only.
- D3.6 registry: `electricity_total_demand_mean` is an area-weighted mean (NUTS-3 value = area-weighted mean of its LAUs, median ratio 1.001), not a sum; the four "days per year" indicators are right-skewed and now use `log1p`.
- `prepare` keeps LAU rows with a population from the LAU list (they have no Eurostat population series).

**Results**
- LAU M4 benchmark (each indicator hidden at all LAUs, predicted from NUTS-3 and the other LAU indicators; RMSE on the robust-scaled model scale; skill = 1 − MSE / MSE of carrying the NUTS-3 value down):

| Indicator | LAU cells | Carry-down | LevelMean | SoftImpute | HierTypedMF | Skill HierTypedMF |
|---|---|---|---|---|---|---|
| Elevation (mean) | 96,678 | 0.668 | 1.115 | 0.665 | **0.563** | 0.29 |
| Road density | 97,099 | 0.906 | 1.104 | 0.855 | **0.824** | 0.17 |
| Electricity demand | 96,969 | 0.851 | 1.142 | 0.874 | **0.814** | 0.09 |
| Population density | 97,097 | 1.639 | 1.958 | 1.722 | **1.570** | 0.08 |
| Heat days | 96,869 | 0.489 | 0.973 | 0.525 | **0.473** | 0.06 |
| Tropical nights | 96,869 | **0.557** | 1.047 | 0.615 | 0.562 | −0.02 |

  Before the registry fix, electricity demand had skill −4.4 (the sum constraint forced LAU values to 1/46 of the NUTS-3 value) and tropical nights −0.27.
- `regsim report configs/example.yaml` (65 indicators): V1 PC 1 + PC 2 48 % of variance, led by temperature and heating degree days; V3 trustworthiness PCA 0.68, UMAP 0.93; V4 typology on the 9 indicators observed at NUTS-3 more stable (ARI 0.67) than on all indicators (0.46); nearest NUTS-3 regions to Vienna: Karlsruhe, Bonn, Wrocław, Brussels, Nuremberg.
- NUTS-2 typology with the D3.6 data: bootstrap stability ≤ 0.55 for k = 3–12 (M5 statement without D3.6: ≤ 0.62); the regions remain a continuum.

**Tests**: 119 unit tests pass (including REST contract tests and the Dash callbacks through their HTTP interfaces); M0–M6 acceptance pass.

**Open**
- The D3.6 Köppen adjacency for zone pairs not covered by the demonstrators and the categorical scoring rules (to ask the D3.6 authors).
- KNN baselines at LAU scale (quadratic cost) are not part of the LAU benchmark.

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
- Typology of the NUTS-2 regions (`configs/example.yaml`, GMM; computed before the D3.6 scenario fix of M6, i.e. without D3.6 indicators): bootstrap stability 0.62 for k = 3 and 0.36–0.44 for k = 4–12, while BIC decreases up to k ≈ 7–12. The regions form a continuum rather than well-separated types; typologies are reported with their stability for this reason.

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
