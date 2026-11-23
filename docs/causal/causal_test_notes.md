# Empirical test of the candidate causal graph on annual European panels

## Data and specification

- Eurostat annual panels 2008–2024 (2025 where published): 16 indicators at NUTS-2 (348 regions) and 6 national-only indicators (45 countries). Log or log1p transforms as in the regsim indicator registry.
- Edges tested: 46 distinct candidate edges with both ends observed (38 with a prior sign). Grades A–D from the literature step, demographic identities, and SD-model/deliverable edges.
- Four statistics per edge:
  1. between-region Spearman correlation of region means;
  2. within-country between-region correlation (NUTS-2 only, country means removed);
  3. within-region dynamic effect: Δy_t on Δx_t, Δx_{t−1} and Δy_{t−1}, with region and country×year fixed effects (NUTS-2) or region and year fixed effects (national); cumulative effect β₀+β₁, standardised, cluster-robust by region;
  4. lag-0 and lag-1 single-lag variants (kept in the results table for comparison).
- The cumulative lag-0+1 effect replaces the earlier choice of one lag from the prior: single-lag estimates of the same edge changed sign between lag 0 and lag 1 (heating degree days → household electricity +0.27 vs −0.31), which reflects mean reversion rather than causal timing.

## Results

| Test | Outcome |
|---|---|
| Between-region sign agrees with prior | 27 of 38 edges (71 %) |
| Within-country sign agrees with prior | 23 of 25 NUTS-2 edges (92 %) |
| Within-region effect significant (\|t\|>1.96) | 12 of 38 edges |
| … of which agrees / contradicts prior | 7 / 5 |
| Grade A/B edges supported / contradicted / undecided | 2 / 3 / 7 |

Cross-sectional structure largely agrees with the priors; year-to-year co-movement within regions decides about a third of the edges, and the decided set includes contradictions of grade A edges. Each contradiction has an identifiable cause:

- **GDP → life expectancy** (A, +; within-region −0.11, t −4.1). Short-run procyclical mortality, documented for US states by Ruhm (2000, doi:10.1162/003355300554872). The literature grade refers to the long-run, cross-country relationship. Priors need a horizon attribute (short-run / long-run), not only a lag.
- **GDP → renewables share** (A, +; −0.15, t −2.2). Ratio denominator: GDP growth raises final energy consumption, the denominator of the share.
- **R&D expenditure → GDP** (C, +; −0.14, t −3.0) and the reverse edge (−0.15, t −4.0). R&D intensity is expressed as % of GDP; a GDP shock lowers the ratio mechanically.
- **Employment rate → fertility** (B, −; +0.08, t 2.3). The literature effect refers to female employment and cross-country comparisons; within regions, fertility moves procyclically.
- **GDP → employment rate** (D, −). LLM-proposed sign; data contradict it (+0.11, t 3.0) and literature support was absent. Downgrade.

Renewables share → GHG per capita (A) agrees in sign at lag 0 (−0.18, t −2.4) but the cumulative estimate is not significant (t −1.1) with 45 national units; national-only edges have too little within-unit variation to decide.

## Data-driven screening

All 240 ordered pairs of the 16 NUTS-2 indicators were screened with the cumulative within-region model, Bonferroni threshold |t| > 3.71.

- 30 pairs pass; 5 are in the candidate graph, 5 more in reverse orientation.
- 23 of the 30 are accounting or measurement links: shared population stock (10), labour-force-survey base (4), per-capita denominators and age-structure composition (4), employment/unemployment (2), GDP/household income (2), R&D/GDP ratio (1).
- 7 substantive pairs remain. GDP ↔ unemployment (−0.13 and −0.12, both directions) is Okun's law (Ball, Leigh and Loungani 2013, NBER w18668), absent from the candidate graph; household income → unemployment (−0.18) and income ↔ employment are its income-side equivalents. These go back to the literature loop as new candidates.
- Recall of graded NUTS-2 edges (A–C) by the screen: 1 of 15. Data-driven discovery at this resolution does not substitute for the prior graph; it flags omissions and accounting artefacts.

Consequence for the registry: every edge needs a `mechanism_type` field (causal, accounting identity, shared denominator, shared survey base) so that discovery output and consistency checks can exclude non-causal co-movement.

## Hold-out forecasts

Forecast of standardised annual change for the last two available years per NUTS-2 indicator (32 indicator-years), trained on earlier years only; metric RMSE divided by the training SD of annual change (median across indicator-years).

| Model | Median relative RMSE |
|---|---|
| no change | 1.07 |
| region trend | 1.04 |
| own AR(1) on changes | 0.92 |
| all-variable VAR (ridge) | 1.02 |
| graph-constrained VAR (own lag + candidate parents) | 0.94 |

The graph constraint beats the unconstrained VAR in 75 % of indicator-years and the own-lag model in 38 %. Candidate parents act as useful regularisation against spurious cross-effects but add little predictive information at a one-year horizon. Forecast skill is therefore not the main argument for the causal layer; its value lies in intervention and explanation questions (Step 6).

## Heterogeneity

Grade A/B NUTS-2 edges estimated separately by macro-region (North, West, South, East) and GDP tercile; Cochran's Q across macro-regions.

- GDP → cars per 1,000: I² 80 %, p 0.002. Effect concentrated in eastern (+0.22) and low-GDP regions (+0.19), near zero in northern, western and high-GDP regions, consistent with motorisation saturation (Dargay and Gately 1999, doi:10.1016/S0965-8564(98)00026-3).
- Old-age dependency → GDP: I² 70 %, p 0.018; positive only in the South.
- Other A/B edges: I² 0–55 %, not significant.

Edge parameters should be partially pooled by region type (as specified in the design, option C2) and conditioned on level variables such as GDP, which links the causal layer back to the dynamic typology: the same similarity space that groups regions defines where an elasticity transfers.

## Limitations

Annual changes over 2008–2024 identify short-run co-movement only; long-run effects need levels with interactive fixed effects or long-difference estimates. Country×year fixed effects remove national shocks and also national policy effects. NUTS-2 boundaries changed in 2016, 2021 and 2024; series were used as published under NUTS 2024 codes.

## Files

- causal_panel_results.csv — per-edge statistics and E2 status
- causal_panel_screening.csv — 240-pair screen with classification
- causal_forecast_skill.csv — hold-out forecast errors
- causal_edge_heterogeneity.csv — subgroup estimates
