# Decisions

Record answers to HANDOVER.md §4 here (date, decision, by whom).

| ID | Date | Decision | By |
|---|---|---|---|
| D1 | 2026-10-13 | Harmonise geo codes to **NUTS 2024** and **LAU 2024**. Data published under NUTS 2016/2021 are recoded via the GISCO correspondence tables. | M. Zach |
| D2 | 2026-10-13 | Repository `martnzach-wq/CollaborativeFilteringFramework`; package name **`regsim`**; licence **Apache-2.0**. | M. Zach |
| D3 | 2026-10-13 | Time enters v1 with **two period windows**, 2013/2017 and 2018/2022, as geo-time entities with a random-walk penalty (design §4.3). | M. Zach |
| D4 | 2026-10-13 | Default lens: **all model indicators** (one per `same_as` group, all sources). **Polarity off** by default; available as a display option in views, never used in calculations. | M. Zach |
| D5 | 2026-10-13 | v1 sources: **Eurostat** and **GISCO** (primary), **KNOWING D3.6 GeoPackage** (imported in full; only its non-Eurostat indicators enter the model), **JRC ARDECO** as fallback for population, GDP, employment and GVA, flagged as estimates. WDI and AlphaEarth are deferred to the extension beyond Europe. | M. Zach |
| D6 | 2026-10-13 | **PyTorch is a core dependency.** `HierTypedMF` uses autograd for the typed losses and the Laplace posterior. | M. Zach |
| D7 | 2026-10-14 | Interactive front end: **Dash** (Plotly), with linked map–embedding views through callbacks. It is an optional extra (`app`) on top of the views and the REST service; the KNOWING DSS can use the REST service independently. | M. Zach |
| D8 | 2026-10-13 | TRF classes (TRF1–TRF5) are reported **only in D3.6 compatibility mode**; regsim similarities are reported as percentile scores without classes. | M. Zach |
| D9 | 2026-10-13 | D3.6 gates are **soft** in regsim mode: a triggered gate adds a warning flag with its reason to the pair; scores and rankings are unchanged. Hard exclusion (D3.6 `eligible = 0`) applies only in D3.6 compatibility mode. | M. Zach |

## Consequences

- D1: the M1 acceptance criterion (1,929 units, 18,863 cells, ±2 % after code harmonisation) is evaluated after recoding the prototype matrix to NUTS 2024.
- D3: the M1 build and the example config cover 2013–2022; the prototype reference numbers refer to the 2018/2022 window only.
- D5: of the about 60 D3.6 indicators, 13 are copies of Eurostat tables (SOEC01/03/05/07/11, URB02–03, ENERG01–03, TRNPT06/08; D3.6 Table 20), and GEOG03–05 and URBN01 come from GISCO. In the GeoPackage v2.5 only 41 of 79 indicators are populated and none of the Eurostat copies is among them; `population_density` matches Eurostat `pop_density` (Spearman 0.976, median ratio 0.993 at NUTS-3) and is linked with `same_as`. The indicator registry links such duplicates with a source precedence (Eurostat > ARDECO > D3.6); the store keeps all copies with their `source`, the model matrix uses one copy per indicator, and the redundancy is reported. D3.6 copies remain available for the D3.6 compatibility mode (M4).
- D7: `regsim.app` (Dash) uses the plotly versions of the views V1–V8; `dash` is in the optional extra `app`.
- D9: the rule engine (M5) evaluates every gate for each pair and returns flags; the D3.6 mode additionally sets `eligible`.
- D6: `torch` moves from the optional extras into the core dependencies. CI installs the CPU wheels.

## Changes to acceptance criteria

| Milestone | Date | Change | By |
|---|---|---|---|
| M1 | 2026-10-13 | Under NUTS 2024 (D1) the UK is excluded. The prototype matrix is compared after subtracting the UK cells found in current Eurostat tables (2018–2022, NUTS-0/2/3): observed cells within ±2 %, indicators without UK data within ±5 % each; the number of units is reported only, since the prototype counted some regions under old and new codes. | M. Zach |
| M2 | 2026-10-13 | Cold start (scheme M2) is compared with a tolerance of ±0.10 instead of ±0.05, since its RMSE varies between seed pairs with a standard deviation of 0.07–0.10 (20 seed pairs). Ranking and the ±0.05 tolerance for M1 and M4 are unchanged. | M. Zach |
