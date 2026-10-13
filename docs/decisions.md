# Decisions

Record answers to HANDOVER.md §4 here (date, decision, by whom).

| ID | Date | Decision | By |
|---|---|---|---|
| D1 | 2026-10-13 | Harmonise geo codes to **NUTS 2024** and **LAU 2024**. Data published under NUTS 2016/2021 are recoded via the GISCO correspondence tables. | M. Zach |
| D2 | 2026-10-13 | Repository `martnzach-wq/CollaborativeFilteringFramework`; package name **`regsim`**; licence **Apache-2.0**. | M. Zach |
| D3 | 2026-10-13 | Time enters v1 with **two period windows**, 2013/2017 and 2018/2022, as geo-time entities with a random-walk penalty (design §4.3). | M. Zach |
| D4 | open | Default lens; whether polarity is shown | |
| D5 | 2026-10-13 | v1 sources: **Eurostat** and **GISCO** (primary), **KNOWING D3.6 GeoPackage** (imported in full; only its non-Eurostat indicators enter the model), **JRC ARDECO** as fallback for population, GDP, employment and GVA, flagged as estimates. WDI and AlphaEarth are deferred to the extension beyond Europe. | M. Zach |
| D6 | 2026-10-13 | **PyTorch is a core dependency.** `HierTypedMF` uses autograd for the typed losses and the Laplace posterior. | M. Zach |
| D7 | open | Interactive front end | |
| D8 | open | TRF classes for regsim similarities | |
| D9 | open | Severity of D3.6 gates in regsim mode | |

## Consequences

- D1: the M1 acceptance criterion (1,929 units, 18,863 cells, ±2 % after code harmonisation) is evaluated after recoding the prototype matrix to NUTS 2024.
- D3: the M1 build and the example config cover 2013–2022; the prototype reference numbers refer to the 2018/2022 window only.
- D5: of the about 60 D3.6 indicators, 13 are copies of Eurostat tables (SOEC01/03/05/07/11, URB02–03, ENERG01–03, TRNPT06/08; D3.6 Table 20), and GEOG03–05 and URBN01 come from GISCO. In the GeoPackage v2.5 only 41 of 79 indicators are populated and none of the Eurostat copies is among them; `population_density` matches Eurostat `pop_density` (Spearman 0.976, median ratio 0.993 at NUTS-3) and is linked with `same_as`. The indicator registry links such duplicates with a source precedence (Eurostat > ARDECO > D3.6); the store keeps all copies with their `source`, the model matrix uses one copy per indicator, and the redundancy is reported. D3.6 copies remain available for the D3.6 compatibility mode (M4).
- D6: `torch` moves from the optional extras into the core dependencies. CI installs the CPU wheels.

## Changes to acceptance criteria

| Milestone | Date | Change | By |
|---|---|---|---|
| M1 | 2026-10-13 | Under NUTS 2024 (D1) the UK is excluded. The prototype matrix is compared after subtracting the UK cells found in current Eurostat tables (2018–2022, NUTS-0/2/3): observed cells within ±2 %, indicators without UK data within ±5 % each; the number of units is reported only, since the prototype counted some regions under old and new codes. | M. Zach |
| M2 | 2026-10-13 | Cold start (scheme M2) is compared with a tolerance of ±0.10 instead of ±0.05, since its RMSE varies between seed pairs with a standard deviation of 0.07–0.10 (20 seed pairs). Ranking and the ±0.05 tolerance for M1 and M4 are unchanged. | M. Zach |
