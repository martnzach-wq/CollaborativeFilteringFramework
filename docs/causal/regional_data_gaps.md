# Regional data for indicators that were only national

## Sources

| Indicator (national in regsim so far) | Regional source | Level, years | Access | Use in M7 |
|---|---|---|---|---|
| GHG per capita | JRC EDGAR_2025_GHG NUTS2: total and by sector (transport, buildings, energy, industry, agriculture, waste), downscaled from national inventories with spatial proxies | NUTS 2024 level 2, 1990–2024 | public download (edgar.jrc.ec.europa.eu, file server jeodpp.jrc.ec.europa.eu); IEA-based CO2 component under CC BY-NC-ND 4.0 | added; per capita with Eurostat population (2014–2024 in the regsim store) |
| Heating and cooling degree days | Eurostat nrg_chddr2_a (annual), nrg_chddr2_m (monthly) | NUTS-2 (292 regions), 2008–2025; monthly series available | Eurostat API | added; the monthly series supports sub-annual timing tests |
| Gini | Eurostat ilc_di11_r, income quintile share ratio S80/S20 | NUTS-2 (243 regions), 2008–2025, gaps by country | Eurostat API | added as inequality proxy |
| Renewables share | none at NUTS-2 in Eurostat; national energy balances only | — | — | remains national; candidates: national statistical offices, regional energy balances of individual countries, JRC regional energy data |
| Household electricity per capita | none at NUTS-2 in Eurostat | — | — | remains national; candidates: DSO data, national statistics |

## Re-tests with regional data

Within-region model as in the M7 panel module: cumulative effect of current and lagged change, lagged change of the effect, cluster-robust by region. Each edge was estimated with two fixed-effect sets:
- region + country×year, which removes national shocks;
- region + year, which keeps national variation.

| Edge (registry id) | β std, country×year FE | t | β std, year FE | t | Regions | Status |
|---|---|---|---|---|---|---|
| cars per 1,000 → transport GHG per capita (C37) | 0.21 | 4.2 | 0.17 | 6.7 | 308 | supported |
| cars per 1,000 → total GHG per capita (C37) | 0.04 | 1.2 | 0.06 | 1.9 | 312 | undecided |
| heating degree days → buildings GHG per capita (C40 proxy) | 0.44 | 10.4 | 0.46 | 11.7 | 286 | supported |
| heating degree days → total GHG per capita | 0.00 | 0.0 | 0.10 | 2.2 | 286 | supported only with year FE |
| cooling degree days → total GHG per capita (C41 proxy) | 0.03 | 0.7 | −0.06 | −1.1 | 286 | undecided |
| GDP → GHG per capita (S161) | 0.09 | 2.0 | 0.17 | 4.4 | 280 | supported |
| density → GHG per capita (S006) | −0.03 | −0.7 | −0.05 | −1.4 | 312 | undecided |
| population growth → GHG per capita (S014) | −0.03 | −0.9 | 0.02 | 0.4 | 324 | undecided |
| unemployment → S80/S20 (C03) | −0.02 | −0.4 | 0.01 | 0.3 | 220 | undecided |
| S80/S20 → poverty risk (C11) | 0.53 | 6.2 | 0.49 | 6.1 | 233 | supported, shared survey base |

Regional data move four energy and inequality edges from "not testable" (national only) to "supported". The national panel had decided none of them.

## Validity of the regional series

- **EDGAR NUTS-2 emissions are modelled.** National inventories are distributed to grid cells with spatial proxies (population, road networks, point sources) and aggregated to NUTS-2. Within-country variation over time is therefore partly produced by the proxies. Before the HDD → buildings and cars → transport results are used as E2 evidence, the downscaling method has to be checked: if regional weather or vehicle data enter the proxies, the test is circular. Until then the registry records these results as E2 with the flag `modelled_effect_data`.
- **S80/S20 and poverty risk come from the same survey.** Both are taken from EU-SILC, and the at-risk-of-poverty component of the poverty indicator is computed from the same income distribution. The edge gets `mechanism_type: shared_survey_base`.
- **Short emission panel.** The EDGAR panel spans 2014–2024 in the present store, because the NUTS-2 population series begins in 2014. Extending population back to 2000 would add 14 years.

## Remaining data needs

- Regional renewables share and household electricity: no harmonised European source at NUTS-2. Options are national sources for the demonstrator countries (Estonia, Spain/Catalonia, Austria) or electricity distribution data.
- Regional car-km and active-mode km, needed to test the SD mobility edges: national travel surveys, or traffic counts aggregated to NUTS-3.
- Monthly series (degree days, registrations, unemployment) for timing tests below one year.

## Files

- regional_retests.csv: estimates for both fixed-effect sets
- edgar_nuts2_ghg_panel.csv: EDGAR totals, transport and buildings emissions, population and per-capita values
- regional_gapfill_eurostat.csv: degree days and S80/S20 at NUTS-2
