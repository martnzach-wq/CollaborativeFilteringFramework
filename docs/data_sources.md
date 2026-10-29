# Data sources

## v1 (Europe)

| Source | Content | Levels | Access | Notes |
|---|---|---|---|---|
| Eurostat dissemination API | Regional statistics (demography, economy, labour, education, health, transport, R&D, energy) | NUTS-0 to NUTS-3, some cities (Urban Audit) | `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}` (JSON-stat), SDMX 2.1/3.0 | 413 = asynchronous, retry; validate dimension names per dataset |
| Eurostat GISCO | NUTS boundaries (2021, 2024), LAU boundaries, correspondence tables, DEGURBA | NUTS-0 to LAU | `https://gisco-services.ec.europa.eu/distribution/v2/` | 20M scale sufficient for maps; 1M or 3M for area/population weights |
| KNOWING D3.6 database | 79 indicators registered, 41 populated in v2.5 (climate normals, terrain, typologies, land, transport, energy demand) (climate, land, socio-economy, governance, transport, energy), Gower scores per demonstrator | NUTS-2, NUTS-3, LAU | Zenodo doi:10.5281/zenodo.20376109 (GeoPackage v2.5, 295 MB zipped, 2.3 GB unzipped; NUTS 2024 codes plus UK under NUTS 2021); tables `knowing.indicator`, `knowing.indicator_values`, `knowing.indicator_normalised_values`, `knowing.regions`, `knowing.precomputed_scores` | See caveats below |
| JRC ARDECO | Long NUTS-3 time series (population, GDP, employment, GVA), harmonised to NUTS 2024, nowcast for recent years | NUTS-0 to NUTS-3 | GraphQL `https://territorial.ec.europa.eu/ardeco-api-v2/graphql` (metadata), CSV `rest/export/{variable}?version=2024&level_id={level}&{dimensions}` | Dimension values differ from Eurostat (`sex=TOTAL`); one level per request |
| World Bank WDI | National indicators | NUTS-0 / countries | WDI API | Bridge to global extension |

## Extension beyond Europe

| Source | Use |
|---|---|
| GADM 4.1 | Administrative hierarchy outside Europe (`GeoScheme` plug-in) |
| Global DEGURBA (Dijkstra et al., 2021), GHSL | Harmonised settlement typology and population |
| AlphaEarth Foundations satellite embeddings (Brown et al., 2025) | 64-dimensional annual embeddings at 10 m, 2017–2024, Google Earth Engine; aggregate to units as side information for cold start |
| SatCLIP, MOSAIKS | Alternative location embeddings |
| Gridded GDP/HDI (Kummu et al., 2018) | Covariates for disaggregation |

## Caveats on the D3.6 database (from an earlier check of the export)

- ERA5-Land heat indicators do not resolve the urban heat island: Vienna 11.8 heat days and 5.0 tropical nights in the database versus about 21 and 21 at GeoSphere Austria stations for 1991–2020.
- Min–max normalisation over Europe compresses differences; Vienna–Tallinn similarity on heat indicators alone is 95.9.
- Majority Köppen codes look implausible for some demonstrators (Tallinn Dfa, Naples BSk, Granollers Cwc) and drive the hard Köppen gate.
- Urban T2 indicators for green space, NDVI and housing were not yet populated.
- Each demonstrator is scored for one CIC only (Tallinn `EE@LAU:4:0784` urban, Granollers `ES@LAU:4:08096` river, Naples `IT@LAU:4:063049` and `ITF33` coastal, South Westphalia `DEB3K` agroforestry).
- An independent recomputation from `indicator_normalised_values` reproduced the urban T2 score exactly; T1 and the composite deviated by 3–9 points because categorical partial distances (Köppen adjacency, hierarchical DEGURBA credit) are not fully specified in the deliverable. Use this as the tolerance for the M4 acceptance test, and ask the D3.6 authors for the exact categorical rules.

## Prototype indicator set

`configs/indicators_eurostat_v1.yaml` lists the 25 indicators of the prototype with dataset codes, filters, native level, transform and aggregation rule.

## Statistik Austria (source `statat`)
- Table `OGD_bevstandjbab2002_BevStand_<year>` on data.statistik.gv.at: residents on 1 January by sex, municipality (current boundaries) and single year of age; code lists `_C-GRGEMAKT-0` and `_C-GALTEJ112-0`. Licence CC BY 4.0.
- Vienna is published by district (901xx–923xx) and summed to LAU 90001. Mergers after LAU 2024 are listed under `successors` in `configs/indicators_statat_v1.yaml`.
