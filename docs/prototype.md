# Feasibility prototype: Eurostat NUTS-0/2/3 benchmark

Reference material for the implementation. Code: `proto_regsim.py` (estimators), `bench.py` (masking schemes). Results: `prototype_benchmark.csv` (one row per run), `prototype_benchmark_summary.csv`, indicator list `prototype_indicators.csv`.

## Data

- 25 Eurostat indicators (dissemination API, JSON-stat), mean over 2018–2022 per unit: 8 with NUTS-3 as native level (population density, GDP per inhabitant in PPS, median age, old-age dependency, share aged 0–14, population growth, net migration, fertility), 12 with NUTS-2 (labour market, education, life expectancy, cars, household income, road deaths, R&D, poverty risk, hospital beds, broadband, lifelong learning) and 5 national (renewables share, GHG per capita, Gini, heating and cooling degree days).
- 1,929 units: 36 NUTS-0, 337 NUTS-2, 1,556 NUTS-3; NUTS-1 omitted; hierarchy from code prefixes; population weights from `demo_r_pjangrp3`.
- 18,863 observed cells; 61 % of the matrix is missing (NUTS-0 7 %, NUTS-2 29 %, NUTS-3 69 %). At NUTS-3, 17 of 25 indicators are never observed, which is the multi-level problem of R1 in its typical form.
- Transforms: log for strictly positive skewed indicators, log1p for rates with zeros, identity otherwise; robust scaling (median, IQR/1.349) on the native level.

## Methods compared

| Label | Description |
|---|---|
| level mean | mean of training cells at the same level |
| parent carry-down | value of the nearest ancestor (country value copied down) |
| kNN Gower (D3.6-type) | 10 nearest units of the same level by Gower distance (min–max, pairwise deletion), inverse-distance weighted |
| SoftImpute | nuclear-norm completion, all levels as independent rows (λ = 5, rank ≤ 12) |
| MF, broadcast (LEVITATE-type) | ancestor values copied into missing child cells and treated as observations, then MF |
| MF, hierarchical prior | u_g ~ N(u_parent, ·), no aggregation constraint |
| MF, prior + aggregation | hierarchical prior plus constraint that population-weighted child predictions match the observed parent value |

All MF variants: rank 12, λ_u = 0.03, 2,000 Adam iterations, squared loss. Rank and penalties were chosen on a separate validation split (10 % random cells, seed 99) before the benchmark.

## Results (RMSE in robust-scaled units, mean over runs)

| Method | M1 random cells | M3 country blocks | M4 level transfer to NUTS-3 |
|---|---|---|---|
| level mean | 1.174 | 1.273 | 1.214 |
| parent carry-down | 0.719 | 0.783 | 0.613 |
| kNN Gower (D3.6-type) | 0.697 | 0.793 | 1.214 ¹ |
| SoftImpute | 0.669 | 0.842 | 0.810 |
| MF, broadcast (LEVITATE-type) | 0.574 | 0.720 | 0.557 |
| MF, hierarchical prior | 0.430 | 0.833 | 0.460 |
| MF, prior + aggregation | **0.425** | 0.723 | **0.410** |

¹ With the indicator hidden at NUTS-3, no NUTS-3 neighbour carries it; the method falls back to the level mean. A distance-based method cannot predict at a level where the indicator is never observed.

Cold start of NUTS-2 regions (M2), RMSE on the hidden indicators:

| Known indicators k | 1 | 3 | 6 | 10 |
|---|---|---|---|---|
| parent carry-down | 0.875 | 0.838 | 0.814 | 0.872 |
| kNN Gower (D3.6-type) | 1.139 | 1.018 | 0.859 | 0.766 |
| MF, broadcast (LEVITATE-type) | 0.901 | 0.830 | 0.754 | 0.800 |
| MF, prior + aggregation | **0.556** | **0.520** | **0.467** | **0.484** |

## Findings relevant to the design

1. The hierarchical MF with aggregation constraints has the lowest error in three of four schemes and is level with the broadcast variant on country blocks (0.723 vs 0.720).
2. The two coupling mechanisms do different work. The hierarchical prior gives most of the gain for random cells and cold start; the aggregation constraint is what carries the country-block case (prior only: 0.833, with constraint: 0.723), where the national value is the only information on the hidden indicator.
3. Copying parent values (LEVITATE) is a strong baseline at fine levels, and MF on top of it improves on plain carry-down; the explicit hierarchy improves further in M1, M2 and M4.
4. Gower-type nearest neighbours need several shared indicators; with one to three known indicators they are worse than carry-down. This supports profile-based similarity on completed data (S-profile) over pairwise-deletion Gower for entities with few observations.

## Limitations of the prototype

- One cross-section (2018–2022 means); no time dimension.
- NUTS codes taken as published (mixed versions possible); units without a matching parent were dropped.
- Aggregation is applied on the transformed scale (e.g. mean of logs), an approximation; the implementation should aggregate on the original scale according to the `aggregation` metadata.
- In M2 the NUTS-3 children of a hidden NUTS-2 region keep their values, so information flows from children to parent. This is realistic for Eurostat data but optimistic for a new region without observed sub-units.
- No uncertainty, no typed losses (all indicators continuous), single initialisation per fit, small number of seeds.
