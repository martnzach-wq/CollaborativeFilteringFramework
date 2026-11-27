# Transfer of the SD model to regions without domain-model runs

Scope: the KNOWING integrated SD model (mobility, microclimate, energy demand, energy supply) applied to a
region for which no ABM light, MAED-City, IESopt or PALM-4U runs exist, and for which none will be made. The
demonstrators (Tallinn, Naples, South Westphalia, Granollers) supply all behavioural evidence. Decisions D44–D56
in `docs/decisions.md`; results in `docs/causal/demonstrator_calibration_targets.md`.

## 1 Decomposition of a measure effect

    effect = intensity × exposure × response

| Component | Meaning | Source in a new region |
|---|---|---|
| intensity | the dose set by the measure, in natural units (EUR per km, retrofit rate, MWp, share of area greened) | local strategy (section 4) |
| exposure | the share of trips, km, floor area, buildings or area that the measure reaches; bounded by the **potential** | local structural data; regsim indicators; estimates from similar regions |
| response | the change per unit of exposed dose | model structure plus parameters from the demonstrators or the literature |

Responses are of three kinds, which transfer differently:

| Kind | Examples | Transfer |
|---|---|---|
| physical or technical | PV yield per kWp, heat-pump COP, cooling per m² of trees (PALM λ), emission factors | good: parameters with climate and technology corrections |
| behavioural | mode choice, compliance, rebound, investment decisions | partial: through the coping-type distribution (D54), compliance by type (D3.3 Table 30) and the availability of alternatives; residual band 0.27–3.5 × (D56) |
| systemic | electricity mix, imports, prices, grid limits | local: only through scenario inputs (section 4) |

A calibrated lever factor contains exposure and response together wherever the model lacks an explicit exposure
term. This is why the pricing factors did not transfer between the demonstrators (D53): the same toll reached
about 56 % of the car trips in Tallinn and 6 % in Naples.

## 2 Measure catalogue

`SD status`: **lever** = intensity enters the SD model and the response is modelled; **bundled** = in the Python
translation the measure follows the MAED-City or IESopt trajectory in both pathways. In the SD design (D2.4 Table
14) these measures are levers too: refurbishment rate (`EN annual refurbishment rate`), district heating
(`EN finEN total DH`), decarbonisation of industry and services (`EN finEN Industry/Service fossile`), PV, DH heat
pumps, electric boilers, wind and hydro (`ES PV increase`, `ES new heatpumps`, `ES new electrboilers`,
`ES new wind`, `ES new hydro`). They are restored in the Python translation (D58): each sector group of the
energy demand moves from its reference (BAU) trajectory to the CETS row with PI × intensity, each supply group
from its 2019 capacity to the IESopt 2050 value; the BAU trajectories come from the MAED-City BAU results of D3.3,
D3.4a and D3.5. A behavioural response to these measures (for example more private heat pumps through
subsidies) does need an extension (section 6).

| Module | Measure | Intensity (unit) | Exposure / potential | Response | SD status | Response parameters without new runs |
|---|---|---|---|---|---|---|
| mobility | road pricing | EUR per km | charged km per car trip | behavioural | lever (D53 exposure) | semi-elasticity −1 to −6 % car trips per EUR per trip (D55) |
| mobility | area toll | EUR per entry | share of car trips entering the zone | behavioural | lever, exposure missing | per EUR and entry −1.4 % (Tallinn dose-response 2030–2050) |
| mobility | parking management | EUR per trip | share of trips with priced parking | behavioural | lever | semi-elasticities −10 to −18 % per EUR (D55) |
| mobility | traffic calming / capacity | share of road capacity removed | road network in the area | behavioural | lever | semi-elasticities; walk/cycle time bonus (D41) |
| mobility | PT optimisation | share of PT time saved | PT trips, PT network | behavioural | lever | semi-elasticities −5 to −28 % per 100 % time saved |
| mobility | cycling optimisation | share of cycling resistance removed | car trips below 5 km | behavioural | lever | perceived-time reduction (D41) |
| mobility | 15-minute city | share of trips shortened per year | medium and long trips | behavioural | lever | re-choice of mode (D49) |
| mobility | car occupancy | attractiveness | car trips | behavioural | lever | weak, uncertain |
| fleet | combustion-car sales ban, EV subsidy | ban year, EUR | fleet turnover | behavioural + market | EV trajectory with ban compliance (D42) | compliance by coping type |
| microclimate | tree planting, unsealing | trees planted, ha unsealed, or share of the PALM-4U pathway | land-cover classes (placed as in the pathway), city area | physical | levers, decoupled from traffic calming by default (D60) | λ from Tallinn PALM-4U; β CDD per K |
| heat–health | street and courtyard trees, unsealing of streets and courtyards, AC restraint | trees per year, pp of class area per year, AC diffusion factor | land-cover classes and their residents by age | physical | levers of the heat_health submodel where a city configures it (D62) | canopy coefficient (literature prior, station posterior), age-specific DLNM, AC diffusion fitted to the city |
| heat–health | heat-action plan (rest), cooled care facilities, cool zones, passive cooling, building greening, district cooling | implementation share, share of dwellings per year, share connected | residents by age, dwellings | health, physical | levers with an imposed size of effect (stated effect) | meta-analysis (heat-action plan), assumptions (shading, greening) |
| buildings | retrofit | rate (% per year), depth (kWh/m²) | floor area by construction period | technical + behavioural | bundled (MAED) | savings per m² from building typologies (TABULA); compliance by type |
| buildings | heating-system change (heat pumps, DH connection) | units per year, share | buildings suitable, heat density | technical + behavioural | bundled (MAED, IESopt) | COP by climate; compliance by type |
| buildings | cooling limits, AC sales limits | limit values | AC stock, CDD | physical + behavioural | cooling ∝ CDD; behaviour missing | compliance by type (D3.3 Table 30) |
| supply | PV, wind, biomethane | MWp, MW, GWh | roof area, irradiation, wind potential | physical + systemic | bundled (IESopt) | yield per kWp from irradiation (regsim d36) |
| supply | district heating | network length, connections | heat density | technical | bundled (IESopt) | – |
| supply | electrolysis, biofuel blending | MW, blend share | demand in industry and transport | systemic | bundled; blend share as input | – |
| all | emission factors | – | fuel mix | physical | IPCC by fuel (D37) | transferable |

## 3 Transfer of the reference (BAU) pathway

The SD model runs a reference (PI = 0) and an optimised pathway (PI = 1). With the energy levers (D58) the
reference uses the MAED-City BAU trajectories of the demonstrators for energy demand and the 2019 supply
capacities. For a new region the reference has to be built from the components below.

| Component | What the SD model needs | From regsim / European data | Estimable from similar regions (regsim) | Local data needed |
|---|---|---|---|---|
| population 2019 and change | core and surroundings, annual | NUTS-3 population and growth (Eurostat; Statistik Austria at LAU) | – | delineation of core and surroundings; projections (national or city) |
| income, GDP | value of time, car ownership | GDP per inhabitant, household income (NUTS-2/3) | – | – |
| climate | HDD, CDD, heat days, trajectory | HDD/CDD (NUTS-2/3), D3.6 heat days, irradiation | CDD trajectory from similar climate | local climate scenario if available |
| coping types | shares A–F | D54 table | estimate for unsurveyed countries (error 3.9 points per type) | local survey |
| mobility base year | trips, length, time by mode × class × zone | cars per 1,000, network densities (D3.6) | modal split and length structure from similar regions: coarse | travel survey or transport model (essential) |
| mobility reference trend | population, costs, fleet | – | stable by construction in the logit model | national fuel price and EV projections |
| fleet | EV share trajectory | – | – | national registrations and projections |
| energy balance 2019 | final energy by sector and carrier | national balances (Eurostat), not regional | per-capita structure from similar regions: coarse | city energy balance (essential; e.g. Wien Energiebericht) |
| buildings | floor area by period, heating systems | – | coarse from similar regions | building register, heat atlas |
| reference efficiency trend | sector energy intensity over time | – | – | national reference scenario (NECP, EU Reference Scenario) |
| supply 2019 | local generation, DH, imports | renewables share (NUTS-0) | – | utility data, energy balance |
| grid emission factor trajectory | t CO2 per GWh | – | – | national NECP or system operator projection |
| commuting | persons and car share crossing the city boundary, trend | employed residents by place of work, NUTS-2 (`employed_work_*`, LFS; D65) | donor weights of cordon and pricing levers by car share of OD trips and commuter population (D56 extension) | inbound counts or cordon survey (essential for boundary targets) |
| territorial transport accounting | share of fuel sold that is burnt in the territory | – | – | regional emission inventory (fuel export, cross-border fuelling); essential for transport targets |
| local generation and dispatch | share of electricity demand generated locally, operation of gas CHP | – | – | utility data; decides whether local renewables lower territorial emissions |
| households | persons per household | population / private households (`private_households`, NUTS-2; D65) | – | – |
| building heating | heating systems by carrier | Census 2021 (national statistical offices; no Eurostat API table) | coarse from similar regions | building register, useful-energy analysis |
| land cover for the microclimate | sealing, building height, tree canopy by land-cover class, residents by class | Copernicus Urban Atlas, HRL Imperviousness and Tree Cover Density (no connector yet) | – | municipal tree cadastre, land-use map |
| heat–health | age structure, death rates, temperature–mortality curve by age group, care-home residents per person 75+, AC prevalence and its diffusion | age shares 65+ and 75+, NUTS-3 (`share_age_65plus`, `share_age_75plus`; D65); weekly deaths by age (Eurostat demo_r_mweek3) | temperature–mortality curves estimable from weekly deaths for any NUTS-2 region (planned) | care-home places and mortality; AC surveys (national) |

Assessment: with regsim alone a reference pathway can be built for demography, climate, income, coping types and
the age structure and commuting of the heat-health and boundary measures.
Two inputs are indispensable and local: the **mobility base year** (modal split by distance class and zone) and
the **energy balance by sector and carrier**. A reference trend for energy demand and supply comes from
national reference scenarios; it is not estimable from similar regions, since it follows national policy.
Behavioural responses come from the demonstrators with the transfer band of D56.

## 4 Interface for local strategies

A local strategy (for Vienna the Klimafahrplan and the Smart Klima City Strategie) supplies intensity and exposure;
the model supplies the response. The interface is a YAML file validated by `regsim.config.StrategyFile`
(example: `configs/strategies/vienna_klimafahrplan.yaml`).

### 4.1 Entries

| Entry | Content | Use |
|---|---|---|
| `measures` | catalogue measure, intensity series `{year: value}` with unit, exposure (kind, value), source (document, section), confidence | **input**: mapped to the SD levers (`regsim.causal.strategy.apply_strategy`) |
| `pathways` | trajectories of a bundled sector (e.g. final energy of households by carrier, PV capacity), with unit and source | **input**: replace the corresponding MAED-City or IESopt rows; no modelled response |
| `outcomes` | targets stated as results (modal split, CO2 per inhabitant, final energy) | **check**: compared with the model results; never an input |

Strategies often state outcomes ("85 % sustainable modes by 2030", "climate-neutral by 2040") rather than measures.
Outcomes are not inputs: the model computes them from the measures, and the comparison shows whether the
measures stated in the strategy suffice (`check_outcomes`). A target can be solved for a single measure
intensity (the intensity of one lever that meets it), which is reported as the required intensity, not as an
assumption.

### 4.2 Mapping to the SD model

| Catalogue measure | SD field (intensity) | Exposure field |
|---|---|---|
| road pricing | `policy_road_pricing` (EUR/km) | `road_pricing_km_scale` (charged km per car trip / model value) |
| area toll | `policy_area_toll` (EUR per entry) | share of car trips entering the zone (not yet modelled; reported) |
| parking management | `policy_parking_cost_adaptation` (factor on base cost) | – |
| traffic calming | `policy_capacity_adaptation` (factor on capacity) | – |
| PT optimisation | `policy_pt_traveltime_adaptation` (factor on PT time) | – |
| cycling optimisation | `cycling_resistance_reduction` (share) | – |
| 15-minute city | `policy_distance_adaptation` | – |
| car occupancy | `policy_car_passenger_attractiveness` | – |
| EV fleet | `ev_pi_trajectory` (% of fleet) | – |
| greening (unit `trees` or `share_of_pathway`), unsealing (`ha` or `share_of_pathway`) | `MicroclimateConfig.plant_trees` / `policy_plant_trees`, `unseal_ha` / `policy_unseal` (D60) | `city_area_km2` for ha |
| energy demand by sector group (households, services, industry, freight, other transport) | `EnergyDemandConfig.policy_energy[group]` (share of the way from BAU to CETS) | – |
| supply by technology group (PV, wind, hydro, district heating, electrolysis, other) | `EnergySupplyConfig.policy_supply[group]` (share of the way from 2019 to the IESopt 2050 value) | – |
| retrofit rate, heating systems in natural units | translated into the shares above or given as `pathways` | – |

The mapped strategy runs as the optimised pathway (PI = 1); the reference is the same configuration with the
strategy's measures held at their 2019 values. Measures without a lever are listed by `apply_strategy` as
unmapped, with the reason, so that no strategy element is silently dropped.

### 4.3 Responses

Lever factors, logit scale and adjustment come from the demonstrators (D55, D56) and carry the transfer band; a
run reports the central value and both ends. For the area toll and other levers whose calibrated factor still
contains the demonstrators' exposure, the factor is divided by the demonstrator exposure once that exposure is
known; until then the band covers it.

### 4.4 Stated and solved intensities

Strategies often quantify targets but not the intensity of every measure. Each measure therefore carries a role:

- **stated**: the intensity comes from the strategy (or is derived from it, with `confidence: derived` or
  `assumed`); the model supplies the response.
- **solved**: the strategy names the measure without intensity. Its `intensity` holds the *shape* of a lever
  bundle (by default the KNOWING pathway design), and the model finds one common scale per bundle (`solve_group`)
  and target year so that the outcome targets listed under `solve` are met (`solve_strategy`, least squares
  within bounds). Shares of a physical quantity stay at or below 0.9.

`apply_strategy` reports the role of every element (stated, solved, target), so that a result separates what the
strategy states from what the model infers. A solved scale answers "how strong would the named measures have to
be to meet the target, given the transferred responses", which is a consistency check of the strategy, not a
forecast; it carries the transfer band of the responses (D56). Outcome targets used for solving are no longer
independent checks; the other outcomes remain checks.

Independently of the role, each element has an effect class (D63): *modelled* (response estimated by the model),
*stated effect* (response size from an external estimate) or *exogenous path* (trajectory as input). Outcomes that
depend only on exogenous paths (in the SD model: buildings, industry, supply, vehicle fleet) are checks of a
strategy's consistency, not estimates of the effect of its measures.

### 4.5 Backcast without costs

`solve_backcast` (D64) treats every measure and pathway as a multiple m of its stated level: m = 0 is the reference
(the lever's value without the measure, a pathway row's reference override), m = 1 the strategy, values in between
or above move along the same path. Elements share a multiplier through `backcast_group`. Outcomes with a `bound`
("max" or "min") can be hard targets; the solver finds the multipliers that meet them either closest to the
strategy (min Σ (m − 1)², "how must the strategy change") or with the least effort (min Σ m / m_max, "what is the
least that meets the targets"). Shares stay within their physical limits, rows stay non-negative, and `delay_to`
holds the multiplied elements at their reference until a start year (timing questions). For paths towards an end
state (fleet share, phase-out, full implementation) `accelerate` reads a multiplier above 1 as a faster path from a
given year: the value of year y is the stated value of year y0 + (y − y0) m, so the end state comes earlier. Costs are not part of
the objective; with unit costs per measure the same structure takes a cost objective.

## 5 Identification with existing data only

Without new domain-model runs, the separation of intensity, exposure and response rests on:

1. **Dose-response within a demonstrator**: the ABM light single measures of Tallinn in 2030, 2040 and 2050 at
   three intensities (area toll 10 / 15 / 20 EUR: −14 / −21 / −27 % car trips, close to linear).
2. **Cross-sectional semi-elasticities** of the three ABM light demonstrators at their own intensities (D55).
3. **Exposure proxies** from the ABM light exports (charged km per car trip, D53) and from structural data.
4. **Literature parameters** for physical and technical responses (TABULA savings, COP, PV yield, IPCC factors)
   and for the value of time (HEATCO D5, D49).
5. **Behaviour model** compliance by coping type for all measure types in D3.3 Table 30 (AC limits, cooling
   capacity, subsidies, sales ban, parking removal, heat-wave travel, retrofit).

The empirical transfer band (0.27–3.5 × for single mobility effects, D56) stays the measure of what these sources
cannot explain.

## 6 Behavioural responses in energy demand and supply

In D2.4 the energy levers set exogenous rates or trajectories (PI between reference and CETS); the uptake by
households and firms is not modelled. D2.4 marks the endogenous uptake as prepared for the EV change rate, district
cooling and AC limits ("can be made endogenous, not yet implemented"). Private heat pumps are not a lever of their
own: they are part of the household space-heating rows of MAED-City (`hh_space heating_electr` against
`_fossil fuel`); the heat-pump lever of Table 14 is the district-heating heat pump on the supply side.

An endogenous response (e.g. subsidies → private heat pumps) needs an extension with:

| Element | Content | Source without new domain runs |
|---|---|---|
| stock and flows | heating systems by type (fossil boiler, heat pump, DH, biomass, electric) in floor area or dwellings; replacement at end of life plus early replacement | building register, MAED-City base year; lifetimes from the literature |
| decision rule | logit over systems at replacement on annualised cost (investment net of subsidy, energy cost, hurdle rate), as in the mobility logit; partial adjustment for the stock | literature on heating-choice models; the mobility logit as template |
| intensity | subsidy (EUR per unit or share of investment), ban dates for fossil systems, DH expansion | local strategy (section 4) |
| exposure | buildings suitable for heat pumps (age, insulation), buildings within DH reach | building and heat-density data |
| behavioural modulation | compliance by coping type for subsidies and retrofit (D3.3 Table 30), weighted with the local coping distribution (D54) | behaviour model |
| calibration | PI = 1 reproduces the CETS household space-heating rows, PI = 0 the reference; the subsidy response from the literature | MAED-City CETS of the demonstrators |

The same pattern serves the refurbishment rate (decision to retrofit at the renovation cycle), AC uptake and
limits, district-cooling connection and the EV change rate.
