# Calibration targets for the SD model from the demonstrator deliverables

Sources: D1.3 (Coping Types, resubmission Nov 2025), D3.3 (Tallinn), D3.4a (Naples), D3.4b (Granollers),
D3.5 (South Westphalia). Extracted values are in `configs/sd_calibration_targets.csv` (272 rows, long format
with source page per value). Chart values are read from the data labels of the published figures.

## Coverage

| Target | Tallinn | Naples | Granollers | South Westphalia |
|---|---|---|---|---|
| Trips by mode, 2019 (internal + OD) | ABM light | ABM light | trip shares only (city data, 3 mode groups) | ABM light |
| Passenger-km by mode, 2019 | ABM light | ABM light | – | ABM light |
| Trajectories 2030/2040/2050, reference and optimised | ABM light | ABM light | – | ABM light (reference 2030/2040 not labelled in this extract) |
| Single-measure effects, 2040 intensity (car share, CO₂) | 9 measures + all | 9 measures + all | – | 8 measures + all (no area toll) |
| Measure intensities 2030/2040/2050 | Table 1 | Table A-27 | – | Table 23 |
| Coping Types, current | metropolitan, Estonia | metropolitan, Italy | urban, Spain | rural, Germany |
| Energy demand (MAED-City) | input file in SD repository | balance (Table A-13), projections (Fig. A-60/A-62); SD file is an estimate from public data | not modelled (insufficient data) | reconstructed from D3.5 (`build_southwestphalia_xlsx.py`) |
| Energy supply (IESopt), 2019 and 2050 | Tables 9–14; SD file `Supply_Data_Tallinn.xlsx` | Tables A-19–A-24; SD uses synthetic values | not modelled | Tables 15–22; SD file built from them |
| Microclimate (PALM-4U, WRF) | CDD reduction by measures 0.8/1.0/1.2 % (2030/40/50) | local cooling up to 2.8 °C (day), 5.2 °C (night) | local cooling up to 0.8 °C (day), 1.0 °C (night) | CDD/HDD trend from WRF only |

The ABM light base year is calibrated on observed data: the Tallinn Strategic Transport Model (2022) and a
household travel survey (TNS Kantar 2015) for Tallinn; the Landesverkehrsmodell NRW and MiD 2017 for South
Westphalia. D3.4a does not name the calibration data for Naples. Trajectories and measure effects are model
results, so they serve as cross-calibration targets (model against model), not as observations.

## Base year 2019 (internal + OD traffic)

| | Tallinn | Naples | South Westphalia |
|---|---|---|---|
| trips/day (Mio) | 1.26 | 3.13 | 5.60 |
| trip shares car / car passenger / PT / bicycle / walk (%) | 38 / 8 / 35 / 1 / 18 | 38 / 12 / 34 / 5 / 11 | 52 / 11 / 16 / 8 / 14 |
| pkm shares car / car passenger / PT / bicycle / walk (%) | 45 / 9 / 45 / 0 / 2 | 47 / 14 / 36 / 2 / 1 | 68 / 12 / 15 / 5 / 1 |
| car + car passenger, pkm share (%) | 53 | 61 | 80 |
| mean trip length car / PT / walk (km) | 12.7 / 14.1 / 1.3 | 14.8 / 12.3 / 1.5 | 25.8 / 19.4 / 0.9 |

Naples counts motorcycles with cars. Granollers: 251,674 trips/day; private vehicle 36.2 %, public
transport 11.8 %, walking and cycling 52.0 % of trips; 72 % of internal trips on foot.

## Comparison with the current SD configurations

| | SD model (2019) | Deliverable | Consequence |
|---|---|---|---|
| Tallinn, car + passenger share of pkm | 65.7 % (mode-specific trip lengths); 55.3 % (centroid lengths) | 53.3 % | the centroid method matches; the mode-specific method overstates car use |
| Naples, car + passenger share of pkm | 64.9 % | 61.0 % | close; base attractiveness was set to an estimated split |
| Naples, policy levers | area toll 5/12/18/25 EUR, road pricing 0.08/0.20/0.35 EUR/km | area toll 10/15/20 EUR, road pricing 0.10/0.30/0.50 EUR/km (city) | align with Table A-27 |
| Tallinn, Coping Types | national Estonia (12/16/33/14/12/13) | metropolitan Estonia (9/17/32/13/13/17) | use metropolitan |
| Naples, Coping Types | copy of Granollers (4/8/11/36/30/11) | metropolitan Italy (1/6/14/29/37/14) | replace |
| Granollers, Coping Types | national Spain (4/9/10/37/31/9) | urban Spain (3/9/9/37/33/10) | minor |
| South Westphalia | no configuration | full ABM light set | new configuration needed |

Per-capita levels need the population of each model area: the deliverables give trips for "internal + OD
traffic" without the population of the OD zone.

## Inconsistencies in the deliverables

- D3.4a Table A-29 (Naples) lists the metropolitan values of Fig. A-94 in reverse order of types
  (Active Antagonists 14 % instead of 1 %, Fierce Forerunners 1 % instead of 14 %).
- D3.5 Table 25 (South Westphalia) lists the rural column of Fig. 56 in reverse order and is headed
  "Naples"; Fig. 56 is captioned "Italy" but shows Germany.
- The legend of the Spain figure in D3.4b runs in reverse order of the colours; Table (section 5.3.3)
  is consistent with the national values of D1.3.
- The measure-effect figures report "delta car share" without stating whether the share refers to trips
  or passenger-km.

If the behaviour-model runs in D3.4a and D3.5 used the tables, their type-specific results are affected.

## Use in the SD calibration

1. Class R (base attractiveness, damping): fit the 2019 shares of trips and passenger-km per mode
   (`recalibrate_residuals`).
2. Validation of dynamics: optimised trajectories 2030/2040/2050 under the measure intensities of the
   deliverables (not the current SD policy series).
3. Lever validation: the deliverables report add-one effects at 2040 intensity; `lever_table` reports
   leave-one-out effects of the full package. Comparable SD runs add one lever at 2040 intensity to the
   reference.
4. `cross_calibrate` across the three ABM demonstrators; Granollers is a transfer target with trip shares
   as its only observed check.

## Energy supply and microclimate

The SD supply module reads capacities and generation for 2019 and 2050 (`ES_inputs`); the IESopt tables
of the deliverables report exactly these two years, so Tallinn, Naples and South Westphalia can be filled
from them. The supply module used to set wind and hydro generation to zero; it now reads them where a city
file lists them (South Westphalia: wind 2.2 TWh in 2019, 13.0 TWh in 2050).

The SD model receives the microclimate only through the cooling degree days (CDD, attractiveness of active
modes and PT) and the street-greening lever. The MAED-City heat-wave tables give CDD for the base year, the
CETS 2050 and the heat-wave year:

| | 2019 | CETS 2050 | heat wave 2050 | SD configuration |
|---|---|---|---|---|
| Tallinn | 8.6 | 15 | 38 | 100 (Vensim default) |
| Naples | 326 | 610 | 685 | 120 |
| South Westphalia | 21 | 37 | 74 | 21 |
| Granollers | – | – | – | 380 |

The PALM-4U runs report local temperature changes of the greening measures, not a CDD change for the city,
except for Tallinn (CDD −0.8/−1.0/−1.2 % in 2030/2040/2050). They do not quantify the effect of street
greening on mode choice, so the street-greening lever has no calibration target.

## Energy demand check (2019)

With the centroid method and the corrected industry total (manufacturing was counted twice when the file
contains the `Industry_*` carrier rows), the SD model gives for South Westphalia 57.7 TWh final energy
against 49.3 TWh in D3.5. The difference is passenger transport (23.0 against 14.6 TWh): the SD model
converts car and car-passenger kilometres with the Tallinn factor (0.749 kWh/km), while MAED-City uses
vehicle kilometres (load factor 1.13) and German fleet values. City-specific conversion factors are class-O
inputs still missing for Naples, Granollers and South Westphalia.

## Microclimate submodel (D26)

`microclimate.py` in the SD repository translates `Submodel_MicroClimate_v4.mdl` (monthly time step,
Time 120/252/372/492 = 2019/2030/2040/2050) to the annual grid. Hot-day 2 m temperature per Corine land-cover
class is linear in four urban features (sealed area, building volume, young and mature trees; D2.4 section
4.2.4).

| Calibration on `mc4_cal_tallinn.csv` (19 classes × 4 years) | tmax | tmin |
|---|---|---|
| sealed area (K per %) | +0.045 | +0.036 |
| building area × height / 10 | −0.065 | −0.052 |
| young trees per % of city area | −0.0002 | −0.0003 |
| mature trees per % of city area / 100 | −0.005 | −0.014 |
| city-average change 2019–2050, PALM-4U | −0.23 K | −0.21 K |
| city-average change 2019–2050, submodel | −0.16 K | −0.14 K |

The coefficients come from the changes within each class, which is the response to interventions the SD
model needs; a pooled fit across classes reproduces the class levels (weighted R² 0.72 and 0.91) but only a
quarter of the temporal response. The relative CDD change per K of city-average cooling, cross-calibrated to
the D3.3 reductions (−0.8/−1.0/−1.2 % in 2030/2040/2050), is 0.09 per K. Without measures (PI = 0) Tallinn's
CDD in 2050 is 15.2 instead of 15.0. Naples, Granollers and South Westphalia have no class-level PALM-4U
table; their CDD follow MAED-City only.

## Naples reconstruction against the draft MAED-City export (D28)

Rows of the reconstruction from D3.4a against the draft export (not final): totals of households (−1 %),
services (−2 to +1 %), manufacturing (±1 % except 2030) and freight agree; the mean absolute deviation over all
rows and keyframes is 15 %. The largest differences are end-use splits that D3.4a does not publish: the draft
puts a quarter of household firewood into "other" uses (the rule puts all of it into space heating) and
reports industry fossil fuels about twice the manufacturing row of Table A-13.

## Base year after the trip-class estimate (D30)

| pkm shares 2019 car / car passenger / PT / bicycle / walk | SD model | ABM light |
|---|---|---|
| Tallinn | 46 / 9 / 42 / 0 / 3 | 45 / 9 / 45 / 0 / 2 |
| Naples | 46 / 14 / 36 / 3 / 2 | 47 / 14 / 36 / 2 / 1 |
| South Westphalia | 64 / 12 / 17 / 6 / 1 | 68 / 12 / 15 / 5 / 1 |

Car energy 2019 against MAED-City (D29): Tallinn 1.9 against 1.2 TWh, Naples 4.7 against 2.5 TWh, South
Westphalia 19.1 against 13.8 TWh. The ABM light trips include OD traffic, which the SD model divides by the
resident population, while MAED-City counts the residents' consumption; aligning the boundaries is part of the
class-R recalibration.

## Passenger transport in the energy balance (D31)

Trips that cross the city boundary count with 0.5 (GPC induced-activity method: origin and destination share
each such trip; trips within the city count fully). Car energy 2019 against MAED-City: Tallinn 1.25 against
1.23 TWh, South Westphalia 16.8 against 13.8 TWh, Naples 4.2 against 2.5 TWh. The remaining gap in Naples and
South Westphalia points to an OD share above the placeholders (the OD-zone populations are not yet known).

## Class-R recalibration (D32, D35)

The climate effect on active modes and PT no longer enters the attractiveness (−CDD/100, −CDD/200): per cooling
degree day 0.02/365 of active trips and 0.01/365 of PT trips move to car and car passenger, relative to the 2019
climate. Evidence: cycling rises with temperature up to about 20–25 °C and falls on hot days ([Böcker et al.
2019](https://consensus.app/papers/details/w:ec6e3df7-9807-4977-ab41-117b49d6d467/); [Hogeweij et al.
2026](https://consensus.app/papers/details/w:76fbf6e4-09ed-4150-bd52-741fddbce06f/): decline above UTCI 32 °C);
very hot days move walking and transit trips to the car ([Belloc et al.
2025](https://consensus.app/papers/details/w:00984244-138d-4b99-a97c-97264058f0b5/)); mobility falls by up to 10 %
on hot days ([Renninger et al. 2026](https://consensus.app/papers/details/w:702fc3fb-ef05-4cc6-93a9-a2911889d175/)).

Attractiveness offsets per mode (car as reference) and the mode-change damping are then fitted with bounded
least squares (offsets ±15, several starts; `sd_knowing.recalibrate_class_r`) to the passenger-km shares of the
ABM light reference and optimised pathways 2030/2040/2050; Granollers, without ABM light, to its previous model
trajectories.

| | damping | offsets car passenger / PT / bicycle / walk | squared error |
|---|---|---|---|
| Tallinn | 0.101 | −2.24 / +5.45 / +0.54 / +10.69 | 0.011 |
| Naples | 0.080 | +0.51 / −1.22 / −15.0 / −15.0 | 0.006 |
| South Westphalia | 0.466 | −1.08 / +2.68 / −0.68 / −3.57 | 0.008 |
| Granollers | 0.840 | 0.00 / −1.90 / −3.80 / −3.80 | 0.000 |

| car share of pkm, 2050 | SD reference | ABM reference | SD optimised | ABM optimised |
|---|---|---|---|---|
| Tallinn | 0.42 | 0.44 | 0.21 | 0.24 |
| Naples | 0.44 | 0.46 | 0.24 | 0.25 |
| South Westphalia | 0.65 | 0.66 | 0.50 | 0.51 |

For Granollers the fit returns exactly the former terms −CDD/200 and −CDD/100 (CDD 380) as constant offsets: in
the original model they acted as calibration constants, not as a climate response. Naples reaches the lower bound
for walking and cycling; the generalised-cost dynamics overstate the growth of active modes there.

## Longer OD trips (D36)

Tallinn's transport model has its own trip lengths for the OD zone (centroids 1.65/6.79/26.4 km against
0.99/5.72/13.7 km internally); per mode its OD trips are 2.7 to 4.5 times as long as internal ones. Naples and
South Westphalia now use these ratios: the internal and OD mean lengths per mode are set so that their
trip-weighted mean equals the ABM light mean. Car energy 2019 against MAED-City: Naples 2.66 against 2.49 TWh
(+7 %), South Westphalia 13.77 against 13.75 TWh with an OD population of 551,000 (OD share of car pkm 56 %).
Class R refitted: Naples squared error 0.005 (car share 2050 reference 0.44 against 0.46, optimised 0.24 against
0.25), South Westphalia 0.008 (0.66 against 0.66, 0.50 against 0.51).

The Naples transport model covers the metropolitan area (D3.4a Fig. A-84) but its totals count only trips
with one end in the city ("internal + OD traffic Naples city"), so through traffic is not included. D3.2 gives
more than 3.5 million inhabitants for the metropolitan city and 914,758 for the municipality.

## Checks against IESopt and MAED-City (2026-11-25)

**Energy supply 2050 (NY) against IESopt.** Two errors in `energy_supply.py` were fixed: the district-heating
gap was computed before the heat pumps, so electric boilers delivered the heat-pump heat a second time (Tallinn
2050: 865 instead of 38 GWh boiler heat), and the electrolyser demand was not read. After the fixes:

| 2050 | DH electricity SD / IESopt | net import SD / IESopt |
|---|---|---|
| Tallinn | 234 / 233 GWh | 4,431 / 4,516 GWh |
| Naples | 58 / 58 GWh | 2,475 / 2,387 GWh |
| South Westphalia | 625 / 625 GWh | 613 / −221 GWh |

South Westphalia's remaining difference comes from the core electricity demand (SD +4 %, passenger EVs from
the mobility submodel).

**Direct CO2 against the MAED-City KPI tables** (ratio SD / MAED-City, 2019, 2030, 2035, 2040, 2045, 2050), after
D37:

| | 2019 | 2030 | 2035 | 2040 | 2045 | 2050 |
|---|---|---|---|---|---|---|
| Tallinn | 0.72 | 0.98 | 1.01 | 1.03 | 1.03 | 1.07 |
| Naples | 1.02 | 1.06 | 1.04 | 1.07 | 1.00 | 0.94 |
| South Westphalia | 1.00 | 1.08 | 1.11 | 1.18 | 1.56 | 1.19 |

Before D37 the ratios were 0.81/0.96/1.07 in 2019 and Naples fell to 0.30 by 2050 (no aviation). Tallinn 2019:
the MAED-City value (2.06 t/cap) is higher than the fossil fuels of its own balance give (about 0.65 Mt), most
likely because it includes the district-heating gas boilers, which the SD model reports separately (em_dh).
South Westphalia 2045: the SD model treats all non-electric car energy as fossil, while MAED-City blends
biofuels and hydrogen into transport.

**Final energy 2019:** Tallinn 7,963 against 7,936 GWh, Naples 9,326 against 9,165 GWh, South Westphalia 49,372
against 49,329 GWh.

**Tallinn industry rows:** the carrier rows sum to 1,354 GWh against 1,552 GWh of the sector rows; the
difference is manufacturing biofuel (211 GWh, D3.3 Table 3), which the carrier rows omit. No emissions are
missing.

## Biofuel and hydrogen in car fuels

The non-electric car fuels carry a non-fossil share read from MAED-City: Naples 3.1 % (2019), 6 % (2030), 15 %
(2040), 22 % (2045), 32 % (2050); South Westphalia 4.2 %, 5 %, 10 %, 26 %, 30 %; Tallinn 0 (no transport biofuel in
D3.3 Table 3). Aviation ("other" fuels) stays fossil. Fitting the shares to the MAED-City CO2 instead would give
erratic paths (Naples 2045/2050: 0), since the fit absorbs every other difference.

Direct CO2, ratio SD / MAED-City (2019, 2030, 2035, 2040, 2045, 2050): Tallinn 0.72, 0.98, 1.01, 1.03, 1.03, 1.07;
Naples 1.00, 1.03, 0.99, 1.00, 0.91, 0.86; South Westphalia 0.99, 1.05, 1.07, 1.13, 1.29, 0.89.

## Lever ranking against the ABM light single-measure effects

SD lever effects (leave-one-out from the full package, PI = 1, car share of pkm, 2040; D23) against the ABM light
effects of single measures at 2040 intensity (change of car share against the reference):

| | Spearman | measures |
|---|---|---|
| Tallinn | 0.81 | 7 |
| South Westphalia | 0.83 | 6 |
| Naples | 0.50 | 7 |

| measure (% change of car share) | Tallinn ABM / SD | Naples ABM / SD | South Westphalia ABM / SD |
|---|---|---|---|
| area toll | -20.9 / -20.7 | -5.1 / -16.3 | – |
| road pricing | -6.1 / -6.7 | -12.3 / -6.7 | -12.7 / -10.2 |
| parking management | -2.1 / -0.5 | -3.6 / -0.3 | -3.4 / -0.6 |
| public transport optimisation | -2.6 / -0.3 | -4.5 / -0.2 | -3.6 / -0.3 |
| cycling optimisation | -3.1 / -0.6 | -1.7 / -0.5 | -1.2 / -0.5 |
| traffic calming | -3.1 / -0.0 | -1.2 / -0.2 | -0.7 / 0.0 |
| car occupancy incentives | -0.3 / 0.4 | -2.2 / 0.5 | 0.6 / 0.5 |

Pricing levers agree in size where the SD model and ABM light share the policy units (area toll Tallinn −20.9 /
−20.7 %, road pricing Tallinn −6.1 / −6.7 %, South Westphalia −12.7 / −10.2 %). The Naples area toll is three
times too strong in the SD model (−16.3 against −5.1 %): the same charge (10/15/20 EUR) meets a lower value of
time. Parking, PT, cycling and traffic calming act five to twenty times more weakly in the SD model than in
ABM light. The car-occupancy lever has the opposite sign because the SD car share includes car passengers.

## Cross-calibration and uncertainty (protocol validation and step 5)

Class-R parameters of one city transferred to another (`sd_knowing.cross_calibrate_class_r`, wrapping
`sd_bridge.cross_calibrate`), root mean square error of the 30 pkm shares:

| donor → target | transferred | own calibration |
|---|---|---|
| Tallinn → Naples | 0.080 | 0.013 |
| Tallinn → South Westphalia | 0.028 | 0.016 |
| Naples → Tallinn | 0.058 | 0.019 |
| Naples → South Westphalia | 0.051 | 0.016 |
| South Westphalia → Tallinn | 0.055 | 0.019 |
| South Westphalia → Naples | 0.119 | 0.013 |

Transferred class-R parameters raise the error two to nine times; Tallinn → South Westphalia transfers best. Class R
is therefore city-specific and has to be recalibrated for every new region (protocol rule), not copied.

Joint draws of class R (normal approximation of the least-squares fit) and class-O inputs (OD population ±25 %,
car energy per vehicle-km ±10 %, load factor ±5 %, CDD ±10 %), 100 draws, 90 % intervals in 2050
(`sd_knowing.propagate_uncertainty`):

| 2050, PI = 1 | car share of pkm | final energy (GWh) | direct CO2 (kt) |
|---|---|---|---|
| Tallinn | 0.211 (0.193–0.231) | 6916 (6850–7020) | 49 (43–59) |
| Naples | 0.244 (0.230–0.261) | 5914 (5736–6136) | 170 (159–183) |
| South Westphalia | 0.500 (0.465–0.535) | 30717 (30203–31322) | 223 (188–263) |

The walking offset is poorly identified (standard deviation 4–8), since walking carries 1–3 % of pkm; the damping is
well determined. The uncertainty of the 2050 direct CO2 under the full package is about ±15 % (Naples ±7 %).

## Per-intervention calibration (D38–D40)

Each intervention has its own calibration factor, as the `TR Cal` constants of the Vensim model. Fitted one by
one to the ABM light single-measure effects, the factors reproduce every effect, but the full package then
overshoots (car share of pkm 2050 optimised: 0.10 / 0.14 / 0.24 against 0.24 / 0.25 / 0.51 for Tallinn / Naples /
South Westphalia): the SD model adds the measures up over time, while ABM light shows that they overlap. The
factors are therefore fitted jointly with the class-R offsets and the damping (`sd_knowing.recalibrate_joint`).

| | squared error pathways | squared error effects | car share 2050 opt. SD / ABM |
|---|---|---|---|
| Tallinn | 0.0058 | 0.0021 | 0.22 / 0.24 |
| Naples | 0.0010 | 0.0010 | 0.25 / 0.25 |
| South Westphalia | 0.0073 | 0.0015 | 0.49 / 0.51 |

Single-measure effects after the joint fit (% change of car share 2040, SD against ABM light): pricing measures
within 1 point (Tallinn area toll −20.3 / −20.9, road pricing −6.6 / −6.1; Naples road pricing −12.1 / −12.3, area
toll −4.8 / −5.1; South Westphalia road pricing −12.9 / −12.7). The cycling factor goes to its lower bound in all
three cities: the SD cycling lever cannot reproduce its small single effect and the pathways at the same time.
The value of time now follows D39 (Tallinn 3.79, Naples 3.24, South Westphalia 5.11 EUR/h).

Values after the revision of D41 and D42 (pathway and effect errors above).

### Active modes and EV fleet (D41, D42)

Cycling optimisation, traffic calming and the 15-minute city reduce the perceived travel time of walking and
cycling as in ABM light. The single-measure runs hold these series with their lever; before that, reference and
measure run both carried them and the cycling effect read 0 at any factor. With the factor at 1 the cycling effect
is now −0.9 % (Tallinn), −0.04 % (Naples) and −0.3 % (South Westphalia).

The joint fit still sets the cycling factor to its lower bound in all three cities. Without the cycling lever, the
SD bicycle share of pkm in the optimised pathway already reaches or exceeds ABM light (Tallinn 2040 3.3 % against
2.0 %, South Westphalia 2050 7.5 % against 5.9 %, Naples 2050 4.3 % against 3.7 %): the push measures (capacity,
pricing, parking) divert car trips to the bicycle, while in ABM light the bicycle gains mainly from cycling
optimisation. Any cycling effect then overshoots the pathway shares by more than it reduces the effect error.

Tallinn's EV fleet share at PI = 1 is 18 / 45 / 80 % (2030 / 2040 / 2050), BAU 12.3 / 25.8 / 43.3 %. The
sales-ban compliance by coping type scales the gap relative to the 2019 coping-type distribution; with the static
distribution the factor is 1.

### Destination-weighted mode change (D43) and coping sensitivity (D44)

With the mode change weighted by the destination share, the cycling factor leaves its lower bound:

| | a | squared error pathways | squared error effects | cycling factor | cycling effect SD / ABM (%) | car share 2050 opt. |
|---|---|---|---|---|---|---|
| Tallinn | 0.25 | 0.0065 | 0.0014 | 1.85 | −1.9 / −3.1 | 0.22 |
| Naples | 1 | 0.0005 | 0.0007 | 10.7 | −0.9 / −1.7 | 0.24 |
| South Westphalia | 1 | 0.0082 | 0.0011 | 0.75 | −0.4 / −1.2 | 0.49 |

With pure share weighting (a = 1) Tallinn's cycling factor rose to 84, since the bicycle starts from a share near
0; a = 0.25 and a = 0.5 give nearly equal fits. Tallinn's factor of 1.85 reduces the perceived cycling time by 83 %
in 2040, beyond the measure table's mean of 45 %.

Tallinn coping sensitivity (D44), 2050: car share of pkm 0.425 → 0.360 (PI = 0) and 0.215 → 0.179 (PI = 1), EV
share at PI = 1 80 → 82 %. The effect follows from the coping-type attractiveness profiles: all types start
from the same mode split in 2019 and diverge to car trip shares of 0.77 (type A) to 0.14 (type D) by 2050.

### Coping-type initial splits (D45)

Joint fit with type-specific initial splits (squared error pathways / effects): Tallinn 0.0067 / 0.0022, Naples
0.0005 / 0.0007, South Westphalia 0.0078 / 0.0009; cycling factor 1.68 / 9.9 / 0.61, cycling effect −1.2 / −0.9 /
−0.2 % against ABM light −3.1 / −1.7 / −1.2 %. Car share of pkm 2050 optimised 0.21 / 0.24 / 0.49.

The odds ratio of car use of type A stays near its 2019 value in Naples and South Westphalia (1.74 and 1.81 in
2050 BAU) but grows to 3.13 in Tallinn, whose branch applies the attractiveness profile with strength 1 instead of
0.22. Tallinn coping sensitivity (D44) with these splits, 2050: car share of pkm 0.428 → 0.338 (PI = 0) and 0.213 →
0.163 (PI = 1), EV share 80 → 82 %.

### Stable coping-type differences in Tallinn (D46)

The additive value-of-time profile gave type D 0.79 EUR/h and type A 6.79 EUR/h; with the profile relative to the
base and strength 0.768, the car trip shares by type stay near their 2019 values (A–F 2019 0.55 / 0.51 / 0.48 /
0.31 / 0.20 / 0.19, BAU 2050 0.59 / 0.54 / 0.57 / 0.30 / 0.20 / 0.20; odds ratio of type A 2.00). Joint fit: squared
error pathways 0.0077, effects 0.0016; cycling factor 2.09, cycling effect −1.4 % against −3.1 %.

Coping sensitivity (D44) 2050: car share of pkm 0.428 → 0.368 (PI = 0) and 0.216 → 0.176 (PI = 1), EV share 80 →
82 %. The change now equals the composition effect of the shifted type shares at the per-type car shares.

### Tallinn on the ABM light base year (D47, D48)

With length and door-to-door time per mode and class, the 2019 pkm shares equal ABM light (car 0.443, car
passenger 0.088, PT 0.446, bicycle 0.003, walk 0.020). Car energy 2019 falls from 1,249 to 1,129 GWh (MAED-City
1,225), since ABM light's OD car trips are shorter (21 km) than the former class centroid (26 km).

Joint fit with coping-type stability, by value of time (squared error pathways / effects / coping):

| VoT (EUR/h) | pathways | effects | coping | damping | profile strength |
|---|---|---|---|---|---|
| 3.79 | 0.0046 | 0.0029 | 0.0008 | 0.47 | 1.36 |
| 7.58 | 0.0031 | 0.0020 | 0.0000 | 5.0 (bound) | 0.25 |
| 11.37 | 0.0018 | 0.0026 | 0.0000 | 3.19 | 0.94 |

With 11.37 EUR/h: car share of pkm 2050 0.424 (reference; ABM light 0.435) and 0.242 (optimised; 0.241). The
pricing levers keep factors of 0.06 at damping 3.2: the configured road-pricing and toll series act about five
times more strongly than in ABM light. Single-measure effects of cycling (−0.2 % against −3.1 %) and the 15-minute
city (+0.5 % against −2.1 %) remain unmatched. Coping sensitivity (D44) 2050: car share 0.424 → 0.338 (PI = 0),
0.242 → 0.179 (PI = 1).

### Tallinn on the full ABM light export (D49, D50)

The single-measure effects of ABM light are changes in car trips: all measures 2040 −35.6 % (D3.3 Fig. 17:
−37.3 %), car pkm −31.9 %. Cycling optimisation is −1.6 % in 2040, not −3.1 %.

Value of time, Estonia 2019: HEATCO D5 Table 0.4 (EUR2002, factor prices) car commute short 4.99, other short
4.18, business 12.82; with 30/60/10 % weights and the HEATCO update (GDP per capita elasticity 0.7; Eurostat
2002–2019 real GDP per capita ×1.80, HICP ×1.73; factor 2.62): 13.9 EUR/h (bus 10.2). Rail Baltica CBA (EY 2017,
2015 prices): private 0.19 EUR/min = 11.4 EUR/h, business 25.8 EUR/h.

Joint fit (three starting points, same optimum): squared error pathways 0.0156 (pkm and trip shares), effects
0.0141 (24 effects), damping 0.34, profile strength 1.5 (bound). Car share of pkm optimised 2030/2040/2050
0.403 / 0.314 / 0.212 (ABM light 0.365 / 0.296 / 0.242), reference 2050 0.429 (0.434); bicycle trip share
optimised 2050 0.055 (0.056). The SD mode change accumulates over time, so the effects lag in 2030 (area toll
−5.5 % against −14.1 %) and overshoot in 2050 (−30.2 % against −26.7 %); ABM light responds in equilibrium to
each year's measure intensity.

### Logit mode choice in Tallinn (D51)

Joint fit on the full ABM light export (two starting points, nearly equal): squared error pathways 0.0032 and
effects 0.0022 against 0.0156 and 0.0141 with the pairwise flows; coping-type stability holds without a
constraint (odds ratio of type A 1.93 in the 2050 reference).

| car share of pkm | 2030 | 2040 | 2050 |
|---|---|---|---|
| optimised SD / ABM light | 0.355 / 0.365 | 0.285 / 0.296 | 0.236 / 0.242 |
| reference SD / ABM light | 0.435 / 0.440 | 0.436 / 0.436 | 0.445 / 0.434 |

Single-measure effects (car trips, SD / ABM light, %): area toll −12.6 / −14.1, −20.3 / −20.7, −26.2 / −26.7; road
pricing −1.8 / −2.0, −5.5 / −6.1, −9.2 / −10.3; cycling −0.1 / −0.2, −1.2 / −1.6, −2.2 / −3.2; 15-minute city
−0.6 / −0.4, −1.6 / −2.1, −2.6 / −1.1. Car-occupancy incentives overshoot (2050 −2.9 / −0.4); traffic calming and
parking lag in 2030 (−0.9 / −2.0, −1.0 / −2.2). Coping sensitivity (D44) 2050: car share of pkm 0.445 → 0.403
(PI = 0), 0.236 → 0.208 (PI = 1), now the composition effect alone.

### Naples and South Westphalia on the ABM light exports (D52)

| | Naples | South Westphalia |
|---|---|---|
| squared error pathways / effects | 0.0072 / 0.0007 | 0.0199 / 0.0011 |
| logit scale (1/EUR) / adjustment | 0.005 (bound) / 1.0 | 0.009 / 1.0 |
| car share of pkm optimised 2030 / 2040 / 2050, SD | 0.414 / 0.340 / 0.266 | 0.650 / 0.602 / 0.546 |
| ABM light | 0.407 / 0.324 / 0.251 | 0.638 / 0.608 / 0.591 |
| car share of pkm reference 2050, SD / ABM light | 0.475 / 0.455 | 0.680 / 0.661 |

In both cities the logit scale goes to (or near) its lower bound and the lever factors grow in proportion (Naples
parking 86, cycling 93; South Westphalia parking 100, with road pricing, cycling and car passenger at 0.01): only
the products scale × factor are identified. The Tallinn scale (0.25) fixed in Naples gives pathways 0.0118 and
effects 0.0016. ABM light's reference drops from the base year to 2030 in Naples (car 0.474 → 0.458), which the SD
model does not reproduce; in South Westphalia ABM light's optimised car-passenger pkm share falls to 0.034 (SD
0.093). The first South Westphalia base-year export was inconsistent with the scenarios (car pkm share 0.37 against
0.68 in 2030); the corrected one agrees (0.678).

Car energy 2019 against MAED-City: Tallinn 1,154 / 1,225 GWh (−6 %), Naples 2,558 / 2,493 (+3 %), South
Westphalia 12,137 / 13,750 (−12 %).

### Cross test of the logit parameters

Each city run with the parameters of the others (`docs/causal/sd_cross_test_logit.csv`): donor logit scale and
adjustment, effective lever sensitivities scale × factor (15-minute city: factor), as from one donor or as the
geometric mean of the two others; the target keeps its base year, populations, value of time and coping
types. Normalising the scale by the mean car generalised cost (7.7–9.0 EUR per trip) changes little.

| target | own: effects RMS (points) / car opt 2050 | mean of others: effects RMS / car opt 2050 | ABM light car opt 2050 |
|---|---|---|---|
| Tallinn | 1.0 / 0.236 | 6.2 / 0.321 | 0.242 |
| Naples | 1.0 / 0.266 | 6.6 / 0.302 | 0.251 |
| South Westphalia | 1.2 / 0.546 | 2.7 / 0.478 | 0.591 |

Single-measure effects 2040 with the mean of the others (SD / ABM light, %): PT optimisation −3.0 / −2.6
(Tallinn), −4.2 / −4.3 (Naples), −1.1 / −0.7 (South Westphalia); 15-minute city −2.0 / −2.1, −1.8 / −0.4,
−2.0 / −1.7; parking −7.1 / −3.0, −2.8 / −3.3, −1.5 / −5.3. Pricing fails: area toll −6.0 / −20.7 (Tallinn),
−18.8 / −5.6 (Naples); road pricing −0.3 / −6.1 (Tallinn), −0.2 / −13.3 (Naples). The same nominal toll and
charge per km give ABM light effects that differ by a factor of 4 to 10 between the cities, so the pricing
factors carry the scope of each scheme (share of car trips or km charged). Tallinn's scale (0.25, a non-linear
response) transferred to South Westphalia removes most car traffic (car share 0.12); the near-linear scales of
Naples and South Westphalia transfer better to each other.

### Pricing proxies (D53)

ABM light 2040, road pricing (`abm_base_year/abm_light_pricing_proxies.csv`):

| | charge (EUR/km) | in-area car vkm per car trip | charge per car trip (EUR) | car trips | per EUR per trip | in-area car vkm |
|---|---|---|---|---|---|---|
| Tallinn | 0.30 | 8.3 | 2.49 | −6.1 % | −2.5 % | −7.2 % |
| Naples | 0.30 | 7.0 | 2.10 | −13.3 % | −6.3 % | −17.7 % |
| South Westphalia | 0.15 | 8.9 | 1.34 | −1.4 % | −1.0 % | +2.9 % |

The response per EUR differs by a factor of six between the cities at similar charged km per trip. In South
Westphalia the in-area car vkm rise under road pricing. If the area toll met the same response per EUR as road
pricing, the 2040 toll of 15 EUR per entry would affect about 56 % of the car trips in Tallinn (−20.7 %) and 6 %
in Naples (−5.6 %).

### Energy check and the OD share

Car energy 2019 with OD traffic counted 0 / 0.5 / 1.0 (GWh): Tallinn 655 / 1,154 / 1,654 (MAED-City 1,225), Naples
788 / 2,558 / 4,327 (2,493), South Westphalia 4,620 / 12,137 / 19,653 (13,750). MAED-City corresponds to OD shares
of 0.57, 0.48 and 0.61.

### Transfer to Granollers (no ABM light)

Granollers run with the logit mode choice and transferred parameters against its previous flow model
(`docs/causal/sd_granollers_transfer.csv`); base year from its trip rates, speed tables, access times and
mode-specific lengths, value of time 13.5 EUR/h (HEATCO D5, Spain, updated to 2019).

| car share of pkm 2050 / effect of all measures on car trips 2040 | reference | optimised | all measures 2040 |
|---|---|---|---|
| previous flow model | 0.387 | 0.068 | −58 % |
| logit, Naples and South Westphalia | 0.578 | 0.559 | −3 % |
| logit, all three | 0.573 | 0.552 | −5 % |
| logit, Tallinn | 0.491 | 0.431 | −29 % |

The previous model loses a third of its car share without measures and nearly all with them; street greening
(−25 %), road pricing (−16 %) and car-passenger attractiveness (−10 %) dominate its single effects. Street
greening acts in the reference too (not scaled by the PI), and cycling and car passenger act on the
attractiveness, so their factors from the ABM light cities do not apply. The near-linear parameters of Naples and
South Westphalia barely move Granollers, whose short trips (3.8 km by car) and low parking cost (0.18 EUR) give
small cost changes; the Tallinn parameters give −29 %, within the range of the ABM light cities (all measures 2040:
Tallinn −35.6 %, Naples −24.2 %, South Westphalia −10.5 % car trips).

### Coping-type shares for regions without a survey (D54)

Leave-one-out over the six surveyed countries (shares estimated from the other five, bandwidth by inner
leave-one-out, at least 1): mean absolute error per type 1.9–6.9 points (mean 3.9) against 4.6–9.1 points (mean
7.0) for the plain mean of the other countries; Aitchison distance 0.38–1.08 against 0.86–1.61. The settlement
offsets (metropolitan, urban, rural: demonstrator column minus national sample) rest on one or two pairs; the
metropolitan offset from Naples worsens Tallinn (Aitchison 0.56 against 0.41 without offset) and the one from
Tallinn improves Naples (0.56 against 0.70), so the offset is an option, not a default.

### Transfer by semi-elasticities (D55)

Semi-elasticities 2040 (% car trips per unit; Tallinn / Naples / South Westphalia): road pricing per EUR per trip
−2.4 / −6.4 / −1.1, parking per EUR per trip −10.2 / −16.6 / −17.8, PT time per 100 % reduction −17 / −28 / −5,
cycling resistance −3.6 / −5.0 / −1.2, capacity −21 / −8 / −6.

Cross test, each city from the other two (`docs/causal/sd_cross_test_semi_elasticities.csv`), against the
transfer of scale × factor (mean of others):

| target | effects RMS (points) | car share of pkm optimised 2050 | ABM light |
|---|---|---|---|
| Tallinn | 5.8 (6.2) | 0.259 (0.321) | 0.242 |
| Naples | 6.5 (6.6) | 0.277 (0.302) | 0.251 |
| South Westphalia | 2.2 (2.7) | 0.250 (0.478) | 0.591 |

The rule improves the two cities and fails for South Westphalia: its ABM light effects are weak throughout
(all measures 2040 −10.5 % against −36 % and −24 %), and both donors are cities. The semi-elasticities differ by
region type more than by measure intensity, so the donors must resemble the target.

Granollers (`docs/causal/sd_granollers_transfer.csv`): transferred effects 2040 road pricing −3.7 %, PT −2.5 %,
traffic calming −1.7 %, cycling −1.2 % (reached −0.3 %; the factor hits the time floor), car occupancy −0.9 %,
parking −0.5 % (base parking cost 0.18 EUR); all measures −9.8 % against −58 % in the previous flow model. Car
share of pkm 2050: reference 0.584 (previous 0.387), optimised 0.512 (previous 0.068).

### Similarity-weighted donors and transfer band (D56)

Cross test of the single-measure effects 2040 (RMS, points; weighted / equal donors): Tallinn 6.5 / 5.6, Naples
6.2 / 6.5, South Westphalia 1.6 / 2.2. The weighting changes little: Tallinn and Naples have nearly the same profile
but differ in their response to pricing, and South Westphalia has no polycentric donor. The range of the
single-donor transfers contains the ABM light effect in 14–50 % of the measures; the ratio ABM light / transferred
has the 10 / 90 % quantiles 0.27 / 3.47 (median 0.61), used as transfer band.

Granollers (weights Tallinn 0.71, Naples 0.29, South Westphalia 0; scale 0.078, adjustment 0.77): all measures
2040 −12.5 % car trips (band −3 % to −39 %), car share of pkm 2050 optimised 0.502 (0.569 to 0.292), reference
0.591.

### Integrated results after the mobility revision

Energy and emissions of the current SD branch against the state before the mobility revision (commit f349f68,
after D37) and the MAED-City pathway (`docs/causal/sd_integrated_comparison.csv`). Direct CO2, optimised pathway
(PI = 1), 2019 / 2030 / 2040 / 2050, kt:

| | before | now | MAED-City | now / MAED-City |
|---|---|---|---|---|
| Tallinn | 651 / 514 / 248 / 50 | 627 / 435 / 243 / 56 | 905 / 525 / 241 / 47 | 0.69 / 0.83 / 1.01 / 1.19 |
| Naples | 1,387 / 976 / 538 / 187 | 1,341 / 905 / 486 / 173 | 1,360 / 920 / 503 / 199 | 0.99 / 0.98 / 0.97 / 0.87 |
| South Westphalia | 9,380 / 6,394 / 2,638 / 300 | 8,832 / 5,894 / 2,397 / 208 | 9,380 / 5,921 / 2,236 / 252 | 0.94 / 1.00 / 1.07 / 0.82 |
| Granollers | 176 / 130 / 63 / 18 | 182 / 139 / 76 / 27 | – | – |

MAED-City values follow from the earlier ratios (before / MAED-City, rounded to two digits). Total CO2 2050
(PI = 1): Tallinn 139 → 144, Naples 212 → 198, South Westphalia 306 → 211, Granollers 23 → 33 kt. Final energy
changes by −1 to −3 % (Granollers +2 to +6 %). The faster response of the logit mode choice lowers Tallinn's
2030 transport emissions (246 against 325 kt). Granollers' emissions rise because the previous model removed most
car traffic (car share of pkm 2050 0.068 against 0.50).

### Granollers: transfer band through energy and emissions

The three lever-factor sets of the transfer band (weak end, central, strong end) run through the full model
(`docs/causal/sd_granollers_band_integrated.csv`). Optimised pathway 2050:

| | reference | weak end | central | strong end | previous flow model |
|---|---|---|---|---|---|
| car share of pkm | 0.591 | 0.569 | 0.501 | 0.292 | 0.068 |
| car energy (GWh) | 108 | 104 | 93 | 57 | 17 |
| transport CO2 (kt) | 19.2 | 18.8 | 17.4 | 13.0 | 8 |
| direct CO2 (kt) | 28.7 | 28.2 | 26.9 | 22.5 | 18 |
| total CO2 (kt) | 34.8 | 34.4 | 33.0 | 28.5 | 23 |
| final energy (GWh) | 1,362 | 1,359 | 1,349 | 1,318 | 1,273 |

The mobility measures change Granollers' total emissions 2050 by 1–18 % against the reference; the decline from
312 kt (2019) to about 33 kt comes from fleet electrification and energy supply.

### Energy levers and the MAED-City BAU (D58)

BAU growth rates per sector group (per year): Tallinn households 0.16 %, services 0.96 %, industry 1.37 %,
freight and other −0.48 %; Naples −1.0 %, 0.8 %, 0.9 %, −0.4 %; South Westphalia buildings −0.35 %, industry
0.71 %, freight and other −0.15 %. Reference pathway (PI = 0) against the MAED-City BAU, 2050:

| | final energy SD / MAED-City BAU (GWh) | electricity share SD / MAED-City | direct CO2 2019 → 2050 SD (kt) | MAED-City BAU CO2 2019 → 2050 (kt) |
|---|---|---|---|---|
| Tallinn | 9,138 / 9,770 | 0.40 / 0.41 | 627 → 580 | 901 → 994 |
| Naples | 7,963 / 8,490 | 0.38 / 0.37 | 1,341 → 876 | 1,369 → 1,144 |
| South Westphalia | 45,369 / 50,200 | 0.23 / 0.19 | 8,832 → 7,262 | 9,340 → 8,347 |

The gap in final energy lies in passenger transport: the SD reference electrifies the car fleet (EV share 2050
43 / 46 / 52 %, efficiency three times higher), while MAED-City's BAU keeps fossil cars (passenger transport 2050
Tallinn 870 against 1,466 GWh, South Westphalia 7,769 against 12,600 GWh). Total CO2 2050 (reference / optimised):
Tallinn 793 / 144, Naples 2,161 / 198, South Westphalia 7,400 / 211 kt; before D58 the reference used the CETS
energy system (Tallinn 282 kt).

### Reference EV fleet from the MAED-City BAU (D59)

Reference pathway 2050 after D59 against the MAED-City BAU: final energy Tallinn 9,436 / 9,770 GWh (−3.4 %),
Naples 8,783 / 8,490 (+3.5 %), South Westphalia 49,876 / 50,200 (−0.6 %); passenger transport Tallinn 1,168 /
1,466 GWh, South Westphalia 12,277 / 12,600 GWh; direct CO2 Tallinn 694 / 994 kt (base-year gap: MAED-City counts
the DH gas boilers), Naples 1,068 / 1,144 kt, South Westphalia 8,378 / 8,347 kt. Mobility recalibration (squared
error pathways / effects): Tallinn 0.0036 / 0.0023, Naples 0.0069 / 0.0006, South Westphalia 0.0197 / 0.0011;
car share of pkm 2050 reference / optimised: Tallinn 0.420 / 0.237 (ABM light 0.434 / 0.242), Naples 0.475 / 0.265,
South Westphalia 0.678 / 0.546.
