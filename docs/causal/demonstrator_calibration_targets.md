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
