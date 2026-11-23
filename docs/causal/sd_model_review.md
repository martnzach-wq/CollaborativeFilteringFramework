# KNOWING SD model: Granollers elasticities and review of the Tallinn anomalies

## Runs

The KNOWING integrated Python model was run for the TALLINN and GRANOLLERS configurations (`city_config.py`), BAU and full policy intervention (PI = 1), with the same perturbations as before:
- drivers scaled by +10 %, elasticity at 2050;
- each policy lever removed under PI = 1, giving the relative change at 2050.

The NAPLES configuration could not be run because its input file EN_MEADCityInput_Naples.xlsx is not in the repository. Results are in sd_city_elasticities.csv; the diagnostic runs are in sd_anomaly_tests.csv.

## Driver elasticities, BAU 2050

| Driver | Output | Tallinn | Granollers | Reference |
|---|---|---|---|---|
| fuel car cost per km | car-km per capita | +0.01 | −0.28 (constant cost input) | −0.3 long run (Goodwin, Dargay and Hanly 2004) |
| fuel car cost per km | active-mode km per capita | −0.34 | +0.03 | positive expected |
| cooling degree days | car-km per capita | 0.00 | +0.93 | positive, small (heat reduces walking and cycling) |
| cooling degree days | active-mode km per capita | 0.00 | −0.35 | negative |
| parking cost | car-km per capita | 0.00 | −0.06 | negative |
| value of time | car-km per capita | 0.00 | +0.51 | positive |
| population (internal only) | car-km per capita | −0.58 | −0.71 | ≈ 0 for a per-capita output |
| population (internal only) | total emissions | −0.47 | −0.19 | ≥ 0 |
| grid emission factor | total emissions | +0.17 | +0.16 | positive |

Lever effects (PI = 1, lever removed, car-km per capita 2050):

| Lever | Tallinn | Granollers |
|---|---|---|
| road pricing | −0.42 | −0.72 |
| area toll | −0.73 | not configured |
| street green | not configured | −0.68 |
| cycling attractiveness | −0.05 | −0.21 |
| car-passenger attractiveness | +0.04 | +0.18 |
| parking cost adaptation | −0.03 | −0.01 |
| PT travel time adaptation | −0.02 | −0.02 |

Granollers responds to all drivers with the expected signs. Fuel cost (−0.28) is close to the long-run literature value. Tallinn shows the three anomalies reported earlier, and the responses to parking cost and value of time are zero there.

## Causes of the Tallinn anomalies

| Item | Finding | Test | Type |
|---|---|---|---|
| R1 Cooling degree days, parking cost: no effect | Tallinn applies a GC-neutral base-attractiveness calibration at 2019 (`apply_cdd_baseattr_compensation = True`; `MobilityModel._compute_baseattr_calibration`). The calibration absorbs every input that is constant over time, so the level of CDD or of parking cost cannot change mode shares. Granollers has no such calibration. | With the calibration switched off, Tallinn responds with the expected signs: CDD → active km −0.13, parking → car-km −0.08 | model property (design choice) |
| R2 Fuel car cost → active-mode km −0.34 | The calibration is anchored at the 2019 fuel cost. Scaling the whole fuel-cost series also scales the 2019 anchor, so only the post-2019 path acts. As the EV share rises, the weight of fuel cost in car cost per km falls. A higher fuel-cost level therefore makes car travel relatively cheaper over time compared with the calibrated start, which reduces active travel. | Calibration off: fuel cost → car-km −0.11, active km +0.09 | model property; elasticities of time-constant inputs are not interpretable in Tallinn |
| R3 Value of time: no effect | The Tallinn branch of `_generalised_cost` hard-codes `value_of_time_base = 15` and ignores the configured value (mobility.py, line 508). | Code inspection | implementation issue |
| R4 Population → total emissions −0.47 | Per-capita distance is computed as (internal + origin–destination trip distance) / internal population, and energy demand multiplies per-capita km by a separate population series from the energy input workbook. Scaling only `population_internal` therefore lowers per-capita km, and with it total transport energy, while the energy model's population stays fixed. | Scaling internal and OD population together: car-km per capita +0.015, total emissions +0.012 | perturbation artefact plus a model property: total emissions do not respond to population, because energy population is an exogenous input not linked to the mobility population |
| R5 Granollers fuel cost perturbation zero in the first run | Granollers uses `car_cost_per_km_fuel_constant`; the policy series is empty. | Perturbing the constant gives −0.28 | perturbation set-up |

## Items for the modellers

1. **Calibration and transfer.** The GC-neutral calibration makes Tallinn insensitive to time-constant inputs. Elasticities of level inputs (climate, parking cost, fuel cost level) from Tallinn should not be transferred to other cities or compared with cross-sectional data. Possible actions:
   - recalibrate with the inputs held at a reference level and expose them as deviations from that reference;
   - or document that only changes after 2019 act.
2. **Value of time in the Tallinn branch.** Read `m.value_of_time_base` instead of the constant 15, or document the constant.
3. **Population consistency.** Link the energy-model population to the mobility population, or provide one population input. Scale internal and OD population together in scenario runs.
4. **Fuel-cost input harmonisation.** Tallinn uses a cost series, Granollers a constant. A common input definition would make scenario and perturbation runs comparable.
5. **Naples input file.** EN_MEADCityInput_Naples.xlsx is referenced by the configuration but missing from the repository.

## Consequences for the regsim SD bridge

- **Edge import.** Only edges whose perturbation responds in both cities, or whose zero response has a documented cause, are imported as `model` evidence. The Tallinn level-input elasticities (R1, R2) are imported with status `model_review` and a note on the calibration.
- **Mapping.**
  - Granollers fuel cost → car-km (−0.28) agrees with the literature and can serve as a model-based strength prior for a fuel-price edge once a regional fuel-price indicator is added.
  - CDD → active-mode km (−0.35 in Granollers) supports the literature edge from heat days to active travel.
  - Road-pricing and area-toll effects are city-specific lever effects. They go to the intervention table (D3.6 interventions), not to the indicator graph.
- **Perturbation protocol.** The bridge should:
  - perturb consistent groups of inputs (all population inputs together);
  - perturb constants and series alike;
  - run with and without the calibration where a calibration exists;
  - report all of these in the `compare` output.
