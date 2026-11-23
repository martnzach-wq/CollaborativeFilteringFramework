# Transfer of SD calibration parameters from demonstrators to other regions

## Question

How confidently can a calibration parameter of the KNOWING SD model, such as value of time or base attractiveness of transport modes, be set for a region on the basis of one to three calibrated demonstrators?

## Findings

1. **The calibrated cities differ widely, and not all calibrations are independent** (sd_calibration_parameters.csv). Mode-change damping is 0.264 in Tallinn, 0.84 in Granollers and 0.0376 in Naples. Value of time is 15 (hard-coded in the Tallinn branch), 5.10 and 5.5. Naples' base attractiveness and coping-type distribution are close to Granollers'. In effect there are two independent calibrations.
2. **Parameters are defined within a model variant.** Tallinn uses a GC-neutral base-attractiveness calibration and flat attractiveness per mode; Granollers and Naples use attractiveness per mode and zone and a congestion term. Base attractiveness and damping absorb whatever the model does not represent, and several parameter combinations reproduce the same mode split.
3. **Between-region variation cannot be estimated from two or three regions.** With three regions, the 95 % interval for the between-region standard deviation spans 0.5–6.3 times the estimate, and no moderator can be estimated. With about 280 regions, only 3 of 7 effects showed heterogeneity that held up in held-out countries (effect_similarity.md).
4. **Most regions lie outside the demonstrators' coverage** (demonstrator_coverage.csv). The lens is GDP, household income, density, cars per 1,000, heating and cooling degree days, standardised.
   - 82 % of 231 NUTS-2 regions are outside the demonstrators' range on at least one dimension (GDP 53 %, cars 43 %, cooling degree days 24 %).
   - Median distance to the nearest demonstrator is 0.87, against 1.22 between two random regions.
   - Domain classes: close (< 0.5) 10 regions, moderate (0.5–0.75) 53, distant (0.75–1.0) 92, extrapolation (≥ 1.0) 73.
5. **Lever rankings agree, levels do not.** Effect on car-km per capita, Tallinn vs Granollers:
   - road pricing −0.42 vs −0.72;
   - cycling attractiveness −0.05 vs −0.21;
   - parking adaptation −0.03 vs −0.01.

   The order of the two strongest levers is the same in both cities; the sizes differ by up to a factor of four.
6. **One additional demonstrator would double coverage.** A western or central European regional city region raises the share of regions within 0.75 of a demonstrator from 0.29 to about 0.58 (next_demonstrator_candidates.csv; best candidates DED2 Dresden, DE94 Weser-Ems, FRF1 Alsace, NL13 Drenthe).

## Protocol

### Parameter classes

`configs/sd_parameter_classes.yaml` assigns every model parameter to one class.

| Class | Examples | Rule for a target region | Confidence |
|---|---|---|---|
| O, observable input | population (internal and OD), trip rates, fuel and EV cost, parking fees, PT fares, degree days, network capacity | take from data; where missing, impute with regsim and propagate the prediction interval | data quality or regsim interval |
| B, behavioural with external evidence | value of time, cost sensitivities | function from literature evidence (value of time from income and trip characteristics, Wardman, Chintakayala and de Jong 2016, doi:10.1016/j.tra.2016.08.019); demonstrators used as checks | medium; interval from the meta-analysis |
| R, residual calibration constant | base-attractiveness corrections, mode-change damping, coping-type distribution, GC-neutral correction | never transferred; recalibrated to the target region's observed or regsim-imputed mode split and trip-length distribution | that of the calibration targets; widened when targets are imputed |
| S, structural switch | `apply_cdd_baseattr_compensation`, generic vs Tallinn branch, congestion on/off | one model variant for all transfers; never mixed between donor and target | n/a |

### Steps for a non-demonstrator region

1. **Domain check.** Compute the distance to the nearest demonstrator in the causal lens of the SD outputs. Report the domain class (close, moderate, distant, extrapolation) with every result.
2. **Class O.** Assemble inputs from the store, using regsim imputations with intervals where values are missing.
3. **Class B.** Set behavioural parameters from the literature function, with interval.
4. **Class R.** Recalibrate to target-region calibration targets (mode split, trip lengths). If these are imputed, recalibrate across their distribution, sampling from the regsim predictive distribution.
5. **Uncertainty propagation.** Run the model over joint draws of classes O, B and R targets.
6. **Reporting.**
   - Lever rankings and directions are reported for all domain classes.
   - Absolute levels are reported only for close and moderate regions, or where class R has been recalibrated to observed (not imputed) targets.
   - Every result carries the domain class and the share of imputed inputs.

### Validation

- **Cross-calibration.** Calibrate on one demonstrator, transfer by the protocol to another, and compare with its observed mode shares and its own lever responses (six directed pairs with three cities). Report the error by parameter class and the agreement in lever ranking. The result indicates the order of magnitude of transfer error; it is not a confidence level for other regions.
- **Between-region spread for hierarchical priors** comes from outside the demonstrators: literature, other calibrated models, or effect-similarity results. It is never estimated from two or three points.

### Demonstrator selection

`sd_bridge.next_demonstrator` ranks candidate regions by the reduction in median distance to the nearest demonstrator. The same function can rank candidate cities from partner projects for which calibrated models exist.

## Files

- sd_calibration_parameters.csv: calibration parameters of the three city configurations
- demonstrator_coverage.csv: distance to the nearest demonstrator, outside-range flags and domain class per NUTS-2 region
- next_demonstrator_candidates.csv: candidates ranked by coverage gain
