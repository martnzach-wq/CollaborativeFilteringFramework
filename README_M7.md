# regsim M7 causal addendum (version 3)

Unpack at the repository root. Files under `src/`, `configs/` and `tests/` are new and do not overwrite existing modules. Start with `HANDOVER_M7_causal.md`.

## Topics and documents

| Topic | Document (docs/causal/) | Milestone |
|---|---|---|
| Methods review | causal_related_work.md | background |
| Existing causal knowledge (SD models, LEVITATE, KNOWING D3.6) | causal_inventory.md | M7.1 |
| Literature hypotheses, round 1 | literature_hypotheses.md | M7.2 |
| Design of the causal layer | causal_layer_design.md | all |
| Panel tests, data screen, forecasts, heterogeneity | causal_test_notes.md | M7.3 |
| Questions the layer answers | causal_usefulness.md | all |
| Main report | causal_extension_report.md | all |
| Temporal precedence | causal_extension_addendum.md §5 | M7.3 |
| Effect similarity | effect_similarity.md | M7.6 |
| Literature round 2 (strict measure check, horizons) | literature_round2.md | M7.2 |
| SD model review (Tallinn, Granollers) | sd_model_review.md | M7.4 |
| SD parameter transfer to other regions | sd_parameter_transfer.md | M7.4 |
| Regional data for national-only indicators | regional_data_gaps.md | data |
| Summary of version-2 and version-3 results | causal_extension_addendum.md | all |

## Configuration

- `configs/causal_edges.yaml`: edge registry, schema 0.2, 91 edges
- `configs/indicator_reference_periods.yaml`: reference periods for timing tests
- `configs/sd_parameter_classes.yaml`: classes of KNOWING SD parameters for transfer
