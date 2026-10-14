# Changelog

## 0.7.0 (M6)
- Prediction tasks P1–P7; views V1–V8 and diagnostic views (matplotlib and plotly); `regsim report`, `fit`, `predict`, `serve`, `app`.
- REST service (FastAPI) with pydantic response models; Dash explorer with linked map and embedding (decision D7).
- LAU extension (`configs/lau.yaml`) and `level_transfer_benchmark`.
- Fixes: D3.6 baseline values stored as observed; D3.6 electricity demand aggregated as area-weighted mean; "days per year" indicators log-transformed; LAU rows kept by their LAU-list population; model cache keyed by registry content.

## 0.6.0 (M5)
- Rule engine with `configs/rules_d36.yaml` (G001–G010, Table 10 and published severities, derived Köppen adjacency); gates in the D3.6 mode and in `regsim similar --mode d36`; per-tier coverage in `D36Scorer`; `load_values`, `published_gates`, `apply_gates`.
- `Conformalised` (Mondrian split-conformal intervals), `applicability` (applicability domain).
- Typology runs (GMM, Ward, HDBSCAN, spatial Ward) with stability, labels, tracking and configuration hash; `regsim typology`.
- Decision D9: gates are soft in regsim mode.

## 0.5.0 (M4)
- D3.6 compatibility mode with inferred scoring rules (`configs/d36_scoring.yaml`); entity similarity S-profile, S-latent, S-gower with percentile scores, contributions and intervals; transfer-specific weights; indicator similarity I-model, I-observed, I-partial, I-predictive; neighbour-prediction skill; `regsim similar`.
- `HierTypedMF.predict_dist(noise=...)`.

## 0.4.0 (M3)
- `HierTypedMF` with typed losses, hierarchical prior, aggregation constraints, side information, Laplacian and random-walk penalties, Laplace posterior, fold-in, MinT-type reconciliation; `Tuned` nested CV; geo-time `prepare`; `configs/benchmark_m3.yaml`.
- `ModelData.parent` holds membership fractions; ratio indicators aggregate with their denominator's weights.

## 0.3.0 (M2)
- Model matrix preparation, seven baselines, masking schemes M1–M4, metrics, benchmark runner, model card, `regsim evaluate`, `configs/benchmark_prototype.yaml`.
- M2 acceptance: cold-start tolerance ±0.10 (see `docs/decisions.md`).

## 0.2.1
- ARDECO connector and fallback registry (GDP per inhabitant, employment rate, population); shared cached HTTP client.

## 0.2.0 (M1)
- Eurostat connector, NUTS/LAU 2024 geo registry with NUTS 2021 → 2024 correspondence, indicator registry with source links, Parquet/DuckDB observation store, D3.6 GeoPackage adapter, `regsim build`.
- Registry: `fertility_rate` filtered to `TOTFERRT`; D3.6 registry for the 41 populated indicators.
- M1 acceptance criterion adapted to NUTS 2024 (see `docs/decisions.md`).

## 0.1.0 (M0)
- Synthetic generator `regsim.synthetic.generate_hierarchical` with known factors, mixed types, fractional parents, period random walk and missingness schemes MCAR, block-MNAR and level-missing.
- Decisions D1 (NUTS/LAU 2024), D2 (Apache-2.0), D3 (two period windows) and D6 (PyTorch core) applied.
- Ruff, pre-commit and GitHub Actions (Linux, Windows).

## 0.0.1
- Hand-over skeleton, specification and prototype reference.
