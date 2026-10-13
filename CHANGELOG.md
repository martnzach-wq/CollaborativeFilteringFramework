# Changelog

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
