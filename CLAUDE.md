# CLAUDE.md — conventions for the regsim repository

## Language and documentation
- British English everywhere: code comments, docstrings, documentation, figure labels (modelling, normalisation, behaviour, centre, colour).
- State choices and results directly; give at most one reason where a reader would ask for it. No defensive text about limitations in docstrings; limitations go to `docs/limitations.md`.
- Headings are neutral and descriptive.

## Code
- Python ≥ 3.12, `src/` layout, type hints throughout, `ruff` for lint and format, NumPy-style docstrings.
- Core dependencies: numpy, scipy, pandas, pyarrow, duckdb, scikit-learn, matplotlib, pyyaml, pydantic. Optional extras: `geo` (geopandas, shapely), `viz` (plotly, umap-learn), `api` (fastapi, uvicorn), `torch` (only if decision D6 allows it).
- No hard-coded geo-schemes: code paths refer to `GeoScheme` and the hierarchy graph, never to NUTS string prefixes.
- No thematic indicator classes inside models, similarity or typology. `tags` are metadata for search and lens construction only.
- Every estimator implements `regsim.models.base.Estimator`. Every random operation takes a `seed` or `numpy.random.Generator`.
- Configs are YAML validated by pydantic models in `regsim.config`; every run writes its resolved config and a hash next to its outputs.
- Figures: `fig, ax = plt.subplots()`; `fig.savefig(...)`; never `plt.savefig`. Each embedding view reports explained variance or trustworthiness.

## Data
- Raw downloads go to `data/raw/` (git-ignored), processed observations to `data/store/` as Parquet. Never commit data files above 5 MB.
- Every observation keeps `source`, `source_version` and flags.
- Values are stored on the original scale; transforms are applied at model time from the indicator registry.

## Tests
- `pytest`; unit tests per module, acceptance tests per milestone in `tests/acceptance/` (marked `skip` until the milestone is started).
- Synthetic data from `regsim.synthetic` for model tests; no network access in unit tests (connectors are tested against recorded responses in `tests/fixtures/`).

## Working rules
- Ask the user before deciding any item listed in `HANDOVER.md` §4, and record the answer in `docs/decisions.md`.
- Keep `docs/progress.md` current at the end of each milestone: what was done, test status, open issues.
- Do not change acceptance criteria without asking.
