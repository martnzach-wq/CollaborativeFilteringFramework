# Prototype reference

Run order (from this folder, with `eurostat_raw/` downloaded as described in `docs/data_sources.md`):
1. `build_matrix.build()` returns Z, mask, hierarchy and weights.
2. `bench.py` expects `Z, M, levels, parent, aggS, inds, native, geos` in the namespace and writes `prototype_benchmark.csv`.

The download specification (dataset codes and filters) is in `configs/indicators_eurostat_v1.yaml`.
