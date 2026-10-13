# Limitations

## Data layer (M1)

- **NUTS version correspondence.** Only NUTS 2021 → 2024 is supported (Eurostat change table). Values that Eurostat still publishes under older codes (NUTS 2016 and 2013, e.g. `BE221`, `HR04`, `FR2`, `NO01`) are dropped and listed in `build_report.json`; they concern earlier years of series that Eurostat republished under current codes.
- **Correspondence weights.** For changed regions the weights are LAU population overlaps where the LAU codes of both years cover at least 95 % of every changed region (Finland, Portugal); otherwise polygon overlaps of the 1:1 million boundaries weighted by 2021 population (Germany, Latvia, the Netherlands), which assumes uniform density within each 2021 region.
- **Aggregation rules in recoding.** `wmean_area` and `ratio` indicators are recoded with population weights; `none` indicators are recoded only where the mapping is 1:1.
- **United Kingdom.** NUTS 2024 has no UK regions, so UK data are dropped (decision D1).
- **Population.** Units without Eurostat population (BA, UA, XK) use area shares as edge weights.
- **LAU.** Only the LAU year matching the NUTS version is linked; LAU 2021 → 2024 changes are not mapped.
- **ARDECO.** NUTS-3 employment rates are experimental ARDECO estimates and enter the model as observations flagged `ardeco`. ARDECO has no GVA counterpart in the Eurostat registry, so GVA is not used. Restrict an entry with `source.levels` to exclude levels.
