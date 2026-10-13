# Decisions

Record answers to HANDOVER.md §4 here (date, decision, by whom).

| ID | Date | Decision | By |
|---|---|---|---|
| D1 | 2026-10-13 | Harmonise geo codes to **NUTS 2024** and **LAU 2024**. Data published under NUTS 2016/2021 are recoded via the GISCO correspondence tables. | M. Zach |
| D2 | 2026-10-13 | Repository `martnzach-wq/CollaborativeFilteringFramework`; package name **`regsim`**; licence **Apache-2.0**. | M. Zach |
| D3 | 2026-10-13 | Time enters v1 with **two period windows**, 2013/2017 and 2018/2022, as geo-time entities with a random-walk penalty (design §4.3). | M. Zach |
| D4 | open | Default lens; whether polarity is shown | |
| D5 | open | Sources beyond Eurostat in v1 | |
| D6 | 2026-10-13 | **PyTorch is a core dependency.** `HierTypedMF` uses autograd for the typed losses and the Laplace posterior. | M. Zach |
| D7 | open | Interactive front end | |
| D8 | open | TRF classes for regsim similarities | |
| D9 | open | Severity of D3.6 gates in regsim mode | |

## Consequences

- D1: the M1 acceptance criterion (1,929 units, 18,863 cells, ±2 % after code harmonisation) is evaluated after recoding the prototype matrix to NUTS 2024.
- D3: the M1 build and the example config cover 2013–2022; the prototype reference numbers refer to the 2018/2022 window only.
- D6: `torch` moves from the optional extras into the core dependencies. CI installs the CPU wheels.
