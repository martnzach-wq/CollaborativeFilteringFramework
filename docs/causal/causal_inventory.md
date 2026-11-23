# Inventory of existing causal knowledge

Outputs: `causal_edges_seed.csv` (481 edges), `sd_dependency_edges.csv` (structural dependency graph of the two published Vensim sub-models), `sd_tallinn_elasticities.csv` (perturbation runs of the Python SD model), `causal_seed_gaps.csv`, `fig_knowledge_graph.png`.

## Sources and what they contribute

| Source | Edges | Directed | Signed | Quantified | Mapped to regsim at both ends |
|---|---|---|---|---|---|
| KNOWING integrated SD model (Python, Tallinn), perturbation of 8 drivers and 10 levers on 8 outputs (2050) | 136 | yes | yes | yes (elasticities, lever effects) | 12 |
| Vensim mobility and energy-supply sub-models, equation dependency graph | 178 | yes | no | no | 0 (SD variable names) |
| LEVITATE D4.2 Fig. 1, goal relations | 23 | no | in goal space | no | 12 |
| LEVITATE D4.2 Sect. 2.1, cited literature | 4 | yes | yes | 2 elasticities | 4 |
| KNOWING D3.6 Annex A, intervention–indicator links | 318 | yes (intervention → indicator) | no | no | 0 (cause is an intervention) |

## Findings

1. **Coverage is thin.** Of 325 pairs among the 26 time-varying regsim indicators (25 Eurostat indicators plus D3.6 electricity demand), 19 are linked by any seed edge and 5 by a quantified, non-zero edge. Fifteen indicators have no edge at all: median age, old-age dependency, share aged 0–14, net migration, fertility, unemployment, employment rate, tertiary attainment, life expectancy, household income, R&D expenditure, hospital beds, broadband, lifelong learning and heating degree days.
2. **The SD model and the regsim registry describe different variables.** The SD model works with trips, modal split, EV share, car costs, travel time and policy levers; regsim holds statistical context indicators. Only 12 of 136 SD edges connect two regsim indicators. Adding modal split, EV share, final energy and sectoral emissions to the registry (Eurostat national and urban-audit series, D3.6 TRNPT06/TRNPT08/ENERG02–04 once populated) would make the SD structure testable.
3. **D3.6 Annex A is a moderator map, not a causal graph.** It lists which indicators an intervention acts on (direct) and which context conditions its effect (indirect). In the causal layer these become intervention nodes and effect modifiers, which is what transportability analysis needs.
4. **Model-implied signs need checking.** The perturbation runs give plausible effects for the levers (area toll −0.73 and road pricing −0.42 on 2050 car km per capita) but also results that contradict expectation: a 10 % higher fuel car cost lowers active-mode km (elasticity −0.34), cooling degree days have no effect on any output, and population lowers total emissions (−0.47), probably through per-capita normalisation. Such edges are exactly where data and literature have to arbitrate.
5. **LEVITATE goal relations are undirected and polarity-aligned.** They can enter as sign constraints on correlations (synergy or conflict between goals), not as directed effects. The figure caption swaps red and green; the reading consistent with the text (green = goals support each other) is used.
