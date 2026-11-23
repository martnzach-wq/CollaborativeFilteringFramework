# Literature-derived causal hypotheses

Outputs: `causal_candidates_llm.csv` (54 hypothesised edges), `causal_edges_literature.csv` (121 study claims after screening), `causal_edges_prior.csv` (aggregated edges with grades), `literature_handcheck.csv`, `causal_candidate_graph.csv` (seed + literature, used in the empirical test), `fig_evidence_map.png`.

## Procedure

1. **Hypothesis generation.** A language model, acting as an imperfect expert (Long et al., 2023), proposed 54 directed edges among the 26 time-varying regsim indicators and six D3.6 context indicators, excluding the 5 pairs already quantified by the seed sources. For each edge it gave an expected sign, mechanism, typical lag, likely confounders, a search query and a prior plausibility. Four edges that hold physically by definition (temperature → degree days, elevation → heating degree days) were not searched; five demographic accounting relations (e.g. fertility → share aged 0–14) are kept as identities.
2. **Search.** OpenAlex (articles since 1995 with abstract), 10 results per edge; a second round with two alternative queries for the 27 edges with fewer than two relevant claims. 870 abstracts in total.
3. **Screening and extraction.** One structured extraction per abstract (relevance, direction, sign, magnitude, elasticity, lag, spatial level, geography, method, identification, Europe relevance, proxies).
4. **Verification.** A second, independent pass with a stronger model re-derived sign, design, claim strength (causal claim, association, disclaimed) and match of measured quantities. Agreement with pass 1: sign 83%, design 77%. Verified values are used.
5. **Grading.** Each claim is weighted by design (experiment 3, quasi-experiment or meta-analysis 2.5, panel fixed effects 2, time series 1.5, cross-section 1, simulation 0.75, descriptive 0.5) × spatial level (regional/national 1, city 0.8, individual 0.7) × Europe relevance (1 or 0.6) × claim strength (1, 0.7, 0.3) × quantity match (1, 0.8, 0.4). Per edge: consensus sign, sign consistency (weight share of the consensus sign), and grade A (weight ≥ 4, consistency ≥ 0.75, at least one quasi-experiment/meta-analysis or three panel studies), B (weight ≥ 2, consistency ≥ 0.7), C (consistency ≥ 0.6), D (no study or conflicting evidence).
6. **Hand check.** Six claims, stratified over grades and designs, were checked against full texts where open access (3 of 6) or against the abstract (`literature_handcheck.csv`).

## Results

- Of 45 non-identity edges, 4 reach grade A (GDP → life expectancy, household income → cars per 1,000, renewables share → GHG per capita, GDP → renewables share), 8 grade B, 15 grade C and 18 grade D.
- Where the literature gives a consensus sign, it agrees with the language-model prior in 28 of 31 edges. The exceptions are GDP → employment rate, net migration → median age and unemployment → Gini; the last two are classic cases where the sign depends on context (age structure of migrants, tax-benefit system).
- Elasticities are rarely stated in abstracts: income → car ownership 0.47 (also GDP → cars), heating degree days → electricity demand 0.46, old-age dependency → GDP −0.55, GDP → life expectancy 0.05.
- Grade D has two different meanings: no study at all (employment → household income, GDP → R&D, density → broadband, density → hospital beds; mostly accounting or supply relations studied under other names) and conflicting studies (net migration ↔ unemployment, GDP → fertility, unemployment → Gini). The second group marks genuine heterogeneity, which the regional panel can test.

## Reliability and limits

- Pass 1 got 1 of 6 hand-checked signs wrong (rail access → car ownership read as "+" from "influenced"); the verification pass corrected it. One further claim had a misclassified design, and one study disclaims causality in its own abstract. Two-pass extraction with a check of the sign definition is therefore necessary.
- Relevance-ranked search favours recent and regional-case papers over seminal studies and meta-analyses; ranking by citations or adding known meta-analyses would raise grades for well-studied edges (e.g. income elasticity of car ownership, Okun-type GDP–employment relations).
- Abstract-level extraction misses most magnitudes. Full-text extraction for grade A/B edges is the next step if strength priors are needed.
