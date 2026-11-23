# Design of a causal layer for regsim

Basis: regsim v0.8.0 (`martnzach-wq/CollaborativeFilteringFramework`), the method review (`causal_related_work.md`), the inventory of existing causal knowledge (`causal_inventory.md`) and the literature-derived edge priors (`literature_hypotheses.md`). `fig_causal_architecture.png` shows the components and the two directions of information flow.

## 1 Aim

regsim represents indicators by their joint, symmetric structure: shared latent factors, model-implied and partial correlations, and a random walk between two period windows. The causal layer adds directed, signed and, where possible, quantified relations between indicators, interventions and SD-model variables. It works in two directions:

- **knowledge → data:** relations from SD models, project deliverables and literature constrain and explain the statistical model, and allow what-if propagation;
- **data → knowledge:** regional panels test the sign and estimate the strength and heterogeneity of these relations, and propose new candidate relations, which are sent back to the literature search and to the SD modellers.

No static classification of indicators is introduced. The causal graph is a set of hypotheses with provenance and evidence grades; the "parents" of an indicator in the graph can be used as a lens, but the lens remains a user choice.

## 2 Causal knowledge registry

### 2.1 Nodes

| Node type | Examples | Link to regsim |
|---|---|---|
| Indicator | `cars_per_1000`, `ghg_per_capita` | registry indicator id |
| Context (time-invariant) | D3.6 climate, terrain, network densities | registry indicator id, cause only |
| Intervention | area toll, street greening, building retrofit (D3.6 Annex A; SD levers) | event table (region, start year, intensity) |
| SD variable | `tr percentageofcarownership`, `TR ModalSplit` | mapping table (SD variable ↔ indicator, transform, scale, level) |
| Latent | development level, national institutions | regsim factors u_g (not named) |

### 2.2 Edges

One record per relation, with all sources kept:

| Field | Content |
|---|---|
| `edge_id`, `cause`, `effect` | node ids |
| `kind` | mechanism, accounting identity, intervention effect, effect modifier, goal relation (undirected) |
| `sign_prior` | probability that the effect is positive (from consensus and consistency; 0.5 = unknown) |
| `strength_prior` | distribution of the elasticity or standardised coefficient (mean, sd, unit), empty if unknown |
| `lag_prior` | years (distribution or class: < 1, 1–3, 3–10, > 10) |
| `validity` | levels, geography, region types, period in which the evidence holds |
| `sources` | list: source type (SD model, deliverable, literature, expert, LLM hypothesis), reference (DOI, model file and version), design, weight |
| `grade` | A–D (literature), "model-implied", "identity", "hypothesis" |
| `status` | hypothesis, supported, contested, contradicted, rejected; set by the data tests (§4) |
| `data_evidence` | latest test results: within-region coefficient and interval, sign agreement, invariance p-value, heterogeneity τ |

The registry is a YAML file under version control (`configs/causal_edges.yaml`) plus a Parquet table in the store for the evidence history. The seed and literature edges produced in this work (`causal_edges_seed.csv`, `causal_edges_prior.csv`) are the first content.

### 2.3 Literature module

The pipeline used here becomes a reusable module (`regsim.causal.literature`): hypothesis generation for uncovered pairs, OpenAlex/CrossRef search with query expansion, two-pass structured extraction (extraction and independent verification of sign, design, claim strength and quantity match), weighting, grading, and a hand-check sample. Each run is stored with its prompts, model ids, query strings and retrieved records, so that grades can be reproduced and updated. Ranking by citation count and seeding with known meta-analyses is added to reduce the bias towards recent case studies.

## 3 Coupling options

The options are ordered by effort; C1–C2 and C5 are recommended for the first implementation.

### C1 Consistency checks on the existing model (low effort)

For every edge with both ends in the registry:

- compare `sign_prior` with the cross-sectional partial correlation (I-partial, conditional on all other indicators) and with the within-region correlation of changes between the two windows (geo-time entities);
- for LEVITATE goal relations, compare the polarity-aligned correlation with the stated synergy or conflict;
- flag contradictions and report, per edge, whether the association is "prior-consistent".

C1 changes no model. It turns every regsim correlation output into one of three labels: consistent with a graded prior, contradicting it, or without prior.

### C2 Causal panel module on annual data (medium effort, core of the extension)

For indicators in the graph, annual series by region (NUTS-2 first; NUTS-3 where available) are modelled as

Δz_{g,t,k} = α_{g,k} + δ_{c(g),t,k} + Σ_{j ∈ pa(k)} β_{jk,s(g)} · Δz_{g,t−ℓ,j} + γ_k · Δz_{g,t−1,k} + λ_{g,k}·f_t + ε_{g,t,k}

with region effects α, country × year effects δ (common national shocks and policies), parents pa(k) from the registry, lag ℓ from `lag_prior`, partial pooling of β over region types s(g) (β_{jk,s} ~ N(β_jk, τ²_jk)), and an interactive factor term λ·f_t whose loadings are taken from regsim's entity factors u_g (common correlated effects). Priors: β_jk ~ N(m, s²) from `strength_prior`, with the sign probability entering as a skewed prior; edges without prior use a shrinkage prior (horseshoe) so that the same model also screens for missing edges. Missing annual values are filled by regsim with their predictive variance (multiple imputation), not treated as observed.

Outputs per edge: posterior of β_jk, heterogeneity τ_jk, region-type effects, prior–data conflict (posterior shift relative to prior sd), and a status update. Estimation: hierarchical Bayesian (NumPyro or PyMC) or a penalised two-step estimator for speed.

### C3 Factor-augmented structural model inside HierTypedMF (high effort, later)

z = μ + b + U Vᵀ + B z̃ + ε, with B a sparse matrix restricted to registry edges (acyclic part) and U Vᵀ the confounding factors. This would make imputation and what-if share one model, but identification of B next to free factors needs strong priors and instruments. It is deferred until C2 shows which edges are stable.

### C4 SD-model bridge (medium effort)

- **Edges from SD models:** perturbation runs of SD models (as done here for the Tallinn configuration) produce signed, quantified total effects of drivers and levers on outputs; static dependency graphs from the Vensim equations give the structure. Both are imported with source type "SD model" and the city configuration as validity.
- **Inputs and targets:** regsim provides completed regional input values (population, density, income, car ownership, degree days) and calibration targets with intervals for SD calibration in follower regions.
- **Parameter transfer:** calibrated SD parameters of demonstrators become new regsim columns (fold-in of an indicator). Transfer-specific similarity (S-transfer) then learns which indicators predict a parameter, and predicts it for followers with an interval. This makes the D2.5/D3.6 statement "interpolate calibration parameters by similarity" testable.
- **Feedback to modellers:** where data and literature contradict a model-implied sign (e.g. fuel cost → active travel in the current Tallinn configuration), the edge is flagged for model review.

### C5 Intervention effects by counterfactual matrix completion (medium effort)

For interventions with known implementation (event table from city plans, LOCALISED/Covenant of Mayors data, or D3.6 Annex A follow-up), the post-intervention cells of treated regions are masked and predicted by HierTypedMF from untreated regions (Athey et al., 2021). The difference between observed and counterfactual values, pooled over treated regions, estimates the intervention effect on each Annex A indicator; Annex A's indirect indicators serve as effect modifiers. This gives the intervention edges their strength and is the most direct route to evidence on what transfers.

## 4 Bidirectional loop and evidence tiers

1. **Priors in:** SD models, deliverables, literature (graded), expert input; LLM hypotheses only as `hypothesis` with grade D.
2. **Tests:** C1 sign checks, C2 within-region estimates with heterogeneity and invariance across countries and region types (invariant causal prediction), C5 intervention effects.
3. **Updates:** status and data evidence per edge; contradicted edges keep their sources and are shown as contested, not deleted.
4. **New hypotheses out:** edges selected by the shrinkage prior, or residual dependencies after C2, are sent to the literature module as targeted queries and to SD modellers as candidate structure.

Every causal output carries an evidence tier:

| Tier | Meaning | Typical source | May claim |
|---|---|---|---|
| E0 | association | regsim correlations, similarity | "move together across regions" |
| E1 | prior-consistent association | C1 | "association in the direction expected from graded evidence" |
| E2 | within-region dynamic evidence | C2 | "changes in X precede changes in Y within regions, net of national shocks and common factors" |
| E3 | quasi-experimental | C5, literature grade A with quasi-experiments | "estimated effect of the intervention or driver" |
| E4 | structural counterfactual | C3, C4 what-if | "effect under the stated causal model" |

## 5 Identification assumptions

| Output | Main assumptions | Typical failure |
|---|---|---|
| C1 sign check | partial correlation reflects the direct relation | common causes not in the indicator set; colliders |
| C2 panel | no time-varying confounder beyond country × year effects and the factor term; lags respect timing; no anticipation | regional policies correlated with both indicators; reverse causation within the year; Nickell bias for short panels (use bias correction or GMM) |
| C5 counterfactual | untreated regions are not affected (no spillover); treatment does not change the factor structure; treatment timing not driven by expected outcomes | neighbouring regions treated; selection of pioneers |
| C4 SD transfer | model structure valid in the follower; parameter variation explained by registry indicators | structural differences (e.g. district heating, mode choice set) |
| Cross-level use | relation estimated at one level holds at another | ecological fallacy, MAUP |

## 6 New regsim functions

| Function | Question | Tier |
|---|---|---|
| `causal.check_edges` (C1) | Are observed associations consistent with the evidence base? | E1 |
| `causal.panel.fit` (C2) | How strong is X → Y within regions, and where does it differ? | E2 |
| `causal.events.effect` (C5) | What did intervention I change where it was implemented? | E3 |
| `predict.what_if` (P8) | If X changes by Δ in region g, what follows for its descendants (with intervals)? | E2–E4 |
| `predict.explain_difference` (P9) | Which causes account for the difference in Y between regions g and h? (path decomposition along the graph) | E2 |
| `predict.vision_check` (P10) | Are the targets of a vision jointly consistent with the causal relations and their lags? | E2–E4 |
| `similarity.s_causal` | Similarity on the causes and effect modifiers of a target (lens = parents and moderators) | transportability |
| `causal.sd_bridge` (C4) | Import SD edges; export inputs/targets; transfer calibrated parameters | E4 |

## 7 Data requirements

- Annual regional series for the panel module: Eurostat regional tables (most from 2000 or 2008, NUTS-2 complete, NUTS-3 for demography and GDP), ARDECO for long economic series.
- Registry additions to make SD structure testable: modal split, EV share, final energy per capita, sectoral emissions (national Eurostat tables, Urban Audit for cities, D3.6 TRNPT06/TRNPT08/ENERG02–04 once populated).
- Event table for interventions, starting with KNOWING demonstrators and LOCALISED or Covenant of Mayors records.
