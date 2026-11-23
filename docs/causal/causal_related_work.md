# Causal methods for multi-level regional indicator data

Scope: methods that could add causal structure to regsim (repository `martnzach-wq/CollaborativeFilteringFramework`, v0.8.0), either by bringing in prior causal knowledge or by estimating causal relations from the observed indicator data. DOIs were checked against CrossRef and arXiv identifiers against the arXiv API; the list is in `causal_references.csv`. `fig_causal_methods.png` rates each method family against the questions it can answer.

## 1 Starting point: what regsim represents today

regsim fits z_gk = μ_k + b_g + u_g·v_k with a hierarchical prior on u_g, aggregation constraints across levels and, since M3, two period windows (2013/2017, 2018/2022) linked by a random-walk penalty. Indicator similarity is available as model-implied correlation (I-model), observed Spearman correlation (I-observed), partial correlation from the graphical lasso (I-partial) and predictive gain (I-predictive). All four are symmetric measures of association. Three features of the data make it hard to read them causally:

- **Common causes.** The latent factors u_g are, by construction, the shared drivers of many indicators (development level, settlement structure, climate, national institutions). Two indicators loading on the same factor are correlated whether or not one affects the other. In panel econometrics such factors are called interactive fixed effects (Bai, 2009) or a multifactor error structure (Pesaran, 2006); they are the main source of confounding in cross-regional comparisons.
- **Cross-level aggregation.** A correlation between regions at NUTS-2 need not hold within regions or at LAU level (ecological fallacy; Robinson, 1950; Greenland and Morgenstern, 1989), and depends on the zoning (Fotheringham and Wong, 1991).
- **Short time.** Two five-year windows give a direction of change per region but no information on lags or feedback. Annual series exist for many Eurostat indicators from about 2000 (ARDECO from 1980 for core economic variables), which is the minimum for dynamic analysis.

## 2 Knowledge-based causal structures

**Causal loop diagrams and SD models.** System dynamics models encode causal structure explicitly as stocks, flows and signed links. The KNOWING SD models (D2.4) and the LEVITATE goal network (D4.2 Fig. 1) are such structures. Known weaknesses of causal loop diagrams are ambiguous link polarity when several mechanisms act at once and missing stock–flow distinctions (Richardson, 1986); a signed link in a CLD is therefore a hypothesis about the net effect of a mechanism, not a coefficient.

**Causal maps from text.** System dynamics has a method tradition of coding causal statements from documents into causal maps, with procedures for reliability and confidence (Kim and Andersen, 2012; Eker and Zimmermann, 2016). At larger scale, causal and semantic relations have been extracted from millions of documents, e.g. SemMedDB for biomedical predications (Kilicoglu et al., 2012) and CauseNet for causal relations from web text (Heindorf et al., 2020).

**Language models as imperfect experts.** Large language models reproduce many known causal relations between named variables and can propose candidate edges and directions (Kıcıman et al., 2023; Long et al., 2023a). Their output is best treated as a prior that has to be checked against data and literature: Long et al. (2023b) use the model as an "imperfect expert" to restrict the equivalence class found by a discovery algorithm, and Jiralerspong et al. (2024) use it to order candidate edges. For regsim this supports a two-stage approach: an LLM extracts structured causal claims from retrieved abstracts (cause, effect, sign, magnitude, method, setting), and a separate grading step weighs them.

**Evidence grading.** The GRADE approach rates the certainty of evidence by study design, risk of bias, inconsistency, indirectness and imprecision (Guyatt et al., 2008). A comparable grade per edge, based on identification strategy (experiment, quasi-experiment, panel with fixed effects, cross-section, model or expert judgement), number of independent studies, sign consistency and transferability to European regions, makes prior edges comparable across sources.

## 3 Causal discovery from observational data

**Cross-sectional discovery.** Constraint-based algorithms (PC; FCI, which allows hidden confounders) infer a class of graphs from conditional independences (Spirtes et al., 2001); score-based (greedy equivalence search), continuous-optimisation (NOTEARS; Zheng et al., 2018) and non-Gaussian methods (LiNGAM) are alternatives, reviewed by Glymour et al. (2019). They assume independent samples and faithfulness. Regions violate the first assumption through spatial dependence and shared national context, and at NUTS level the sample has at most about 1,500 units for tens of indicators.

**Heterogeneity as information.** Invariant causal prediction (Peters et al., 2016) uses several environments: a set of causes is accepted if the conditional distribution of the effect given these causes is the same in all environments. Countries, region types or periods can serve as environments, which turns regional heterogeneity from a nuisance into a test of causal stability.

**Time-series discovery.** Granger causality tests whether past values of one series improve the prediction of another (Granger, 1969); panel versions allow heterogeneous coefficients across units (Dumitrescu and Hurlin, 2012). PCMCI combines lagged conditional-independence tests with a selection step suited to many autocorrelated variables (Runge et al., 2019a), and Runge et al. (2019b) review the assumptions in Earth system science, where similar methods were applied to climate networks (Ebert-Uphoff and Deng, 2012). Convergent cross mapping targets deterministic coupled dynamics where Granger tests fail (Sugihara et al., 2012). All need series long compared with the lags of interest; annual regional data give 10–25 points per region, so pooling across regions is required.

## 4 Causal estimation in panels of regions

**Fixed effects and dynamic panels.** Two-way fixed effects remove time-invariant regional differences and common shocks, but with heterogeneous effects the estimator can weight units negatively (de Chaisemartin and D'Haultfœuille, 2020), and with a lagged dependent variable it is biased in short panels (Nickell, 1981), which GMM estimators address (Arellano and Bond, 1991). Panel vector autoregressions estimate feedback between several indicators (Holtz-Eakin et al., 1988; Abrigo and Love, 2016).

**Latent-factor panels.** Common correlated effects (Pesaran, 2006) and interactive fixed effects (Bai, 2009) remove unobserved common factors with region-specific loadings. Generalised synthetic control (Xu, 2017), synthetic control (Abadie et al., 2010), synthetic difference-in-differences (Arkhangelsky et al., 2021) and matrix completion for panel data (Athey et al., 2021; Bai and Ng, 2021) estimate the counterfactual outcome of treated units from untreated ones. The last point is directly relevant: regsim's HierTypedMF is a matrix-completion estimator, so masking the post-intervention cells of regions that implemented a policy and predicting them from the rest is a counterfactual estimate of the policy effect, under the assumption that the policy does not change the factor structure of the untreated regions.

**Structural and factor models.** Structural equation models combine latent variables with directed paths (Bollen, 1989). Dynamic factor models and factor-augmented VARs identify structural shocks in many-variable systems (Stock and Watson, 2016; Kilian and Lütkepohl, 2017). The "deconfounder" proposes to estimate a latent factor model of many causes and use it as a substitute confounder (Wang and Blei, 2019); its identification claims are contested, but the idea of conditioning causal estimates on regsim's latent factors is the same as in interactive fixed effects.

## 5 Combining models, knowledge and data

**Knowledge-guided machine learning** constrains data-driven models with known structure or physics, which improves extrapolation and interpretability (Karpatne et al., 2017; Reichstein et al., 2019).

**Calibration of SD models against data** treats parameter estimation as a test of model structure (Oliva, 2003) and has been developed into maximum-likelihood and bootstrap procedures (Struben et al., 2015), indirect inference (Hosseinichimeh et al., 2016) and Bayesian calibration (Andrade and Duggan, 2021); methods are collected in Rahmandad et al. (2015) and reporting standards in Rahmandad and Sterman (2012). Loop dominance analysis explains which feedback structure drives behaviour (Schoenberg et al., 2020).

**Transportability and data fusion** give formal conditions under which a causal effect estimated in one population (a demonstrator) transfers to another (a follower region), given a causal graph that marks where the populations differ (Pearl and Bareinboim, 2014; Bareinboim and Pearl, 2016). This is the step that the D3.6 similarity score leaves open: it requires knowing which differences between regions affect the mechanism.

**Spatial confounding.** Adding spatially structured random effects can change or bias fixed-effect estimates (Reich et al., 2006; Hodges and Reich, 2010); Reich et al. (2021) review spatial causal inference. For regsim, the spatial Laplacian and hierarchical prior must be treated with care when coefficients are interpreted causally.

**Framework.** The ladder of association, intervention and counterfactual (Pearl, 2009) and the potential-outcomes framework (Imbens and Rubin, 2015; Schölkopf, 2019 for the link to machine learning) set what each output may claim.

## 6 Consequences for the regsim extension

1. Prior structure from SD models, project documents and literature is indispensable, since the data alone identify neither direction nor strength in most cases. LLM-assisted extraction makes a literature prior feasible at the scale of the regsim registry, provided extracted claims are graded and sampled for manual checks.
2. The latent factors of regsim are the natural control for common causes. Causal estimates should be computed on the within-region, factor-adjusted variation (interactive fixed effects), not on cross-sectional correlations.
3. Annual panels are needed for dynamics; the two five-year windows are too coarse. Pooling across regions with partial pooling by region type gives both an average effect and its heterogeneity.
4. Heterogeneity across countries, region types and periods is a test of invariance and a link to transferability.
5. Policy effects with observed implementation dates can be estimated with regsim itself as a counterfactual (matrix completion) estimator.
6. SD models and the statistical layer meet in calibration: regsim supplies completed regional inputs and targets, and calibrated SD parameters become a new "indicator" whose transfer is predicted by transfer-specific similarity.
