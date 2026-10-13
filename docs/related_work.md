# Related work for a multi-level indicator and geo-entity similarity framework

All references with a DOI were checked against CrossRef (title, first author, year); entries without a DOI (JMLR, NeurIPS, arXiv) are listed with their stable URL. The full list is in `references.csv`. Section 9 maps each method family to the requirements of the new framework.

## 1 Collaborative filtering and low-rank matrix completion

The LEVITATE approach belongs to model-based collaborative filtering (Herlocker et al., 1999). Matrix factorisation with user and item biases, fitted on observed cells only, became the reference method through the Netflix Prize (Koren et al., 2009). Its convex counterpart, nuclear-norm matrix completion, recovers a low-rank matrix from a small random subset of entries under incoherence conditions (Candès and Recht, 2009); SoftImpute provides a scalable iterative-SVD solver with a regularisation path (Mazumder et al., 2010; Hastie et al., 2015). Probabilistic and Bayesian matrix factorisation adds posterior uncertainty for each predicted cell and automatic regularisation (Salakhutdinov and Mnih, 2008).

Three extensions are directly relevant:

- **Mixed data types.** Generalised low rank models (GLRM) attach a loss per column (quadratic, Huber, logistic, ordinal hinge, multinomial) to one shared factorisation (Udell et al., 2016). This is the latent-model analogue of the type-specific partial distances in D3.6 and removes the need to force all indicators onto one rating scale.
- **Side information and cold start.** Inductive matrix completion predicts factors from features of rows and columns (Jain and Dhillon, 2013), factorisation machines generalise this to arbitrary sparse feature vectors (Rendle, 2010), and cold-start strategies for new users and items are reviewed by Schein et al. (2002). Active learning selects which ratings to elicit next (Elahi et al., 2016); in our context: which indicator to measure next for a region.
- **Several related matrices.** Collective matrix factorisation shares factors across matrices that have one dimension in common (Singh and Gordon, 2008), e.g. a NUTS-3 × indicator and a LAU × indicator matrix that share indicator factors. Graph regularisation pulls factors of adjacent rows together (Kalofolias et al., 2014; Rao et al., 2015), which encodes spatial adjacency or the NUTS hierarchy. Tensor factorisation (Kolda and Bader, 2009) and low-rank spatio-temporal tensor learning (Bahadori et al., 2014) treat entity × indicator × time jointly.

Matrix completion assumes, in its standard form, that entries are missing at random. Missing-not-at-random patterns bias both predictions and rankings (Marlin and Zemel, 2009); low-rank estimators with explicit MNAR models exist (Sportisse et al., 2020). Factor-model results for panel data give conditions under which missing blocks (e.g. whole countries without an indicator) can be completed (Bai and Ng, 2021), and matrix completion has been used to estimate counterfactual outcomes in panel data (Athey et al., 2021). The last point links matrix completion to the LEVITATE "vision" entity and to backcasting: a vision is a partially specified counterfactual row.

## 2 Imputation of multivariate data

PCA-based iterative imputation (Josse and Husson, 2012, 2016) was the statistical route in LEVITATE. Standard alternatives are k-nearest-neighbour imputation (Troyanskaya et al., 2001), random-forest imputation for mixed-type data (Stekhoven and Bühlmann, 2012) and multiple imputation by chained equations (van Buuren and Groothuis-Oudshoorn, 2011). Tree-based and chained-equation imputers are strong baselines on tabular data with non-linear relations but do not give a shared latent space; they belong in the benchmark, not in the core model.

## 3 Multi-level data: small-area estimation, change of support, reconciliation

The relation between NUTS-0, NUTS-2, NUTS-3 and LAU values is a change-of-support problem (Gotway and Young, 2002), with the modifiable areal unit problem as its best-known symptom (Fotheringham and Wong, 1991). Three bodies of work offer tools:

- **Small-area estimation** combines a direct estimate for a small unit with a regression prediction from covariates, weighted by their precision (Fay and Herriot, 1979; Rao and Molina, 2015). This is the statistical template for "borrowing strength" from a parent region instead of copying its value.
- **Dasymetric disaggregation** predicts fine-scale values from covariates, e.g. population with random forests (Stevens et al., 2015) or gridded GDP and HDI from sub-national statistics (Kummu et al., 2018).
- **Hierarchical reconciliation** forces predictions at different levels of a hierarchy to be coherent, i.e. children aggregate to their parent, with minimum-variance adjustments (Hyndman et al., 2011; Wickramasuriya et al., 2019). Learning from aggregate observations generalises this to training a fine-scale model from coarse targets (Zhang et al., 2020).

For the framework, these suggest treating a coarse-level observation as a *constraint on the aggregate* of finer-level predictions, with the aggregation rule (sum, population-weighted mean, area-weighted mean) stored as indicator metadata.

## 4 Regional typologies and clustering

Geodemographic classifications cluster small areas on census variables into a hierarchical typology with human-readable labels, e.g. the UK Output Area Classification (Vickers and Rees, 2007; Gale et al., 2016); self-organising maps are a frequent alternative (Kohonen, 1990). A global urban energy typology was derived by clustering cities on a small set of drivers (Creutzig et al., 2015). Spatially constrained regionalisation produces contiguous clusters: SKATER (Assunção et al., 2006), max-p regions (Duque et al., 2012), ClustGeo (Chavent et al., 2018). Density-based clustering finds clusters of varying shape and leaves outliers unassigned (Campello et al., 2013); consensus clustering measures stability under resampling (Monti et al., 2003).

Co-clustering partitions rows and columns of a matrix simultaneously (Dhillon, 2001; Madeira and Oliveira, 2004). This is the formal counterpart of a *dynamic* typology in which groups of regions are defined by the groups of indicators that distinguish them, without a prior thematic classification of indicators. Applied to the latent factors, co-clustering of entities and indicators replaces the fixed LEVITATE dimensions and the fixed D3.6 tiers.

## 5 Similarity, analogues and transferability

- **Mixed-type distance.** Gower's coefficient (Gower, 1971) and Aitchison geometry for compositions (Aitchison, 1982) are the basis of D3.6. In high dimensions, distances concentrate and nearest-neighbour contrast fades (Beyer et al., 1999; Aggarwal et al., 2001), which explains part of the score compression seen in D3.6.
- **Climate and city analogues.** Analogue studies match each city to the present-day location whose climate it will resemble in future (Fitzpatrick and Dunn, 2019; Bastin et al., 2019); Mahalanobis-type distances with standardisation by local interannual variability are used to detect novel climates (Mahony et al., 2017). These works handle correlated variables explicitly and show that analogues should be searched in a decorrelated space.
- **Hydrological regionalisation.** Predicting runoff in ungauged catchments compares transfer by spatial proximity, by physical similarity and by regression (Oudin et al., 2008; Razavi and Coulibaly, 2013). The main lesson for transferability is that the similarity that works best is the one that predicts the transferred quantity, and that it is assessed by leave-one-out on donor catchments.
- **Metric learning** fits the distance itself so that neighbours share a target property (Kulis, 2013; Weinberger and Saul, 2009). It replaces fixed tier weights by weights learned from predictive performance.
- **Transportability.** Causal inference distinguishes similarity of populations from transportability of effects: an effect transfers if the mechanisms that differ between populations do not affect it, which needs structural knowledge (Pearl and Bareinboim, 2014; Bareinboim and Pearl, 2016). For KNOWING this means that similarity on context indicators is necessary but not sufficient for the transfer of intervention effects, and that a similarity should be specific to the transferred quantity.
- **Applicability domain.** In QSAR modelling, predictions are accompanied by a check whether the query lies within the domain covered by the training data (Netzeva et al., 2005). This is a continuous, data-driven alternative to the hard gates of D3.6.

## 6 Embeddings and side information for cold start

Entity embeddings of categorical variables (Guo and Berkhahn, 2016) are the concept that LEVITATE refers to. For geographic entities with few or no indicators, general-purpose location embeddings derived from satellite imagery are now available: MOSAIKS random features (Rolf et al., 2021), SatCLIP location encoders (Klemmer et al., 2025), and the AlphaEarth Foundations embedding field with annual 64-dimensional layers at 10 m for 2017–2024 (Brown et al., 2025). Aggregated to administrative units, such embeddings are side information for inductive completion, available globally and therefore the most direct route to an extension beyond Europe. The global application of the degree of urbanisation (Dijkstra et al., 2021) provides a harmonised settlement typology outside Europe.

## 7 Uncertainty quantification

Bayesian factorisation yields posterior intervals (Salakhutdinov and Mnih, 2008); conformal prediction yields distribution-free intervals with guaranteed marginal coverage for any point predictor (Vovk et al., 2005; Angelopoulos and Bates, 2023). Group-conditional (Mondrian) calibration by geographic level or country addresses the heterogeneous error structure of multi-level data.

## 8 Dimensionality reduction and visualisation

The biplot shows rows and columns of a matrix in one low-dimensional display, with a choice of scaling that determines whether row distances or column correlations are approximated (Gabriel, 1971); it is the correct frame for the LEVITATE joint plot of indicators and geo-entities. Non-linear projections (t-SNE: van der Maaten and Hinton, 2008; UMAP: McInnes et al., 2018) preserve local neighbourhoods but distort global distances. Distortion is quantified by Shepard diagrams (Kruskal, 1964) and by trustworthiness and continuity (Venna and Kaski, 2001). Trajectories over time need aligned coordinates; generalised Procrustes analysis aligns successive configurations (Gower, 1975). Conditional dependence between indicators can be displayed as a sparse partial-correlation network estimated with the graphical lasso (Friedman et al., 2008).

## 9 Method families against requirements

The matrix in `figures/fig_method_matrix.png` rates each family against eleven requirements derived from the task (0 = not addressed, 1 = partly, 2 = addressed). The ratings are a qualitative assessment by the author of this review, not measured values. No single family covers all requirements. The combination with the widest coverage is a **typed, hierarchical, side-information matrix factorisation** (rows 1–3 and 5 of the matrix) with **observed-data Gower distance** kept as a transparent companion measure, **co-clustering and consensus clustering** for typologies, and **conformal intervals** with an **applicability-domain** check.

## 10 Project documents

- LEVITATE Consortium. D4.2 *Definition of desirable scenarios* (final). Horizon 2020 project LEVITATE.
- KNOWING Consortium (2026). D3.6 *Transferability framework*, Horizon Europe Grant Agreement No. 101056841. Database: doi:10.5281/zenodo.20376109.
- KNOWING Consortium. D2.5 *Backcasting/Transferability Concept*, as named in D3.6 (proof of concept with 16 variables, PCA and k-means on a 5 km grid).
- LOCALISED project. *Integrated database on adaptation and mitigation measures in Europe*, https://zenodo.org/records/10993450 (named in D3.6 as the basis for an intervention database).
