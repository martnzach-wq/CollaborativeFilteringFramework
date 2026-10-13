"""Views V1-V8 (docs/visualisation.md). Each returns a matplotlib Figure; `interactive=True` returns plotly."""
from __future__ import annotations


def biplot(model, lens, level, colour_by=None, n_arrows=12, interactive=False): raise NotImplementedError("M6")     # V1
def indicator_similarity(model, observations, level, interactive=False): raise NotImplementedError("M6")           # V2
def distortion(model, lens, level, methods=("pca", "umap"), n_sample=400): raise NotImplementedError("M6")         # V3
def maps(typology=None, similarity=None, boundaries=None): raise NotImplementedError("M6")                          # V4
def type_profiles(typology): raise NotImplementedError("M6")                                                         # V5
def difference_profile(model, geo_a, geo_b, lens): raise NotImplementedError("M6")                                   # V6
def trajectories(model, geo_ids, periods, vision=None): raise NotImplementedError("M6")                             # V7
def model_card(benchmark): raise NotImplementedError("M6")                                                           # V8
