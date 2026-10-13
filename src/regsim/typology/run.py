"""Typology runs (design §6): clustering of completed profiles with stability, labels and tracking.

A typology is the output of a :class:`TypologyRun` on a fitted model. The run's configuration hash covers the
run parameters, the model parameters and a fingerprint of the data, so that runs are reproducible and
comparable.

- Representation: whitened completed profile on the lens (default) or latent projection (``"latent"``).
- Algorithms: Gaussian mixture (soft membership, BIC), Ward, HDBSCAN (outliers keep label -1) and contiguity-
  constrained Ward (``ward_spatial``, needs ``data.adjacency``).
- Number of clusters: the smallest BIC among the k with bootstrap stability (mean adjusted Rand index) of at
  least ``min_stability``; the most stable k if none qualifies. BIC is that of a diagonal Gaussian mixture.
- Stability: consensus matrix (share of bootstrap and posterior runs in which two entities share a cluster);
  per-entity stability is the mean consensus with the other members of its cluster.
- Labels: per cluster, the indicators with the largest standardised centroid deviation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from regsim.models.base import ModelData
from regsim.schema import Lens
from regsim.similarity.entities import ProfileSpace, noise_variance, s_latent, s_profile

ALGORITHMS = ("gmm", "ward", "hdbscan", "ward_spatial")


@dataclass(frozen=True)
class TypologyRun:
    """Configuration of a typology run."""

    lens: Lens = field(default_factory=Lens)
    representation: str = "whitened_profile"  # or "latent"
    algorithm: str = "gmm"  # gmm | ward | hdbscan | ward_spatial
    k: int | None = None  # None: chosen by stability and BIC
    k_range: tuple[int, int] = (3, 12)
    min_cluster_size: int = 10  # hdbscan
    n_boot: int = 30
    n_posterior: int = 10
    min_stability: float = 0.7
    n_label: int = 3
    seed: int = 0

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if self.representation not in {"whitened_profile", "latent"}:
            raise ValueError("representation must be 'whitened_profile' or 'latent'")

    def config_hash(self, model=None, data: ModelData | None = None) -> str:
        """Hash of the run parameters, the model parameters and a fingerprint of the data."""
        d = asdict(self)
        d["lens"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.lens).items()}
        if model is not None and hasattr(model, "get_params"):
            d["model"] = {k: repr(v) for k, v in sorted(model.get_params().items())}
        if data is not None:
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(np.where(data.mask, data.Z, 0.0)).tobytes())
            h.update(np.ascontiguousarray(data.mask).tobytes())
            h.update("\n".join(data.geo_ids).encode())
            d["data"] = h.hexdigest()
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class Typology:
    """Result of a typology run (rows: the lens entities)."""

    run: TypologyRun
    config_hash: str
    geo_ids: list[str]
    labels: np.ndarray
    membership: np.ndarray
    k: int
    stability: float
    entity_stability: np.ndarray
    selection: pd.DataFrame
    centroids: pd.DataFrame  # standardised centroid deviations (clusters x indicators)
    cluster_table: pd.DataFrame
    consensus: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cluster": self.labels,
                "membership": self.membership.max(axis=1) if self.membership.size else np.nan,
                "stability": self.entity_stability,
            },
            index=pd.Index(self.geo_ids, name="geo_id"),
        )


def _representation(model, data: ModelData, run: TypologyRun) -> ProfileSpace:
    return (s_latent if run.representation == "latent" else s_profile)(model, data, run.lens)


def _cluster(X: np.ndarray, run: TypologyRun, k: int | None, seed: int, connectivity=None):
    """Labels and soft memberships of one clustering."""
    if run.algorithm == "gmm":
        gm = GaussianMixture(k, covariance_type="diag", n_init=2, random_state=seed).fit(X)
        return gm.predict(X), gm.predict_proba(X), gm
    if run.algorithm == "hdbscan":
        hd = HDBSCAN(min_cluster_size=run.min_cluster_size, copy=True).fit(X)
        labels = hd.labels_
        n = max(labels.max() + 1, 1)
        memb = np.zeros((len(X), n))
        ok = labels >= 0
        memb[np.flatnonzero(ok), labels[ok]] = hd.probabilities_[ok]
        return labels, memb, hd
    ag = AgglomerativeClustering(
        n_clusters=k, linkage="ward", connectivity=connectivity if run.algorithm == "ward_spatial" else None
    ).fit(X)
    return ag.labels_, np.eye(k)[ag.labels_], ag


def _assign(X_ref: np.ndarray, labels_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Assign rows of ``X`` to the nearest centroid of the clusters ``labels_ref`` of ``X_ref``."""
    ids = np.unique(labels_ref[labels_ref >= 0])
    C = np.array([X_ref[labels_ref == c].mean(axis=0) for c in ids])
    return ids[np.argmin(((X[:, None, :] - C[None]) ** 2).sum(axis=-1), axis=1)]


def _resample_labels(X, run, k, rng, connectivity):
    """Labels of all rows from a clustering of a bootstrap resample."""
    idx = np.unique(rng.choice(len(X), len(X), replace=True))
    conn = connectivity[idx][:, idx] if connectivity is not None else None
    lab, _, est = _cluster(X[idx], run, k, int(rng.integers(2**31)), conn)
    if run.algorithm == "gmm":
        return est.predict(X)
    if (lab >= 0).sum() == 0:
        return np.full(len(X), -1)
    return _assign(X[idx], lab, X)


def _stability(X, run, k, labels, seed, connectivity, samples=None) -> tuple[float, np.ndarray]:
    """Mean ARI against ``labels`` over bootstrap (and posterior) runs, and the consensus matrix."""
    rng = np.random.default_rng(seed)
    runs = [_resample_labels(X, run, k, rng, connectivity) for _ in range(run.n_boot)]
    for S in samples if samples is not None else []:
        lab, _, est = _cluster(S, run, k, int(rng.integers(2**31)), connectivity)
        runs.append(lab)
    if not runs:
        return float("nan"), np.eye(len(X))
    ari = [adjusted_rand_score(labels, r) for r in runs]
    cons = np.zeros((len(X), len(X)))
    for r in runs:
        cons += (r[:, None] == r[None, :]) & (r[:, None] >= 0)
    return float(np.mean(ari)), cons / len(runs)


def _labels_table(Z: np.ndarray, labels: np.ndarray, names: list[str], n_label: int):
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    ids = np.unique(labels[labels >= 0])
    dev = pd.DataFrame(
        [(Z[labels == c].mean(axis=0) - mu) / sd for c in ids],
        index=pd.Index(ids, name="cluster"),
        columns=names,
    )
    rows = []
    for c in ids:
        top = dev.loc[c].abs().sort_values(ascending=False).index[:n_label]
        label = ", ".join(f"{'high' if dev.loc[c, t] > 0 else 'low'} {t}" for t in top)
        within = (Z[labels == c].std(axis=0) / sd).mean()
        rows.append(
            {"cluster": c, "size": int((labels == c).sum()), "label": label, "dispersion": float(within)}
        )
    return dev, pd.DataFrame(rows).set_index("cluster")


def run_typology(model, data: ModelData, run: TypologyRun, samples: np.ndarray | None = None) -> Typology:
    """Run a typology on a fitted model.

    Parameters
    ----------
    model : Estimator
        Fitted model (completed profiles from ``predict``).
    data : ModelData
        The model's data.
    run : TypologyRun
        Configuration.
    samples : ndarray, optional
        Posterior samples of the profiles ``(s, n, p)`` (e.g. ``model.predict_dist(noise=False)``); the first
        ``run.n_posterior`` enter the consensus matrix.
    """
    space = _representation(model, data, run)
    rows = space.lens.rows
    X = space.Y[rows]
    conn = None
    if run.algorithm == "ward_spatial":
        if data.adjacency is None:
            raise ValueError("ward_spatial needs data.adjacency")
        conn = data.adjacency[rows][:, rows]
    S = None
    if samples is not None and run.n_posterior:
        S = [(s[:, space.lens.cols] - space.mean) @ space.whitener for s in samples[: run.n_posterior]]
        S = [s[rows] for s in S]
    sel = []
    if run.algorithm == "hdbscan":
        ks = [None]
    elif run.k is not None:
        ks = [run.k]
    else:
        ks = list(range(run.k_range[0], run.k_range[1] + 1))
    results = {}
    for k in ks:
        labels, memb, _ = _cluster(X, run, k, run.seed, conn)
        stab, cons = _stability(X, run, k, labels, run.seed + 1, conn, S)
        bic = GaussianMixture(k, covariance_type="diag", random_state=run.seed).fit(X).bic(X) if k else np.nan
        sel.append({"k": k if k else int(labels.max() + 1), "stability": stab, "bic": bic})
        results[k] = (labels, memb, stab, cons)
    selection = pd.DataFrame(sel)
    if len(ks) == 1:
        best = ks[0]
    else:
        ok = selection[selection.stability >= run.min_stability]
        best = int(
            ok.loc[ok.bic.idxmin(), "k"] if len(ok) else selection.loc[selection.stability.idxmax(), "k"]
        )
    labels, memb, stab, cons = results[best]
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    with np.errstate(invalid="ignore"):
        ent = np.where(same.any(axis=1), (cons * same).sum(axis=1) / np.maximum(same.sum(axis=1), 1), np.nan)
    ent = np.where(labels >= 0, ent, np.nan)
    Z = model.predict()[rows][:, space.lens.cols]
    dev, table = _labels_table(Z, labels, space.lens.names, run.n_label)
    table["stability"] = [float(np.nanmean(ent[labels == c])) for c in table.index]
    return Typology(
        run=run,
        config_hash=run.config_hash(model, data),
        geo_ids=[data.geo_ids[i] for i in rows],
        labels=labels,
        membership=memb,
        k=int(labels.max() + 1),
        stability=stab,
        entity_stability=ent,
        selection=selection,
        centroids=dev,
        cluster_table=table,
        consensus=cons,
    )


def track(a: Typology, b: Typology, key=None) -> tuple[dict[int, int], pd.DataFrame]:
    """Align the clusters of ``b`` to those of ``a`` (Hungarian matching on standardised centroids over the
    common indicators) and the transition matrix of the entities in both (``key`` maps entity ids to a common
    id, e.g. ``lambda g: g.split('@')[0]`` across periods).

    Returns the mapping {cluster of b: cluster of a} and the transition counts (rows: a, columns: b mapped).
    """
    common = [c for c in a.centroids.columns if c in b.centroids.columns]
    A, B = a.centroids[common].to_numpy(), b.centroids[common].to_numpy()
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1))
    ra, rb = linear_sum_assignment(D)
    mapping = {int(b.centroids.index[j]): int(a.centroids.index[i]) for i, j in zip(ra, rb, strict=True)}
    nxt = max(a.centroids.index.max(), -1) + 1
    for c in b.centroids.index:
        if int(c) not in mapping:
            mapping[int(c)] = int(nxt)
            nxt += 1
    key = key or (lambda g: g)
    la = pd.Series(a.labels, index=[key(g) for g in a.geo_ids])
    lb = pd.Series([mapping.get(int(x), -1) for x in b.labels], index=[key(g) for g in b.geo_ids])
    both = la.index.intersection(lb.index)
    trans = pd.crosstab(la[both].rename("a"), lb[both].rename("b"))
    return mapping, trans


__all__ = ["TypologyRun", "Typology", "run_typology", "track", "noise_variance"]
