"""Synthetic hierarchical low-rank data with known factors (milestone M0).

Entities sit on nested levels. Entity factors follow the hierarchical prior of design §4.2,

    u_g = sum_p w_gp u_p + delta_g,  delta_g ~ N(0, sigma_level(g)^2 I),

and the latent value of entity g, indicator k and period t is

    z(g, k, t) = mu_k + (u_g + tau_(g, t)) . v_k + eps,

with a random walk tau over periods (design §4.3). Latent values are mapped to the original scale by
indicator type; aggregable indicators at coarse levels are recomputed bottom-up from their children on the
original scale, so that parent values are exact aggregates as in official statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from regsim.schema import Aggregation, DType, GeoEdge, GeoUnit, Indicator, Transform

SCHEME = "SYN"
SUPPORTED_DTYPES = (DType.CONTINUOUS, DType.SHARE, DType.COUNT, DType.ORDINAL)
MISSINGNESS = ("none", "mcar", "block_mnar", "level")
N_ORDINAL = 5

_TYPE_META: dict[DType, tuple[Transform, Aggregation]] = {
    DType.CONTINUOUS: (Transform.IDENTITY, Aggregation.WMEAN_POP),
    DType.SHARE: (Transform.LOGIT, Aggregation.WMEAN_POP),
    DType.COUNT: (Transform.LOG1P, Aggregation.SUM),
    DType.ORDINAL: (Transform.RANK, Aggregation.NONE),
}


@dataclass
class SyntheticData:
    """Output of :func:`generate_hierarchical`.

    Rows are ordered by level (all level-0 entities first). Arrays with a period axis have shape
    ``(n_periods, n_entities, n_indicators)``.

    Attributes
    ----------
    U : ndarray, shape (n_entities, rank)
        True entity factors (period-independent part).
    tau : ndarray, shape (n_periods, n_entities, rank)
        True period offsets of the entity factors.
    V : ndarray, shape (n_indicators, rank)
        True indicator factors.
    mu : ndarray, shape (n_indicators,)
        True indicator offsets.
    level : ndarray of int, shape (n_entities,)
        Depth of each entity in the hierarchy.
    parent : list of list of (int, float)
        Parents of each entity with population weights (shares of the child in the parent); empty at level 0.
    membership : dict of (int, int) to float
        Fraction of child c that lies in parent p, keyed by ``(c, p)``; 1 except for fractional parents.
    population, area : ndarray, shape (n_entities,)
        Population and area; coarse levels are sums over their children.
    latent : ndarray
        Noise-free latent values ``mu + (U + tau) V^T`` (transformed scale).
    X : ndarray
        Complete values on the original scale.
    observed : ndarray of bool
        Observation mask.
    indicators : list of Indicator
        Metadata with dtype, transform and aggregation per column.
    periods : list of str
        Period labels.
    geo_ids : list of str
        Scheme-qualified entity identifiers.
    seed : int
        Seed used for generation.
    """

    U: np.ndarray
    tau: np.ndarray
    V: np.ndarray
    mu: np.ndarray
    level: np.ndarray
    parent: list[list[tuple[int, float]]]
    membership: dict[tuple[int, int], float]
    population: np.ndarray
    area: np.ndarray
    latent: np.ndarray
    X: np.ndarray
    observed: np.ndarray
    indicators: list[Indicator]
    periods: list[str]
    geo_ids: list[str]
    seed: int

    @property
    def n_entities(self) -> int:
        return self.U.shape[0]

    @property
    def n_indicators(self) -> int:
        return self.V.shape[0]

    def X_observed(self) -> np.ndarray:
        """Original-scale values with unobserved cells set to NaN."""
        return np.where(self.observed, self.X, np.nan)

    def units(self) -> list[GeoUnit]:
        return [
            GeoUnit(
                geo_id=g,
                scheme=SCHEME,
                level=int(lv),
                country=self.geo_ids[self.root_of(i)],
                area_km2=float(self.area[i]),
                population=float(self.population[i]),
            )
            for i, (g, lv) in enumerate(zip(self.geo_ids, self.level, strict=True))
        ]

    def edges(self) -> list[GeoEdge]:
        out = []
        for c, parents in enumerate(self.parent):
            for p, w in parents:
                out.append(
                    GeoEdge(
                        child=self.geo_ids[c],
                        parent=self.geo_ids[p],
                        weight_pop=w,
                        weight_area=self.membership[(c, p)] * self.area[c] / self.area[p],
                    )
                )
        return out

    def root_of(self, i: int) -> int:
        """Level-0 ancestor along the parent with the largest weight."""
        while self.parent[i]:
            i = max(self.parent[i], key=lambda pw: pw[1])[0]
        return i

    def aggregation_matrix(self, weight: str = "pop") -> sparse.csr_matrix:
        """Sparse matrix A (entities x entities) with A[p, c] = weight of child c in parent p.

        ``weight="pop"`` gives population shares (rows sum to 1) for weighted means; ``weight="membership"``
        gives the fractional membership used for sums of extensive quantities.
        """
        rows, cols, vals = [], [], []
        for c, parents in enumerate(self.parent):
            for p, w in parents:
                rows.append(p)
                cols.append(c)
                vals.append(w if weight == "pop" else self.membership[(c, p)])
        n = self.n_entities
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def to_observations(self) -> pd.DataFrame:
        """Observed cells in the long format of design §2.1 (original scale)."""
        t, g, k = np.nonzero(self.observed)
        return pd.DataFrame(
            {
                "geo_id": np.asarray(self.geo_ids, dtype=object)[g],
                "indicator_id": np.asarray([ind.indicator_id for ind in self.indicators], dtype=object)[k],
                "period": np.asarray(self.periods, dtype=object)[t],
                "scenario": "observed",
                "value": self.X[t, g, k],
                "component": None,
                "se": np.nan,
                "source": "synthetic",
                "source_version": f"seed={self.seed}",
                "flags": "",
            }
        )


def generate_hierarchical(
    n_per_level: tuple[int, ...] = (30, 300, 1500),
    n_indicators: int = 25,
    rank: int = 5,
    sigma_level: tuple[float, ...] = (1.0, 0.5, 0.3),
    noise: float = 0.1,
    dtypes: dict[str, float] | None = None,
    missingness: str = "level",
    missing_rate: float = 0.3,
    n_periods: int = 1,
    drift: float = 0.1,
    fractional_share: float = 0.0,
    seed: int | np.random.Generator = 0,
) -> SyntheticData:
    """Generate entities on nested levels with known low-rank structure.

    Parameters
    ----------
    n_per_level : tuple of int
        Number of entities per level, coarse to fine; each level needs at least as many entities as the
        level above.
    n_indicators : int
        Number of indicators.
    rank : int
        Rank of the latent factors.
    sigma_level : tuple of float
        Standard deviation of the factor innovation delta_g per level.
    noise : float
        Standard deviation of the cell noise on the latent scale.
    dtypes : dict of str to float, optional
        Shares of indicator types among ``continuous``, ``share``, ``count`` and ``ordinal``; default all
        continuous.
    missingness : {"none", "mcar", "block_mnar", "level"}
        ``mcar``: cells missing at random with probability ``missing_rate``. ``block_mnar``: for each
        indicator, whole level-0 subtrees missing below level 0, plus value-dependent dropping (high values
        missing more often). ``level``: each indicator is observed down to a random finest level, then
        thinned at random with ``missing_rate / 3``; this reproduces the Eurostat pattern where many
        indicators exist only at coarse levels.
    missing_rate : float
        Target share of missing cells for ``mcar`` and ``block_mnar``.
    n_periods : int
        Number of period windows; labels ``"P0"``, ``"P1"``, ...
    drift : float
        Standard deviation of the random-walk step of the entity factors between periods.
    fractional_share : float
        Share of entities at the finest level that have two parents with fractional weights.
    seed : int or numpy.random.Generator
        Seed; identical seeds give identical output.

    Returns
    -------
    SyntheticData
    """
    if len(sigma_level) != len(n_per_level):
        raise ValueError("sigma_level needs one value per level")
    if any(b < a for a, b in zip(n_per_level, n_per_level[1:], strict=False)):
        raise ValueError("each level needs at least as many entities as the level above")
    if missingness not in MISSINGNESS:
        raise ValueError(f"missingness must be one of {MISSINGNESS}")
    rng = np.random.default_rng(seed)
    seed_int = seed if isinstance(seed, int) else -1

    n_levels = len(n_per_level)
    offsets = np.concatenate([[0], np.cumsum(n_per_level)])
    n = int(offsets[-1])
    level = np.repeat(np.arange(n_levels), n_per_level)

    # Hierarchy: every parent gets at least one child; optional second parent at the finest level.
    parent: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    membership: dict[tuple[int, int], float] = {}
    for lv in range(1, n_levels):
        n_par, n_chi = n_per_level[lv - 1], n_per_level[lv]
        first = np.concatenate([rng.permutation(n_par), rng.integers(0, n_par, n_chi - n_par)])
        for j in range(n_chi):
            c, p = offsets[lv] + j, offsets[lv - 1] + first[j]
            parent[c] = [(int(p), 1.0)]
            membership[(int(c), int(p))] = 1.0
    if n_levels > 1 and fractional_share > 0:
        lv = n_levels - 1
        n_par = n_per_level[lv - 1]
        chosen = rng.choice(n_per_level[lv], int(round(fractional_share * n_per_level[lv])), replace=False)
        for j in chosen:
            c = int(offsets[lv] + j)
            p1 = parent[c][0][0]
            p2 = int(offsets[lv - 1] + rng.integers(0, n_par))
            if p2 == p1:
                continue
            f = float(rng.uniform(0.2, 0.5))
            parent[c] = [(p1, 1.0 - f), (p2, f)]
            membership[(c, p1)], membership[(c, p2)] = 1.0 - f, f

    # Population and area: drawn at the finest level, summed upwards by membership.
    population = np.zeros(n)
    area = np.zeros(n)
    fine = slice(offsets[-2], offsets[-1])
    population[fine] = rng.lognormal(11.0, 0.8, n_per_level[-1])
    area[fine] = rng.lognormal(6.5, 1.0, n_per_level[-1])
    for lv in range(n_levels - 1, 0, -1):
        for c in range(offsets[lv], offsets[lv + 1]):
            for p, _ in parent[c]:
                population[p] += membership[(c, p)] * population[c]
                area[p] += membership[(c, p)] * area[c]
    # Parent weights as population shares of the child in each parent.
    for c in range(n):
        parent[c] = [(p, membership[(c, p)] * population[c] / population[p]) for p, _ in parent[c]]

    # Factors from the hierarchical prior (parent mean weighted by membership).
    U = np.zeros((n, rank))
    for lv in range(n_levels):
        rows = np.arange(offsets[lv], offsets[lv + 1])
        U[rows] = sigma_level[lv] * rng.standard_normal((len(rows), rank))
        if lv > 0:
            for c in rows:
                U[c] += sum(membership[(c, p)] * U[p] for p, _ in parent[c])
    V = rng.standard_normal((n_indicators, rank)) / np.sqrt(rank)
    mu = rng.normal(0.0, 0.5, n_indicators)
    tau = np.zeros((n_periods, n, rank))
    for t in range(1, n_periods):
        tau[t] = tau[t - 1] + drift * rng.standard_normal((n, rank))

    indicators = _make_indicators(n_indicators, dtypes, rng)
    latent = mu + np.einsum("tnr,kr->tnk", U[None] + tau, V)
    noisy = latent + noise * rng.standard_normal(latent.shape)

    X = np.empty_like(noisy)
    thresholds = np.quantile(noisy, np.linspace(0, 1, N_ORDINAL + 1)[1:-1], axis=(0, 1))
    for k, ind in enumerate(indicators):
        X[..., k] = _to_original(noisy[..., k], ind.dtype, population, thresholds[:, k], rng)

    # Coarse values of aggregable indicators are exact aggregates of their children.
    for lv in range(n_levels - 2, -1, -1):
        rows = range(offsets[lv + 1], offsets[lv + 2])
        for k, ind in enumerate(indicators):
            if ind.aggregation is Aggregation.NONE:
                continue
            num = np.zeros((n_periods, n))
            den = np.zeros(n)
            for c in rows:
                for p, _ in parent[c]:
                    m = membership[(c, p)]
                    if ind.aggregation is Aggregation.SUM:
                        num[:, p] += m * X[:, c, k]
                        den[p] = 1.0
                    else:
                        num[:, p] += m * population[c] * X[:, c, k]
                        den[p] += m * population[c]
            sel = np.arange(offsets[lv], offsets[lv + 1])
            X[:, sel, k] = num[:, sel] / den[sel]

    observed = _mask(missingness, missing_rate, X, noisy, level, parent, n_levels, rng)
    for k, ind in enumerate(indicators):
        lv_obs = sorted({int(lv) for lv in level[observed[..., k].any(axis=0)]})
        indicators[k] = Indicator(
            indicator_id=ind.indicator_id,
            dtype=ind.dtype,
            transform=ind.transform,
            aggregation=ind.aggregation,
            native_levels=tuple(str(lv) for lv in lv_obs),
            source="synthetic",
            categories=ind.categories,
        )

    width = len(str(max(n_per_level)))
    geo_ids = [f"{SCHEME}:L{lv}_{j:0{width}d}" for lv in range(n_levels) for j in range(n_per_level[lv])]
    data = SyntheticData(
        U=U,
        tau=tau,
        V=V,
        mu=mu,
        level=level,
        parent=parent,
        membership=membership,
        population=population,
        area=area,
        latent=latent,
        X=X,
        observed=observed,
        indicators=indicators,
        periods=[f"P{t}" for t in range(n_periods)],
        geo_ids=geo_ids,
        seed=seed_int,
    )
    return data


def _make_indicators(n_indicators: int, dtypes: dict[str, float] | None, rng: np.random.Generator) -> list:
    shares = {DType.CONTINUOUS: 1.0} if not dtypes else {DType(k): float(v) for k, v in dtypes.items()}
    unsupported = set(shares) - set(SUPPORTED_DTYPES)
    if unsupported:
        raise ValueError(f"unsupported dtypes for synthetic data: {sorted(d.value for d in unsupported)}")
    total = sum(shares.values())
    counts = {d: int(np.floor(n_indicators * s / total)) for d, s in shares.items()}
    for d in sorted(shares, key=lambda d: -(n_indicators * shares[d] / total - counts[d])):
        if sum(counts.values()) == n_indicators:
            break
        counts[d] += 1
    types = [d for d in SUPPORTED_DTYPES for _ in range(counts.get(d, 0))]
    types = [types[i] for i in rng.permutation(len(types))]
    width = len(str(n_indicators))
    out = []
    for k, d in enumerate(types):
        transform, aggregation = _TYPE_META[d]
        cats = tuple(str(c) for c in range(N_ORDINAL)) if d is DType.ORDINAL else ()
        out.append(Indicator(f"ind_{k:0{width}d}", d, transform, aggregation, categories=cats))
    return out


def _to_original(
    z: np.ndarray, dtype: DType, population: np.ndarray, thresholds: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if dtype is DType.CONTINUOUS:
        return z
    if dtype is DType.SHARE:
        return expit(z)
    if dtype is DType.COUNT:
        return rng.poisson(population / 1000.0 * np.exp(np.clip(z, -10, 10))).astype(float)
    return np.searchsorted(thresholds, z).astype(float)  # ordinal codes 0 .. N_ORDINAL - 1


def _mask(
    kind: str,
    rate: float,
    X: np.ndarray,
    z: np.ndarray,
    level: np.ndarray,
    parent: list[list[tuple[int, float]]],
    n_levels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = X.shape
    if kind == "none":
        return np.ones(shape, dtype=bool)
    if kind == "mcar":
        return rng.random(shape) >= rate
    if kind == "level":
        finest = rng.integers(0, n_levels, shape[2])
        obs = level[None, :, None] <= finest[None, None, :]
        return obs & (rng.random(shape) >= rate / 3)
    # block_mnar: half of the target rate from level-0 subtree blocks, half from value-dependent dropping.
    root = np.arange(len(level))
    for i in range(len(level)):
        r = i
        while parent[r]:
            r = max(parent[r], key=lambda pw: pw[1])[0]
        root[i] = r
    roots = np.unique(root)
    blocked = rng.random((len(roots), shape[2])) < rate / 2
    block = blocked[np.searchsorted(roots, root)] & (level > 0)[:, None]
    zs = (z - z.mean(axis=(0, 1))) / z.std(axis=(0, 1))
    p_drop = expit(2.0 * zs + _logit_offset(rate / 2))
    drop = rng.random(shape) < p_drop
    return ~(block[None] | drop)


def _logit_offset(p: float) -> float:
    """Intercept b with mean of expit(2 Z + b) close to p for Z ~ N(0, 1)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)) * np.sqrt(1 + np.pi * 4 / 8))
