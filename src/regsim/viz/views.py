"""Views V1-V8 and further diagnostic views (docs/visualisation.md).

Each view returns a matplotlib Figure, or a plotly Figure with ``interactive=True``. Quantitative statements are
computed in the full space; 2D views state their explained variance or trustworthiness. Lens, level and period
appear in the titles. Statistics of a view are attached as ``fig.regsim_stats`` (matplotlib) or
``fig.layout.meta`` (plotly).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, spearmanr

from regsim.models.base import ModelData
from regsim.schema import Lens
from regsim.similarity.entities import resolve_lens, s_profile
from regsim.similarity.indicators import i_model, i_observed

CATEGORICAL = plt.get_cmap("tab20").colors
DIVERGING = "RdBu_r"
GREY = "#d9d9d9"


# ---------------------------------------------------------------------------------------------------- helpers
def lens_label(data: ModelData, lens: Lens | None = None, level: int | None = None) -> str:
    """Short description of the lens for titles: indicator count, level and period."""
    li = resolve_lens(data, lens)
    parts = [f"{len(li.cols)} indicators"]
    if level is not None:
        parts.append(f"NUTS-{level}")
    if data.period is not None:
        parts.append("periods " + ", ".join(sorted(set(map(str, data.period)))))
    return ", ".join(parts)


def _rows_at(data: ModelData, lens: Lens | None, level: int | None) -> tuple[np.ndarray, np.ndarray]:
    li = resolve_lens(data, lens)
    rows = li.rows if level is None else li.rows[data.levels()[li.rows] == level]
    return rows, li.cols


def _standardised(model, data, lens, level):
    rows, cols = _rows_at(data, lens, level)
    X = model.predict()[np.ix_(rows, cols)]
    sd = X.std(axis=0)
    return rows, cols, (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _pca(X: np.ndarray, k: int = 2):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k], Vt[:k].T, S**2 / np.sum(S**2)


def _colours(labels) -> list:
    labels = np.asarray(labels)
    return [GREY if lab < 0 else CATEGORICAL[int(lab) % len(CATEGORICAL)] for lab in labels]


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _short(geo_id: str) -> str:
    return geo_id.split(":", 1)[-1]


def add_arrow(
    fig,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str,
    colour: str = "#444",
    hover: str | None = None,
) -> None:
    """Plotly arrow from (x0, y0) to (x1, y1) with its label at the tip (plotly puts annotation text at the tail);
    ``hover`` is shown when pointing at the label."""
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowcolor=colour,
    )
    dx, dy = x1 - x0, y1 - y0
    fig.add_annotation(
        x=x1,
        y=y1,
        text=label,
        hovertext=hover,
        showarrow=False,
        font={"size": 10, "color": colour if colour not in ("#444", "#999") else "#333"},
        xanchor="left" if dx >= 0 else "right",
        yanchor="bottom" if dy >= 0 else "top",
    )


def _plotly():
    import plotly.graph_objects as go

    return go


def _hex(c) -> str:
    if isinstance(c, str):
        return c
    return "#" + "".join(f"{int(255 * x):02x}" for x in c[:3])


# ---------------------------------------------------------------------------------------------------- V1
def biplot_coords(model, data: ModelData, lens=None, level: int = 3):
    """Rows, scores, loadings (scaled into the score range), indicator names and explained variance of the PCA of
    the standardised completed profiles (V1, row-principal scaling)."""
    rows, cols, X = _standardised(model, data, lens, level)
    scores, loadings, ev = _pca(X)
    names = [data.column_ids[j] for j in cols]
    scale = 0.9 * min(np.abs(scores[:, d]).max() / max(np.abs(loadings[:, d]).max(), 1e-12) for d in (0, 1))
    return rows, scores, loadings * scale, names, ev


def biplot(
    model, data: ModelData, lens=None, level: int = 3, colour_by=None, n_arrows: int = 12, interactive=False
):
    """V1: regions and indicator loadings in the first two principal components of the standardised completed
    profiles (row-principal scaling: point distances approximate region distances)."""
    rows, cols, X = _standardised(model, data, lens, level)
    scores, loadings, ev = _pca(X)
    names = [data.column_ids[j] for j in cols]
    length = np.sqrt((loadings**2).sum(axis=1))
    top = np.argsort(length)[::-1][:n_arrows]
    scale = 0.9 * min(np.abs(scores[:, d]).max() / max(np.abs(loadings[top, d]).max(), 1e-12) for d in (0, 1))
    ids = [data.geo_ids[i] for i in rows]
    labels = (
        None if colour_by is None else pd.Series(colour_by).reindex(ids).fillna(-1).astype(int).to_numpy()
    )
    title = f"V1 biplot ({lens_label(data, lens, level)})"
    stats = {"explained_variance": [float(ev[0]), float(ev[1])], "arrows": [names[i] for i in top]}
    xl, yl = f"PC 1 ({100 * ev[0]:.0f} %)", f"PC 2 ({100 * ev[1]:.0f} %)"
    if interactive:
        go = _plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode="markers",
                marker={"size": 5, "color": None if labels is None else [_hex(c) for c in _colours(labels)]},
                text=ids,
                customdata=ids,
                hovertemplate="%{text}<extra></extra>",
                name="regions",
            )
        )
        for i in top:
            x, y = loadings[i] * scale
            add_arrow(fig, 0.0, 0.0, x, y, names[i])
        fig.update_layout(title=title, xaxis_title=xl, yaxis_title=yl, meta=stats, dragmode="lasso")
        return fig
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        scores[:, 0], scores[:, 1], s=6, c="#9ecae1" if labels is None else _colours(labels), alpha=0.7
    )
    for i in top:
        x, y = loadings[i] * scale
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "#444"})
        ax.text(x * 1.05, y * 1.05, names[i], fontsize=7)
    ax.set(xlabel=xl, ylabel=yl, title=title)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)
    fig.regsim_stats = stats
    return fig


# ---------------------------------------------------------------------------------------------------- V2
def indicator_similarity(model, data: ModelData, level: int = 2, min_pairs: int = 30, interactive=False):
    """V2: model-implied correlation (I-model) next to the observed Spearman correlation at ``level``, both in the
    order of a hierarchical clustering of I-model; grey where fewer than ``min_pairs`` common entities exist."""
    rows = np.flatnonzero(data.levels() == level)
    R = i_model(model, data, rows)
    order = leaves_list(linkage(squareform(np.clip(1 - R.to_numpy(), 0, None), checks=False), "average"))
    names = [R.index[i] for i in order]
    obs = i_observed(data, rows, min_pairs)
    Obs = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    N = pd.DataFrame(0, index=R.index, columns=R.columns)
    for r in obs.itertuples():
        Obs.loc[r.indicator_a, r.indicator_b] = Obs.loc[r.indicator_b, r.indicator_a] = r.rho
        N.loc[r.indicator_a, r.indicator_b] = N.loc[r.indicator_b, r.indicator_a] = r.n
    for c in Obs.index:
        Obs.loc[c, c] = 1.0
    R, Obs, N = R.loc[names, names], Obs.loc[names, names], N.loc[names, names]
    stats = {"share_observed_pairs": float(np.isfinite(Obs.to_numpy()).mean())}
    title = f"V2 indicator similarity (NUTS-{level})"
    if interactive:
        from plotly.subplots import make_subplots

        go = _plotly()
        fig = make_subplots(
            1, 2, subplot_titles=("model-implied (I-model)", f"observed (≥ {min_pairs} pairs)")
        )
        fig.add_trace(go.Heatmap(z=R.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r"), 1, 1)
        fig.add_trace(
            go.Heatmap(
                z=Obs.values,
                x=names,
                y=names,
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
                customdata=N.values,
                hovertemplate="%{y} / %{x}: %{z:.2f} (n = %{customdata})<extra></extra>",
                showscale=False,
            ),
            1,
            2,
        )
        fig.update_layout(title=title, meta=stats)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    cmap = plt.get_cmap(DIVERGING).copy()
    cmap.set_bad(GREY)
    for ax, M, t in [
        (axes[0], R, "model-implied (I-model)"),
        (axes[1], Obs, f"observed (≥ {min_pairs} pairs)"),
    ]:
        im = ax.imshow(np.ma.masked_invalid(M.to_numpy()), cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [], fontsize=6)
        ax.set_title(t)
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation")
    fig.suptitle(title)
    fig.regsim_stats = stats
    return fig


# ---------------------------------------------------------------------------------------------------- V3
def embed_2d(Y: np.ndarray, method: str, seed: int = 0) -> tuple[np.ndarray, float | None]:
    """2D embedding (``pca`` or ``umap``) and, for PCA, the explained variance of the two components."""
    if method == "pca":
        scores, _, ev = _pca(Y)
        return scores, float(ev[:2].sum())
    if method == "umap":
        import umap

        return umap.UMAP(random_state=seed, n_jobs=1).fit_transform(Y), None
    raise ValueError(method)


def distortion(
    model,
    data: ModelData,
    lens=None,
    level: int = 3,
    methods=("pca", "umap"),
    n_sample: int = 400,
    k: int = 10,
    seed: int = 0,
    interactive=False,
):
    """V3: Shepard diagrams (whitened full-space distance against 2D distance) with trustworthiness (k
    neighbours) and the rank correlation of distances."""
    from sklearn.manifold import trustworthiness

    space = s_profile(model, data, lens)
    rows = space.lens.rows[data.levels()[space.lens.rows] == level]
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(rows, min(n_sample, rows.size), replace=False))
    Y = space.Y[rows]
    d_full = pdist(Y)
    stats, panels = {}, []
    for m in methods:
        try:
            E, ev = embed_2d(Y, m, seed)
        except ImportError:
            continue
        d2 = pdist(E)
        stats[m] = {
            "trustworthiness": float(trustworthiness(Y, E, n_neighbors=k)),
            "rank_correlation": float(spearmanr(d_full, d2).statistic),
            "explained_variance": ev,
        }
        panels.append((m, d2))
    title = f"V3 distortion of 2D projections ({lens_label(data, lens, level)}, {len(rows)} regions)"
    sub = [
        f"{m.upper()}: trust. {stats[m]['trustworthiness']:.2f}, rank corr. {stats[m]['rank_correlation']:.2f}"
        for m, _ in panels
    ]
    idx = rng.choice(d_full.size, min(20000, d_full.size), replace=False)
    if interactive:
        from plotly.subplots import make_subplots

        go = _plotly()
        fig = make_subplots(1, len(panels), subplot_titles=sub)
        for i, (_, d2) in enumerate(panels, start=1):
            fig.add_trace(
                go.Scattergl(
                    x=d_full[idx],
                    y=d2[idx],
                    mode="markers",
                    marker={"size": 2, "opacity": 0.3},
                    showlegend=False,
                ),
                1,
                i,
            )
        fig.update_xaxes(title_text="distance in full space")
        fig.update_yaxes(title_text="distance in 2D", col=1)
        fig.update_layout(title=title, meta=stats)
        return fig
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, (_, d2), t in zip(axes[0], panels, sub, strict=True):
        ax.scatter(d_full[idx], d2[idx], s=1, alpha=0.2)
        ax.set(xlabel="distance in full space", ylabel="distance in 2D", title=t)
    fig.suptitle(title)
    fig.regsim_stats = stats
    return fig


# ---------------------------------------------------------------------------------------------------- V4
def load_geometry(path: str | Path, level: int | None = None, simplify: float = 1000.0) -> dict:
    """GISCO region polygons (EPSG:3035, ``Shape`` as WKB) as {NUTS id: shapely geometry}, simplified to
    ``simplify`` metres."""
    import shapely

    g = pd.read_parquet(path, columns=["NUTS_ID", "LEVL_CODE", "Shape"])
    if level is not None:
        g = g[g.LEVL_CODE.eq(level)]
    geoms = shapely.from_wkb(g.Shape.to_numpy())
    if simplify:
        geoms = shapely.simplify(geoms, simplify, preserve_topology=True)
    return dict(zip(g.NUTS_ID, geoms, strict=True))


EXTENT = (2.5e6, 7.4e6, 1.35e6, 5.45e6)  # x and y limits of the European mainland view (EPSG:3035)


def _rings(geom) -> list[np.ndarray]:
    polys = getattr(geom, "geoms", [geom])
    return [np.asarray(p.exterior.coords) for p in polys if not p.is_empty]


def maps(
    panels: dict[str, pd.Series],
    geometry: dict,
    flags: pd.Series | None = None,
    ncols: int = 2,
    interactive=False,
    names: dict | None = None,
):
    """V4: choropleth panels {title: values by geo id}; integer or string series are drawn as categories,
    floats as a continuous scale. ``flags`` (geo id -> 'edge' / 'outside') outlines regions outside the
    applicability domain; ``names`` (geo id -> region name) labels the hover texts of the interactive map."""
    if interactive:
        return _maps_plotly(panels, geometry, names)
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)
    for ax, (title, s) in zip(axes.flat, panels.items(), strict=False):
        verts, colours = [], []
        categorical = not pd.api.types.is_float_dtype(s)
        if categorical:
            cats = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(sorted(s.dropna().unique()))}
        else:
            cmap = plt.get_cmap("viridis")
            lo, hi = np.nanmin(s), np.nanmax(s)
        for gid, geom in geometry.items():
            v = s.get(gid, np.nan)
            col = GREY
            if pd.notna(v):
                col = cats[v] if categorical else cmap((v - lo) / (hi - lo if hi > lo else 1))
            for ring in _rings(geom):
                verts.append(ring)
                colours.append(col)
        ax.add_collection(PolyCollection(verts, facecolors=colours, edgecolors="white", linewidths=0.1))
        if flags is not None:
            out = [
                r
                for gid, f in flags.items()
                if f in ("edge", "outside") and gid in geometry
                for r in _rings(geometry[gid])
            ]
            ax.add_collection(PolyCollection(out, facecolors="none", edgecolors="black", linewidths=0.4))
        ax.set_xlim(EXTENT[:2])
        ax.set_ylim(EXTENT[2:])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        if not categorical:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
            fig.colorbar(sm, ax=ax, shrink=0.6)
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    return fig


def _maps_plotly(panels: dict[str, pd.Series], geometry: dict, names: dict | None = None):
    """Plotly maps: outlines filled per class (one trace per class) and selectable centroid markers."""
    from plotly.subplots import make_subplots

    go = _plotly()
    fig = make_subplots(1, len(panels), subplot_titles=list(panels))
    for i, (_, s) in enumerate(panels.items(), start=1):
        categorical = not pd.api.types.is_float_dtype(s)
        cls = s if categorical else pd.cut(s, 8, labels=False)
        palette = (
            {c: _hex(CATEGORICAL[k % len(CATEGORICAL)]) for k, c in enumerate(sorted(cls.dropna().unique()))}
            if categorical
            else {k: _hex(plt.get_cmap("viridis")(k / 7)) for k in range(8)}
        )
        for c, colour in palette.items():
            xs, ys = [], []
            for gid in cls.index[cls == c]:
                for ring in _rings(geometry.get(gid, _EMPTY)):
                    xs += [*ring[:, 0], None]
                    ys += [*ring[:, 1], None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    fill="toself",
                    fillcolor=colour,
                    line={"width": 0.3, "color": "white"},
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                1,
                i,
            )
        ids = [g for g in s.index if g in geometry]
        cen = np.array([geometry[g].representative_point().coords[0] for g in ids]).reshape(-1, 2)
        fig.add_trace(
            go.Scatter(
                x=cen[:, 0],
                y=cen[:, 1],
                mode="markers",
                marker={"size": 3, "opacity": 0.01},
                customdata=ids,
                text=[
                    f"{names[g]} ({g}): {_fmt(s[g])}" if names and g in names else f"{g}: {_fmt(s[g])}"
                    for g in ids
                ],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            1,
            i,
        )
    fig.update_xaxes(visible=False, range=EXTENT[:2])
    fig.update_yaxes(visible=False, range=EXTENT[2:], scaleanchor="x")
    fig.update_layout(dragmode="lasso", margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return fig


class _Empty:
    geoms: list = []


_EMPTY = _Empty()


# ---------------------------------------------------------------------------------------------------- V5
def type_profiles(typology, interactive=False):
    """V5: standardised centroid deviation per type and indicator, with automatic label, size, dispersion and
    stability per type."""
    dev = typology.centroids
    tab = typology.cluster_table
    ylab = [
        f"{c}: {tab.loc[c, 'label']} (n={tab.loc[c, 'size']}, disp. {tab.loc[c, 'dispersion']:.2f}, "
        f"stab. {tab.loc[c, 'stability']:.2f})"
        for c in dev.index
    ]
    title = f"V5 type profiles (k = {typology.k}, stability {typology.stability:.2f}, hash {typology.config_hash})"
    if interactive:
        go = _plotly()
        fig = go.Figure(
            go.Heatmap(z=dev.values, x=list(dev.columns), y=ylab, zmin=-2, zmax=2, colorscale="RdBu_r")
        )
        fig.update_layout(title=title)
        return fig
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * dev.shape[1]), 0.5 * dev.shape[0] + 3))
    im = ax.imshow(dev.to_numpy(), cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(dev.shape[1]), dev.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(dev.shape[0]), ylab, fontsize=7)
    fig.colorbar(im, ax=ax, label="standardised centroid deviation")
    ax.set_title(title, fontsize=10)
    return fig


# ---------------------------------------------------------------------------------------------------- V6
def difference_profile(
    model,
    data: ModelData,
    geo_a: str,
    geo_b: str,
    lens=None,
    alpha: float = 0.1,
    interactive=False,
    labels: dict | None = None,
    hover: dict | None = None,
):
    """V6: per-indicator difference of two entities on the model scale, sorted; filled markers where both values
    are observed, open where at least one is predicted; bars are calibrated (1 - alpha) intervals of the
    difference (``model.predict_std``). ``labels`` (indicator id -> short label) and ``hover`` (id -> full name)
    replace the indicator ids in the display."""
    labels = labels or {}
    hover = hover or {}
    a, b = data.geo_ids.index(geo_a), data.geo_ids.index(geo_b)
    lens_cols = resolve_lens(data, lens).cols
    discrete = {"categorical", "ordinal"}
    cols = np.array([j for j in lens_cols if str(data.column_meta[j].dtype) not in discrete], dtype=int)
    P = model.predict()
    cat = [j for j in lens_cols if str(data.column_meta[j].dtype) in discrete]
    differ = [
        labels.get(data.column_ids[j], data.column_ids[j])
        for j in cat
        if round(float(P[a, j] if not data.mask[a, j] else data.Z[a, j]))
        != round(float(P[b, j] if not data.mask[b, j] else data.Z[b, j]))
    ]
    S = model.predict_std() if hasattr(model, "predict_std") else np.full(P.shape, np.nan)
    va = np.where(data.mask[a], data.Z[a], P[a])[cols]
    vb = np.where(data.mask[b], data.Z[b], P[b])[cols]
    sa = np.where(data.mask[a], 0.0, S[a])[cols]
    sb = np.where(data.mask[b], 0.0, S[b])[cols]
    diff = va - vb
    err = norm.ppf(1 - alpha / 2) * np.sqrt(sa**2 + sb**2)
    both = (data.mask[a] & data.mask[b])[cols]
    order = np.argsort(diff)
    ids = [data.column_ids[j] for j in cols]
    names = [labels.get(c, c) for c in ids]
    title = f"V6 difference profile {_short(geo_a)} − {_short(geo_b)} ({lens_label(data, lens)})"
    if cat:
        title += f"\ncategorical: {len(cat) - len(differ)} of {len(cat)} equal" + (
            f"; different: {', '.join(differ)}" if differ else ""
        )
    if interactive:
        go = _plotly()
        fig = go.Figure(
            go.Scatter(
                x=diff[order],
                y=[names[i] for i in order],
                mode="markers",
                error_x={"type": "data", "array": np.nan_to_num(err[order])},
                marker={"symbol": ["circle" if both[i] else "circle-open" for i in order], "size": 9},
                text=[
                    f"{hover.get(ids[i], ids[i])} ({ids[i]})<br>difference {diff[i]:.2f} ± {err[i]:.2f}"
                    f"<br>{'both observed' if both[i] else 'at least one predicted'}"
                    for i in order
                ],
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=title.replace("\n", "<br>"),
            xaxis_title="difference (robust-scaled)",
            height=max(400, 18 * len(order) + 160),
            yaxis={"dtick": 1, "automargin": True},
        )
        return fig
    fig, ax = plt.subplots(figsize=(8, 0.25 * len(cols) + 2))
    y = np.arange(len(cols))
    ax.errorbar(diff[order], y, xerr=np.nan_to_num(err[order]), fmt="none", ecolor="#999", lw=1)
    ax.scatter(diff[order][both[order]], y[both[order]], color="#08519c", zorder=3, label="both observed")
    ax.scatter(
        diff[order][~both[order]],
        y[~both[order]],
        facecolors="white",
        edgecolors="#08519c",
        zorder=3,
        label="at least one predicted",
    )
    ax.axvline(0, color="#ccc")
    ax.set_yticks(y, [names[i] for i in order], fontsize=7)
    ax.set(xlabel="difference (robust-scaled)", title=title)
    ax.legend(fontsize=7, loc="lower right")
    return fig


# ---------------------------------------------------------------------------------------------------- V7
def trajectories(
    model, data: ModelData, geo_ids: list[str], level: int = 2, relative: bool = False, interactive=False
):
    """V7: geo-time entities of ``geo_ids`` in a PCA space shared by all periods at ``level``; arrows from the
    first to the last period. ``relative=True`` removes the mean displacement of all entities at the level."""
    if data.period is None:
        raise ValueError("trajectories need geo-time data (several period windows)")
    rows, cols, X = _standardised(model, data, None, level)
    scores, _, ev = _pca(X)
    ids = [data.geo_ids[i] for i in rows]
    base = [g.split("@")[0] for g in ids]
    per = [g.split("@")[1] for g in ids]
    first, last = min(per), max(per)
    pos = {(b, p): s for b, p, s in zip(base, per, scores, strict=True)}
    disp = {b: pos[(b, last)] - pos[(b, first)] for b in set(base) if (b, first) in pos and (b, last) in pos}
    drift = np.mean(list(disp.values()), axis=0) if relative else np.zeros(2)
    med = float(np.median([np.linalg.norm(v - drift) for v in disp.values()]))
    title = (
        f"V7 trajectories {first} → {last} (NUTS-{level}, PC 1+2 {100 * ev[:2].sum():.0f} %, "
        f"median displacement {med:.2f}{', relative to mean drift' if relative else ''})"
    )
    stats = {
        "median_displacement": med,
        "explained_variance": [float(ev[0]), float(ev[1])],
        "displacement": {g: float(np.linalg.norm(disp[g] - drift)) for g in geo_ids if g in disp},
    }
    sel = [g for g in geo_ids if g in disp]
    if interactive:
        go = _plotly()
        fig = go.Figure(
            go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode="markers",
                marker={"size": 3, "color": GREY},
                text=ids,
                hoverinfo="text",
                showlegend=False,
            )
        )
        for g in sel:
            p0, p1 = pos[(g, first)], pos[(g, first)] + disp[g] - drift
            add_arrow(fig, p0[0], p0[1], p1[0], p1[1], _short(g), colour="#08519c")
        fig.update_layout(
            title=title,
            xaxis_title=f"PC 1 ({100 * ev[0]:.0f} %)",
            yaxis_title=f"PC 2 ({100 * ev[1]:.0f} %)",
            meta=stats,
        )
        return fig
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(scores[:, 0], scores[:, 1], s=3, color=GREY)
    for g in sel:
        p0, p1 = pos[(g, first)], pos[(g, first)] + disp[g] - drift
        ax.scatter(*p0, s=12, color="#9ecae1")
        ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->", "color": "#08519c"})
        ax.text(*p1, _short(g), fontsize=7)
    ax.set(xlabel=f"PC 1 ({100 * ev[0]:.0f} %)", ylabel=f"PC 2 ({100 * ev[1]:.0f} %)")
    ax.set_title(title, fontsize=9)
    fig.regsim_stats = stats
    return fig


# ---------------------------------------------------------------------------------------------------- V8
def benchmark_chart(results: pd.DataFrame, interactive=False):
    """V8a: mean hold-out RMSE by method and masking scheme."""
    tab = results.groupby(["scheme", "method"]).rmse.mean().unstack("scheme")
    tab = tab.loc[tab.mean(axis=1).sort_values().index]
    if interactive:
        go = _plotly()
        fig = go.Figure([go.Bar(name=s, x=tab.index, y=tab[s]) for s in tab.columns])
        fig.update_layout(barmode="group", title="V8 hold-out RMSE by method and scheme", yaxis_title="RMSE")
        return fig
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / tab.shape[1]
    x = np.arange(tab.shape[0])
    for i, s in enumerate(tab.columns):
        ax.bar(x + i * w, tab[s], w, label=s)
    ax.set_xticks(x + 0.4 - w / 2, tab.index, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="RMSE (robust-scaled)", title="V8 hold-out RMSE by method and scheme")
    ax.legend(fontsize=7)
    return fig


def cold_start_chart(results: pd.DataFrame, interactive=False):
    """V8b: cold-start RMSE against the number of known indicators per method."""
    cs = results[results.scheme.str.startswith("M2")].groupby(["method", "k"]).rmse.mean().unstack("method")
    if interactive:
        go = _plotly()
        fig = go.Figure([go.Scatter(x=cs.index, y=cs[m], mode="lines+markers", name=m) for m in cs.columns])
        fig.update_layout(title="V8 cold start", xaxis_title="known indicators", yaxis_title="RMSE")
        return fig
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in cs.columns:
        ax.plot(cs.index, cs[m], marker="o", label=m)
    ax.set(xlabel="known indicators of the new region", ylabel="RMSE (robust-scaled)", title="V8 cold start")
    ax.legend(fontsize=7)
    return fig


def model_card(results: pd.DataFrame, interactive=False):
    """V8: benchmark and cold-start charts."""
    return benchmark_chart(results, interactive), cold_start_chart(results, interactive)


# ---------------------------------------------------------------------------------------------------- further views
def coverage_matrix(data: ModelData, interactive=False):
    """Observed cells (entities sorted by level and country x indicators)."""
    order = np.lexsort(
        (data.country if data.country is not None else np.zeros(len(data.geo_ids)), data.levels())
    )
    M = data.mask[order].astype(float)
    title = f"Coverage: {100 * data.mask.mean():.0f} % of cells observed"
    if interactive:
        go = _plotly()
        fig = go.Figure(
            go.Heatmap(
                z=M,
                x=data.column_ids,
                y=[data.geo_ids[i] for i in order],
                showscale=False,
                colorscale=[[0, "white"], [1, "#08519c"]],
            )
        )
        fig.update_layout(title=title)
        return fig
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(M, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(data.column_ids)), data.column_ids, rotation=90, fontsize=6)
    ax.set(ylabel="entities (by level and country)", title=title)
    return fig


def consensus_heatmap(typology, interactive=False):
    """Consensus matrix of a typology run, ordered by cluster."""
    order = np.argsort(typology.labels, kind="stable")
    C = typology.consensus[np.ix_(order, order)]
    title = f"Consensus matrix (k = {typology.k}, mean ARI {typology.stability:.2f})"
    if interactive:
        go = _plotly()
        fig = go.Figure(go.Heatmap(z=C, zmin=0, zmax=1, colorscale="Greys"))
        fig.update_layout(title=title)
        return fig
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(C, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title)
    return fig


def calibration_plot(
    pred: np.ndarray, sd: np.ndarray, truth: np.ndarray, test: np.ndarray, alpha: float = 0.1
):
    """Observed against predicted held-out values with (1 - alpha) intervals and the empirical coverage."""
    z = norm.ppf(1 - alpha / 2)
    p, s, t = pred[test], sd[test], truth[test]
    cov = float(np.mean(np.abs(t - p) <= z * s))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(p, t, xerr=z * s, fmt="o", ms=2, alpha=0.3, elinewidth=0.5)
    lim = [min(p.min(), t.min()), max(p.max(), t.max())]
    ax.plot(lim, lim, color="#999")
    ax.set(
        xlabel="predicted",
        ylabel="observed",
        title=f"Calibration: {100 * (1 - alpha):.0f} % intervals cover {100 * cov:.1f} %",
    )
    fig.regsim_stats = {"coverage": cov}
    return fig


def transitions(trans: pd.DataFrame, interactive=False):
    """Type transitions between two typologies (rows: from, columns: to); Sankey diagram if interactive."""
    if interactive:
        go = _plotly()
        src = [f"a{r}" for r in trans.index]
        dst = [f"b{c}" for c in trans.columns]
        nodes = src + dst
        links = [
            (i, len(src) + j, int(trans.iat[i, j]))
            for i in range(len(src))
            for j in range(len(dst))
            if trans.iat[i, j] > 0
        ]
        fig = go.Figure(
            go.Sankey(
                node={"label": nodes},
                link={
                    "source": [a for a, _, _ in links],
                    "target": [b for _, b, _ in links],
                    "value": [v for _, _, v in links],
                },
            )
        )
        fig.update_layout(title="Type transitions")
        return fig
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(trans.to_numpy(), cmap="Blues")
    for (i, j), v in np.ndenumerate(trans.to_numpy()):
        if v:
            ax.text(j, i, int(v), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(trans.shape[1]), trans.columns)
    ax.set_yticks(range(trans.shape[0]), trans.index)
    ax.set(xlabel="to", ylabel="from", title="Type transitions")
    return fig
