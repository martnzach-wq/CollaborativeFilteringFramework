"""Worked example of regsim: similarity, prediction and typology (Vienna) and the causal layer (Estonia).

Run from the repository root after the builds and fits:

    regsim build configs/example.yaml        # NUTS-0/2/3 store, two period windows
    regsim build configs/causal.yaml         # annual NUTS-0/2 panels 2008-2024
    regsim causal fit --config configs/causal.yaml
    regsim causal effects --config configs/causal.yaml
    REGSIM_SD_MODEL=<checkout of KNOWING-integrated-SD-model> python examples/walkthrough.py

Figures go to docs/examples/figures, key numbers to docs/examples/walkthrough_results.json
(docs/examples/walkthrough.md explains them).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs/examples")
FIG = OUT / "figures"
RESULTS: dict = {}
warnings.filterwarnings("ignore")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def names_of(store_path: str) -> dict:
    from regsim.store.store import ObservationStore

    u = ObservationStore(store_path).units()
    return dict(zip(u.geo_id, u.name, strict=True))


# ----------------------------------------------------------------------------------- part 1: Vienna
def labels_of(cfg) -> dict:
    """Indicator id -> readable label from the indicator registry, shortened for axis labels."""
    from regsim.registry import IndicatorRegistry

    reg = IndicatorRegistry.from_yaml(*cfg.data.registries)
    return {i: e.label if len(e.label) <= 34 else e.label[:32] + "…" for i, e in reg.entries.items()}


def part_similarity() -> None:
    from regsim.api.runtime import fit, resolve_id, short_id
    from regsim.config import RunConfig
    from regsim.predict.tasks import _to_model, analogues, fill_gaps, vision_entity
    from regsim.schema import Lens
    from regsim.similarity.entities import nearest, s_profile
    from regsim.typology.run import TypologyRun, run_typology
    from regsim.viz import views as V

    cfg = RunConfig.from_yaml("configs/example.yaml")
    f = fit(cfg)
    model, data = f.model, f.data
    names = names_of(cfg.data.store)
    ind = labels_of(cfg)
    wien = resolve_id(data, "AT130")

    # 1 data coverage
    save(V.coverage_matrix(data), "01_coverage")
    RESULTS["data"] = {
        "entities": int(data.Z.shape[0]),
        "indicators": int(data.Z.shape[1]),
        "observed_share": float(data.mask.mean()),
        "period": cfg.data.periods[-1],
    }

    # 2 nearest regions to Vienna (NUTS-3, all indicators)
    nb = nearest(s_profile(model, data, Lens(levels=(3,))), wien, k=10)
    RESULTS["nearest_vienna"] = [
        {"region": names.get(g, g), "code": short_id(g), "score": round(float(s), 1), "differs_most_in": c}
        for g, s, c in zip(nb.geo_id, nb.score, nb.top_contributors, strict=True)
    ]

    # 3 difference profile Vienna - nearest region
    partner = nb.geo_id.iloc[0]
    save(V.difference_profile(model, data, wien, partner, labels=ind), "02_difference_profile")
    RESULTS["difference_pair"] = [names.get(wien), names.get(partner)]

    # 4 gap filling for Vienna: predicted values with 90 % intervals
    gaps = fill_gaps(model, wien)
    pred = gaps[~gaps.observed].sort_values("indicator")
    RESULTS["vienna_gaps"] = {
        "observed": int(gaps.observed.sum()),
        "predicted": [
            {"indicator": r.indicator, "value": round(float(r.value), 3), "lower": round(float(r.lower), 3),
             "upper": round(float(r.upper), 3)}
            for r in pred.itertuples()
        ],
    }  # fmt: skip

    # 5 typology of NUTS-2 regions, map and profiles
    ty = run_typology(f.base, data, TypologyRun(lens=Lens(levels=(2,)), n_boot=10, seed=0))
    labels = pd.Series(ty.labels, index=ty.geo_ids)
    geometry = V.load_geometry(Path(cfg.data.raw_dir) / "gisco" / "NUTS_RG_01M_2024_3035.parquet", level=2)
    title = f"NUTS-2 typology: k = {ty.k}, bootstrap stability (ARI) {ty.stability:.2f}"
    type_names = {t: f"type {t}" for t in sorted(labels.unique())}
    save(
        V.maps({title: labels.rename(index=short_id)}, geometry, ncols=1, legend=type_names),
        "03_typology_map",
    )
    save(V.type_profiles(ty), "04_type_profiles")
    RESULTS["typology"] = {
        "k": int(ty.k),
        "stability": float(ty.stability),
        "vienna_type": int(labels.get(resolve_id(data, "AT13"), -1)),
        "types": ty.cluster_table.reset_index().astype(str).to_dict("records"),
    }

    # 6 biplot (V1) at NUTS-2, coloured by type
    save(V.biplot(f.base, data, level=2, colour_by=labels, labels=ind, colour_names=type_names), "05_biplot")

    # 7 vision: Vienna with 60 % tertiary attainment, and real analogues of that vision
    vision = vision_entity(model, wien, {"tertiary_attainment": 60.0})
    profile = dict(zip(vision.indicator, vision.vision, strict=True))
    ana = analogues(model, profile, k=6, level=3)
    ana = ana[ana.geo_id != wien]  # the vision stays close to Vienna itself
    # compare changes on the model scale (transformed, robust-scaled), not relative to near-zero values
    col = {c: j for j, c in enumerate(data.column_ids)}
    vision["shift"] = [
        float(_to_model(data, col[i], v) - _to_model(data, col[i], p))
        if i in col and np.isfinite(v)
        else np.nan
        for i, v, p in zip(vision.indicator, vision.vision, vision.present, strict=True)
    ]
    top = vision[~vision.target].dropna(subset=["shift"])
    top = top.reindex(top["shift"].abs().sort_values(ascending=False).index).head(6)
    RESULTS["vision"] = {
        "target": {"tertiary_attainment": [float(vision.set_index("indicator").present["tertiary_attainment"]), 60.0]},
        "largest_changes": [
            {"indicator": r.indicator, "present": round(float(r.present), 2), "vision": round(float(r.vision), 2),
             "lower": round(float(r.lower), 2), "upper": round(float(r.upper), 2), "shift": round(float(r.shift), 2)}
            for r in top.itertuples()
        ],
        "analogues": [names.get(g, g) for g in ana.geo_id],
    }  # fmt: skip


# ----------------------------------------------------------------------------------- part 2: Estonia
def part_causal() -> None:
    import pickle

    from regsim.api.runtime import short_id
    from regsim.causal.check import run_check
    from regsim.causal.events import effect, load_events
    from regsim.causal.run import panel, run_what_if
    from regsim.config import RunConfig
    from regsim.similarity.effect import s_effect
    from regsim.viz import effect_views as EV
    from regsim.viz import views as V

    cfg = RunConfig.from_yaml("configs/causal.yaml")
    names = names_of(cfg.data.store)
    labels = labels_of(cfg)

    # 8 consistency of the edge registry with the data (C1)
    _, rates, _ = run_check(cfg)
    r = rates[rates.grade == "all"].set_index("test")
    RESULTS["c1"] = {k: {"agree": int(r.loc[k, "agree"]), "n": int(r.loc[k, "n"])} for k in r.index}

    # 9 what-if: +10 % GDP in Estonia (short run)
    w_all = run_what_if(cfg, "EE00", {"gdp_pps_hab": 0.10})
    rates = {"net_migration_rate"}  # rates around zero: absolute change, not a relative one
    RESULTS["what_if_absolute"] = {
        i: round(float(c), 3) for i, c in zip(w_all.indicator, w_all.change_abs, strict=True) if i in rates
    }
    w = w_all[w_all.change.notna() & ~w_all.indicator.isin(rates)].copy()
    w = w.reindex(w.change.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(w) + 1.2))
    y = np.arange(len(w))
    ax.barh(y, 100 * w.change, color=["#1f5fa8" if "effect_model" in s else "#8aa9cf" for s in w.sources])
    ax.errorbar(100 * w.change, y, xerr=[100 * (w.change - w.lower), 100 * (w.upper - w.change)], fmt="none",
                color="black", capsize=2)  # fmt: skip
    ax.set_yticks(y, [labels.get(i, i) for i in w.indicator])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("change in % (short run, 90 % interval)")
    ax.set_title("What if GDP per inhabitant in Estonia rises by 10 %?", fontsize=10)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color="#1f5fa8", label="path uses the region-varying effect model"), Patch(color="#8aa9cf", label="pooled panel estimates (country group) only")],
        fontsize=8, loc="lower right",
    )  # fmt: skip
    save(fig, "06_what_if_estonia")
    RESULTS["what_if"] = [
        {"indicator": i, "change_pct": round(100 * c, 2), "lower": round(100 * lo, 2), "upper": round(100 * hi, 2),
         "sources": s, "tier": t}
        for i, c, lo, hi, s, t in zip(w.indicator, w.change, w.lower, w.upper, w.sources, w.tier, strict=True)
    ]  # fmt: skip

    # 10 effect similarity Estonia vs Catalonia, effect map
    folder = Path(cfg.output_dir) / f"causal-effects-{cfg.config_hash()}"
    fits = pickle.loads((folder / "fits.pkl").read_bytes())
    table = s_effect(fits, "NUTS2024:EE00", "NUTS2024:ES51", delta=0.1)
    save(EV.pairwise_comparison(table, {"NUTS2024:EE00": "Estonia", "NUTS2024:ES51": "Cataluña"}, labels),
         "07_effect_comparison")  # fmt: skip
    region_effects = pd.read_csv(folder / "region_effects.csv")
    geometry = V.load_geometry(Path(cfg.data.raw_dir) / "gisco" / "NUTS_RG_01M_2024_3035.parquet", level=2)
    save(EV.effect_map(region_effects, "gdp_pps_hab->life_expectancy", geometry, labels), "08_effect_map")
    RESULTS["effect_similarity"] = [
        {"link": t.link, "p_equivalent": None if pd.isna(t.p_equivalent) else round(float(t.p_equivalent), 2),
         "status": t.status}
        for t in table.itertuples()
    ]  # fmt: skip

    # 11 intervention: Estonian motor vehicle tax (announced 2024, in force 2025)
    p = panel(cfg)
    events = load_events("configs/events.yaml")
    eff = effect(None, events, "cars_per_1000", analogues=20, placebo=200, seed=0, panel=p)
    save(EV.event_plot(p, "cars_per_1000", eff, {"NUTS2024:EE00": "Estonia"}, label="log cars per 1,000"),
         "09_event_estonia")  # fmt: skip
    s = eff.attrs["summary"]
    RESULTS["event"] = s.astype(object).where(s.notna(), None).to_dict("records")

    # 12 SD model levers in three cities and where demonstrator results transfer
    sd_path = os.environ.get("REGSIM_SD_MODEL")
    if sd_path:
        from regsim.causal import sd_knowing as K
        from regsim.causal.run import lens_matrix
        from regsim.causal.sd_bridge import domain_check, load_parameter_classes

        m = K.load(sd_path)
        lev = pd.concat([K.lever_table(m, c) for c in ("TALLINN", "GRANOLLERS", "NAPLES")])
        lev = lev[lev.output == "car_km_cap"]
        wide = lev.pivot(index="lever", columns="city", values="effect")
        wide = wide.reindex(wide.abs().max(axis=1).sort_values().index)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        (100 * wide.rename(index=lambda s: s.removeprefix("policy_").replace("_", " "))).plot.barh(ax=ax)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("effect of the lever on car-km per inhabitant in 2050 (%)")
        ax.set_ylabel("")
        ax.set_title("Policy levers in the KNOWING SD model (full policy, lever removed)", fontsize=10)
        save(fig, "10_sd_levers")
        RESULTS["sd_levers"] = (100 * wide).round(1).astype(object).where(wide.notna(), None).to_dict()
        lens = load_parameter_classes()["domain_lens"]
        dom = domain_check(lens_matrix(cfg, lens), None, cfg.causal.sd_demonstrators, lens)
        title = "Distance to the nearest SD demonstrator (Tallinn, Granollers, Naples)"
        order = {
            "close": "1 close",
            "moderate": "2 moderate",
            "distant": "3 distant",
            "extrapolation": "4 extrapolation",
        }
        save(V.maps({title: dom.domain_class.map(order)}, geometry, ncols=1), "11_sd_domain")
        RESULTS["sd_domain"] = dom.domain_class.value_counts().to_dict()
        RESULTS["sd_domain_vienna"] = dom.loc["AT13"].domain_class if "AT13" in dom.index else None
    RESULTS["names"] = {"EE00": names.get("NUTS2024:EE00"), "ES51": names.get("NUTS2024:ES51")}
    _ = short_id


if __name__ == "__main__":
    part_similarity()
    part_causal()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "walkthrough_results.json").write_text(
        json.dumps(RESULTS, indent=1, default=str), encoding="utf-8"
    )
    print("results written to", OUT / "walkthrough_results.json")
