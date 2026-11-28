"""Strategy dashboard (D66) on a synthetic adapter: margins, bands, contributions, figures and app layout."""

import pandas as pd
import pytest

from regsim.app.strategy_adapter import (
    LeverSpec,
    OptionSpec,
    SeriesSpec,
    StrategyAdapter,
    TargetSpec,
    contributions,
    margins,
    target_status,
)

YEARS = list(range(2019, 2051))


class Toy(StrategyAdapter):
    title = {"de": "Testdashboard", "en": "Test dashboard"}
    sectors = {"mob": {"de": "Mobilität", "en": "Mobility"}, "heat": {"de": "Hitze", "en": "Heat"}}
    levers = [
        LeverSpec("pricing", "mob", {"de": "Bepreisung", "en": "Pricing"}, level={"en": "EUR 2 per trip"}),
        LeverSpec(
            "trees",
            "heat",
            {"de": "Bäume", "en": "Trees"},
            effect="modelled",
            confidence="assumed",
            gaps=("G1",),
        ),
    ]
    presets = {
        "plan": {"label": {"en": "Plan"}, "multipliers": {"pricing": 1.0, "trees": 1.0}},
        "double": {"label": {"en": "Double"}, "multipliers": {"pricing": 2.0, "trees": 2.0}},
    }
    targets = [
        TargetSpec("emissions", 2030, -0.5, {"en": "emissions"}, relative_to=2019),
        TargetSpec("heat", 2050, 0.0, {"en": "heat"}, relative_to=2025),
        TargetSpec("car_share", 2030, 0.15, {"en": "car share"}, hard=False),
    ]
    pathways = [SeriesSpec("emissions", {"en": "Emissions"}, {"en": "% vs 2019"}, relative_to=2019)]
    stacks = {"mob": [SeriesSpec("emissions", {"en": "all"}, {"en": "kt"})]}
    options = [OptionSpec("climate", {"en": "climate"}, (("a", {"en": "A"}), ("b", {"en": "B"})), "a")]
    response_bands = ("weak", "strong")
    climate_series = [SeriesSpec("heat", {"en": "Heat"}, {"en": "deaths"})]

    def run(self, multipliers, options):
        f = {"weak": 0.5, "strong": 1.5}.get(options.get("band"), 1.0)
        p, tr_ = multipliers.get("pricing", 1.0), multipliers.get("trees", 1.0)
        e = [
            100.0 * (1 - f * 0.3 * p * max(0, y - 2019) / 11) if y <= 2030 else 100.0 * (1 - f * 0.3 * p)
            for y in YEARS
        ]
        h = [100.0 * (1 + 0.01 * (y - 2025) - 0.004 * tr_ * max(0, y - 2025)) for y in YEARS]
        return pd.DataFrame({"emissions": e, "heat": h, "car_share": [0.2] * len(YEARS)}, index=YEARS)

    def run_climate_band(self, multipliers, options):
        base = self.run(multipliers, options)
        rows = [
            {"member": m, "year": y, "quantity": "heat", "value": base.loc[y, "heat"] * k}
            for m, k in (("cool", 0.9), ("hot", 1.2))
            for y in YEARS
            if y >= 2025
        ]
        return pd.DataFrame(rows)


def test_margins_and_status_with_bands():
    a = Toy()
    central = a.run(a.defaults(), {})
    m = margins(central, a.targets)
    assert m.loc[0, "model"] == pytest.approx(-0.3)  # -30 % in 2030
    assert m.loc[0, "margin"] == pytest.approx(-0.2)  # target -50 %: missed by 20 pp
    st = target_status(
        central,
        a.targets,
        {b: a.run(a.defaults(), {"band": b}) for b in a.response_bands},
        a.run_climate_band(a.defaults(), {}),
    )
    assert st.loc[0, "margin_strong"] == pytest.approx(-0.05)
    assert st.loc[1, "margin_climate_min"] == pytest.approx(
        st.loc[1, "margin"]
    )  # scaling keeps relative change
    assert not st.loc[2, "hard"]


def test_contributions_point_to_the_helpful_lever():
    a = Toy()
    c = contributions(a, a.defaults(), {}, [x for x in a.targets if x.hard])
    em = c[(c.quantity == "emissions")].set_index("lever").contribution
    assert em["pricing"] > 0 and em["trees"] == pytest.approx(0.0)
    heat = c[(c.quantity == "heat")].set_index("lever").contribution
    assert heat["trees"] > 0


def test_figures_and_app_build():
    pytest.importorskip("dash")
    from regsim.app.strategy_dash import (
        create_strategy_app,
        fig_climate,
        fig_contributions,
        fig_series,
        status_rows,
    )

    a = Toy()
    central = a.run(a.defaults(), {})
    bands = {b: a.run(a.defaults(), {"band": b}) for b in a.response_bands}
    fig = fig_series(a.pathways[0], central, "de", bands, None, a.targets)
    assert len(fig.data) == 4  # band (2), model, targets
    fig = fig_climate(a.climate_series[0], a.run_climate_band(a.defaults(), {}), central, "en")
    assert len(fig.data) == 4
    c = contributions(a, a.defaults(), {}, a.targets[:1])
    assert len(fig_contributions(c, "emissions", 2030, {}, "x").data) == 1
    rows = status_rows(target_status(central, a.targets), a.targets, "de")
    assert rows[0]["status"].startswith("✗") and rows[2]["kind"] == "berichtet"
    app = create_strategy_app(a, lang="en")
    assert app.server is not None
    text = str(app.layout)
    assert "Test dashboard" in text and "Pricing" in text and "G1" in text
