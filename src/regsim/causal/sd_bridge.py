"""Bridge between regsim and system dynamics models (option C4).

Functions
---------
- `parse_vensim_edges`: structural edges (cause -> effect) from a Vensim `.mdl` file.
  M7 extracted 109 edges from Submodel_Mobility_v10.mdl and 69 from
  Submodel_EnergySupply_v4.mdl.
- `perturbation_elasticities`: run a Python SD model with one input perturbed at a time
  and return output elasticities at a target year (M7: KNOWING model, TALLINN, 2050).
- `map_to_registry`: map SD variables to regsim indicators via a mapping table with a
  `relation` column (same, proxy, inverse_proxy). Inverse proxies flip the sign.
- `compare`: compare SD elasticities with panel and literature priors; flag sign
  conflicts and differences above a factor threshold as `model_review`.
- `export_inputs`: regsim estimates (imputed values, intervals) for SD model inputs of a
  follower region.
- `transfer_parameters`: transfer calibrated parameters from a pioneer to a follower,
  weighted by `similarity.causal_lens` on the parameter's causes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def parse_vensim_edges(mdl_path: str | Path) -> pd.DataFrame:
    """Return columns model, cause, effect, effect_is_stock, cause_is_constant."""
    raise NotImplementedError("M7.4")


def perturbation_elasticities(
    run_model: Callable[[dict], pd.DataFrame],
    baseline: dict,
    inputs: list[str],
    outputs: list[str],
    rel_step: float = 0.1,
    year: int = 2050,
) -> pd.DataFrame:
    """Central-difference elasticities d log(output) / d log(input) at `year`."""
    raise NotImplementedError("M7.4")


def map_to_registry(sd_edges: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("M7.4")


def compare(sd_elasticities: pd.DataFrame, registry, factor: float = 3.0) -> pd.DataFrame:
    raise NotImplementedError("M7.4")


def export_inputs(model, geo_id: str, inputs: list[str], alpha: float = 0.1) -> pd.DataFrame:
    raise NotImplementedError("M7.4")


def transfer_parameters(params: pd.DataFrame, pioneer: str, follower: str, registry, model) -> pd.DataFrame:
    """Set parameters for a follower region by class (configs/sd_parameter_classes.yaml):
    O from data or regsim imputation with interval; B from the literature function;
    R recalibrated to follower targets (never copied); S identical to the donor variant.
    Returns one row per parameter with value, interval, class and source."""
    raise NotImplementedError("M7.4")


# ---- parameter-transfer protocol (docs/causal/sd_parameter_transfer.md) --------------


def load_parameter_classes(path: str | Path = "configs/sd_parameter_classes.yaml") -> dict:
    """Classes O (observable), B (behavioural, literature), R (residual calibration),
    S (structural switch), P (policy lever) with their parameter paths."""
    raise NotImplementedError("M7.4")


def domain_check(model, geo_id: str, demonstrators: list[str], lens: list[str]) -> dict:
    """Distance to the nearest demonstrator in the standardised lens, outside-range flags per
    dimension, and domain class (close < 0.5, moderate < 0.75, distant < 1.0, extrapolation).
    M7 reference: 82 % of NUTS-2 regions are outside the range of EE00/ES51/ITF3 on at least
    one dimension; median nearest distance 0.87."""
    raise NotImplementedError("M7.4")


def recalibrate_residuals(
    run_model: Callable[[dict], pd.DataFrame],
    city_config,
    targets: pd.DataFrame,
    residual_params: list[str],
    n_draws: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Calibrate class-R parameters to target mode split and trip lengths. When targets are
    regsim imputations, repeat over draws from their predictive distribution and return the
    parameter distribution."""
    raise NotImplementedError("M7.4")


def cross_calibrate(cities: dict, run_model: Callable[[dict], pd.DataFrame], observed: dict) -> pd.DataFrame:
    """Leave-one-city-out transfer test: calibrate on donor, transfer by protocol to target,
    compare with target observations and target lever responses. Reports error by parameter
    class and Spearman agreement of lever rankings for every directed pair."""
    raise NotImplementedError("M7.4")


def lever_ranking(lever_effects: pd.DataFrame) -> pd.DataFrame:
    """Rank levers by effect per city and output; agreement across cities (M7 reference:
    road pricing > cycling attractiveness > parking adaptation on car-km in Tallinn and
    Granollers, while magnitudes differ up to a factor of four)."""
    raise NotImplementedError("M7.4")


def report_rules(domain_class: str, residuals_from_observed: bool) -> dict:
    """Which outputs may be reported: rankings and directions always; absolute levels only for
    close/moderate regions or when class-R parameters were recalibrated to observed targets."""
    return {
        "rankings": True,
        "directions": True,
        "levels": domain_class in ("close", "moderate") or residuals_from_observed,
    }


def next_demonstrator(
    lens_matrix: pd.DataFrame, demonstrators: list[str], candidates: list[str] | None = None
) -> pd.DataFrame:
    """Rank candidate regions by the reduction in median distance to the nearest demonstrator
    (M7 reference: a central or western European regional city region such as DED2, DE94,
    FRF1 or NL13 raises the share of regions within 0.75 from 0.29 to about 0.58)."""
    raise NotImplementedError("M7.4")
