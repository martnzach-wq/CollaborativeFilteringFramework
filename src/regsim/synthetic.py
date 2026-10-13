"""Synthetic hierarchical low-rank data with known factors (milestone M0)."""
from __future__ import annotations


def generate_hierarchical(
    n_per_level: tuple[int, ...] = (30, 300, 1500),
    n_indicators: int = 25,
    rank: int = 5,
    sigma_level: tuple[float, ...] = (1.0, 0.5, 0.3),
    noise: float = 0.1,
    dtypes: dict[str, float] | None = None,
    missingness: str = "level",
    seed: int = 0,
) -> dict:
    """Generate entities on nested levels with u_child = u_parent + N(0, sigma^2) and z = mu + U V^T + noise.

    Returns a dict with true factors (U, V), parent index, levels, population weights, the complete matrix,
    the observed mask and indicator metadata. `missingness` in {"mcar", "block_mnar", "level"}: "level"
    reproduces the Eurostat pattern where many indicators are observed only at coarse levels.
    """
    raise NotImplementedError("M0")
