import numpy as np

from regsim.synthetic import generate_hierarchical


def test_m0_acceptance():
    """Acceptance: generator reproducible by seed, three levels, mixed types, all missingness schemes."""
    dtypes = {"continuous": 0.5, "share": 0.2, "count": 0.2, "ordinal": 0.1}
    for missingness in ["mcar", "block_mnar", "level"]:
        a = generate_hierarchical(dtypes=dtypes, missingness=missingness, n_periods=2, seed=11)
        b = generate_hierarchical(dtypes=dtypes, missingness=missingness, n_periods=2, seed=11)
        assert np.array_equal(a.X, b.X) and np.array_equal(a.observed, b.observed)
        assert len(set(a.level.tolist())) == 3
        assert len({ind.dtype for ind in a.indicators}) == 4
