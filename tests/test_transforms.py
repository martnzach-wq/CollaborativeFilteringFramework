import numpy as np

from regsim.prep.transforms import clr, forward, inverse, robust_scale


def test_roundtrip():
    x = np.array([0.1, 1.0, 10.0])
    for k in ["identity", "log", "log1p"]:
        assert np.allclose(inverse(forward(x, k), k), x)
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(inverse(forward(p, "logit"), "logit"), p)


def test_clr_sums_to_zero():
    c = clr(np.array([[0.2, 0.3, 0.5]]))
    assert abs(c.sum()) < 1e-12


def test_robust_scale():
    med, s = robust_scale(np.array([1.0, 2.0, 3.0, 4.0, np.nan]))
    assert med == 2.5 and s > 0
