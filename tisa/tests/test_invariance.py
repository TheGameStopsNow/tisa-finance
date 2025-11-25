import numpy as np
from hypothesis import given, strategies as st

from tisa.distance import TISADistance


def arrays():
    return st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=5, max_size=20).map(np.array)


@given(arrays())
def test_reverse_invert_invariance(x):
    dist = TISADistance(normalize=True, numba=False)
    y = x[::-1] * -1
    detail = dist.detailed(x, y)
    assert detail["best_transform"] in {"rev", "rev_inv", "inv", "orig"}
    if np.allclose(x, y[::-1] * -1):
        assert detail["distance"] < 1e-6


@given(arrays(), st.floats(min_value=-0.01, max_value=0.01))
def test_noise_stability(x, noise):
    dist = TISADistance(normalize=True, numba=False)
    y = x + noise
    d1 = dist.pairwise(x, x)
    d2 = dist.pairwise(x, y)
    assert d2 >= d1
