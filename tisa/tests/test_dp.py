import numpy as np

from tisa.distance import TISADistance
from tisa.dp_align import dp_align
from tisa.segments import segmentize


def test_worked_example_zero_cost():
    X = np.array([0, 1, 3, 2, 1], dtype=float)
    Y = np.array([2, 1, 0, 1, 3], dtype=float)
    dist = TISADistance(normalize=False, numba=False)
    detail = dist.detailed(X, Y)
    assert detail["best_transform"] == "rev_inv"
    assert abs(detail["distance"]) < 1e-9


def test_dp_backtracking_consistency():
    X = np.array([0, 1, 3, 2, 1], dtype=float)
    seg_x = segmentize(X, tau_v=0.0, tau_t=1)
    Y = np.array([0, 2, 3, 1, 0], dtype=float)
    seg_y = segmentize(Y, tau_v=0.0, tau_t=1)
    result = dp_align(seg_x, seg_y, alpha=1.0, beta=1.0, gamma=0.5, use_numba=False)
    assert result.dp_matrix[-1, -1] == result.cost
    assert result.mapping
