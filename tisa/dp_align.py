"""Dynamic programming alignment core for TISA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None  # type: ignore

from .segments import Segment

EPSILON = 1e-8


@dataclass
class DPResult:
    cost: float
    dp_matrix: np.ndarray
    backpointers: np.ndarray
    mapping: List[Tuple[int, int, str]]


def _compute_norm_constants(seg_x: Sequence[Segment], seg_y: Sequence[Segment]) -> Tuple[float, float]:
    dt_values = [s.dt for s in seg_x] + [s.dt for s in seg_y]
    dv_values = [abs(s.dv) for s in seg_x] + [abs(s.dv) for s in seg_y]
    T = float(max(1.0, float(np.median(dt_values)))) if dt_values else 1.0
    V = float(np.median(dv_values)) if dv_values else 1.0
    if V < EPSILON:
        V = EPSILON
    return T, V


def _arrays_from_segments(segments: Sequence[Segment]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.asarray([s.dt for s in segments], dtype=np.float64)
    dv = np.asarray([s.dv for s in segments], dtype=np.float64)
    direction = np.asarray([s.direction for s in segments], dtype=np.int8)
    return dt, dv, direction


def _gap_cost(dv: float, gamma: float, V: float) -> float:
    return gamma * ((abs(dv) / V) ** 2)


def _dp_numpy(
    dt_x: np.ndarray,
    dv_x: np.ndarray,
    dir_x: np.ndarray,
    dt_y: np.ndarray,
    dv_y: np.ndarray,
    dir_y: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    T: float,
    V: float,
    band: int | None,
    best_so_far: float,
) -> Tuple[np.ndarray, np.ndarray]:
    L = dt_x.shape[0]
    K = dt_y.shape[0]
    INF = 1e18
    dp = np.full((L + 1, K + 1), INF, dtype=np.float64)
    bt = np.full((L + 1, K + 1), -1, dtype=np.int8)
    dp[0, 0] = 0.0

    for i in range(1, L + 1):
        cost = _gap_cost(dv_x[i - 1], gamma, V)
        dp[i, 0] = dp[i - 1, 0] + cost
        bt[i, 0] = 1  # gap in Y (skip X segment)
    for j in range(1, K + 1):
        cost = _gap_cost(dv_y[j - 1], gamma, V)
        dp[0, j] = dp[0, j - 1] + cost
        bt[0, j] = 2  # gap in X

    best_row = np.inf
    for i in range(1, L + 1):
        row_best = np.inf
        j_start = 1
        j_stop = K + 1
        if band is not None:
            j_start = max(1, i - band)
            j_stop = min(K + 1, i + band + 1)
        for j in range(j_start, j_stop):
            # gap X
            gap_x = dp[i - 1, j] + _gap_cost(dv_x[i - 1], gamma, V)
            best_cost = gap_x
            best_move = 1

            # gap Y
            gap_y = dp[i, j - 1] + _gap_cost(dv_y[j - 1], gamma, V)
            if gap_y < best_cost:
                best_cost = gap_y
                best_move = 2

            # match
            if dir_x[i - 1] == dir_y[j - 1]:
                dt_err = ((dt_x[i - 1] - dt_y[j - 1]) / T) ** 2
                dv_err = ((dv_x[i - 1] - dv_y[j - 1]) / V) ** 2
                match_cost = dp[i - 1, j - 1] + alpha * dt_err + beta * dv_err
                if match_cost < best_cost:
                    best_cost = match_cost
                    best_move = 3

            dp[i, j] = best_cost
            bt[i, j] = best_move
            if best_cost < row_best:
                row_best = best_cost
        if row_best < best_row:
            best_row = row_best
        if best_row > best_so_far:
            # early abandon
            break
    return dp, bt


if njit is not None:

    @njit(cache=True)  # type: ignore[misc]
    def _dp_numba(
        dt_x,
        dv_x,
        dir_x,
        dt_y,
        dv_y,
        dir_y,
        alpha,
        beta,
        gamma,
        T,
        V,
        band,
        best_so_far,
    ):
        L = dt_x.shape[0]
        K = dt_y.shape[0]
        INF = 1e18
        dp = np.empty((L + 1, K + 1), dtype=np.float64)
        bt = np.empty((L + 1, K + 1), dtype=np.int8)
        for i in range(L + 1):
            for j in range(K + 1):
                dp[i, j] = INF
                bt[i, j] = -1
        dp[0, 0] = 0.0
        for i in range(1, L + 1):
            cost = gamma * ((abs(dv_x[i - 1]) / V) ** 2)
            dp[i, 0] = dp[i - 1, 0] + cost
            bt[i, 0] = 1
        for j in range(1, K + 1):
            cost = gamma * ((abs(dv_y[j - 1]) / V) ** 2)
            dp[0, j] = dp[0, j - 1] + cost
            bt[0, j] = 2
        best_row = 1e18
        for i in range(1, L + 1):
            row_best = 1e18
            j_start = 1
            j_stop = K + 1
            if band >= 0:
                if i - band > j_start:
                    j_start = i - band
                if i + band + 1 < j_stop:
                    j_stop = i + band + 1
            for j in range(j_start, j_stop):
                gap_x = dp[i - 1, j] + gamma * ((abs(dv_x[i - 1]) / V) ** 2)
                best_cost = gap_x
                best_move = 1
                gap_y = dp[i, j - 1] + gamma * ((abs(dv_y[j - 1]) / V) ** 2)
                if gap_y < best_cost:
                    best_cost = gap_y
                    best_move = 2
                if dir_x[i - 1] == dir_y[j - 1]:
                    dt_err = ((dt_x[i - 1] - dt_y[j - 1]) / T) ** 2
                    dv_err = ((dv_x[i - 1] - dv_y[j - 1]) / V) ** 2
                    match_cost = dp[i - 1, j - 1] + alpha * dt_err + beta * dv_err
                    if match_cost < best_cost:
                        best_cost = match_cost
                        best_move = 3
                dp[i, j] = best_cost
                bt[i, j] = best_move
                if best_cost < row_best:
                    row_best = best_cost
            if row_best < best_row:
                best_row = row_best
            if best_row > best_so_far:
                break
        return dp, bt
else:  # pragma: no cover
    def _dp_numba(*args, **kwargs):
        raise RuntimeError("numba is not available")


def _backtrack(bt: np.ndarray) -> List[Tuple[int, int, str]]:
    i, j = bt.shape[0] - 1, bt.shape[1] - 1
    mapping: List[Tuple[int, int, str]] = []
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 3:
            mapping.append((i - 1, j - 1, "match"))
            i -= 1
            j -= 1
        elif move == 1:
            mapping.append((i - 1, j, "x_gap"))
            i -= 1
        elif move == 2:
            mapping.append((i, j - 1, "y_gap"))
            j -= 1
        else:
            # outside computed band, treat as gap
            if i > 0:
                mapping.append((i - 1, j, "x_gap"))
                i -= 1
            elif j > 0:
                mapping.append((i, j - 1, "y_gap"))
                j -= 1
    mapping.reverse()
    return mapping


def dp_align(
    seg_x: Sequence[Segment],
    seg_y: Sequence[Segment],
    alpha: float,
    beta: float,
    gamma: float,
    band: int | None = None,
    use_numba: bool = True,
    best_so_far: float = np.inf,
) -> DPResult:
    """Run DP alignment and return the result."""

    if not seg_x or not seg_y:
        raise ValueError("DP requires non-empty segment sequences")

    T, V = _compute_norm_constants(seg_x, seg_y)
    dt_x, dv_x, dir_x = _arrays_from_segments(seg_x)
    dt_y, dv_y, dir_y = _arrays_from_segments(seg_y)
    if band is None:
        band = int(max(5, 0.3 * np.sqrt(len(seg_x) * len(seg_y))))
    if use_numba and njit is not None:
        dp, bt = _dp_numba(
            dt_x,
            dv_x,
            dir_x,
            dt_y,
            dv_y,
            dir_y,
            alpha,
            beta,
            gamma,
            T,
            V,
            band,
            best_so_far,
        )
    else:
        dp, bt = _dp_numpy(
            dt_x,
            dv_x,
            dir_x,
            dt_y,
            dv_y,
            dir_y,
            alpha,
            beta,
            gamma,
            T,
            V,
            band,
            best_so_far,
        )
    mapping = _backtrack(bt)
    return DPResult(cost=float(dp[-1, -1]), dp_matrix=dp, backpointers=bt, mapping=mapping)


__all__ = ["dp_align", "DPResult"]
