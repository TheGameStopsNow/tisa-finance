"""Segmentization utilities for TISA.

This module implements deterministic extrema-based segmentation exactly as
specified in the build directive. The implementation carefully handles flat
regions and enforces alternating segment directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class Segment:
    """Container for a single segment between extrema.

    Attributes
    ----------
    start: int
        Inclusive index of the segment start.
    end: int
        Inclusive index of the segment end.
    dt: int
        Number of points in the segment (end - start + 1).
    dv: float
        Amplitude change across the segment.
    direction: int
        +1 for upward, -1 for downward.
    """

    start: int
    end: int
    dt: int
    dv: float
    direction: int

    def as_tuple(self) -> tuple[int, int, int, float, int]:
        return (self.start, self.end, self.dt, self.dv, self.direction)


def _extrema_indices(series: np.ndarray) -> List[int]:
    """Return deterministic extrema indices including endpoints.

    The algorithm follows the strict rules from the specification:
    maxima satisfy x[t-1] < x[t] > x[t+1]; minima satisfy the inverse. Flat
    stretches are handled by selecting the midpoint index of the plateau to
    maintain reverse-invariance, then enforcing alternating extrema.
    """

    n = series.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    extrema: List[int] = [0]
    prev_slope = 0
    plateau_start: int | None = None

    # Tolerance for floating noise when deciding slope sign. Use scale-aware eps.
    scale = float(np.max(np.abs(series))) if n else 1.0
    eps = 1e-12 * (scale if scale > 1.0 else 1.0)

    for idx in range(1, n):
        diff = series[idx] - series[idx - 1]
        slope = 0
        if diff > eps:
            slope = 1
        elif diff < -eps:
            slope = -1

        if slope == 0:
            if plateau_start is None:
                plateau_start = idx - 1
        else:
            # plateau ending
            if plateau_start is not None:
                # Choose plateau midpoint for reverse-invariant tie-breaking
                turn_idx = (plateau_start + (idx - 1)) // 2
                plateau_start = None
            else:
                turn_idx = idx - 1

            if prev_slope > 0 and slope < 0:
                if turn_idx != extrema[-1]:
                    extrema.append(turn_idx)
            elif prev_slope < 0 and slope > 0:
                if turn_idx != extrema[-1]:
                    extrema.append(turn_idx)

            prev_slope = slope

    # Handle potential extremum at the end when slope changes to zero.
    if plateau_start is not None and prev_slope != 0:
        # Plateau continued to the end; select midpoint for reverse-invariance
        turn_idx = (plateau_start + (n - 1)) // 2
        if turn_idx != extrema[-1] and turn_idx not in extrema:
            extrema.append(turn_idx)

    if extrema[-1] != n - 1:
        extrema.append(n - 1)

    # Ensure alternation by removing duplicates of direction.
    cleaned: List[int] = [extrema[0]]
    for idx in extrema[1:]:
        if idx <= cleaned[-1]:
            continue
        cleaned.append(idx)
    return cleaned


def _merge_segments(segments: Iterable[Segment], values: np.ndarray) -> List[Segment]:
    merged: List[Segment] = []
    for seg in segments:
        if seg.direction == 0:
            if merged:
                last = merged[-1]
                dv = values[seg.end] - values[last.start]
                merged[-1] = Segment(
                    start=last.start,
                    end=seg.end,
                    dt=seg.end - last.start + 1,
                    dv=dv,
                    direction=1 if dv > 0 else -1 if dv < 0 else 0,
                )
            continue
        if merged and merged[-1].direction == seg.direction:
            last = merged[-1]
            dv = values[seg.end] - values[last.start]
            direction = 1 if dv > 0 else -1 if dv < 0 else 0
            merged[-1] = Segment(
                start=last.start,
                end=seg.end,
                dt=seg.end - last.start + 1,
                dv=dv,
                direction=direction,
            )
        else:
            merged.append(seg)
    # Remove any residual zero-direction segments
    filtered = [s for s in merged if s.direction != 0]
    if not filtered and merged:
        # if all zero directions (flat series) treat as single segment with direction 0
        seg = merged[-1]
        direction = 1 if seg.dv > 0 else -1 if seg.dv < 0 else 0
        return [Segment(seg.start, seg.end, seg.end - seg.start + 1, seg.dv, direction)]
    return filtered


def segmentize(series: np.ndarray, tau_v: float = 0.0, tau_t: int = 1) -> List[Segment]:
    """Deterministically segmentize a series into monotone segments.

    Parameters
    ----------
    series:
        Input sequence. Converted to ``np.ndarray`` of ``float64``.
    tau_v:
        Minimum absolute amplitude required to keep a segment.
    tau_t:
        Minimum duration (#points) required to keep a segment.
    """

    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("segmentize expects a 1D array")
    n = arr.shape[0]
    if n == 0:
        return []

    extrema = _extrema_indices(arr)
    segments: List[Segment] = []
    for start, end in zip(extrema[:-1], extrema[1:]):
        dv = arr[end] - arr[start]
        dt = end - start + 1
        if dt < tau_t:
            continue
        if abs(dv) < tau_v:
            continue
        direction = 1 if dv > 0 else -1 if dv < 0 else 0
        seg = Segment(start=start, end=end, dt=dt, dv=dv, direction=direction)
        segments.append(seg)

    if not segments:
        # fallback single segment covering the series
        dv = arr[-1] - arr[0]
        direction = 1 if dv > 0 else -1 if dv < 0 else 0
        return [Segment(0, n - 1, n, dv, direction)]

    merged = _merge_segments(segments, arr)
    if not merged:
        dv = arr[-1] - arr[0]
        direction = 1 if dv > 0 else -1 if dv < 0 else 0
        return [Segment(0, n - 1, n, dv, direction)]

    return merged


__all__ = ["Segment", "segmentize"]
