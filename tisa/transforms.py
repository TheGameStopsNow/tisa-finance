"""Series transformation helpers for TISA."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .segments import Segment, segmentize

TRANSFORM_MODES = {"orig", "rev", "inv", "rev_inv"}


def transform_series(y: np.ndarray, mode: str) -> np.ndarray:
    """Return transformed copy of series according to mode."""

    if mode not in TRANSFORM_MODES:
        raise ValueError(f"Unknown transform mode '{mode}'")
    arr = np.asarray(y, dtype=float)
    if mode == "orig":
        return arr.copy()
    if mode == "rev":
        return arr[::-1].copy()
    if mode == "inv":
        return (-arr).copy()
    if mode == "rev_inv":
        return (-arr[::-1]).copy()
    raise RuntimeError("unreachable")


def all_transforms(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Return dictionary of all transform variants."""

    return {mode: transform_series(y, mode) for mode in sorted(TRANSFORM_MODES)}


def segmentize_transforms(y: np.ndarray, tau_v: float = 0.0, tau_t: int = 1) -> Dict[str, Iterable[Segment]]:
    """Segmentize all transforms of series using provided thresholds."""

    segments = {}
    for mode, arr in all_transforms(y).items():
        segments[mode] = segmentize(arr, tau_v=tau_v, tau_t=tau_t)
    return segments


__all__ = ["transform_series", "all_transforms", "segmentize_transforms", "TRANSFORM_MODES"]
