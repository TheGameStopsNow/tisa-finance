"""Distortion utilities applied to real market data windows."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class Distortion:
    name: str
    params: Dict[str, float]
    data: np.ndarray

    def to_record(self) -> Dict[str, object]:
        rec = {"name": self.name, "params": self.params}
        rec["length"] = int(self.data.shape[0])
        return rec


def reverse(x: np.ndarray) -> Distortion:
    return Distortion("reverse", {}, x[::-1].copy())


def invert(x: np.ndarray) -> Distortion:
    return Distortion("invert", {}, (-x).copy())


def rev_inv(x: np.ndarray) -> Distortion:
    return Distortion("rev_inv", {}, (-x[::-1]).copy())


def time_warp_piecewise(x: np.ndarray, segments: List[Tuple[int, int]] | None = None, factors: Iterable[float] | None = None, seed: int | None = None) -> Distortion:
    rng = random.Random(seed)
    n = len(x)
    if segments is None:
        seg_count = rng.randint(2, 3)
        idxs = sorted(rng.sample(range(1, n - 1), seg_count))
        segments = []
        prev = 0
        for idx in idxs + [n - 1]:
            segments.append((prev, idx))
            prev = idx
    if factors is None:
        factors = [0.5, 0.75, 1.25, 1.5]
    warped = []
    for (start, end) in segments:
        seg = x[start : end + 1]
        factor = rng.choice(list(factors))
        new_len = max(2, int(round(len(seg) * factor)))
        warped.extend(np.interp(
            np.linspace(0, len(seg) - 1, num=new_len),
            np.arange(len(seg)),
            seg,
        ))
    return Distortion("time_warp_piecewise", {"segments": segments}, np.asarray(warped))


def amplitude_scale_local(x: np.ndarray, segments: List[Tuple[int, int]] | None = None, scales: Iterable[float] | None = None, seed: int | None = None) -> Distortion:
    rng = random.Random(seed)
    n = len(x)
    if segments is None:
        seg_count = rng.randint(1, 3)
        segments = []
        for _ in range(seg_count):
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            segments.append((a, b))
    if scales is None:
        scales = [0.8, 1.2]
    y = x.copy()
    for (start, end) in segments:
        scale = rng.choice(list(scales))
        y[start : end + 1] *= scale
    return Distortion("amplitude_scale_local", {"segments": segments}, y)


def jitter_minor(x: np.ndarray, sigma: float = 0.02, seed: int | None = None) -> Distortion:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma * np.std(x), size=len(x))
    return Distortion("jitter_minor", {"sigma": sigma}, x + noise)


DISTORTION_FUNCS: Dict[str, Callable[..., Distortion]] = {
    "reverse": reverse,
    "invert": invert,
    "rev_inv": rev_inv,
    "time_warp_piecewise": time_warp_piecewise,
    "amplitude_scale_local": amplitude_scale_local,
    "jitter_minor": jitter_minor,
}


def apply_all(x: np.ndarray, seed: int | None = None) -> List[Distortion]:
    rng = random.Random(seed)
    distortions = [reverse(x), invert(x), rev_inv(x)]
    distortions.append(time_warp_piecewise(x, seed=rng.randint(0, 10_000)))
    distortions.append(amplitude_scale_local(x, seed=rng.randint(0, 10_000)))
    distortions.append(jitter_minor(x, seed=rng.randint(0, 10_000)))
    return distortions


__all__ = ["Distortion", "DISTORTION_FUNCS", "apply_all"]
