"""Public distance API for TISA."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .dp_align import DPResult, dp_align
from .segments import Segment, segmentize
from .transforms import TRANSFORM_MODES, segmentize_transforms


def _z_normalize(arr: np.ndarray) -> np.ndarray:
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-12:
        std = 1.0
    return (arr - mean) / std


class TISADistance:
    """Compute the transform-invariant segment alignment distance."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        tau_v: float = 0.0,
        tau_t: int = 1,
        normalize: bool = True,
        numba: bool = True,
        band: int | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau_v = float(tau_v)
        self.tau_t = int(tau_t)
        self.normalize = bool(normalize)
        self.use_numba = bool(numba)
        self.band = band

    def _prepare(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("TISADistance expects 1D numpy arrays")
        if self.normalize:
            x = _z_normalize(x)
            y = _z_normalize(y)
        return x, y

    def _segmentize(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[Segment], Dict[str, List[Segment]]]:
        seg_x = segmentize(x, tau_v=self.tau_v, tau_t=self.tau_t)
        segs_y = segmentize_transforms(y, tau_v=self.tau_v, tau_t=self.tau_t)
        return seg_x, segs_y

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Return the transform-invariant distance between two series."""

        return self.detailed(X, Y)["distance"]

    def _compute_segment_similarity(self, seg_x: List[Segment], seg_y: List[Segment]) -> float:
        """Compute a similarity score between two segment sequences.
        
        Returns a score in [0, 1] where 1 is very similar, 0 is very different.
        """
        if not seg_x or not seg_y:
            return 0.0
        
        # Segment count similarity
        count_sim = 1.0 - abs(len(seg_x) - len(seg_y)) / max(len(seg_x), len(seg_y))
        
        # Direction pattern similarity (fraction of matching directions)
        len_min = min(len(seg_x), len(seg_y))
        if len_min > 0:
            dir_matches = sum(1 for i in range(len_min) if seg_x[i].direction == seg_y[i].direction)
            direction_sim = dir_matches / len_min
        else:
            direction_sim = 0.0
        
        # Amplitude distribution similarity
        total_amp_x = sum(abs(s.dv) for s in seg_x)
        total_amp_y = sum(abs(s.dv) for s in seg_y)
        if total_amp_x > 0 and total_amp_y > 0:
            amp_ratio = min(total_amp_x, total_amp_y) / max(total_amp_x, total_amp_y)
        else:
            amp_ratio = 0.0
        
        # Combined similarity (weighted average)
        similarity = 0.4 * count_sim + 0.4 * direction_sim + 0.2 * amp_ratio
        return float(np.clip(similarity, 0, 1))
    
    def _adaptive_band(self, seg_x: List[Segment], seg_y: List[Segment]) -> int:
        """Compute adaptive band width based on segment similarity.
        
        Returns a band width optimized for the similarity level.
        """
        if self.band is not None:
            return self.band  # User-specified band takes precedence
        
        similarity = self._compute_segment_similarity(seg_x, seg_y)
        L, K = len(seg_x), len(seg_y)
        diagonal = np.sqrt(L * K)
        
        # Adaptive band sizing based on similarity (more conservative)
        if similarity > 0.8:  # Very high similarity
            band_factor = 0.15  # Moderate band (was 0.05, too tight)
        elif similarity > 0.5:  # Medium-high similarity
            band_factor = 0.25  # Wide-moderate band (was 0.15)
        else:  # Lower similarity
            band_factor = 0.35  # Wide band (was 0.3, slightly wider for safety)
        
        return int(max(10, band_factor * diagonal))  # Increased minimum from 5 to 10

    def _should_skip_transform(self, seg_x: List[Segment], seg_y: List[Segment]) -> bool:
        """Quick heuristic to skip unlikely transforms.
        
        Returns True if this transform is very unlikely to be the best match.
        """
        if not seg_x or not seg_y:
            return True
        
        # Skip if segment count difference is too large
        count_diff = abs(len(seg_x) - len(seg_y))
        max_count = max(len(seg_x), len(seg_y))
        if max_count > 0 and count_diff / max_count > 0.5:  # >50% difference
            return True
        
        # Skip if total amplitude  difference is extreme
        total_amp_x = sum(abs(s.dv) for s in seg_x)
        total_amp_y = sum(abs(s.dv) for s in seg_y)
        if total_amp_x > 0 and total_amp_y > 0:
            amp_ratio = max(total_amp_x, total_amp_y) / min(total_amp_x, total_amp_y)
            if amp_ratio > 3.0:  # 3x difference
                return True
        
        return False

    def detailed(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, object]:
        """Return rich information about the alignment.

        The result dictionary includes the winning transform, distance, mapping,
        and per-transform DP costs.
        """

        x, y = self._prepare(X, Y)
        seg_x, segs_y = self._segmentize(x, y)

        best_transform = None
        best_cost = float("inf")
        best_result: DPResult | None = None
        best_segments_y: List[Segment] | None = None
        transform_details: Dict[str, Dict[str, object]] = {}
        skipped_transforms = []

        # Transform-aware early stopping: test in order of expected cost
        # Typically: orig < rev < inv < rev_inv
        transform_order = ["orig", "rev", "inv", "rev_inv"]
        
        for mode in transform_order:
            seg_y = segs_y[mode]
            
            # Early pruning: skip unlikely transforms
            if self._should_skip_transform(seg_x, seg_y):
                skipped_transforms.append(mode)
                continue
            
            if not seg_y:
                continue
            
            # Adaptive band sizing
            band = self._adaptive_band(seg_x, seg_y)
            
            result = dp_align(
                seg_x,
                seg_y,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                band=band,
                use_numba=self.use_numba,
                best_so_far=best_cost,
            )
            transform_details[mode] = {
                "distance": result.cost,
                "mapping": result.mapping,
            }
            if result.cost < best_cost:
                best_cost = result.cost
                best_transform = mode
                best_result = result
                best_segments_y = seg_y
            
            # Early exit: if we found a nearly perfect match, don't test remaining transforms
            # Only exit if cost is extremely low (near-zero alignment error)
            # This is very conservative to avoid missing the true best transform
            if best_cost < len(seg_x) * 0.001:  # Extremely strict threshold (was 0.01)
                # Skip remaining transforms
                remaining = [t for t in transform_order if t not in transform_details and t not in skipped_transforms]
                if remaining:
                    skipped_transforms.extend(remaining)
                break

        if best_transform is None or best_result is None or best_segments_y is None:
            raise RuntimeError("Alignment failed to find a valid transform")

        return {
            "distance": best_cost,
            "best_transform": best_transform,
            "dp_costs": {mode: info["distance"] for mode, info in transform_details.items()},
            "seg_X": [asdict(s) for s in seg_x],
            "seg_Y": [asdict(s) for s in best_segments_y],
            "mapping": best_result.mapping,
            "dp_matrix": best_result.dp_matrix,
            "skipped_transforms": skipped_transforms,  # For debugging/analysis
        }


def tisa_metric(X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
    """Convenience wrapper for sklearn-style metric usage."""

    return TISADistance(**kwargs).pairwise(X, Y)


__all__ = ["TISADistance", "tisa_metric"]
