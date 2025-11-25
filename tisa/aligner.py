"""Alignment utilities built on top of :class:`TISADistance`."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .distance import TISADistance
from .segments import Segment


def _linear_warp(seg_x: Segment, seg_y: Segment) -> List[Tuple[int, int]]:
    len_x = seg_x.end - seg_x.start + 1
    len_y = seg_y.end - seg_y.start + 1
    if len_x == 1 and len_y == 1:
        return [(seg_x.start, seg_y.start)]
    if len_x == 1:
        return [(seg_x.start, seg_y.start)]
    if len_y == 1:
        return [(seg_x.start + i, seg_y.start) for i in range(len_x)]
    y_positions = np.linspace(seg_y.start, seg_y.end, num=len_x)
    warp = []
    for offset, y_idx in enumerate(np.rint(y_positions).astype(int)):
        warp.append((seg_x.start + offset, int(y_idx)))
    return warp


class TISAAligner(TISADistance):
    """Alignment class returning detailed warping paths."""

    def align(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, object]:
        detail = self.detailed(X, Y)
        best_transform = detail["best_transform"]
        seg_x_dicts = detail["seg_X"]
        seg_y_dicts = detail["seg_Y"]
        seg_x = [Segment(**d) for d in seg_x_dicts]
        seg_y = [Segment(**d) for d in seg_y_dicts]

        warp_path: List[Tuple[int, Optional[int]]] = []
        for i, j, kind in detail["mapping"]:
            if kind == "match":
                warp_path.extend(_linear_warp(seg_x[i], seg_y[j]))
            elif kind == "x_gap":
                seg = seg_x[i]
                for idx in range(seg.start, seg.end + 1):
                    warp_path.append((idx, None))
            elif kind == "y_gap":
                seg = seg_y[j]
                for idx in range(seg.start, seg.end + 1):
                    warp_path.append((None, idx))

        return {
            "distance": detail["distance"],
            "best_transform": best_transform,
            "mapping": detail["mapping"],
            "warp_path": warp_path,
            "segments_X": seg_x_dicts,
            "segments_Y": seg_y_dicts,
        }


__all__ = ["TISAAligner"]
