"""Transform-Invariant Segment Alignment (TISA).

This package implements the alignment model described in the build directive.
It exposes distance and alignment utilities together with a CLI.
"""

from .distance import TISADistance, tisa_metric
from .aligner import TISAAligner

__all__ = ["TISADistance", "TISAAligner", "tisa_metric"]
