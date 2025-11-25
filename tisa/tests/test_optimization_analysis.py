"""Test to analyze optimization impact."""

import numpy as np
import pytest

from tisa.distance import TISADistance


def test_optimization_analysis():
    """Analyze which transforms are being skipped by optimizations."""
    np.random.seed(42)
    base = np.cumsum(np.random.randn(252))
    
    test_cases = [
        ("Original", base),
        ("Scaled", base * 1.5),
        ("Shifted", base + 10),
        ("Reversed", base[::-1]),
        ("Inverted", -base),
    ]
    
    for name, series in test_cases:
        tisa = TISADistance()
        result = tisa.detailed(base, series)
        skipped = result.get("skipped_transforms", [])
        print(f"\n{name}: best={result['best_transform']}, skipped={skipped}, distance={result['distance']:.2f}")
    
    # This isn't a real assertion test, just for analysis
    assert True


if __name__ == "__main__":
    test_optimization_analysis()
