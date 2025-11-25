"""Quality validation tests for TISA.

These tests ensure that optimizations don't degrade alignment quality.
They test TISA's ability to correctly identify transforms and produce
consistent alignments.
"""

import numpy as np
import pytest

from tisa.distance import TISADistance


@pytest.fixture
def base_series():
    """Generate a base series with clear structure."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 252)
    trend = 0.1 * t
    seasonal = 5 * np.sin(t)
    noise = np.random.randn(252) * 0.5
    return trend + seasonal + noise


class TestTransformDetection:
    """Test TISA's ability to correctly identify transformations."""
    
    def test_detect_original(self, base_series):
        """Should detect 'orig' transform for identical series (with noise)."""
        transformed = base_series + np.random.randn(len(base_series)) * 0.1
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        assert result["best_transform"] == "orig"
    
    def test_detect_reverse(self, base_series):
        """Should detect 'rev' transform for reversed series."""
        transformed = base_series[::-1]
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        assert result["best_transform"] == "rev"
    
    def test_detect_invert(self, base_series):
        """Should detect 'inv' transform for inverted series."""
        transformed = -base_series
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        assert result["best_transform"] == "inv"
    
    def test_detect_rev_inv(self, base_series):
        """Should detect 'rev_inv' transform for reversed and inverted series."""
        transformed = -base_series[::-1]
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        assert result["best_transform"] == "rev_inv"


class TestAlignmentQuality:
    """Test alignment quality using known transformations."""
    
    def test_scale_invariance(self, base_series):
        """TISA should handle amplitude scaling well."""
        transformed = base_series * 2.5
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        # Should find a good match (low distance relative to series variance)
        assert result["distance"] < 100  # Reasonable threshold
    
    def test_shift_invariance(self, base_series):
        """TISA should handle vertical shifts well."""
        transformed = base_series + 50
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        assert result["distance"] < 100
    
    def test_scale_and_shift(self, base_series):
        """TISA should handle combined scale and shift."""
        transformed = base_series * 1.5 + 20
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        assert result["distance"] < 100
    
    def test_noise_robustness(self, base_series):
        """TISA should be robust to moderate noise."""
        noise = np.random.randn(len(base_series)) * 0.5
        transformed = base_series + noise
        tisa = TISADistance()
        result = tisa.detailed(base_series, transformed)
        # Should still match original transform
        assert result["best_transform"] == "orig"
        # Distance should be reasonable (baseline: ~84)
        assert result["distance"] < 100


class TestAlignmentConsistency:
    """Test that TISA produces consistent results."""
    
    def test_symmetric_distance(self, base_series):
        """Distance should be symmetric (approximately)."""
        transformed = base_series * 1.5 + 10
        tisa = TISADistance()
        dist_ab = tisa.pairwise(base_series, transformed)
        dist_ba = tisa.pairwise(transformed, base_series)
        # Should be very close (within numerical tolerance)
        assert abs(dist_ab - dist_ba) < 1.0
    
    def test_deterministic(self, base_series):
        """Same inputs should produce same outputs."""
        transformed = base_series * 1.2 + 5
        tisa = TISADistance()
        result1 = tisa.detailed(base_series, transformed)
        result2 = tisa.detailed(base_series, transformed)
        assert result1["distance"] == result2["distance"]
        assert result1["best_transform"] == result2["best_transform"]
    
    def test_triangle_inequality_soft(self, base_series):
        """TISA should roughly satisfy triangle inequality."""
        # Create three related series
        series_b = base_series * 1.2
        series_c = base_series * 1.5
        
        tisa = TISADistance()
        dist_ab = tisa.pairwise(base_series, series_b)
        dist_bc = tisa.pairwise(series_b, series_c)
        dist_ac = tisa.pairwise(base_series, series_c)
        
        # Soft triangle inequality (with some slack for transform-invariance)
        assert dist_ac <= (dist_ab + dist_bc) * 1.5


def compute_quality_score(base_series, transformed_series, expected_transform):
    """Compute a quality score for TISA alignment.
    
    Returns a dict with:
    - correct_transform: bool
    - distance: float
    - num_matches: int (number of matched segments)
    """
    tisa = TISADistance()
    result = tisa.detailed(base_series, transformed_series)
    
    correct_transform = (result["best_transform"] == expected_transform)
    num_matches = sum(1 for _, _, op in result["mapping"] if op == "match")
    
    return {
        "correct_transform": correct_transform,
        "distance": result["distance"],
        "num_matches": num_matches,
        "transform": result["best_transform"],
    }


# Baseline quality test for regression detection
def test_baseline_quality():
    """Establish baseline quality metrics.
    
    This test should pass with current TISA and any optimized versions.
    If this test starts failing after optimizations, quality has degraded.
    """
    np.random.seed(42)
    base = np.cumsum(np.random.randn(252))
    
    test_cases = [
        (base, "orig", 50),
        (base[::-1], "rev", 50),
        (-base, "inv", 50),
        (base * 1.5 + 10, "orig", 50),
    ]
    
    for transformed, expected_transform, max_distance in test_cases:
        score = compute_quality_score(base, transformed, expected_transform)
        assert score["correct_transform"], f"Failed to detect {expected_transform}"
        assert score["distance"] < max_distance, f"Distance too high for {expected_transform}"
