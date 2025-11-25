"""Performance benchmarks using pytest-benchmark.

Run with: pytest tisa/tests/test_performance.py --benchmark-only
Compare: pytest tisa/tests/test_performance.py --benchmark-compare
"""

import numpy as np
import pytest
from tslearn.metrics import dtw, dtw_path

from tisa.distance import TISADistance


@pytest.fixture
def sample_series():
    """Generate reproducible sample data for benchmarking."""
    np.random.seed(42)
    n = 252  # One year of daily data
    returns = np.random.randn(n) * 0.01
    price = 100 + np.cumsum(returns)
    return price


@pytest.fixture
def small_series():
    """Generate small series for quick benchmarks."""
    np.random.seed(123)
    x = np.cumsum(np.random.randn(50))
    return x


@pytest.fixture
def tisa_instance():
    """Create a TISA instance with default parameters."""
    return TISADistance()


# ============================================================================
# TISA Benchmarks
# ============================================================================

def test_benchmark_tisa_pairwise_daily(benchmark, sample_series, tisa_instance):
    """Benchmark TISA pairwise distance on daily data (252 bars)."""
    transformed = sample_series * 1.5 + 10
    result = benchmark(tisa_instance.pairwise, sample_series, transformed)
    assert isinstance(result, float) and result >= 0


def test_benchmark_tisa_detailed_daily(benchmark, sample_series, tisa_instance):
    """Benchmark TISA detailed alignment on daily data (252 bars)."""
    transformed = sample_series * 1.5 + 10
    result = benchmark(tisa_instance.detailed, sample_series, transformed)
    assert "distance" in result and result["distance"] >= 0


def test_benchmark_tisa_pairwise_small(benchmark, small_series):
    """Benchmark TISA pairwise distance on small series (50 bars)."""
    y = small_series * 1.2 + 5
    tisa = TISADistance()
    result = benchmark(tisa.pairwise, small_series, y)
    assert isinstance(result, float) and result >= 0


def test_benchmark_tisa_numba_vs_python(benchmark, sample_series):
    """Benchmark TISA with numba disabled (useful for comparison)."""
    tisa = TISADistance(numba=False)
    transformed = sample_series * 1.5 + 10
    result = benchmark(tisa.pairwise, sample_series, transformed)
    assert isinstance(result, float) and result >= 0


# ============================================================================
# DTW Benchmarks (for comparison)
# ============================================================================

def test_benchmark_dtw_distance_daily(benchmark, sample_series):
    """Benchmark standard DTW distance on daily data (252 bars)."""
    transformed = sample_series * 1.5 + 10
    result = benchmark(dtw, sample_series, transformed)
    assert isinstance(result, float) and result >= 0


def test_benchmark_dtw_path_daily(benchmark, sample_series):
    """Benchmark DTW with path computation on daily data (252 bars)."""
    transformed = sample_series * 1.5 + 10
    result = benchmark(dtw_path, sample_series, transformed)
    path, dist = result
    assert isinstance(dist, float) and dist >= 0


def test_benchmark_dtw_distance_small(benchmark, small_series):
    """Benchmark standard DTW distance on small series (50 bars)."""
    y = small_series * 1.2 + 5
    result = benchmark(dtw, small_series, y)
    assert isinstance(result, float) and result >= 0


def test_benchmark_dtw_constrained_daily(benchmark, sample_series):
    """Benchmark constrained DTW (Sakoe-Chiba) on daily data (252 bars)."""
    transformed = sample_series * 1.5 + 10
    radius = int(0.1 * len(sample_series))  # 10% window
    result = benchmark(
        dtw,
        sample_series,
        transformed,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=radius
    )
    assert isinstance(result, float) and result >= 0
