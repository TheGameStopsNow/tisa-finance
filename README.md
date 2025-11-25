# TISA
**Transform-Invariant Segment Alignment for financial time series**

TISA is a novel algorithm for aligning and comparing financial time series in a way that is invariant to non-linear transformations (like different volatility regimes or trends) and robust to noise. It combines segmentation with dynamic programming to find the optimal alignment between two series.

## Theory

Standard measures like Euclidean distance or Dynamic Time Warping (DTW) often fail on financial data because they are sensitive to amplitude scaling and noise. TISA addresses this by:

1.  **Segmentation**: Breaking the time series into linear segments to reduce noise and dimensionality.
2.  **Transform Invariance**: Allowing each segment in the source series to be transformed (scaled and shifted) to match the target series.
3.  **Dynamic Programming**: Finding the globally optimal sequence of segments and transforms that minimizes the alignment cost.

For more details, see the [documentation](tisa/docs/README.md).

## Install

Requires Python 3.9+.

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install TISA
pip install -U pip
pip install -e .[dev]
```

Optional: create a `.env` file with your Polygon.io API key for data fetching:

```
POLYGON_API_KEY=YOUR_KEY_HERE
```

## Quick Start

### Python API

```python
import numpy as np
from tisa.distance import TISADistance

# Generate synthetic data
x = np.cumsum(np.random.randn(252))
y = x * 1.5 + 10  # Scaled and shifted version

# Compute distance
dist = TISADistance().pairwise(x, y)
print(f"TISA Distance: {dist:.4f}")

# Get detailed alignment info
detail = TISADistance().detailed(x, y)
print(f"Best Transform: {detail['best_transform']}")
```

### CLI

Fetch data (auto-uses Polygon if `POLYGON_API_KEY` is present, else yfinance):

```bash
tisa fetch --tickers SPY,QQQ --interval 1d --start 2022-01-01 --end 2025-01-01 --source auto
```

Compute distance / alignment from CSV files:

```bash
tisa distance --fileA data/SPY_1d.csv --fileB data/QQQ_1d.csv
tisa align --fileA data/SPY_1d.csv --fileB data/QQQ_1d.csv --out reports/align_SPY_QQQ.json
```

## Benchmarks

TISA comes with a benchmarking suite to evaluate performance against other metrics.

```bash
# Run daily benchmark
tisa bench --config tisa/benchmarks/multi_daily.yaml --out reports/benchmark_daily
```

See [tisa/benchmarks/README.md](tisa/benchmarks/README.md) for more info.

## Performance Benchmarks

Track TISA performance over time using `pytest-benchmark`:

```bash
# Run performance benchmarks
pytest tisa/tests/test_performance.py --benchmark-only

# Compare with previous runs
pytest tisa/tests/test_performance.py --benchmark-compare

# Save baseline
pytest tisa/tests/test_performance.py --benchmark-save=baseline
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

[MIT](LICENSE)