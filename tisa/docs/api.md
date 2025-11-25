# API Reference

## `tisa.distance.TISADistance`

```python
TISADistance(alpha=1.0, beta=1.0, gamma=0.5, tau_v=0.0, tau_t=1, normalize=True, numba=True, band=None)
```

- `pairwise(X, Y) -> float`: compute the transform-invariant distance.
- `detailed(X, Y) -> dict`: detailed results including winning transform,
  segment lists, DP matrix, and mapping.

## `tisa.aligner.TISAAligner`

Extends `TISADistance` with:

- `align(X, Y) -> dict`: returns a mapping and per-point warp path alongside
  metadata.

## `tisa.io`

- `fetch_series(ticker, interval, start, end, auto_save=True) -> DataFrame`
- `load_series_from_csv(path, column='Close') -> np.ndarray`
- `rolling_windows(series, window, stride)` generator
- `load_options_series(...)` placeholder for future work

## `tisa.benchmarks.run_benchmarks`

- `run_benchmark(config_path, out_dir) -> dict`: runs the benchmark suite and
  writes metrics, summary, and manifest files.

## CLI

Install with `pip install -e .` then run `tisa --help`.
