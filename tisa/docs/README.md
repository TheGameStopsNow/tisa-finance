# TISA

This package implements Transform-Invariant Segment Alignment (TISA) and utilities for benchmarking and plotting.

## Presets

- Daily preset: tuned for ~252-bar daily windows.
  - alpha=1.0, beta=1.0, gamma=0.75, band=20
  - Usage: `tisa bench --preset daily --config tisa/benchmarks/multi_daily.yaml --out reports/benchmark_daily`

- Intraday preset: tuned for ~1000-bar 1m windows.
  - alpha=1.0, beta=1.0, gamma=0.5, band=3, tau_t≈2 (auto in runner)
  - Usage: `tisa bench --preset intraday --config tisa/benchmarks/multi_intraday.yaml --out reports/benchmark_intraday`

You can override any parameter via flags in `tisa/benchmarks/run_benchmarks.py` if needed.
