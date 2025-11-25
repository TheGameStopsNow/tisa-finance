"""Benchmark runner comparing TISA against DTW baselines."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from tslearn.metrics import dtw_path

from ..aligner import TISAAligner
from ..io import load_series_from_csv
from ..viz import plot_benchmarks
from .distortions import Distortion, apply_all


@dataclass
class BenchmarkRecord:
    ticker: str
    window_id: int
    distortion: str
    method: str
    distance: float
    mae: float
    runtime_ms: float
    pearson_r: float

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__


def _alignment_mae(base: np.ndarray, other: np.ndarray, path: Sequence[Tuple[int, int]]) -> float:
    if not path:
        return float(np.mean(np.abs(base - other[: len(base)])))
    diffs = []
    for i, j in path:
        diffs.append(abs(float(base[i]) - float(other[j])))
    return float(np.mean(diffs)) if diffs else float("nan")


def _aligned_corr(base: np.ndarray, other: np.ndarray, path: Sequence[Tuple[int, int]]) -> float:
    if not path:
        return float("nan")
    x = np.asarray([base[i] for i, _ in path], dtype=float)
    y = np.asarray([other[j] for _, j in path], dtype=float)
    if x.size < 2:
        return 1.0
    return float(pearsonr(x, y)[0])


def _dtw_metrics(base: np.ndarray, series: np.ndarray, constraint: str | None = None, radius: int | None = None) -> Tuple[float, List[Tuple[int, int]]]:
    kwargs = {}
    if constraint == "sakoe_chiba":
        kwargs["global_constraint"] = "sakoe_chiba"
        kwargs["sakoe_chiba_radius"] = radius
    path, dist = dtw_path(base, series, **kwargs)
    return float(dist), path


def _tisa_metrics(aligner: TISAAligner, base: np.ndarray, series: np.ndarray) -> Tuple[float, List[Tuple[int, int]], Dict[str, object]]:
    detail = aligner.align(base, series)
    warp = [(i, j) for i, j in detail["warp_path"] if i is not None and j is not None]
    return float(detail["distance"]), warp, detail


def _fastdtw_metrics(base: np.ndarray, series: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    # Fallback implementation based on downsampling followed by DTW.
    if len(base) > 50:
        factor = int(np.ceil(len(base) / 50))
        base_coarse = base[::factor]
        series_coarse = series[::factor]
    else:
        factor = 1
        base_coarse = base
        series_coarse = series
    path_coarse, _ = dtw_path(base_coarse, series_coarse)
    # Expand coarse path to full resolution using nearest neighbors.
    path = []
    for i_c, j_c in path_coarse:
        i_start = i_c * factor
        j_start = j_c * factor
        for di in range(factor):
            i = min(i_start + di, len(base) - 1)
            j = min(j_start + di, len(series) - 1)
            path.append((i, j))
    if not path:
        dist = float(np.mean(np.abs(base - series[: len(base)])))
    else:
        dist = float(np.mean([abs(base[i] - series[j]) for i, j in path]))
    return dist, path


def run_benchmark(
    config_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    band: int | None = None,
    numba: bool = True,
    tau_v: float = 0.0,
    tau_t: int = 1,
) -> Dict[str, object]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r", encoding="utf8") as fh:
        config = yaml.safe_load(fh)

    # If config window suggests daily (≈252), prefer tuned defaults unless overridden
    window_size = int(config.get("window", 252))
    tuned_alpha, tuned_beta, tuned_gamma, tuned_band = alpha, beta, gamma, band
    tuned_tau_v, tuned_tau_t = tau_v, tau_t
    if alpha == 1.0 and beta == 1.0 and gamma == 0.5 and band is None and 200 <= window_size <= 400:
        tuned_alpha, tuned_beta, tuned_gamma, tuned_band = 1.0, 1.0, 0.75, 20
    # Intraday heuristic: larger windows tend to be 1m data; tighten band, default gamma
    if alpha == 1.0 and beta == 1.0 and band is None and window_size >= 500:
        tuned_alpha, tuned_beta = 1.0, 1.0
        tuned_gamma = gamma if gamma != 0.5 else 0.5
        tuned_band = 3
        if tau_t == 1 or tau_t == 2:
            tuned_tau_t = 2
    aligner = TISAAligner(alpha=tuned_alpha, beta=tuned_beta, gamma=tuned_gamma, band=tuned_band, numba=numba, tau_v=tuned_tau_v, tau_t=tuned_tau_t)
    records: List[BenchmarkRecord] = []
    manifest_distortions: List[Dict[str, object]] = []

    datasets = config.get("datasets", [])
    window_size = int(config.get("window", 252))
    stride = int(config.get("stride", 63))

    for dataset in datasets:
        ticker = dataset["ticker"]
        path = dataset.get("file")
        if not path:
            continue
        try:
            series = load_series_from_csv(path)
        except FileNotFoundError:
            continue
        windows = []
        for start in range(0, len(series) - window_size + 1, stride):
            windows.append(series[start : start + window_size])
        for window_id, base in enumerate(windows):
            distortions = apply_all(base, seed=window_id)
            manifest_distortions.append(
                {
                    "ticker": ticker,
                    "window_id": window_id,
                    "base_start": int(window_id * stride),
                    "distortions": [d.to_record() for d in distortions],
                }
            )
            for distortion in distortions:
                distorted = distortion.data
                methods = []
                start = time.perf_counter()
                tisa_dist, tisa_path, detail = _tisa_metrics(aligner, base, distorted)
                runtime = (time.perf_counter() - start) * 1000
                mae = _alignment_mae(base, distorted, tisa_path)
                corr = _aligned_corr(base, distorted, tisa_path)
                records.append(
                    BenchmarkRecord(ticker, window_id, distortion.name, "tisa", tisa_dist, mae, runtime, corr)
                )

                start = time.perf_counter()
                dtw_dist, dtw_path_points = _dtw_metrics(base, distorted)
                runtime = (time.perf_counter() - start) * 1000
                mae = _alignment_mae(base, distorted, dtw_path_points)
                corr = _aligned_corr(base, distorted, dtw_path_points)
                records.append(
                    BenchmarkRecord(ticker, window_id, distortion.name, "dtw", dtw_dist, mae, runtime, corr)
                )

                radius = max(1, int(0.1 * len(base)))
                start = time.perf_counter()
                cdtw_dist, cdtw_path_points = _dtw_metrics(base, distorted, constraint="sakoe_chiba", radius=radius)
                runtime = (time.perf_counter() - start) * 1000
                mae = _alignment_mae(base, distorted, cdtw_path_points)
                corr = _aligned_corr(base, distorted, cdtw_path_points)
                records.append(
                    BenchmarkRecord(ticker, window_id, distortion.name, "cdtw", cdtw_dist, mae, runtime, corr)
                )

                start = time.perf_counter()
                fast_dist, fast_path_points = _fastdtw_metrics(base, distorted)
                runtime = (time.perf_counter() - start) * 1000
                mae = _alignment_mae(base, distorted, fast_path_points)
                corr = _aligned_corr(base, distorted, fast_path_points)
                records.append(
                    BenchmarkRecord(ticker, window_id, distortion.name, "fastdtw", fast_dist, mae, runtime, corr)
                )

    df = pd.DataFrame([r.to_dict() for r in records])
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    summary = {
        "records": len(records),
        "tickers": sorted({r.ticker for r in records}),
        "methods": sorted({r.method for r in records}),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2)

    manifest = {
        "config": config,
        "distortions": manifest_distortions,
        "summary": summary,
        "note": "All distortions are transformations of real market data (no synthetic waveforms).",
        "tisa_params": {"alpha": tuned_alpha, "beta": tuned_beta, "gamma": tuned_gamma, "band": tuned_band, "numba": numba, "tau_v": tuned_tau_v, "tau_t": tuned_tau_t},
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2)

    if not df.empty:
        plot_benchmarks(df, out_dir / "plots")

    return {"metrics": str(metrics_path), "summary": str(summary_path), "manifest": str(manifest_path)}


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Run TISA benchmarks")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--band", type=int, default=None)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--tau-v", type=float, default=0.0)
    parser.add_argument("--tau-t", type=int, default=1)
    args = parser.parse_args()
    run_benchmark(
        args.config,
        args.out,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        band=args.band,
        numba=not args.no_numba,
        tau_v=args.tau_v,
        tau_t=args.tau_t,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
