"""Command line interface for TISA."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from .aligner import TISAAligner
from .benchmarks.run_benchmarks import run_benchmark
from .distance import TISADistance
from .io import fetch_series, load_series_from_csv
from .transforms import transform_series
from .viz import plot_alignment


def _load_series(path: str) -> np.ndarray:
    return load_series_from_csv(path)


def cmd_fetch(args: argparse.Namespace) -> None:
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    for ticker in tickers:
        fetch_series(
            ticker=ticker,
            interval=args.interval,
            start=args.start,
            end=args.end,
            source=args.source,
        )


def cmd_distance(args: argparse.Namespace) -> None:
    x = _load_series(args.fileA)
    y = _load_series(args.fileB)
    dist = TISADistance(normalize=not args.no_normalize).pairwise(x, y)
    print(f"TISA distance: {dist:.6f}")


def cmd_align(args: argparse.Namespace) -> None:
    x = _load_series(args.fileA)
    y = _load_series(args.fileB)
    aligner = TISAAligner(normalize=not args.no_normalize)
    detail = aligner.align(x, y)
    y_trans = transform_series(y, detail["best_transform"])
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as fh:
        json.dump(detail, fh, indent=2)
    plot_alignment(x, y_trans, detail, out_path, warped=detail.get("warp_path"))
    print(f"Alignment written to {out_path}")


def cmd_bench(args: argparse.Namespace) -> None:
    kwargs: dict[str, Any] = {}
    if args.preset == "daily":
        kwargs.update(dict(alpha=1.0, beta=1.0, gamma=0.75, band=20, numba=not args.no_numba))
    elif args.preset == "intraday":
        kwargs.update(dict(alpha=1.0, beta=1.0, gamma=0.5, band=3, numba=not args.no_numba))
    run_benchmark(args.config, args.out, **kwargs)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="tisa", description="Transform-Invariant Segment Alignment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch price data via yfinance or Polygon")
    p_fetch.add_argument("--tickers", required=True, help="Ticker symbol")
    p_fetch.add_argument("--interval", default="1d")
    p_fetch.add_argument("--start", default=None)
    p_fetch.add_argument("--end", default=None)
    p_fetch.add_argument("--source", choices=["auto", "polygon", "yfinance"], default="auto")
    p_fetch.set_defaults(func=cmd_fetch)

    p_distance = sub.add_parser("distance", help="Compute TISA distance between two CSV files")
    p_distance.add_argument("--fileA", required=True)
    p_distance.add_argument("--fileB", required=True)
    p_distance.add_argument("--no-normalize", action="store_true")
    p_distance.set_defaults(func=cmd_distance)

    p_align = sub.add_parser("align", help="Align two CSV files and save report")
    p_align.add_argument("--fileA", required=True)
    p_align.add_argument("--fileB", required=True)
    p_align.add_argument("--out", required=True)
    p_align.add_argument("--no-normalize", action="store_true")
    p_align.set_defaults(func=cmd_align)

    p_bench = sub.add_parser("bench", help="Run benchmark suite")
    p_bench.add_argument("--config", required=True)
    p_bench.add_argument("--out", required=True)
    p_bench.add_argument("--preset", choices=["none","daily","intraday"], default="none")
    p_bench.add_argument("--no-numba", action="store_true")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
