"""Data loading and fetching utilities for TISA."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
import os
import time
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

try:  # pragma: no cover - optional at runtime
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class SeriesMetadata:
    ticker: str
    interval: str
    start: str
    end: str
    fetched_at: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def _ensure_yfinance():  # pragma: no cover - trivial
    if yf is None:
        raise ImportError("yfinance is required for fetching data")


def _load_env_polygon_key() -> str | None:
    key = os.getenv("POLYGON_API_KEY")
    if key:
        return key
    env_path = pathlib.Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf8").splitlines():
                if line.strip().startswith("POLYGON_API_KEY="):
                    _, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    if value:
                        os.environ.setdefault("POLYGON_API_KEY", value)
                        return value
        except Exception:
            return None
    return None


def _format_date(dt: str | datetime | None) -> str:
    if dt is None:
        # Polygon requires an end date; use today for None
        return datetime.utcnow().date().isoformat()
    if isinstance(dt, datetime):
        return dt.date().isoformat()
    return str(dt)


def _polygon_interval(interval: str) -> tuple[int, str]:
    interval = interval.lower()
    if interval in {"1d", "1day", "day", "d"}:
        return 1, "day"
    if interval in {"1h", "hour", "60m"}:
        return 1, "hour"
    if interval in {"15m", "15min", "15"}:
        return 15, "minute"
    if interval in {"5m", "5min", "5"}:
        return 5, "minute"
    if interval in {"1m", "minute", "m"}:
        return 1, "minute"
    raise ValueError(f"Unsupported interval for Polygon: {interval}")


def fetch_series(
    ticker: str,
    interval: str = "1d",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    auto_save: bool = True,
    out_dir: pathlib.Path | str = DATA_DIR,
    source: str = "auto",
) -> pd.DataFrame:
    """Fetch OHLCV data for ticker and optionally save to CSV."""

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_polygon = False
    if source == "polygon":
        use_polygon = True
    elif source == "yfinance":
        use_polygon = False
    else:  # auto
        use_polygon = _load_env_polygon_key() is not None

    if use_polygon:
        key = _load_env_polygon_key()
        if not key:
            raise RuntimeError("POLYGON_API_KEY not set in environment or .env")
        mult, timespan = _polygon_interval(interval)
        start_str = _format_date(start)
        end_str = _format_date(end)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{timespan}/{start_str}/{end_str}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={key}"
        )
        rows: list[dict] = []
        next_url = url
        while next_url:
            resp = requests.get(next_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            for r in results:
                rows.append(
                    {
                        "Datetime": datetime.utcfromtimestamp(r.get("t", 0) / 1000).isoformat(),
                        "open": r.get("o"),
                        "high": r.get("h"),
                        "low": r.get("l"),
                        "close": r.get("c"),
                        "volume": r.get("v"),
                    }
                )
            next_url = data.get("next_url")
            if next_url:
                if "apiKey=" not in next_url:
                    next_url += ("&" if "?" in next_url else "?") + f"apiKey={key}"
                time.sleep(0.1)
        if not rows:
            raise RuntimeError(f"No data returned for {ticker} from Polygon")
        df = pd.DataFrame(rows)
        df = df.sort_values("Datetime")
        df.index = pd.to_datetime(df["Datetime"])  # keep a proper index
    else:
        _ensure_yfinance()
        df = yf.download(ticker, interval=interval, start=start, end=end, auto_adjust=False, progress=False)
        df = df.sort_index()
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if auto_save:
        fname = out_dir / f"{ticker}_{interval}.csv"
        df.to_csv(fname)
        meta = SeriesMetadata(
            ticker=ticker,
            interval=interval,
            start=str(start) if start else "",
            end=str(end) if end else "",
            fetched_at=datetime.utcnow().isoformat(),
        )
        with open(fname.with_suffix(".json"), "w", encoding="utf8") as fh:
            fh.write(meta.to_json())
    return df


def load_series_from_csv(path: str | pathlib.Path, column: str = "Close") -> np.ndarray:
    """Load a price series from CSV file.

    Column lookup is case-insensitive; falls back to matching lowercase name
    when an exact match is not present (e.g., "close" vs "Close").
    """

    df = pd.read_csv(path)
    col = column
    if col not in df.columns:
        # case-insensitive fallback
        lowered = {c.lower(): c for c in df.columns}
        alt = lowered.get(column.lower())
        if alt is not None:
            col = alt
        else:
            raise ValueError(f"Column '{column}' not present in CSV (available: {list(df.columns)})")
    return df[col].to_numpy(dtype=float)


def load_options_series(*args, **kwargs) -> np.ndarray:  # pragma: no cover - placeholder
    """Placeholder for future options data support."""

    raise NotImplementedError("Option series loading is not implemented yet")


def rolling_windows(series: np.ndarray, window: int, stride: int) -> Iterable[np.ndarray]:
    """Yield rolling windows from the series."""

    n = len(series)
    if window > n:
        return []
    for start in range(0, n - window + 1, stride):
        yield series[start : start + window]


__all__ = [
    "fetch_series",
    "load_series_from_csv",
    "load_options_series",
    "rolling_windows",
    "SeriesMetadata",
]
