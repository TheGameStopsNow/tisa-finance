"""Visualization utilities for TISA."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.autolayout": True})


def _ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    detail: Dict[str, object],
    outpath: str | pathlib.Path,
    warped: Sequence[Tuple[int, int]] | None = None,
) -> Dict[str, str]:
    """Create overlay, segment map, and DP heatmap plots."""

    outpath = pathlib.Path(outpath)
    _ensure_parent(outpath)
    base = outpath.stem
    overlay_path = outpath.with_name(f"{base}_overlay.png")
    segmap_path = outpath.with_name(f"{base}_segmap.png")
    dp_path = outpath.with_name(f"{base}_dp_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(X, label="X", color="C0")
    ax.plot(Y, label="Y", color="C1", alpha=0.7)
    if warped:
        paired = [(x, y) for x, y in warped if x is not None and y is not None]
        if paired:
            warp_x, warp_y = zip(*paired)
            ax.scatter(warp_x, np.take(Y, warp_y), s=10, c="C2", label="Warped Y")
    ax.set_title(f"TISA Alignment (best={detail['best_transform']})")
    ax.legend()
    fig.savefig(overlay_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Segment Alignment")
    ax.set_xlabel("Segment index (X)")
    ax.set_ylabel("Segment index (Y)")
    for i, j, kind in detail["mapping"]:
        if kind == "match":
            ax.plot([i, j], [i, j], marker="o")
    fig.savefig(segmap_path)
    plt.close(fig)

    dp = detail.get("dp_matrix")
    if isinstance(dp, np.ndarray):
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(dp, origin="lower", aspect="auto", cmap="magma")
        fig.colorbar(cax, ax=ax)
        ax.set_title("DP cost matrix")
        fig.savefig(dp_path)
        plt.close(fig)

    return {
        "overlay": str(overlay_path),
        "segmap": str(segmap_path),
        "dp": str(dp_path),
    }


def plot_benchmarks(summary: pd.DataFrame, out_dir: str | pathlib.Path) -> List[str]:
    out_dir = pathlib.Path(out_dir)
    _ensure_parent(out_dir / "dummy")
    paths: List[str] = []
    if "method" in summary.columns and "mae" in summary.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        summary.groupby("method")["mae"].mean().plot.bar(ax=ax)
        ax.set_ylabel("MAE")
        fig.savefig(out_dir / "mae_bar.png")
        plt.close(fig)
        paths.append(str(out_dir / "mae_bar.png"))
    if "runtime_ms" in summary.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        summary.boxplot(column="runtime_ms", by="method", ax=ax)
        ax.set_ylabel("Runtime (ms)")
        fig.suptitle("")
        ax.set_title("Runtime distribution")
        fig.savefig(out_dir / "runtime_box.png")
        plt.close(fig)
        paths.append(str(out_dir / "runtime_box.png"))
    return paths


__all__ = ["plot_alignment", "plot_benchmarks"]
