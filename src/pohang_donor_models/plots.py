from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def save_feature_importance(frame: pd.DataFrame, path: str | Path, top_n: int = 15) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    plot_frame = frame.head(top_n).sort_values("importance_normalized", ascending=True)
    fig, axis = plt.subplots(figsize=(9, max(4, 0.38 * len(plot_frame))))
    axis.barh(plot_frame["feature"], plot_frame["importance_normalized"])
    axis.set_xlabel("Normalized importance")
    axis.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def save_actual_vs_predicted(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(frame["actual"], frame["predicted"], s=10, alpha=0.35)
    finite = frame[["actual", "predicted"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        low = float(finite.min().min())
        high = float(finite.max().max())
        axis.plot([low, high], [low, high], linestyle="--")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    axis.set_title("Actual vs predicted")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def save_cluster_profile_heatmap(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if frame.empty:
        return
    data = frame.set_index("cluster")
    fig, axis = plt.subplots(figsize=(max(8, 0.7 * data.shape[1]), max(4, 0.7 * data.shape[0])))
    image = axis.imshow(data.to_numpy(dtype=float), aspect="auto")
    axis.set_xticks(range(data.shape[1]), data.columns, rotation=70, ha="right")
    axis.set_yticks(range(data.shape[0]), data.index)
    axis.set_title("Standardized cluster profile")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def save_correlation_heatmap(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if frame.empty:
        return
    size = max(8.0, min(24.0, 0.52 * len(frame.columns)))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(frame.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=75, ha="right", fontsize=7)
    axis.set_yticks(range(len(frame.index)), frame.index, fontsize=7)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Association")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def save_ablation_plot(frame: pd.DataFrame, path: str | Path, top_n: int = 30) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    passed = frame[frame["status"] == "PASS"].dropna(subset=["r2_drop"])
    if passed.empty:
        return
    plot_frame = passed.nlargest(top_n, "r2_drop").sort_values("r2_drop")
    colors = np.where(plot_frame["r2_drop"] >= 0, "#d95f02", "#1b9e77")
    fig, axis = plt.subplots(figsize=(10, max(4, 0.34 * len(plot_frame))))
    axis.barh(plot_frame["omitted_feature"], plot_frame["r2_drop"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Baseline R² - ablated R² (positive = helpful feature)")
    axis.set_title("Leave-one-feature-out ablation")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
