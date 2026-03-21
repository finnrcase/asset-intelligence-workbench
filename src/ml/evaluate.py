"""
Time-aware evaluation utilities for forecasting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_holdout_split(
    frame: pd.DataFrame,
    date_column: str = "feature_date",
    holdout_fraction: float = 0.2,
    min_train_rows: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train/test sets using chronological order only."""

    if frame.empty:
        return frame.copy(), frame.copy()

    ordered = frame.sort_values(date_column).reset_index(drop=True)
    split_index = max(min_train_rows, int(len(ordered) * (1.0 - holdout_fraction)))
    split_index = min(split_index, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def generate_expanding_window_splits(
    frame: pd.DataFrame,
    date_column: str = "feature_date",
    min_train_rows: int = 80,
    test_rows: int = 20,
    step_rows: int = 20,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate expanding-window chronological train/test splits."""

    if frame.empty:
        return []

    ordered = frame.sort_values(date_column).reset_index(drop=True)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_end = min_train_rows
    while train_end + test_rows <= len(ordered):
        train_frame = ordered.iloc[:train_end].copy()
        test_frame = ordered.iloc[train_end : train_end + test_rows].copy()
        splits.append((train_frame, test_frame))
        train_end += step_rows
    return splits


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true >= 0.0) == (y_pred >= 0.0)))


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = float(np.sum((y_true == 1) & (y_pred == 1)))
    predicted_positive = float(np.sum(y_pred == 1))
    if predicted_positive == 0.0:
        return 0.0
    return true_positive / predicted_positive


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = float(np.sum((y_true == 1) & (y_pred == 1)))
    actual_positive = float(np.sum(y_true == 1))
    if actual_positive == 0.0:
        return 0.0
    return true_positive / actual_positive


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positive_rank_sum = float(ranks[y_true == 1].sum())
    auc = (
        positive_rank_sum
        - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)
    return float(auc)


def evaluate_regression_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def evaluate_classification_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": classification_accuracy(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def summarize_metric_history(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries."""

    if not metric_rows:
        return {}

    summary: dict[str, float] = {}
    for metric_name in metric_rows[0].keys():
        values = [row[metric_name] for row in metric_rows if not pd.isna(row[metric_name])]
        summary[metric_name] = float(np.mean(values)) if values else float("nan")
    return summary
