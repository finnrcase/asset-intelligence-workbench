"""
Interpretability helpers for the machine learning weighting engine.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.ml.evaluate import evaluate_regression_predictions
from src.ml.features import feature_group_lookup



def extract_linear_feature_weights(model, feature_columns: list[str]) -> list[dict[str, float | str]]:
    """Return linear feature coefficients in a display-friendly structure."""

    if getattr(model, "coef_", None) is None:
        return []

    rows = []
    for feature_name, coefficient in zip(feature_columns, model.coef_):
        rows.append(
            {
                "feature": feature_name,
                "coefficient": float(coefficient),
                "absolute_weight": abs(float(coefficient)),
            }
        )
    return sorted(rows, key=lambda row: row["absolute_weight"], reverse=True)



def aggregate_linear_weights_by_pillar(
    model,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
) -> list[dict[str, float | str]]:
    """Aggregate absolute linear coefficients into the three model pillars."""

    lookup = feature_group_lookup(feature_groups)
    totals: defaultdict[str, float] = defaultdict(float)
    if getattr(model, "coef_", None) is None:
        return []

    for feature_name, coefficient in zip(feature_columns, model.coef_):
        group_name = lookup.get(feature_name)
        if group_name is None:
            continue
        totals[group_name] += abs(float(coefficient))

    grand_total = sum(totals.values()) or 1.0
    rows = []
    for group_name in ("history", "risk", "sentiment"):
        rows.append(
            {
                "pillar": group_name,
                "weight": float(totals.get(group_name, 0.0) / grand_total),
            }
        )
    return rows



def compute_linear_feature_contributions(
    model,
    row_frame: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, float | str]]:
    """Return per-feature linear contributions for a latest scoring row."""

    if row_frame.empty or getattr(model, "coef_", None) is None:
        return []

    row = row_frame.iloc[0]
    mean_vector = np.asarray(model.feature_mean_, dtype=float)
    scale_vector = np.asarray(model.feature_scale_, dtype=float)
    scale_vector[scale_vector == 0.0] = 1.0
    feature_vector = np.asarray([float(row[column]) for column in feature_columns], dtype=float)
    standardized = (feature_vector - mean_vector) / scale_vector

    rows = []
    for feature_name, standardized_value, coefficient in zip(feature_columns, standardized, model.coef_):
        contribution = float(standardized_value * coefficient)
        rows.append(
            {
                "feature": feature_name,
                "standardized_value": float(standardized_value),
                "coefficient": float(coefficient),
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
            }
        )
    return sorted(rows, key=lambda item: item["absolute_contribution"], reverse=True)



def aggregate_feature_contributions_by_pillar(
    feature_contributions: list[dict[str, float | str]],
    feature_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Aggregate signed feature contributions into history/risk/sentiment pillars."""

    lookup = feature_group_lookup(feature_groups)
    totals = {"history": 0.0, "risk": 0.0, "sentiment": 0.0}
    for row in feature_contributions:
        group_name = lookup.get(str(row["feature"]))
        if group_name is None:
            continue
        totals[group_name] += float(row["contribution"])
    return totals



def compute_permutation_feature_importance(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
) -> list[dict[str, float | str]]:
    """Compute simple holdout permutation importance for a regression model."""

    if x_test.empty or y_test.empty:
        return []

    baseline_prediction = np.asarray(model.predict(x_test), dtype=float)
    baseline_rmse = evaluate_regression_predictions(y_test.to_numpy(dtype=float), baseline_prediction)["rmse"]
    importance_rows: list[dict[str, float | str]] = []

    for feature_name in feature_columns:
        permuted = x_test.copy()
        permuted[feature_name] = permuted[feature_name].sample(frac=1.0, random_state=42).to_numpy()
        permuted_prediction = np.asarray(model.predict(permuted), dtype=float)
        permuted_rmse = evaluate_regression_predictions(
            y_test.to_numpy(dtype=float),
            permuted_prediction,
        )["rmse"]
        importance_rows.append(
            {
                "feature": feature_name,
                "importance": float(max(permuted_rmse - baseline_rmse, 0.0)),
            }
        )

    return sorted(importance_rows, key=lambda row: row["importance"], reverse=True)



def aggregate_importance_by_pillar(
    importance_rows: list[dict[str, float | str]],
    feature_groups: dict[str, list[str]],
) -> list[dict[str, float | str]]:
    """Aggregate permutation importance into pillar-level importance shares."""

    lookup = feature_group_lookup(feature_groups)
    totals: defaultdict[str, float] = defaultdict(float)
    for row in importance_rows:
        group_name = lookup.get(str(row["feature"]))
        if group_name is None:
            continue
        totals[group_name] += float(row["importance"])

    grand_total = sum(totals.values()) or 1.0
    return [
        {"pillar": "history", "importance": float(totals.get("history", 0.0) / grand_total)},
        {"pillar": "risk", "importance": float(totals.get("risk", 0.0) / grand_total)},
        {"pillar": "sentiment", "importance": float(totals.get("sentiment", 0.0) / grand_total)},
    ]



def serialize_rows(rows: list[dict[str, float | str]], top_n: int = 8) -> list[dict[str, float | str]]:
    """Return a JSON-friendly top slice of interpretability rows."""

    return rows[:top_n]
