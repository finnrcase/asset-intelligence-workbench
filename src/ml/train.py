"""
Model training workflow for return and downside-risk forecasting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.features.feature_store import load_training_frame_from_store
from src.ml.evaluate import evaluate_classification_predictions
from src.ml.evaluate import evaluate_regression_predictions
from src.ml.evaluate import generate_expanding_window_splits
from src.ml.evaluate import summarize_metric_history
from src.ml.evaluate import time_holdout_split
from src.ml.models import get_model_factories


REGRESSION_TARGET = "target_forward_return_20d"
CLASSIFICATION_TARGET = "target_negative_return_20d"
DEFAULT_EXCLUDED_COLUMNS = {
    "asset_id",
    "ticker",
    "feature_date",
    "price_date",
    "feature_version",
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
}


def infer_feature_columns(training_frame: pd.DataFrame) -> list[str]:
    """Infer numeric feature columns from the SQL-backed training frame."""

    return [
        column
        for column in training_frame.columns
        if column not in DEFAULT_EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(training_frame[column])
    ]


def prepare_model_frame(
    training_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a chronologically ordered model frame with stable numeric features."""

    if training_frame.empty:
        return training_frame.copy(), feature_columns or []

    frame = training_frame.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    frame = frame.sort_values(["feature_date", "asset_id"]).reset_index(drop=True)

    selected_features = feature_columns or infer_feature_columns(frame)
    for column in selected_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median_value = frame[column].median()
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
        frame[column] = frame[column].fillna(fill_value)

    return frame, selected_features


def _evaluate_regression_models(
    factories: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    x_train = train_frame[feature_columns]
    y_train = train_frame[REGRESSION_TARGET].astype(float)
    x_test = test_frame[feature_columns]
    y_test = test_frame[REGRESSION_TARGET].astype(float).to_numpy()

    for model_name, factory in factories.items():
        model = factory()
        model.fit(x_train, y_train)
        y_pred = np.asarray(model.predict(x_test), dtype=float)
        metrics[model_name] = evaluate_regression_predictions(y_test, y_pred)
    return metrics


def _evaluate_classification_models(
    factories: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    x_train = train_frame[feature_columns]
    y_train = train_frame[CLASSIFICATION_TARGET].astype(int)
    x_test = test_frame[feature_columns]
    y_test = test_frame[CLASSIFICATION_TARGET].astype(int).to_numpy()

    for model_name, factory in factories.items():
        model = factory()
        model.fit(x_train, y_train)
        probabilities = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        predictions = (probabilities >= 0.5).astype(int)
        metrics[model_name] = evaluate_classification_predictions(y_test, predictions, probabilities)
    return metrics


def _summarize_expanding_window_metrics(
    frame: pd.DataFrame,
    feature_columns: list[str],
    factories: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    splits = generate_expanding_window_splits(frame)
    if not splits:
        return {"regression": {}, "classification": {}}

    regression_history: dict[str, list[dict[str, float]]] = {name: [] for name in factories["regression"]}
    classification_history: dict[str, list[dict[str, float]]] = {name: [] for name in factories["classification"]}

    for train_frame, test_frame in splits:
        reg_train = train_frame.dropna(subset=[REGRESSION_TARGET])
        reg_test = test_frame.dropna(subset=[REGRESSION_TARGET])
        clf_train = train_frame.dropna(subset=[CLASSIFICATION_TARGET])
        clf_test = test_frame.dropna(subset=[CLASSIFICATION_TARGET])

        if not reg_train.empty and not reg_test.empty:
            regression_metrics = _evaluate_regression_models(
                factories["regression"], reg_train, reg_test, feature_columns
            )
            for model_name, metric_row in regression_metrics.items():
                regression_history[model_name].append(metric_row)

        if not clf_train.empty and not clf_test.empty:
            classification_metrics = _evaluate_classification_models(
                factories["classification"], clf_train, clf_test, feature_columns
            )
            for model_name, metric_row in classification_metrics.items():
                classification_history[model_name].append(metric_row)

    return {
        "regression": {
            model_name: summarize_metric_history(metric_rows)
            for model_name, metric_rows in regression_history.items()
            if metric_rows
        },
        "classification": {
            model_name: summarize_metric_history(metric_rows)
            for model_name, metric_rows in classification_history.items()
            if metric_rows
        },
    }


def _select_best_regression_model(metrics: dict[str, dict[str, float]]) -> str:
    return min(
        metrics,
        key=lambda model_name: (
            metrics[model_name].get("rmse", float("inf")),
            metrics[model_name].get("mae", float("inf")),
        ),
    )


def _select_best_classification_model(metrics: dict[str, dict[str, float]]) -> str:
    return max(
        metrics,
        key=lambda model_name: (
            -1.0 if pd.isna(metrics[model_name].get("roc_auc")) else metrics[model_name]["roc_auc"],
            metrics[model_name].get("accuracy", 0.0),
        ),
    )


def train_forecasting_models(
    training_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train the forecasting model suite and evaluate it with time-aware splits."""

    prepared_frame, selected_features = prepare_model_frame(training_frame, feature_columns)
    if prepared_frame.empty:
        raise ValueError("A non-empty training frame is required to train forecasting models.")

    factories = get_model_factories()
    holdout_train, holdout_test = time_holdout_split(prepared_frame)
    regression_train = holdout_train.dropna(subset=[REGRESSION_TARGET])
    regression_test = holdout_test.dropna(subset=[REGRESSION_TARGET])
    classification_train = holdout_train.dropna(subset=[CLASSIFICATION_TARGET])
    classification_test = holdout_test.dropna(subset=[CLASSIFICATION_TARGET])

    if regression_train.empty or regression_test.empty:
        raise ValueError("Insufficient forward-return target history for time-aware regression training.")
    if classification_train.empty or classification_test.empty:
        raise ValueError("Insufficient downside-risk target history for time-aware classification training.")

    holdout_metrics = {
        "regression": _evaluate_regression_models(
            factories["regression"], regression_train, regression_test, selected_features
        ),
        "classification": _evaluate_classification_models(
            factories["classification"], classification_train, classification_test, selected_features
        ),
    }
    expanding_window_metrics = _summarize_expanding_window_metrics(
        prepared_frame,
        selected_features,
        factories,
    )

    trained_models = {"regression": {}, "classification": {}}
    full_regression = prepared_frame.dropna(subset=[REGRESSION_TARGET])
    full_classification = prepared_frame.dropna(subset=[CLASSIFICATION_TARGET])

    for model_name, factory in factories["regression"].items():
        model = factory()
        model.fit(full_regression[selected_features], full_regression[REGRESSION_TARGET].astype(float))
        trained_models["regression"][model_name] = model

    for model_name, factory in factories["classification"].items():
        model = factory()
        model.fit(
            full_classification[selected_features],
            full_classification[CLASSIFICATION_TARGET].astype(int),
        )
        trained_models["classification"][model_name] = model

    return {
        "trained_at": datetime.utcnow(),
        "feature_columns": selected_features,
        "models": trained_models,
        "evaluation": {
            "holdout": holdout_metrics,
            "expanding_window": expanding_window_metrics,
        },
        "selected_models": {
            "regression": _select_best_regression_model(holdout_metrics["regression"]),
            "classification": _select_best_classification_model(holdout_metrics["classification"]),
        },
        "row_counts": {
            "total_rows": int(len(prepared_frame)),
            "regression_rows": int(len(full_regression)),
            "classification_rows": int(len(full_classification)),
        },
        "training_frame_dates": {
            "start_date": prepared_frame["feature_date"].min().date(),
            "end_date": prepared_frame["feature_date"].max().date(),
        },
    }


def train_models_from_feature_store(
    session: Session,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Read the stored feature set from SQL and train the forecasting models."""

    training_frame = load_training_frame_from_store(session=session, ticker=ticker)
    return train_forecasting_models(training_frame)
