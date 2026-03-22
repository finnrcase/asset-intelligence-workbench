"""
Model training workflow for the machine learning signal-calibration layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.features.feature_store import load_training_frame_from_store
from src.ml.evaluate import evaluate_classification_predictions
from src.ml.evaluate import evaluate_regression_predictions
from src.ml.evaluate import generate_expanding_window_splits
from src.ml.evaluate import summarize_metric_history
from src.ml.evaluate import time_holdout_split
from src.ml.features import PILLAR_COLUMNS
from src.ml.features import build_ml_feature_frame
from src.ml.interpret import aggregate_importance_by_pillar
from src.ml.interpret import aggregate_linear_weights_by_pillar
from src.ml.interpret import compute_permutation_feature_importance
from src.ml.interpret import extract_linear_feature_weights
from src.ml.models import get_model_factories
from src.ml.targets import NEGATIVE_TARGET_COLUMN
from src.ml.targets import TARGET_COLUMN
from src.ml.targets import TARGET_HORIZON_DAYS
from src.ml.targets import TARGET_NAME
from src.ml.targets import TARGET_SUMMARY


DEFAULT_EXCLUDED_COLUMNS = {
    "asset_id",
    "ticker",
    "feature_date",
    "price_date",
    "feature_version",
    TARGET_COLUMN,
    NEGATIVE_TARGET_COLUMN,
}



def infer_feature_columns(model_frame: pd.DataFrame) -> list[str]:
    """Infer numeric feature columns from the enriched ML model frame."""

    return [
        column
        for column in model_frame.columns
        if column not in DEFAULT_EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(model_frame[column])
    ]



def prepare_model_frame(
    training_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Prepare a clean, chronologically ordered modeling frame."""

    model_frame, feature_groups = build_ml_feature_frame(training_frame)
    if model_frame.empty:
        return model_frame.copy(), feature_columns or [], feature_groups

    selected_features = feature_columns or infer_feature_columns(model_frame)
    frame = model_frame.copy()
    for column in selected_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median_value = frame[column].median()
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
        frame[column] = frame[column].fillna(fill_value)

    return frame.sort_values(["feature_date", "asset_id"]).reset_index(drop=True), selected_features, feature_groups



def _evaluate_regression_models(
    factories: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    x_train = train_frame[feature_columns]
    y_train = train_frame[TARGET_COLUMN].astype(float)
    x_test = test_frame[feature_columns]
    y_test = test_frame[TARGET_COLUMN].astype(float).to_numpy()

    for model_name, factory in factories.items():
        model = factory()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
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
    y_train = (1 - train_frame[NEGATIVE_TARGET_COLUMN].astype(float)).astype(int)
    x_test = test_frame[feature_columns]
    y_test = (1 - test_frame[NEGATIVE_TARGET_COLUMN].astype(float)).astype(int).to_numpy()

    for model_name, factory in factories.items():
        model = factory()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
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
        reg_train = train_frame.dropna(subset=[TARGET_COLUMN])
        reg_test = test_frame.dropna(subset=[TARGET_COLUMN])
        clf_train = train_frame.dropna(subset=[NEGATIVE_TARGET_COLUMN])
        clf_test = test_frame.dropna(subset=[NEGATIVE_TARGET_COLUMN])

        if not reg_train.empty and not reg_test.empty:
            regression_metrics = _evaluate_regression_models(
                factories["regression"],
                reg_train,
                reg_test,
                feature_columns,
            )
            for model_name, metric_row in regression_metrics.items():
                regression_history[model_name].append(metric_row)

        if not clf_train.empty and not clf_test.empty:
            classification_metrics = _evaluate_classification_models(
                factories["classification"],
                clf_train,
                clf_test,
                feature_columns,
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



def train_forecasting_models(
    training_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train the three-pillar weighting engine and nonlinear challenger model."""

    prepared_frame, selected_features, feature_groups = prepare_model_frame(training_frame, feature_columns)
    if prepared_frame.empty:
        raise ValueError("A non-empty training frame is required to train forecasting models.")

    factories = get_model_factories()
    holdout_train, holdout_test = time_holdout_split(prepared_frame)
    regression_train = holdout_train.dropna(subset=[TARGET_COLUMN])
    regression_test = holdout_test.dropna(subset=[TARGET_COLUMN])
    classification_train = holdout_train.dropna(subset=[NEGATIVE_TARGET_COLUMN])
    classification_test = holdout_test.dropna(subset=[NEGATIVE_TARGET_COLUMN])

    if regression_train.empty or regression_test.empty:
        raise ValueError("Insufficient forward-return target history for time-aware regression training.")
    if classification_train.empty or classification_test.empty:
        raise ValueError("Insufficient forward-direction target history for time-aware classification training.")

    holdout_metrics = {
        "regression": _evaluate_regression_models(
            factories["regression"],
            regression_train,
            regression_test,
            selected_features,
        ),
        "classification": _evaluate_classification_models(
            factories["classification"],
            classification_train,
            classification_test,
            selected_features,
        ),
    }
    expanding_window_metrics = _summarize_expanding_window_metrics(
        prepared_frame,
        selected_features,
        factories,
    )

    trained_models = {"regression": {}, "classification": {}}
    full_regression = prepared_frame.dropna(subset=[TARGET_COLUMN])
    full_classification = prepared_frame.dropna(subset=[NEGATIVE_TARGET_COLUMN])

    for model_name, factory in factories["regression"].items():
        model = factory()
        model.fit(full_regression[selected_features], full_regression[TARGET_COLUMN].astype(float))
        trained_models["regression"][model_name] = model

    for model_name, factory in factories["classification"].items():
        model = factory()
        model.fit(full_classification[selected_features], (1 - full_classification[NEGATIVE_TARGET_COLUMN].astype(float)).astype(int))
        trained_models["classification"][model_name] = model

    linear_model = trained_models["regression"]["ridge_regression"]
    nonlinear_model = trained_models["regression"]["random_forest_regressor"]
    linear_feature_weights = extract_linear_feature_weights(linear_model, selected_features)
    linear_pillar_weights = aggregate_linear_weights_by_pillar(linear_model, selected_features, feature_groups)
    nonlinear_feature_importance = compute_permutation_feature_importance(
        nonlinear_model,
        regression_test[selected_features],
        regression_test[TARGET_COLUMN].astype(float),
        selected_features,
    )
    nonlinear_pillar_importance = aggregate_importance_by_pillar(
        nonlinear_feature_importance,
        feature_groups,
    )

    target_std = float(full_regression[TARGET_COLUMN].astype(float).std(ddof=1))
    if pd.isna(target_std) or target_std == 0.0:
        target_std = 0.05

    return {
        "trained_at": datetime.utcnow(),
        "target": {
            "name": TARGET_NAME,
            "column": TARGET_COLUMN,
            "horizon_days": TARGET_HORIZON_DAYS,
            "summary": TARGET_SUMMARY,
        },
        "feature_columns": selected_features,
        "feature_groups": feature_groups,
        "pillar_columns": PILLAR_COLUMNS,
        "models": trained_models,
        "evaluation": {
            "holdout": holdout_metrics,
            "expanding_window": expanding_window_metrics,
        },
        "selected_models": {
            "regression": "ridge_regression",
            "comparison": "random_forest_regressor",
            "classification": "random_forest_classifier",
        },
        "interpretability": {
            "linear_feature_weights": linear_feature_weights,
            "linear_pillar_weights": linear_pillar_weights,
            "nonlinear_feature_importance": nonlinear_feature_importance,
            "nonlinear_pillar_importance": nonlinear_pillar_importance,
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
        "target_statistics": {
            "mean": float(full_regression[TARGET_COLUMN].astype(float).mean()),
            "std": target_std,
        },
    }



def train_models_from_feature_store(
    session: Session,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Read the stored feature set from SQL and train the model suite."""

    training_frame = load_training_frame_from_store(session=session, ticker=ticker)
    return train_forecasting_models(training_frame)
