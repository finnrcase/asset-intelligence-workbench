"""
Prediction workflow for SQL-backed return and downside-risk forecasts.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.database import loaders as database_loaders
from src.database import queries as database_queries
from src.ml.train import prepare_model_frame


def build_prediction_frame(
    feature_frame: pd.DataFrame,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Filter the feature store down to the latest row per asset for prediction."""

    if feature_frame.empty:
        return feature_frame.copy()

    frame = feature_frame.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    if as_of_date is not None:
        frame = frame[frame["feature_date"] <= pd.Timestamp(as_of_date)]
    if frame.empty:
        return frame

    return (
        frame.sort_values(["asset_id", "feature_date"])
        .groupby("asset_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def generate_predictions(
    feature_frame: pd.DataFrame,
    training_result: dict[str, Any],
    as_of_date: date | None = None,
    prediction_horizon_days: int = 20,
) -> pd.DataFrame:
    """Generate probabilistic return/downside forecasts from the selected models."""

    if feature_frame.empty:
        return pd.DataFrame()

    prepared_frame, feature_columns = prepare_model_frame(
        feature_frame,
        training_result["feature_columns"],
    )
    prediction_frame = build_prediction_frame(prepared_frame, as_of_date=as_of_date)
    if prediction_frame.empty:
        return pd.DataFrame()

    regression_model_name = training_result["selected_models"]["regression"]
    classification_model_name = training_result["selected_models"]["classification"]
    regression_model = training_result["models"]["regression"][regression_model_name]
    classification_model = training_result["models"]["classification"][classification_model_name]

    prediction_frame["predicted_return_20d"] = regression_model.predict(prediction_frame[feature_columns])
    prediction_frame["downside_probability_20d"] = classification_model.predict_proba(
        prediction_frame[feature_columns]
    )[:, 1]
    prediction_frame["predicted_negative_return_flag"] = (
        prediction_frame["downside_probability_20d"] >= 0.5
    ).astype(int)
    prediction_frame["prediction_horizon_days"] = prediction_horizon_days
    prediction_frame["regression_model_name"] = regression_model_name
    prediction_frame["classification_model_name"] = classification_model_name
    prediction_frame["as_of_date"] = prediction_frame["feature_date"].dt.date
    prediction_frame["prediction_generated_at"] = datetime.utcnow()

    return prediction_frame[
        [
            "asset_id",
            "ticker",
            "as_of_date",
            "prediction_horizon_days",
            "regression_model_name",
            "classification_model_name",
            "predicted_return_20d",
            "downside_probability_20d",
            "predicted_negative_return_flag",
            "prediction_generated_at",
        ]
    ].copy()


def predict_from_feature_store(
    session: Session,
    training_result: dict[str, Any],
    ticker: str | None = None,
    as_of_date: date | None = None,
    model_run_id: str | None = None,
    write_to_sql: bool = True,
) -> pd.DataFrame:
    """Read stored features from SQL, generate forecasts, and optionally persist them."""

    feature_frame = database_queries.get_ml_training_frame(session=session, ticker=ticker)
    prediction_frame = generate_predictions(
        feature_frame=feature_frame,
        training_result=training_result,
        as_of_date=as_of_date,
    )

    if write_to_sql and not prediction_frame.empty:
        database_loaders.ensure_ml_tables(session)
        database_loaders.load_ml_predictions(
            session=session,
            prediction_rows=prediction_frame,
            model_run_id=model_run_id,
        )

    return prediction_frame
