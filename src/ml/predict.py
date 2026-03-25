"""
Prediction workflow for the machine learning signal-calibration layer.
"""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.database import loaders as database_loaders
from src.database import queries as database_queries
from src.ml.features import latest_rows_by_asset
from src.ml.interpret import aggregate_feature_contributions_by_pillar
from src.ml.interpret import compute_linear_feature_contributions
from src.ml.interpret import serialize_rows
from src.ml.score import classify_directional_signal
from src.ml.score import sanitize_predicted_return
from src.ml.score import sanitize_probability
from src.ml.score import compute_composite_ml_score
from src.ml.score import compute_confidence_score
from src.ml.train import prepare_model_frame
from src.ml.targets import TARGET_NAME


LOGGER = logging.getLogger(__name__)



def build_prediction_frame(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Filter the engineered feature frame down to the latest scoring row per asset."""

    if feature_frame.empty:
        return feature_frame.copy()

    prepared_frame, _, _ = prepare_model_frame(feature_frame, feature_columns)
    return latest_rows_by_asset(prepared_frame, as_of_date=pd.Timestamp(as_of_date) if as_of_date else None)



def _row_to_dataframe(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict()])



def generate_predictions(
    feature_frame: pd.DataFrame,
    training_result: dict[str, Any],
    as_of_date: date | None = None,
    prediction_horizon_days: int = 20,
) -> pd.DataFrame:
    """Generate composite ML scores, directional signals, and interpretability payloads."""

    if feature_frame.empty:
        return pd.DataFrame()

    prepared_frame, feature_columns, feature_groups = prepare_model_frame(
        feature_frame,
        training_result["feature_columns"],
    )
    prediction_frame = latest_rows_by_asset(prepared_frame, as_of_date=pd.Timestamp(as_of_date) if as_of_date else None)
    if prediction_frame.empty:
        return pd.DataFrame()

    linear_model = training_result["models"]["regression"][training_result["selected_models"]["regression"]]
    comparison_model = training_result["models"]["regression"][training_result["selected_models"]["comparison"]]
    classification_model = training_result["models"]["classification"][training_result["selected_models"]["classification"]]

    target_scale = float(training_result["target_statistics"]["std"])
    top_feature_rows = training_result["interpretability"]["nonlinear_feature_importance"]
    pillar_weight_rows = training_result["interpretability"]["linear_pillar_weights"]

    result_rows: list[dict[str, Any]] = []
    for _, row in prediction_frame.iterrows():
        row_frame = _row_to_dataframe(row)
        predicted_return = sanitize_predicted_return(
            float(linear_model.predict(row_frame[feature_columns])[0]),
            target_volatility_scale=target_scale,
        )
        comparison_predicted_return = sanitize_predicted_return(
            float(comparison_model.predict(row_frame[feature_columns])[0]),
            target_volatility_scale=target_scale,
        )
        probability_positive = sanitize_probability(
            float(classification_model.predict_proba(row_frame[feature_columns])[:, 1][0])
        )
        downside_probability = sanitize_probability(1.0 - probability_positive)
        missing_features = [
            column_name
            for column_name in feature_columns
            if pd.isna(row_frame.iloc[0][column_name])
        ]
        LOGGER.info(
            "ML scorer feature coverage for %s: generated=%s missing=%s",
            row["ticker"],
            len(feature_columns) - len(missing_features),
            missing_features,
        )
        feature_contributions = compute_linear_feature_contributions(
            linear_model,
            row_frame,
            feature_columns,
        )
        pillar_contributions = aggregate_feature_contributions_by_pillar(
            feature_contributions,
            feature_groups,
        )
        composite_ml_score = compute_composite_ml_score(
            predicted_return=predicted_return,
            probability_positive=probability_positive,
            history_score=float(row.get("history_score", 0.0)),
            risk_score=float(row.get("risk_score", 0.0)),
            sentiment_score=float(row.get("sentiment_score", 0.0)),
            target_volatility_scale=target_scale,
        )
        confidence_score = compute_confidence_score(
            predicted_return=predicted_return,
            comparison_predicted_return=comparison_predicted_return,
            probability_positive=probability_positive,
            target_volatility_scale=target_scale,
        )
        result_rows.append(
            {
                "asset_id": row["asset_id"],
                "ticker": row["ticker"],
                "as_of_date": pd.Timestamp(row["feature_date"]).date(),
                "prediction_horizon_days": prediction_horizon_days,
                "model_run_id": None,
                "regression_model_name": training_result["selected_models"]["regression"],
                "classification_model_name": training_result["selected_models"]["classification"],
                "selected_model_name": training_result["selected_models"]["regression"],
                "model_family": "machine_learning_signal_calibration",
                "target_name": TARGET_NAME,
                "predicted_return_20d": predicted_return,
                "downside_probability_20d": downside_probability,
                "probability_positive_20d": probability_positive,
                "predicted_negative_return_flag": int(downside_probability >= 0.5),
                "composite_ml_score": composite_ml_score,
                "confidence_score": confidence_score,
                "directional_signal": classify_directional_signal(composite_ml_score),
                "history_score": float(row.get("history_score", 0.0)),
                "risk_score": float(row.get("risk_score", 0.0)),
                "sentiment_score": float(row.get("sentiment_score", 0.0)),
                "history_contribution": float(pillar_contributions.get("history", 0.0)),
                "risk_contribution": float(pillar_contributions.get("risk", 0.0)),
                "sentiment_contribution": float(pillar_contributions.get("sentiment", 0.0)),
                "pillar_weights_json": pillar_weight_rows,
                "feature_importance_json": serialize_rows(top_feature_rows, top_n=8),
                "top_features_json": serialize_rows(feature_contributions, top_n=8),
                "prediction_generated_at": datetime.utcnow(),
            }
        )
        LOGGER.info(
            "ML scorer returned for %s: fields=%s composite=%s history=%s risk=%s sentiment=%s",
            row["ticker"],
            list(result_rows[-1].keys()),
            result_rows[-1]["composite_ml_score"],
            result_rows[-1]["history_score"],
            result_rows[-1]["risk_score"],
            result_rows[-1]["sentiment_score"],
        )

    return pd.DataFrame(result_rows)



def predict_from_feature_store(
    session: Session,
    training_result: dict[str, Any],
    ticker: str | None = None,
    as_of_date: date | None = None,
    model_run_id: str | None = None,
    write_to_sql: bool = True,
) -> pd.DataFrame:
    """Read stored features from SQL, score them, and optionally persist the output."""

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
