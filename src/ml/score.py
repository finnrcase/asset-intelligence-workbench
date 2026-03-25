"""
Signal scoring helpers for the machine learning weighting engine.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sanitize_probability(probability: float | None) -> float:
    """Clamp directional probabilities to a valid unit interval."""

    if probability is None or pd.isna(probability):
        return float("nan")
    return float(np.clip(float(probability), 0.0, 1.0))


def sanitize_predicted_return(
    predicted_return: float | None,
    target_volatility_scale: float | None = None,
    hard_cap: float = 0.60,
) -> float:
    """Bound model-implied returns to a plausible range for downstream use."""

    if predicted_return is None or pd.isna(predicted_return):
        return float("nan")

    dynamic_cap = hard_cap
    if target_volatility_scale is not None and not pd.isna(target_volatility_scale):
        scale = max(float(target_volatility_scale), 0.02)
        dynamic_cap = min(max(scale * 4.0, 0.20), hard_cap)

    return float(np.clip(float(predicted_return), -dynamic_cap, dynamic_cap))


def compute_composite_ml_score(
    predicted_return: float,
    probability_positive: float,
    history_score: float,
    risk_score: float,
    sentiment_score: float,
    target_volatility_scale: float,
) -> float:
    """Convert model outputs and pillar context into a bounded composite score."""

    predicted_return = sanitize_predicted_return(predicted_return, target_volatility_scale)
    probability_positive = sanitize_probability(probability_positive)
    scale = max(target_volatility_scale, 0.02)
    return_signal = math.tanh(predicted_return / scale)
    directional_probability_signal = (2.0 * probability_positive) - 1.0
    pillar_signal = float(np.nanmean([history_score, risk_score, sentiment_score]))
    return 100.0 * (
        (0.45 * return_signal)
        + (0.35 * directional_probability_signal)
        + (0.20 * pillar_signal)
    )



def classify_directional_signal(composite_score: float) -> str:
    """Map the composite score into analyst-friendly directional language."""

    if composite_score >= 20.0:
        return "Favorable"
    if composite_score <= -20.0:
        return "Unfavorable"
    return "Neutral"



def compute_confidence_score(
    predicted_return: float,
    comparison_predicted_return: float,
    probability_positive: float,
    target_volatility_scale: float,
) -> float:
    """Build a bounded confidence indicator from signal magnitude and model agreement."""

    predicted_return = sanitize_predicted_return(predicted_return, target_volatility_scale)
    comparison_predicted_return = sanitize_predicted_return(comparison_predicted_return, target_volatility_scale)
    probability_positive = sanitize_probability(probability_positive)
    scale = max(target_volatility_scale, 0.02)
    return_strength = min(abs(predicted_return) / scale, 1.0)
    probability_strength = min(abs((2.0 * probability_positive) - 1.0), 1.0)
    model_agreement = 1.0 - min(abs(predicted_return - comparison_predicted_return) / scale, 1.0)
    regression_direction_probability = 1.0 / (1.0 + math.exp(-(predicted_return / scale)))
    directional_alignment = 1.0 - min(abs(regression_direction_probability - probability_positive), 1.0)
    confidence = (
        (0.30 * return_strength)
        + (0.20 * probability_strength)
        + (0.25 * model_agreement)
        + (0.25 * directional_alignment)
    )
    return float(np.clip(confidence, 0.0, 1.0))



def summarize_pillar_contributions(contributions: dict[str, float]) -> list[dict[str, float | str]]:
    """Convert pillar contributions into chart-friendly rows."""

    return [
        {"pillar": "History", "contribution": float(contributions.get("history", 0.0))},
        {"pillar": "Risk", "contribution": float(contributions.get("risk", 0.0))},
        {"pillar": "Sentiment", "contribution": float(contributions.get("sentiment", 0.0))},
    ]



def summarize_feature_importance_rows(feature_importance: list[dict[str, float | str]], top_n: int = 8) -> list[dict[str, float | str]]:
    """Return the top-ranked feature importance rows."""

    ordered = sorted(feature_importance, key=lambda row: float(row["importance"]), reverse=True)
    return ordered[:top_n]



def prepare_prediction_history_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare stored prediction history rows for app and report visuals."""

    if prediction_frame.empty:
        return prediction_frame.copy()

    frame = prediction_frame.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    numeric_columns = [
        "predicted_return_20d",
        "downside_probability_20d",
        "probability_positive_20d",
        "composite_ml_score",
        "confidence_score",
        "history_score",
        "risk_score",
        "sentiment_score",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "predicted_return_20d" in frame.columns:
        frame["predicted_return_20d"] = frame["predicted_return_20d"].apply(sanitize_predicted_return)
    for probability_column in ("downside_probability_20d", "probability_positive_20d", "confidence_score"):
        if probability_column in frame.columns:
            frame[probability_column] = frame[probability_column].apply(sanitize_probability)
    return frame.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)
