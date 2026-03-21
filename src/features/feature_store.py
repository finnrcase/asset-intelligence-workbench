"""
Feature-store orchestration for the SQL-backed forecasting workflow.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.database import loaders as database_loaders
from src.database import queries as database_queries
from src.features.engineering import build_sentiment_feature_frame
from src.features.engineering import build_technical_feature_frame
from src.features.targets import attach_targets_to_features
from src.features.targets import build_forward_return_targets


def build_feature_store_frames(
    market_history: pd.DataFrame,
    news_history: pd.DataFrame | None = None,
    horizon_days: int = 20,
) -> dict[str, pd.DataFrame]:
    """Build technical, sentiment, and joined training frames in memory."""

    technical_features = build_technical_feature_frame(market_history)
    targets = build_forward_return_targets(market_history, horizon_days=horizon_days)
    technical_features = attach_targets_to_features(
        technical_features,
        targets,
        horizon_days=horizon_days,
    )

    sentiment_features = build_sentiment_feature_frame(news_history if news_history is not None else pd.DataFrame())
    training_frame = technical_features.merge(
        sentiment_features.drop(columns=["ticker"], errors="ignore"),
        on=["asset_id", "feature_date"],
        how="left",
    )

    return {
        "technical_features": technical_features,
        "sentiment_features": sentiment_features,
        "training_frame": training_frame,
    }


def persist_feature_store(
    session: Session,
    technical_features: pd.DataFrame,
    sentiment_features: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Persist engineered features into SQL tables."""

    database_loaders.ensure_ml_tables(session)
    technical_count = database_loaders.load_technical_features(session, technical_features)
    sentiment_count = 0
    if sentiment_features is not None and not sentiment_features.empty:
        sentiment_count = database_loaders.load_sentiment_features(session, sentiment_features)

    return {
        "technical_features_loaded": technical_count,
        "sentiment_features_loaded": sentiment_count,
    }


def refresh_feature_store(
    session: Session,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    horizon_days: int = 20,
) -> dict[str, Any]:
    """Pull raw source data from SQL, engineer features/targets, and persist them."""

    market_history = database_queries.get_market_feature_source_frame(
        session=session,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    news_history = database_queries.get_sentiment_source_frame(
        session=session,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    frames = build_feature_store_frames(
        market_history=market_history,
        news_history=news_history,
        horizon_days=horizon_days,
    )
    load_summary = persist_feature_store(
        session=session,
        technical_features=frames["technical_features"],
        sentiment_features=frames["sentiment_features"],
    )
    return {**frames, **load_summary}


def load_training_frame_from_store(
    session: Session,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Read the SQL-backed training frame used by model training/evaluation."""

    return database_queries.get_ml_training_frame(
        session=session,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

