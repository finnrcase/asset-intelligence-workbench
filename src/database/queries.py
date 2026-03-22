"""
Reusable database query functions for core asset, price, and ML retrieval.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.database.connection import Asset
from src.database.connection import DataSource
from src.database.connection import HistoricalPrice


def get_asset_list(session: Session, active_only: bool = True) -> list[dict[str, Any]]:
    """Return a sorted list of tracked assets for selection or screening workflows."""

    statement = select(Asset).order_by(Asset.ticker.asc())
    if active_only:
        statement = statement.where(Asset.is_active.is_(True))

    assets = session.scalars(statement).all()
    return [
        {
            "ticker": asset.ticker,
            "asset_name": asset.asset_name,
            "asset_class": asset.asset_class,
            "exchange": asset.exchange,
            "currency": asset.currency,
            "is_active": asset.is_active,
        }
        for asset in assets
    ]


def get_asset_metadata(session: Session, ticker: str) -> dict[str, Any] | None:
    """Return detailed metadata for a single asset ticker."""

    statement = (
        select(Asset, DataSource)
        .outerjoin(DataSource, Asset.primary_source_id == DataSource.id)
        .where(Asset.ticker == ticker.strip().upper())
    )
    row = session.execute(statement).first()
    if row is None:
        return None

    asset, data_source = row
    return {
        "ticker": asset.ticker,
        "asset_name": asset.asset_name,
        "asset_class": asset.asset_class,
        "exchange": asset.exchange,
        "currency": asset.currency,
        "sector": asset.sector,
        "industry": asset.industry,
        "country": asset.country,
        "is_active": asset.is_active,
        "primary_source": data_source.source_name if data_source else None,
        "created_at": asset.created_at,
        "updated_at": asset.updated_at,
    }


def get_price_history(
    session: Session,
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return historical daily price data for a single ticker."""

    statement: Select[tuple[HistoricalPrice, DataSource]] = (
        select(HistoricalPrice, DataSource)
        .join(Asset, HistoricalPrice.asset_id == Asset.id)
        .join(DataSource, HistoricalPrice.source_id == DataSource.id)
        .where(Asset.ticker == ticker.strip().upper())
        .order_by(HistoricalPrice.price_date.asc())
    )

    if start_date is not None:
        statement = statement.where(HistoricalPrice.price_date >= start_date)
    if end_date is not None:
        statement = statement.where(HistoricalPrice.price_date <= end_date)
    if limit is not None:
        statement = statement.limit(limit)

    rows = session.execute(statement).all()
    return [
        {
            "price_date": price.price_date,
            "open_price": price.open_price,
            "high_price": price.high_price,
            "low_price": price.low_price,
            "close_price": price.close_price,
            "adjusted_close": price.adjusted_close,
            "volume": price.volume,
            "source_name": data_source.source_name,
            "ingestion_timestamp": price.ingestion_timestamp,
        }
        for price, data_source in rows
    ]


def get_recent_news_sentiment(
    session: Session,
    ticker: str,
    limit: int = 20,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Return recent stored news sentiment records for a single ticker."""

    statement = """
        SELECT
            a.ticker,
            na.publisher_name,
            na.headline,
            na.summary,
            na.url,
            na.published_at,
            na.sentiment_score,
            na.sentiment_label,
            na.query_text,
            ds.source_name,
            na.ingestion_timestamp
        FROM news_articles na
        JOIN assets a
          ON na.asset_id = a.id
        JOIN data_sources ds
          ON na.source_id = ds.id
        WHERE a.ticker = :ticker
    """
    params: dict[str, Any] = {"ticker": ticker.strip().upper(), "limit": limit}

    if start_date is not None:
        statement += " AND DATE(na.published_at) >= :start_date"
        params["start_date"] = start_date.isoformat()
    if end_date is not None:
        statement += " AND DATE(na.published_at) <= :end_date"
        params["end_date"] = end_date.isoformat()

    statement += " ORDER BY na.published_at DESC LIMIT :limit"
    rows = session.execute(text(statement), params).mappings().all()
    return [dict(row) for row in rows]


def get_market_feature_source_frame(
    session: Session,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Return the raw market-history frame used for feature engineering."""

    statement = """
        SELECT
            a.id AS asset_id,
            a.ticker,
            hp.price_date,
            hp.open_price,
            hp.high_price,
            hp.low_price,
            hp.close_price,
            hp.adjusted_close,
            hp.volume,
            ds.source_name
        FROM historical_prices hp
        JOIN assets a
          ON hp.asset_id = a.id
        JOIN data_sources ds
          ON hp.source_id = ds.id
        WHERE 1 = 1
    """
    params: dict[str, Any] = {}
    if ticker:
        statement += " AND a.ticker = :ticker"
        params["ticker"] = ticker.strip().upper()
    if start_date:
        statement += " AND hp.price_date >= :start_date"
        params["start_date"] = start_date.isoformat()
    if end_date:
        statement += " AND hp.price_date <= :end_date"
        params["end_date"] = end_date.isoformat()
    statement += " ORDER BY a.ticker, hp.price_date"

    rows = session.execute(text(statement), params).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    adjusted = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    close_price = pd.to_numeric(frame["close_price"], errors="coerce")
    frame["analysis_price"] = adjusted.where(adjusted.notna(), close_price)
    return frame


def get_sentiment_source_frame(
    session: Session,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Return article-level sentiment rows used for daily feature aggregation."""

    statement = """
        SELECT
            a.id AS asset_id,
            a.ticker,
            na.published_at,
            na.sentiment_score,
            na.sentiment_label,
            ds.source_name
        FROM news_articles na
        JOIN assets a
          ON na.asset_id = a.id
        JOIN data_sources ds
          ON na.source_id = ds.id
        WHERE 1 = 1
    """
    params: dict[str, Any] = {}
    if ticker:
        statement += " AND a.ticker = :ticker"
        params["ticker"] = ticker.strip().upper()
    if start_date:
        statement += " AND DATE(na.published_at) >= :start_date"
        params["start_date"] = start_date.isoformat()
    if end_date:
        statement += " AND DATE(na.published_at) <= :end_date"
        params["end_date"] = end_date.isoformat()
    statement += " ORDER BY a.ticker, na.published_at"

    try:
        rows = session.execute(text(statement), params).mappings().all()
    except OperationalError as exc:
        if "no such table: news_articles" in str(exc).lower():
            return pd.DataFrame(
                columns=[
                    "asset_id",
                    "ticker",
                    "published_at",
                    "sentiment_score",
                    "sentiment_label",
                    "source_name",
                ]
            )
        raise
    return pd.DataFrame(rows)


def get_ml_training_frame(
    session: Session,
    ticker: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Return the joined SQL feature store used for model training and scoring."""

    statement = """
        SELECT
            tf.asset_id,
            a.ticker,
            tf.feature_date,
            tf.analysis_price,
            tf.close_price,
            tf.adjusted_close,
            tf.volume,
            tf.daily_return,
            tf.return_lag_1d,
            tf.return_lag_5d,
            tf.return_lag_10d,
            tf.rolling_mean_return_5d,
            tf.rolling_mean_return_20d,
            tf.rolling_volatility_10d,
            tf.rolling_volatility_20d,
            tf.realized_volatility_20d,
            tf.recent_realized_volatility_5d,
            tf.momentum_5d,
            tf.momentum_10d,
            tf.momentum_20d,
            tf.ma_distance_10d,
            tf.ma_distance_20d,
            tf.drawdown_from_peak,
            tf.rolling_drawdown_20d,
            tf.downside_volatility_20d,
            tf.intraday_range_pct,
            tf.volume_change_1d,
            tf.volume_ratio_20d,
            tf.volume_zscore_20d,
            tf.target_forward_return_20d,
            tf.target_negative_return_20d,
            tf.feature_version,
            sf.article_count_1d,
            sf.sentiment_mean_1d,
            sf.sentiment_mean_7d,
            sf.sentiment_std_7d,
            sf.negative_article_share_7d,
            sf.positive_article_share_7d,
            sf.article_count_7d,
            sf.source_count_7d,
            sf.source_sentiment_dispersion_7d
        FROM technical_features tf
        JOIN assets a
          ON tf.asset_id = a.id
        LEFT JOIN sentiment_features sf
          ON tf.asset_id = sf.asset_id
         AND tf.feature_date = sf.feature_date
        WHERE 1 = 1
    """
    params: dict[str, Any] = {}
    if ticker:
        statement += " AND a.ticker = :ticker"
        params["ticker"] = ticker.strip().upper()
    if start_date:
        statement += " AND tf.feature_date >= :start_date"
        params["start_date"] = start_date.isoformat()
    if end_date:
        statement += " AND tf.feature_date <= :end_date"
        params["end_date"] = end_date.isoformat()
    statement += " ORDER BY a.ticker, tf.feature_date"

    rows = session.execute(text(statement), params).mappings().all()
    return pd.DataFrame(rows)


def get_ml_predictions(
    session: Session,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """Return stored ML prediction rows for one asset or the full universe."""

    statement = """
        SELECT
            a.ticker,
            mp.as_of_date,
            mp.prediction_horizon_days,
            mp.model_run_id,
            mp.regression_model_name,
            mp.classification_model_name,
            mp.predicted_return_20d,
            mp.downside_probability_20d,
            mp.probability_positive_20d,
            mp.predicted_negative_return_flag,
            mp.composite_ml_score,
            mp.confidence_score,
            mp.directional_signal,
            mp.history_score,
            mp.risk_score,
            mp.sentiment_score,
            mp.prediction_generated_at
        FROM ml_predictions mp
        JOIN assets a
          ON mp.asset_id = a.id
        WHERE 1 = 1
    """
    params: dict[str, Any] = {}
    if ticker:
        statement += " AND a.ticker = :ticker"
        params["ticker"] = ticker.strip().upper()
    statement += " ORDER BY a.ticker, mp.as_of_date"
    rows = session.execute(text(statement), params).mappings().all()
    return [dict(row) for row in rows]


def get_latest_ml_prediction(
    session: Session,
    ticker: str,
) -> dict[str, Any] | None:
    """Return the latest stored ML forecast snapshot for a single asset."""

    statement = """
        SELECT
            a.id AS asset_id,
            a.ticker,
            mp.as_of_date,
            mp.prediction_horizon_days,
            mp.model_run_id,
            mp.regression_model_name,
            mp.classification_model_name,
            mp.selected_model_name,
            mp.model_family,
            mp.target_name,
            mp.predicted_return_20d,
            mp.downside_probability_20d,
            mp.probability_positive_20d,
            mp.predicted_negative_return_flag,
            mp.composite_ml_score,
            mp.confidence_score,
            mp.directional_signal,
            mp.history_score,
            mp.risk_score,
            mp.sentiment_score,
            mp.history_contribution,
            mp.risk_contribution,
            mp.sentiment_contribution,
            mp.pillar_weights_json,
            mp.feature_importance_json,
            mp.top_features_json,
            mp.prediction_generated_at,
            mr.run_timestamp,
            mr.evaluation_summary,
            mr.feature_version,
            tf.realized_volatility_20d,
            tf.recent_realized_volatility_5d,
            tf.downside_volatility_20d,
            tf.rolling_volatility_20d,
            tf.momentum_20d,
            tf.ma_distance_20d,
            tf.drawdown_from_peak,
            tf.volume_ratio_20d,
            sf.sentiment_mean_1d,
            sf.sentiment_mean_7d,
            sf.negative_article_share_7d,
            sf.article_count_7d,
            sf.source_count_7d,
            sf.source_sentiment_dispersion_7d
        FROM ml_predictions mp
        JOIN assets a
          ON mp.asset_id = a.id
        LEFT JOIN ml_model_runs mr
          ON mp.model_run_id = mr.run_id
        LEFT JOIN technical_features tf
          ON mp.asset_id = tf.asset_id
         AND mp.as_of_date = tf.feature_date
        LEFT JOIN sentiment_features sf
          ON mp.asset_id = sf.asset_id
         AND mp.as_of_date = sf.feature_date
        WHERE a.ticker = :ticker
        ORDER BY mp.as_of_date DESC, mp.prediction_generated_at DESC
        LIMIT 1
    """
    row = session.execute(
        text(statement),
        {"ticker": ticker.strip().upper()},
    ).mappings().first()
    return dict(row) if row else None


def get_ml_prediction_history(
    session: Session,
    ticker: str,
    limit: int = 60,
) -> list[dict[str, Any]]:
    """Return recent stored prediction history for a single asset."""

    statement = """
        SELECT
            a.ticker,
            mp.as_of_date,
            mp.prediction_horizon_days,
            mp.regression_model_name,
            mp.classification_model_name,
            mp.predicted_return_20d,
            mp.downside_probability_20d,
            mp.probability_positive_20d,
            mp.predicted_negative_return_flag,
            mp.composite_ml_score,
            mp.confidence_score,
            mp.directional_signal,
            mp.history_score,
            mp.risk_score,
            mp.sentiment_score,
            mp.prediction_generated_at
        FROM ml_predictions mp
        JOIN assets a
          ON mp.asset_id = a.id
        WHERE a.ticker = :ticker
        ORDER BY mp.as_of_date DESC, mp.prediction_generated_at DESC
        LIMIT :limit
    """
    rows = session.execute(
        text(statement),
        {"ticker": ticker.strip().upper(), "limit": limit},
    ).mappings().all()
    return [dict(row) for row in rows]


def get_feature_driver_frame(
    session: Session,
    ticker: str,
    lookback_rows: int = 252,
) -> pd.DataFrame:
    """Return recent joined feature history for lightweight driver interpretation."""

    statement = """
        SELECT
            tf.feature_date,
            tf.realized_volatility_20d,
            tf.recent_realized_volatility_5d,
            tf.downside_volatility_20d,
            tf.rolling_volatility_20d,
            tf.momentum_20d,
            tf.ma_distance_20d,
            tf.drawdown_from_peak,
            tf.volume_ratio_20d,
            sf.sentiment_mean_7d,
            sf.negative_article_share_7d,
            sf.article_count_7d,
            sf.source_count_7d,
            sf.source_sentiment_dispersion_7d
        FROM technical_features tf
        JOIN assets a
          ON tf.asset_id = a.id
        LEFT JOIN sentiment_features sf
          ON tf.asset_id = sf.asset_id
         AND tf.feature_date = sf.feature_date
        WHERE a.ticker = :ticker
        ORDER BY tf.feature_date DESC
        LIMIT :limit
    """
    rows = session.execute(
        text(statement),
        {"ticker": ticker.strip().upper(), "limit": lookback_rows},
    ).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    return frame.sort_values("feature_date").reset_index(drop=True)
