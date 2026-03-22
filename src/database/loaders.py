"""
Reusable data loading functions for asset master data, price history, and ML outputs.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import Asset
from src.database.connection import DataSource
from src.database.connection import HistoricalPrice
from src.utils.config import get_config


CONFIG = get_config()


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(value)


def _execute_sql_file(session: Session, file_name: str) -> None:
    sql_path = Path(CONFIG.project_root) / "sql" / file_name
    sql_text = sql_path.read_text(encoding="utf-8")
    connection = session.connection()
    for statement in [chunk.strip() for chunk in sql_text.split(";") if chunk.strip()]:
        connection.exec_driver_sql(statement)


def _ensure_table_columns(session: Session, table_name: str, expected_columns: dict[str, str]) -> None:
    existing_columns = {
        row["name"]
        for row in session.execute(text(f"PRAGMA table_info({table_name})")).mappings().all()
    }
    for column_name, column_definition in expected_columns.items():
        if column_name in existing_columns:
            continue
        session.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"))



def ensure_ml_tables(session: Session) -> None:
    _execute_sql_file(session, "feature_tables.sql")
    _execute_sql_file(session, "prediction_tables.sql")
    _ensure_table_columns(
        session,
        "sentiment_features",
        {
            "source_count_7d": "INTEGER",
            "source_sentiment_dispersion_7d": "NUMERIC(18, 8)",
        },
    )
    _ensure_table_columns(
        session,
        "ml_predictions",
        {
            "selected_model_name": "TEXT",
            "model_family": "TEXT",
            "target_name": "TEXT",
            "probability_positive_20d": "NUMERIC(18, 8)",
            "composite_ml_score": "NUMERIC(18, 8)",
            "confidence_score": "NUMERIC(18, 8)",
            "directional_signal": "TEXT",
            "history_score": "NUMERIC(18, 8)",
            "risk_score": "NUMERIC(18, 8)",
            "sentiment_score": "NUMERIC(18, 8)",
            "history_contribution": "NUMERIC(18, 8)",
            "risk_contribution": "NUMERIC(18, 8)",
            "sentiment_contribution": "NUMERIC(18, 8)",
            "pillar_weights_json": "TEXT",
            "feature_importance_json": "TEXT",
            "top_features_json": "TEXT",
        },
    )


def get_or_create_data_source(
    session: Session,
    source_name: str,
    source_type: str,
    source_url: str | None = None,
) -> DataSource:
    existing = session.scalar(
        select(DataSource).where(DataSource.source_name == source_name.strip())
    )
    if existing is not None:
        existing.source_type = source_type.strip()
        existing.source_url = source_url
        existing.updated_at = datetime.utcnow()
        session.flush()
        return existing

    data_source = DataSource(
        source_name=source_name.strip(),
        source_type=source_type.strip(),
        source_url=source_url,
    )
    session.add(data_source)
    session.flush()
    return data_source


def upsert_asset_metadata(
    session: Session,
    assets: Sequence[Mapping[str, Any]],
    source_name: str | None = None,
    source_type: str = "market_data",
    source_url: str | None = None,
) -> list[Asset]:
    primary_source = None
    if source_name:
        primary_source = get_or_create_data_source(
            session=session,
            source_name=source_name,
            source_type=source_type,
            source_url=source_url,
        )

    persisted_assets: list[Asset] = []
    for record in assets:
        ticker = _normalize_ticker(record["ticker"])
        asset = session.scalar(select(Asset).where(Asset.ticker == ticker))

        if asset is None:
            asset = Asset(ticker=ticker, asset_name=record["asset_name"].strip())
            session.add(asset)

        asset.asset_name = record["asset_name"].strip()
        asset.asset_class = record.get("asset_class")
        asset.exchange = record.get("exchange")
        asset.currency = record.get("currency", asset.currency or "USD")
        asset.sector = record.get("sector")
        asset.industry = record.get("industry")
        asset.country = record.get("country")
        asset.is_active = record.get("is_active", True)
        asset.updated_at = datetime.utcnow()
        if primary_source is not None:
            asset.primary_source_id = primary_source.id

        session.flush()
        persisted_assets.append(asset)

    return persisted_assets


def load_historical_prices(
    session: Session,
    ticker: str,
    price_rows: Sequence[Mapping[str, Any]],
    source_name: str,
    source_type: str = "market_data",
    source_url: str | None = None,
) -> list[HistoricalPrice]:
    normalized_ticker = _normalize_ticker(ticker)
    asset = session.scalar(select(Asset).where(Asset.ticker == normalized_ticker))
    if asset is None:
        raise ValueError(
            f"Asset '{normalized_ticker}' must exist before loading price history."
        )

    data_source = get_or_create_data_source(
        session=session,
        source_name=source_name,
        source_type=source_type,
        source_url=source_url,
    )

    persisted_prices: list[HistoricalPrice] = []
    for row in price_rows:
        price_date = _coerce_date(row["price_date"])
        existing = session.scalar(
            select(HistoricalPrice).where(
                HistoricalPrice.asset_id == asset.id,
                HistoricalPrice.source_id == data_source.id,
                HistoricalPrice.price_date == price_date,
            )
        )

        price_record = existing or HistoricalPrice(
            asset_id=asset.id,
            source_id=data_source.id,
            price_date=price_date,
            close_price=row["close_price"],
        )
        if existing is None:
            session.add(price_record)

        price_record.open_price = row.get("open_price")
        price_record.high_price = row.get("high_price")
        price_record.low_price = row.get("low_price")
        price_record.close_price = row["close_price"]
        price_record.adjusted_close = row.get("adjusted_close")
        price_record.volume = row.get("volume")
        price_record.ingestion_timestamp = row.get("ingestion_timestamp", datetime.utcnow())

        session.flush()
        persisted_prices.append(price_record)

    return persisted_prices


def load_news_articles(
    session: Session,
    ticker: str,
    articles: Sequence[Mapping[str, Any]],
    source_name: str,
    source_type: str = "news_api",
    source_url: str | None = None,
) -> list[dict[str, Any]]:
    normalized_ticker = _normalize_ticker(ticker)
    asset = session.scalar(select(Asset).where(Asset.ticker == normalized_ticker))
    if asset is None:
        raise ValueError(
            f"Asset '{normalized_ticker}' must exist before loading news sentiment."
        )

    data_source = get_or_create_data_source(
        session=session,
        source_name=source_name,
        source_type=source_type,
        source_url=source_url,
    )

    persisted_articles: list[dict[str, Any]] = []
    for article in articles:
        headline = str(article.get("headline", "")).strip()
        url = str(article.get("url", "")).strip()
        if not headline:
            raise ValueError("News articles require a non-empty `headline`.")
        if not url:
            raise ValueError("News articles require a non-empty `url`.")

        provider_article_id = article.get("provider_article_id")
        provider_article_id = (
            str(provider_article_id).strip() if provider_article_id not in (None, "") else None
        )
        published_at = article["published_at"]
        ingestion_timestamp = article.get("ingestion_timestamp", datetime.utcnow())

        existing = None
        if provider_article_id:
            existing = session.execute(
                text(
                    """
                    SELECT id
                    FROM news_articles
                    WHERE asset_id = :asset_id
                      AND source_id = :source_id
                      AND provider_article_id = :provider_article_id
                    """
                ),
                {
                    "asset_id": asset.id,
                    "source_id": data_source.id,
                    "provider_article_id": provider_article_id,
                },
            ).mappings().first()

        if existing is None:
            existing = session.execute(
                text(
                    """
                    SELECT id
                    FROM news_articles
                    WHERE asset_id = :asset_id
                      AND source_id = :source_id
                      AND url = :url
                    """
                ),
                {
                    "asset_id": asset.id,
                    "source_id": data_source.id,
                    "url": url,
                },
            ).mappings().first()

        payload = {
            "asset_id": asset.id,
            "source_id": data_source.id,
            "provider_article_id": provider_article_id,
            "publisher_name": (
                str(article.get("publisher_name")).strip()
                if article.get("publisher_name") not in (None, "")
                else None
            ),
            "headline": headline,
            "summary": article.get("summary"),
            "url": url,
            "published_at": published_at,
            "sentiment_score": article["sentiment_score"],
            "sentiment_label": article["sentiment_label"].strip(),
            "query_text": article.get("query_text"),
            "ingestion_timestamp": ingestion_timestamp,
        }

        if existing is None:
            session.execute(
                text(
                    """
                    INSERT INTO news_articles (
                        asset_id, source_id, provider_article_id, publisher_name, headline,
                        summary, url, published_at, sentiment_score, sentiment_label,
                        query_text, ingestion_timestamp
                    ) VALUES (
                        :asset_id, :source_id, :provider_article_id, :publisher_name, :headline,
                        :summary, :url, :published_at, :sentiment_score, :sentiment_label,
                        :query_text, :ingestion_timestamp
                    )
                    """
                ),
                payload,
            )
        else:
            payload["id"] = existing["id"]
            session.execute(
                text(
                    """
                    UPDATE news_articles
                    SET asset_id = :asset_id,
                        source_id = :source_id,
                        provider_article_id = :provider_article_id,
                        publisher_name = :publisher_name,
                        headline = :headline,
                        summary = :summary,
                        url = :url,
                        published_at = :published_at,
                        sentiment_score = :sentiment_score,
                        sentiment_label = :sentiment_label,
                        query_text = :query_text,
                        ingestion_timestamp = :ingestion_timestamp
                    WHERE id = :id
                    """
                ),
                payload,
            )

        persisted_articles.append(payload)

    session.flush()
    return persisted_articles


def load_technical_features(session: Session, feature_rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> int:
    ensure_ml_tables(session)
    frame = pd.DataFrame(feature_rows).copy()
    if frame.empty:
        return 0

    records = frame.to_dict(orient="records")
    for record in records:
        session.execute(
            text(
                """
                INSERT INTO technical_features (
                    asset_id, feature_date, analysis_price, close_price, adjusted_close, volume,
                    daily_return, return_lag_1d, return_lag_5d, return_lag_10d,
                    rolling_mean_return_5d, rolling_mean_return_20d,
                    rolling_volatility_10d, rolling_volatility_20d,
                    realized_volatility_20d, recent_realized_volatility_5d,
                    momentum_5d, momentum_10d, momentum_20d,
                    ma_distance_10d, ma_distance_20d,
                    drawdown_from_peak, rolling_drawdown_20d, downside_volatility_20d,
                    intraday_range_pct, volume_change_1d, volume_ratio_20d, volume_zscore_20d,
                    target_forward_return_20d, target_negative_return_20d, feature_version
                ) VALUES (
                    :asset_id, :feature_date, :analysis_price, :close_price, :adjusted_close, :volume,
                    :daily_return, :return_lag_1d, :return_lag_5d, :return_lag_10d,
                    :rolling_mean_return_5d, :rolling_mean_return_20d,
                    :rolling_volatility_10d, :rolling_volatility_20d,
                    :realized_volatility_20d, :recent_realized_volatility_5d,
                    :momentum_5d, :momentum_10d, :momentum_20d,
                    :ma_distance_10d, :ma_distance_20d,
                    :drawdown_from_peak, :rolling_drawdown_20d, :downside_volatility_20d,
                    :intraday_range_pct, :volume_change_1d, :volume_ratio_20d, :volume_zscore_20d,
                    :target_forward_return_20d, :target_negative_return_20d, :feature_version
                )
                ON CONFLICT(asset_id, feature_date) DO UPDATE SET
                    analysis_price = excluded.analysis_price,
                    close_price = excluded.close_price,
                    adjusted_close = excluded.adjusted_close,
                    volume = excluded.volume,
                    daily_return = excluded.daily_return,
                    return_lag_1d = excluded.return_lag_1d,
                    return_lag_5d = excluded.return_lag_5d,
                    return_lag_10d = excluded.return_lag_10d,
                    rolling_mean_return_5d = excluded.rolling_mean_return_5d,
                    rolling_mean_return_20d = excluded.rolling_mean_return_20d,
                    rolling_volatility_10d = excluded.rolling_volatility_10d,
                    rolling_volatility_20d = excluded.rolling_volatility_20d,
                    realized_volatility_20d = excluded.realized_volatility_20d,
                    recent_realized_volatility_5d = excluded.recent_realized_volatility_5d,
                    momentum_5d = excluded.momentum_5d,
                    momentum_10d = excluded.momentum_10d,
                    momentum_20d = excluded.momentum_20d,
                    ma_distance_10d = excluded.ma_distance_10d,
                    ma_distance_20d = excluded.ma_distance_20d,
                    drawdown_from_peak = excluded.drawdown_from_peak,
                    rolling_drawdown_20d = excluded.rolling_drawdown_20d,
                    downside_volatility_20d = excluded.downside_volatility_20d,
                    intraday_range_pct = excluded.intraday_range_pct,
                    volume_change_1d = excluded.volume_change_1d,
                    volume_ratio_20d = excluded.volume_ratio_20d,
                    volume_zscore_20d = excluded.volume_zscore_20d,
                    target_forward_return_20d = excluded.target_forward_return_20d,
                    target_negative_return_20d = excluded.target_negative_return_20d,
                    feature_version = excluded.feature_version
                """
            ),
            record,
        )
    return int(len(records))


def load_sentiment_features(session: Session, feature_rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> int:
    ensure_ml_tables(session)
    frame = pd.DataFrame(feature_rows).copy()
    if frame.empty:
        return 0

    records = frame.to_dict(orient="records")
    for record in records:
        session.execute(
            text(
                """
                INSERT INTO sentiment_features (
                    asset_id, feature_date, article_count_1d, sentiment_mean_1d,
                    sentiment_mean_7d, sentiment_std_7d, negative_article_share_7d,
                    positive_article_share_7d, article_count_7d, source_count_7d,
                    source_sentiment_dispersion_7d
                ) VALUES (
                    :asset_id, :feature_date, :article_count_1d, :sentiment_mean_1d,
                    :sentiment_mean_7d, :sentiment_std_7d, :negative_article_share_7d,
                    :positive_article_share_7d, :article_count_7d, :source_count_7d,
                    :source_sentiment_dispersion_7d
                )
                ON CONFLICT(asset_id, feature_date) DO UPDATE SET
                    article_count_1d = excluded.article_count_1d,
                    sentiment_mean_1d = excluded.sentiment_mean_1d,
                    sentiment_mean_7d = excluded.sentiment_mean_7d,
                    sentiment_std_7d = excluded.sentiment_std_7d,
                    negative_article_share_7d = excluded.negative_article_share_7d,
                    positive_article_share_7d = excluded.positive_article_share_7d,
                    article_count_7d = excluded.article_count_7d,
                    source_count_7d = excluded.source_count_7d,
                    source_sentiment_dispersion_7d = excluded.source_sentiment_dispersion_7d
                """
            ),
            record,
        )
    return int(len(records))


def load_ml_model_run(session: Session, run_record: Mapping[str, Any]) -> None:
    ensure_ml_tables(session)
    payload = dict(run_record)
    if isinstance(payload.get("evaluation_summary"), (dict, list)):
        payload["evaluation_summary"] = json.dumps(payload["evaluation_summary"], default=str)

    session.execute(
        text(
            """
            INSERT INTO ml_model_runs (
                run_id, run_timestamp, regression_model_name, classification_model_name,
                training_start_date, training_end_date, evaluation_summary, feature_version, notes
            ) VALUES (
                :run_id, :run_timestamp, :regression_model_name, :classification_model_name,
                :training_start_date, :training_end_date, :evaluation_summary, :feature_version, :notes
            )
            ON CONFLICT(run_id) DO UPDATE SET
                run_timestamp = excluded.run_timestamp,
                regression_model_name = excluded.regression_model_name,
                classification_model_name = excluded.classification_model_name,
                training_start_date = excluded.training_start_date,
                training_end_date = excluded.training_end_date,
                evaluation_summary = excluded.evaluation_summary,
                feature_version = excluded.feature_version,
                notes = excluded.notes
            """
        ),
        payload,
    )


def load_ml_predictions(
    session: Session,
    prediction_rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    model_run_id: str | None = None,
) -> int:
    ensure_ml_tables(session)
    frame = pd.DataFrame(prediction_rows).copy()
    if frame.empty:
        return 0

    records = frame.to_dict(orient="records")
    for record in records:
        record["model_run_id"] = model_run_id
        if isinstance(record.get("prediction_generated_at"), pd.Timestamp):
            record["prediction_generated_at"] = record["prediction_generated_at"].to_pydatetime()
        for json_field in ("pillar_weights_json", "feature_importance_json", "top_features_json"):
            if isinstance(record.get(json_field), (dict, list)):
                record[json_field] = json.dumps(record[json_field], default=str)
        session.execute(
            text(
                """
                INSERT INTO ml_predictions (
                    asset_id, as_of_date, prediction_horizon_days, model_run_id,
                    regression_model_name, classification_model_name,
                    selected_model_name, model_family, target_name,
                    predicted_return_20d, downside_probability_20d, probability_positive_20d,
                    predicted_negative_return_flag, composite_ml_score, confidence_score,
                    directional_signal, history_score, risk_score, sentiment_score,
                    history_contribution, risk_contribution, sentiment_contribution,
                    pillar_weights_json, feature_importance_json, top_features_json,
                    prediction_generated_at
                ) VALUES (
                    :asset_id, :as_of_date, :prediction_horizon_days, :model_run_id,
                    :regression_model_name, :classification_model_name,
                    :selected_model_name, :model_family, :target_name,
                    :predicted_return_20d, :downside_probability_20d, :probability_positive_20d,
                    :predicted_negative_return_flag, :composite_ml_score, :confidence_score,
                    :directional_signal, :history_score, :risk_score, :sentiment_score,
                    :history_contribution, :risk_contribution, :sentiment_contribution,
                    :pillar_weights_json, :feature_importance_json, :top_features_json,
                    :prediction_generated_at
                )
                ON CONFLICT(
                    asset_id, as_of_date, prediction_horizon_days,
                    regression_model_name, classification_model_name
                ) DO UPDATE SET
                    model_run_id = excluded.model_run_id,
                    selected_model_name = excluded.selected_model_name,
                    model_family = excluded.model_family,
                    target_name = excluded.target_name,
                    predicted_return_20d = excluded.predicted_return_20d,
                    downside_probability_20d = excluded.downside_probability_20d,
                    probability_positive_20d = excluded.probability_positive_20d,
                    predicted_negative_return_flag = excluded.predicted_negative_return_flag,
                    composite_ml_score = excluded.composite_ml_score,
                    confidence_score = excluded.confidence_score,
                    directional_signal = excluded.directional_signal,
                    history_score = excluded.history_score,
                    risk_score = excluded.risk_score,
                    sentiment_score = excluded.sentiment_score,
                    history_contribution = excluded.history_contribution,
                    risk_contribution = excluded.risk_contribution,
                    sentiment_contribution = excluded.sentiment_contribution,
                    pillar_weights_json = excluded.pillar_weights_json,
                    feature_importance_json = excluded.feature_importance_json,
                    top_features_json = excluded.top_features_json,
                    prediction_generated_at = excluded.prediction_generated_at
                """
            ),
            record,
        )
    return int(len(records))

