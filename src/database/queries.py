"""
Reusable database query functions for core asset and price retrieval.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import Select
from sqlalchemy import text
from sqlalchemy import select
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
