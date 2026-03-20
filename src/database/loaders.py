"""
Reusable data loading functions for asset master data and price history.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

from sqlalchemy import text
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.connection import Asset
from src.database.connection import DataSource
from src.database.connection import HistoricalPrice


def _normalize_ticker(ticker: str) -> str:
    """Normalize tickers to a consistent storage format."""

    return ticker.strip().upper()


def _coerce_date(value: date | datetime | str) -> date:
    """Convert supported date inputs into a `date` object."""

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(value)


def get_or_create_data_source(
    session: Session,
    source_name: str,
    source_type: str,
    source_url: str | None = None,
) -> DataSource:
    """Return an existing data source or create it if it does not yet exist."""

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
    """
    Insert or update asset master records.

    Expected keys per asset record:
    `ticker`, `asset_name`, and optionally `asset_class`, `exchange`, `currency`,
    `sector`, `industry`, `country`, `is_active`.
    """

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
            asset = Asset(
                ticker=ticker,
                asset_name=record["asset_name"].strip(),
            )
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
    """
    Insert or update daily historical price observations for a ticker.

    Expected keys per price record:
    `price_date`, `close_price`, and optionally `open_price`, `high_price`,
    `low_price`, `adjusted_close`, `volume`, `ingestion_timestamp`.
    """

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
        price_record.ingestion_timestamp = row.get(
            "ingestion_timestamp", datetime.utcnow()
        )

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
    """
    Insert or update normalized news article records for a ticker.

    Expected keys per article record:
    `headline`, `url`, `published_at`, `sentiment_score`, `sentiment_label`, and
    optionally `provider_article_id`, `publisher_name`, `summary`, `query_text`,
    `ingestion_timestamp`.
    """

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
                        asset_id,
                        source_id,
                        provider_article_id,
                        publisher_name,
                        headline,
                        summary,
                        url,
                        published_at,
                        sentiment_score,
                        sentiment_label,
                        query_text,
                        ingestion_timestamp
                    ) VALUES (
                        :asset_id,
                        :source_id,
                        :provider_article_id,
                        :publisher_name,
                        :headline,
                        :summary,
                        :url,
                        :published_at,
                        :sentiment_score,
                        :sentiment_label,
                        :query_text,
                        :ingestion_timestamp
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
