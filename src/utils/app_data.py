"""
Application data helpers for the Streamlit layer.

These functions bridge the database query layer and the analytics/UI layer by
returning clean pandas structures suitable for charting and metric
calculation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy.exc import OperationalError

from src.database.connection import initialize_database
from src.database.connection import session_scope
from src.database import queries as database_queries


DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_SENTIMENT_PAGE_SIZE = 12
DEFAULT_SENTIMENT_FRESHNESS_HOURS = 24


def load_available_tickers() -> list[dict[str, Any]]:
    """Return the available asset universe for app selection controls."""

    with session_scope() as session:
        assets = database_queries.get_asset_list(session, active_only=True)

    return [
        {
            "ticker": asset["ticker"],
            "label": f"{asset['ticker']} | {asset['asset_name']}",
            "asset_name": asset["asset_name"],
            "asset_class": asset["asset_class"],
            "exchange": asset["exchange"],
            "currency": asset["currency"],
        }
        for asset in assets
    ]


def load_asset_metadata(ticker: str) -> dict[str, Any] | None:
    """Return detailed metadata for a selected ticker."""

    with session_scope() as session:
        return database_queries.get_asset_metadata(session, ticker)


def load_price_history(ticker: str) -> list[dict[str, Any]]:
    """Return stored historical price rows for a selected ticker."""

    with session_scope() as session:
        return database_queries.get_price_history(session, ticker)


def normalize_app_ticker(ticker: str) -> str:
    """Normalize user-entered ticker text into the project storage format."""

    if not ticker or not ticker.strip():
        return ""
    return ticker.strip().upper()


def ticker_exists(ticker: str) -> bool:
    """Return True when a ticker already exists in the local asset universe."""

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return False
    return load_asset_metadata(normalized_ticker) is not None


def ingest_single_ticker(
    ticker: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Ingest one ticker into the local database using the existing market-data pipeline.

    Returns a structured result so the Streamlit layer can show clear user-facing
    status messages without catching provider/database exceptions directly.
    """

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return {
            "success": False,
            "ticker": "",
            "status": "invalid_input",
            "message": "Enter a ticker symbol before attempting to load an asset.",
        }

    initialize_database()

    existing_metadata = load_asset_metadata(normalized_ticker)
    existing_prices = load_price_history(normalized_ticker)
    if existing_metadata is not None and existing_prices:
        return {
            "success": True,
            "ticker": normalized_ticker,
            "status": "database",
            "message": f"{normalized_ticker} was loaded from the local database.",
        }

    try:
        from src.database.loaders import load_historical_prices
        from src.database.loaders import upsert_asset_metadata
        from src.ingestion.market_data import YFINANCE_SOURCE_NAME
        from src.ingestion.market_data import YFINANCE_SOURCE_TYPE
        from src.ingestion.market_data import YFINANCE_SOURCE_URL
        from src.ingestion.market_data import YFinanceMarketDataClient
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ingestion_import_error",
            "message": (
                "The on-demand ingestion components could not be loaded. "
                f"Import detail: {exc}"
            ),
        }

    client = YFinanceMarketDataClient()

    try:
        metadata = client.fetch_asset_metadata(normalized_ticker).as_dict()
        price_rows = client.fetch_normalized_price_rows(
            normalized_ticker,
            lookback_days=lookback_days,
        )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "provider_error",
            "message": (
                f"Unable to resolve or download market data for {normalized_ticker}. "
                f"Provider detail: {exc}"
            ),
        }

    if not metadata.get("asset_name"):
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "missing_metadata",
            "message": f"{normalized_ticker} did not return usable asset metadata.",
        }

    if not price_rows:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "missing_prices",
            "message": f"{normalized_ticker} did not return daily historical price data.",
        }

    try:
        with session_scope() as session:
            upsert_asset_metadata(
                session=session,
                assets=[metadata],
                source_name=YFINANCE_SOURCE_NAME,
                source_type=YFINANCE_SOURCE_TYPE,
                source_url=YFINANCE_SOURCE_URL,
            )
            load_historical_prices(
                session=session,
                ticker=normalized_ticker,
                price_rows=price_rows,
                source_name=YFINANCE_SOURCE_NAME,
                source_type=YFINANCE_SOURCE_TYPE,
                source_url=YFINANCE_SOURCE_URL,
            )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ingestion_error",
            "message": f"Failed to write {normalized_ticker} into the local database. Detail: {exc}",
        }

    return {
        "success": True,
        "ticker": normalized_ticker,
        "status": "ingested",
        "message": f"{normalized_ticker} was fetched from the provider and added to the local database.",
    }


def resolve_asset_for_app(
    selected_ticker: str | None = None,
    manual_ticker: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Resolve an asset for the app from either the dropdown or manual ticker input.

    Manual input takes precedence when supplied. Existing database assets are
    loaded directly; missing assets are ingested on demand.
    """

    manual_normalized = normalize_app_ticker(manual_ticker or "")
    if manual_normalized:
        return ingest_single_ticker(manual_normalized, lookback_days=lookback_days)

    selected_normalized = normalize_app_ticker(selected_ticker or "")
    if not selected_normalized:
        return {
            "success": False,
            "ticker": "",
            "status": "no_selection",
            "message": "Select an existing asset or enter a new ticker to continue.",
        }

    if ticker_exists(selected_normalized):
        return {
            "success": True,
            "ticker": selected_normalized,
            "status": "database",
            "message": f"{selected_normalized} was loaded from the local database.",
        }

    return ingest_single_ticker(selected_normalized, lookback_days=lookback_days)


def prepare_price_history_frame(price_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert queried price rows into a clean time-series DataFrame for the app.

    The app prefers `adjusted_close` when available and falls back to
    `close_price` otherwise.
    """

    if not price_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(price_rows).copy()
    frame["price_date"] = pd.to_datetime(frame["price_date"])

    numeric_columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
        "volume",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["analysis_price"] = frame["adjusted_close"].where(
        frame["adjusted_close"].notna(),
        frame["close_price"],
    )
    frame = frame.sort_values("price_date").set_index("price_date")
    frame.index.name = "price_date"
    return frame


def get_recent_price_table(frame: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return a clean recent-price table for terminal or app display."""

    if frame.empty:
        return pd.DataFrame()

    recent = frame.reset_index().sort_values("price_date", ascending=False).head(rows).copy()
    display_columns = [
        "price_date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adjusted_close",
        "volume",
    ]
    available_columns = [column for column in display_columns if column in recent.columns]
    return recent[available_columns]


def load_recent_news_articles(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return recent stored news sentiment rows for a selected ticker."""

    with session_scope() as session:
        get_recent_news = getattr(database_queries, "get_recent_news_sentiment", None)
        if get_recent_news is None:
            return []
        try:
            return get_recent_news(session, ticker=ticker, limit=limit)
        except OperationalError:
            return []


def prepare_sentiment_frame(article_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare stored sentiment rows into app-friendly pandas structures."""

    if not article_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(article_rows).copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame["ingestion_timestamp"] = pd.to_datetime(
        frame["ingestion_timestamp"],
        errors="coerce",
    )
    frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce")
    frame = frame.dropna(subset=["published_at", "sentiment_score"]).sort_values(
        "published_at",
        ascending=True,
    )
    frame["published_date"] = frame["published_at"].dt.normalize()
    return frame


def get_sentiment_summary(article_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute a lightweight sentiment summary for the selected asset."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty:
        return {
            "article_count": 0,
            "average_sentiment": None,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "latest_published_at": None,
        }

    label_counts = frame["sentiment_label"].value_counts()
    return {
        "article_count": int(len(frame)),
        "average_sentiment": float(frame["sentiment_score"].mean()),
        "positive_count": int(label_counts.get("positive", 0)),
        "negative_count": int(label_counts.get("negative", 0)),
        "neutral_count": int(label_counts.get("neutral", 0)),
        "latest_published_at": frame["published_at"].max(),
    }


def get_sentiment_trend_frame(article_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate article sentiment into a daily trend frame for charting."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty:
        return pd.DataFrame()

    trend = (
        frame.groupby("published_date", as_index=False)
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "count"),
        )
        .sort_values("published_date")
    )
    trend["average_sentiment"] = trend["average_sentiment"].round(4)
    return trend


def get_recent_sentiment_table(article_rows: list[dict[str, Any]], rows: int = 8) -> pd.DataFrame:
    """Return a clean recent-articles table for the app layer."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty:
        return pd.DataFrame()

    recent = frame.sort_values("published_at", ascending=False).head(rows).copy()
    recent["published_at"] = recent["published_at"].dt.strftime("%Y-%m-%d %H:%M")
    display_columns = [
        "published_at",
        "publisher_name",
        "headline",
        "sentiment_label",
        "sentiment_score",
        "source_name",
        "url",
    ]
    return recent[display_columns]


def get_latest_sentiment_timestamp(article_rows: list[dict[str, Any]]) -> pd.Timestamp | None:
    """Return the most recent ingestion timestamp for stored sentiment rows."""

    frame = prepare_sentiment_frame(article_rows)
    if frame.empty or "ingestion_timestamp" not in frame:
        return None

    latest_timestamp = frame["ingestion_timestamp"].max()
    if pd.isna(latest_timestamp):
        return None
    return latest_timestamp


def sentiment_is_fresh(
    article_rows: list[dict[str, Any]],
    freshness_hours: int = DEFAULT_SENTIMENT_FRESHNESS_HOURS,
) -> bool:
    """Return True when stored sentiment is recent enough for app reuse."""

    latest_timestamp = get_latest_sentiment_timestamp(article_rows)
    if latest_timestamp is None:
        return False

    return latest_timestamp >= (pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=freshness_hours))


def sentiment_exists_for_ticker(ticker: str, minimum_articles: int = 1) -> bool:
    """Return True when stored sentiment records exist for the selected ticker."""

    return len(load_recent_news_articles(ticker, limit=minimum_articles)) >= minimum_articles


def ensure_sentiment_for_ticker(
    ticker: str,
    page_size: int = DEFAULT_SENTIMENT_PAGE_SIZE,
    freshness_hours: int = DEFAULT_SENTIMENT_FRESHNESS_HOURS,
) -> dict[str, Any]:
    """
    Ensure recent sentiment exists for a ticker, fetching on demand only when needed.

    The default policy is conservative for app reruns:
    - reuse stored sentiment when present and fresh enough
    - fetch only when no sentiment exists or the stored data is stale
    """

    normalized_ticker = normalize_app_ticker(ticker)
    if not normalized_ticker:
        return {
            "success": False,
            "ticker": "",
            "status": "invalid_input",
            "message": "A valid ticker is required before sentiment can be loaded.",
            "fetched": False,
        }

    stored_rows = load_recent_news_articles(normalized_ticker, limit=page_size)
    if stored_rows and sentiment_is_fresh(stored_rows, freshness_hours=freshness_hours):
        return {
            "success": True,
            "ticker": normalized_ticker,
            "status": "database",
            "message": f"{normalized_ticker} sentiment was loaded from the local database.",
            "fetched": False,
            "article_count": len(stored_rows),
        }

    try:
        from src.ingestion.bootstrap_sentiment import ingest_sentiment_for_ticker
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "ingestion_import_error",
            "message": f"Sentiment ingestion components could not be loaded. Detail: {exc}",
            "fetched": False,
        }

    initialize_database()

    try:
        summary = ingest_sentiment_for_ticker(
            ticker=normalized_ticker,
            page_size=page_size,
        )
    except Exception as exc:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "provider_error",
            "message": (
                f"Unable to fetch recent sentiment for {normalized_ticker}. "
                f"Provider detail: {exc}"
            ),
            "fetched": False,
        }

    if summary.get("articles_loaded", 0) <= 0:
        return {
            "success": False,
            "ticker": normalized_ticker,
            "status": "no_data",
            "message": f"No recent sentiment articles were returned for {normalized_ticker}.",
            "fetched": True,
        }

    return {
        "success": True,
        "ticker": normalized_ticker,
        "status": "ingested",
        "message": (
            f"{normalized_ticker} sentiment was fetched and stored locally "
            f"({summary['articles_loaded']} articles)."
        ),
        "fetched": True,
        "article_count": summary["articles_loaded"],
    }
