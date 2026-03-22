"""
Bootstrap workflow for first-pass news sentiment ingestion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.database.connection import initialize_database
from src.database.connection import session_scope
from src.database.loaders import load_news_articles
from src.database.queries import get_asset_list
from src.database.queries import get_asset_metadata
from src.ingestion.sentiment_data import build_default_news_provider
from src.ingestion.sentiment_data import NewsSentimentProvider


STARTER_SENTIMENT_UNIVERSE = ["AAPL", "MSFT", "SPY", "QQQ", "BTC-USD"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def resolve_sentiment_universe(
    tickers: Sequence[str] | None = None,
    session_factory: Callable[[], Any] = session_scope,
) -> list[dict[str, Any]]:
    """Resolve the sentiment ingestion universe from assets already stored in the database."""

    requested = {ticker.strip().upper() for ticker in (tickers or STARTER_SENTIMENT_UNIVERSE)}
    with session_factory() as session:
        assets = get_asset_list(session, active_only=True)

    return [asset for asset in assets if asset["ticker"] in requested]


def ingest_asset_news_sentiment(
    asset: dict[str, Any],
    provider: NewsSentimentProvider,
    page_size: int = 10,
    session_factory: Callable[[], Any] = session_scope,
) -> dict[str, Any]:
    """Fetch and load recent news sentiment for a single stored asset."""

    with session_factory() as session:
        metadata = get_asset_metadata(session, asset["ticker"])

    company_name = metadata["asset_name"] if metadata else asset["asset_name"]
    articles = provider.fetch_recent_articles(
        ticker=asset["ticker"],
        company_name=company_name,
        page_size=page_size,
    )
    article_rows = [article.as_dict() for article in articles]

    with session_factory() as session:
        persisted = load_news_articles(
            session=session,
            ticker=asset["ticker"],
            articles=article_rows,
            source_name=provider.source_name,
            source_type=provider.source_type,
            source_url=provider.source_url,
        )

    return {
        "ticker": asset["ticker"],
        "asset_name": asset["asset_name"],
        "articles_loaded": len(persisted),
        "provider": provider.source_name,
    }


def ingest_sentiment_for_ticker(
    ticker: str,
    page_size: int = 10,
    provider: NewsSentimentProvider | None = None,
    session_factory: Callable[[], Any] = session_scope,
) -> dict[str, Any]:
    """Fetch and store recent sentiment for a single ticker already present in the database."""

    initialize_database(schema_path=SCHEMA_PATH)
    normalized_ticker = ticker.strip().upper()
    universe = resolve_sentiment_universe(
        tickers=[normalized_ticker],
        session_factory=session_factory,
    )
    if not universe:
        raise ValueError(
            f"{normalized_ticker} must exist in the local asset universe before sentiment can be loaded."
        )

    resolved_provider = provider or build_default_news_provider()
    return ingest_asset_news_sentiment(
        asset=universe[0],
        provider=resolved_provider,
        page_size=page_size,
        session_factory=session_factory,
    )


def bootstrap_sentiment_ingestion(
    tickers: Sequence[str] | None = None,
    page_size: int = 10,
    provider: NewsSentimentProvider | None = None,
    session_factory: Callable[[], Any] = session_scope,
) -> list[dict[str, Any]]:
    """
    Bootstrap recent news sentiment ingestion for a starter asset universe.

    The workflow assumes asset metadata already exists in the database and loads
    recent article sentiment for those stored assets.
    """

    initialize_database(schema_path=SCHEMA_PATH)
    resolved_provider = provider or build_default_news_provider()
    universe = resolve_sentiment_universe(tickers=tickers, session_factory=session_factory)

    summaries: list[dict[str, Any]] = []
    for asset in universe:
        summaries.append(
            ingest_asset_news_sentiment(
                asset=asset,
                provider=resolved_provider,
                page_size=page_size,
                session_factory=session_factory,
            )
        )
    return summaries


def _print_summary(summaries: Sequence[dict[str, Any]]) -> None:
    """Print a concise bootstrap summary for terminal execution."""

    if not summaries:
        print("No stored assets were available for sentiment bootstrap ingestion.")
        return

    print("Sentiment bootstrap complete:")
    for summary in summaries:
        print(
            f"- {summary['ticker']}: loaded {summary['articles_loaded']} articles "
            f"from {summary['provider']}"
        )


if __name__ == "__main__":
    _print_summary(bootstrap_sentiment_ingestion())
