"""
Focused tests for the sentiment bootstrap workflow and app-facing helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.loaders import upsert_asset_metadata
from src.database.queries import get_recent_news_sentiment
from src.ingestion.bootstrap_sentiment import bootstrap_sentiment_ingestion
from src.ingestion.bootstrap_sentiment import resolve_sentiment_universe
from src.ingestion.sentiment_data import GNEWS_SOURCE_NAME
from src.ingestion.sentiment_data import GNEWS_SOURCE_TYPE
from src.ingestion.sentiment_data import GNEWS_SOURCE_URL
from src.ingestion.sentiment_data import ProviderNewsArticle
from src.utils import app_data


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


class FakeNewsProvider:
    """Small deterministic provider for bootstrap testing."""

    source_name = GNEWS_SOURCE_NAME
    source_type = GNEWS_SOURCE_TYPE
    source_url = GNEWS_SOURCE_URL

    def fetch_recent_articles(
        self,
        ticker: str | None = None,
        company_name: str | None = None,
        page_size: int = 20,
    ) -> list[ProviderNewsArticle]:
        return [
            ProviderNewsArticle(
                headline=f"{ticker} beats expectations on strong demand",
                summary=f"{company_name} reported resilient growth.",
                url=f"https://example.com/{ticker.lower()}-article-1",
                published_at=datetime(2026, 3, 19, 12, 0, 0),
                publisher_name="Reuters",
                provider_article_id=f"{ticker}-article-1",
                query_text=f'"{ticker}" OR "{company_name}"',
                sentiment_score=0.75,
                sentiment_label="positive",
            ),
            ProviderNewsArticle(
                headline=f"{ticker} faces risk from slower consumer demand",
                summary="Analysts flagged weaker near-term momentum.",
                url=f"https://example.com/{ticker.lower()}-article-2",
                published_at=datetime(2026, 3, 18, 9, 30, 0),
                publisher_name="Bloomberg",
                provider_article_id=f"{ticker}-article-2",
                query_text=f'"{ticker}" OR "{company_name}"',
                sentiment_score=-0.5,
                sentiment_label="negative",
            ),
        ][:page_size]


class SentimentBootstrapTests(unittest.TestCase):
    """Validate bootstrap ingestion and app data helpers for sentiment."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.engine.begin() as connection:
            for statement in [chunk.strip() for chunk in schema_sql.split(";") if chunk.strip()]:
                connection.exec_driver_sql(statement)

        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

        @contextmanager
        def _session_factory():
            session = self.Session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        self.session_factory = _session_factory

        with self.session_factory() as session:
            upsert_asset_metadata(
                session,
                [
                    {
                        "ticker": "AAPL",
                        "asset_name": "Apple Inc.",
                        "asset_class": "Equity",
                        "exchange": "NASDAQ",
                        "currency": "USD",
                    },
                    {
                        "ticker": "MSFT",
                        "asset_name": "Microsoft Corporation",
                        "asset_class": "Equity",
                        "exchange": "NASDAQ",
                        "currency": "USD",
                    },
                ],
            )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_resolve_sentiment_universe_filters_to_requested_assets(self) -> None:
        universe = resolve_sentiment_universe(
            tickers=["AAPL"],
            session_factory=self.session_factory,
        )
        self.assertEqual(len(universe), 1)
        self.assertEqual(universe[0]["ticker"], "AAPL")

    def test_bootstrap_ingestion_loads_articles_and_supports_app_helpers(self) -> None:
        with patch("src.ingestion.bootstrap_sentiment.initialize_database", return_value=None):
            summaries = bootstrap_sentiment_ingestion(
                tickers=["AAPL"],
                page_size=2,
                provider=FakeNewsProvider(),
                session_factory=self.session_factory,
            )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["ticker"], "AAPL")
        self.assertEqual(summaries[0]["articles_loaded"], 2)

        with self.session_factory() as session:
            stored_rows = get_recent_news_sentiment(session, ticker="AAPL", limit=10)

        self.assertEqual(len(stored_rows), 2)
        summary = app_data.get_sentiment_summary(stored_rows)
        trend = app_data.get_sentiment_trend_frame(stored_rows)
        table = app_data.get_recent_sentiment_table(stored_rows, rows=5)

        self.assertEqual(summary["article_count"], 2)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(trend.shape[0], 2)
        self.assertFalse(table.empty)
        self.assertIn("headline", table.columns)


if __name__ == "__main__":
    unittest.main()
