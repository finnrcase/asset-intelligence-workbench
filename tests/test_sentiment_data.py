"""
Focused tests for the news sentiment ingestion and retrieval foundation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import unittest

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from src.database.loaders import load_news_articles
from src.database.loaders import upsert_asset_metadata
from src.database.queries import get_recent_news_sentiment
from src.ingestion.sentiment_data import build_news_query
from src.ingestion.sentiment_data import GNEWS_SOURCE_NAME
from src.ingestion.sentiment_data import GNEWS_SOURCE_TYPE
from src.ingestion.sentiment_data import GNEWS_SOURCE_URL
from src.ingestion.sentiment_data import normalize_gnews_article
from src.ingestion.sentiment_data import score_news_sentiment
from src.ingestion.sentiment_data import sentiment_label_from_score


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


class SentimentDataTests(unittest.TestCase):
    """Validate sentiment normalization, scoring, storage, and retrieval."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.engine.begin() as connection:
            for statement in [chunk.strip() for chunk in schema_sql.split(";") if chunk.strip()]:
                connection.exec_driver_sql(statement)

        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_build_news_query_combines_ticker_and_company_name(self) -> None:
        query = build_news_query(ticker="aapl", company_name="Apple Inc")
        self.assertEqual(query, '"AAPL" OR "Apple Inc"')

    def test_score_news_sentiment_and_label_output(self) -> None:
        positive_score = score_news_sentiment("Company beats expectations with strong growth")
        negative_score = score_news_sentiment("Analysts warn of weak demand and losses")
        neutral_score = score_news_sentiment("Company announces conference schedule")

        self.assertGreater(positive_score, 0)
        self.assertLess(negative_score, 0)
        self.assertEqual(neutral_score, 0.0)
        self.assertEqual(sentiment_label_from_score(positive_score), "positive")
        self.assertEqual(sentiment_label_from_score(negative_score), "negative")
        self.assertEqual(sentiment_label_from_score(neutral_score), "neutral")

    def test_normalize_gnews_article_returns_loader_ready_shape(self) -> None:
        article = normalize_gnews_article(
            article={
                "id": "gnews-1001",
                "title": "Apple beats estimates as services growth remains strong",
                "description": "Investors reacted positively to the earnings release.",
                "url": "https://example.com/apple-earnings",
                "publishedAt": "2026-03-17T15:00:00Z",
                "source": {"name": "Reuters", "url": "https://reuters.com"},
            },
            query_text='"AAPL" OR "Apple Inc"',
        )

        payload = article.as_dict()
        self.assertEqual(payload["headline"], "Apple beats estimates as services growth remains strong")
        self.assertEqual(payload["publisher_name"], "Reuters")
        self.assertEqual(payload["url"], "https://example.com/apple-earnings")
        self.assertIn(payload["sentiment_label"], {"positive", "neutral", "negative"})
        self.assertIsInstance(payload["published_at"], datetime)
        self.assertTrue(payload["provider_article_id"])

    def test_duplicate_articles_are_upserted_and_query_returns_expected_structure(self) -> None:
        with self.Session() as session:
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
                    }
                ],
            )
            session.commit()

        article_payload = normalize_gnews_article(
            article={
                "id": "gnews-2002",
                "title": "Apple beats estimates as margin outlook improves",
                "description": "The company reported strong demand and resilient growth.",
                "url": "https://example.com/apple-margin-outlook",
                "publishedAt": "2026-03-17T11:15:00Z",
                "source": {"name": "Reuters"},
            },
            query_text='"AAPL" OR "Apple Inc"',
        ).as_dict()

        with self.Session() as session:
            first_load = load_news_articles(
                session=session,
                ticker="AAPL",
                articles=[article_payload],
                source_name=GNEWS_SOURCE_NAME,
                source_type=GNEWS_SOURCE_TYPE,
                source_url=GNEWS_SOURCE_URL,
            )
            second_payload = dict(article_payload)
            second_payload["summary"] = "Updated normalized summary."
            second_load = load_news_articles(
                session=session,
                ticker="AAPL",
                articles=[second_payload],
                source_name=GNEWS_SOURCE_NAME,
                source_type=GNEWS_SOURCE_TYPE,
                source_url=GNEWS_SOURCE_URL,
            )
            session.commit()

            count_row = session.execute(text("SELECT COUNT(*) AS count FROM news_articles")).mappings().one()
            results = get_recent_news_sentiment(session, ticker="AAPL", limit=10)

        self.assertEqual(len(first_load), 1)
        self.assertEqual(len(second_load), 1)
        self.assertEqual(count_row["count"], 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ticker"], "AAPL")
        self.assertEqual(results[0]["source_name"], GNEWS_SOURCE_NAME)
        self.assertEqual(results[0]["headline"], article_payload["headline"])
        self.assertEqual(results[0]["url"], article_payload["url"])
        self.assertIn("sentiment_score", results[0])
        self.assertIn("sentiment_label", results[0])

    def test_same_provider_article_can_be_stored_for_multiple_assets(self) -> None:
        with self.Session() as session:
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
            session.commit()

        shared_article = normalize_gnews_article(
            article={
                "id": "gnews-3003",
                "title": "Mega-cap tech earnings improve outlook for the sector",
                "description": "The article discusses both Apple and Microsoft.",
                "url": "https://example.com/shared-tech-article",
                "publishedAt": "2026-03-17T11:15:00Z",
                "source": {"name": "Reuters"},
            },
            query_text='"AAPL" OR "Apple Inc"',
        ).as_dict()

        msft_article = dict(shared_article)
        msft_article["query_text"] = '"MSFT" OR "Microsoft Corporation"'

        with self.Session() as session:
            load_news_articles(
                session=session,
                ticker="AAPL",
                articles=[shared_article],
                source_name=GNEWS_SOURCE_NAME,
                source_type=GNEWS_SOURCE_TYPE,
                source_url=GNEWS_SOURCE_URL,
            )
            load_news_articles(
                session=session,
                ticker="MSFT",
                articles=[msft_article],
                source_name=GNEWS_SOURCE_NAME,
                source_type=GNEWS_SOURCE_TYPE,
                source_url=GNEWS_SOURCE_URL,
            )
            session.commit()

            count_row = session.execute(
                text("SELECT COUNT(*) AS count FROM news_articles")
            ).mappings().one()
            aapl_results = get_recent_news_sentiment(session, ticker="AAPL", limit=10)
            msft_results = get_recent_news_sentiment(session, ticker="MSFT", limit=10)

        self.assertEqual(count_row["count"], 2)
        self.assertEqual(len(aapl_results), 1)
        self.assertEqual(len(msft_results), 1)
        self.assertEqual(aapl_results[0]["url"], "https://example.com/shared-tech-article")
        self.assertEqual(msft_results[0]["url"], "https://example.com/shared-tech-article")


if __name__ == "__main__":
    unittest.main()
