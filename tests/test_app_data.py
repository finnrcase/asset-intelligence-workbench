import unittest
from contextlib import contextmanager
from unittest.mock import patch

from sqlalchemy.exc import OperationalError

from src.utils import app_data


class AppDataTests(unittest.TestCase):
    def test_classify_market_data_provider_error_detects_rate_limit(self) -> None:
        status, message = app_data._classify_market_data_provider_error(
            RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        )

        self.assertEqual(status, "rate_limited")
        self.assertIn("temporarily rate limited", message)

    @patch("src.utils.app_data.initialize_database", return_value=None)
    @patch("src.utils.app_data.load_price_history", return_value=[])
    @patch("src.utils.app_data.load_asset_metadata", return_value=None)
    def test_ingest_single_ticker_returns_rate_limited_status(
        self,
        _mock_metadata,
        _mock_prices,
        _mock_initialize,
    ) -> None:
        class FakeMarketDataClient:
            def fetch_asset_metadata(self, ticker: str):
                raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

        with patch("src.ingestion.market_data.YFinanceMarketDataClient", return_value=FakeMarketDataClient()):
            result = app_data.ingest_single_ticker("VOO")

        self.assertFalse(result["success"])
        self.assertEqual(result["ticker"], "VOO")
        self.assertEqual(result["status"], "rate_limited")
        self.assertIn("temporarily rate limited", result["message"])

    def test_write_market_data_retries_after_readonly_sqlite_error(self) -> None:
        calls = {"count": 0}

        @contextmanager
        def fake_session_scope():
            calls["count"] += 1
            if calls["count"] == 1:
                raise OperationalError(
                    "UPDATE data_sources SET updated_at=? WHERE data_sources.id = ?",
                    {},
                    Exception("attempt to write a readonly database"),
                )
            yield object()

        with patch("src.utils.app_data.session_scope", fake_session_scope):
            with patch("src.utils.app_data._reset_database_engine_if_available") as mock_reset:
                with patch("src.database.loaders.upsert_asset_metadata", return_value=[]):
                    with patch("src.database.loaders.load_historical_prices", return_value=[]):
                        app_data._write_market_data_to_database(
                            ticker="VOO",
                            metadata={"ticker": "VOO", "asset_name": "Vanguard S&P 500 ETF"},
                            price_rows=[{"price_date": "2026-03-20", "close_price": 1}],
                            source_name="yfinance",
                            source_type="market_data_api",
                            source_url="https://finance.yahoo.com/",
                        )

        self.assertEqual(calls["count"], 2)
        mock_reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
