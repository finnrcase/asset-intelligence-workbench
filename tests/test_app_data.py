import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
