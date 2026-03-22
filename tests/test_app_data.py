import unittest
from unittest.mock import patch

from src.data.ingestion.service import IngestionResult
from src.utils import app_data


class AppDataTests(unittest.TestCase):
    @patch("src.utils.app_data.MARKET_DATA_SERVICE")
    def test_ingest_single_ticker_returns_service_result(self, mock_service) -> None:
        mock_service.ingest_ticker.return_value = IngestionResult(
            success=False,
            ticker="VOO",
            status="rate_limited",
            message="Yahoo market data is temporarily rate limited for this ticker. Please wait a minute and try again. Ticker: VOO.",
            source="provider",
        )

        result = app_data.ingest_single_ticker("VOO")

        self.assertFalse(result["success"])
        self.assertEqual(result["ticker"], "VOO")
        self.assertEqual(result["status"], "rate_limited")
        self.assertIn("temporarily rate limited", result["message"])

    @patch("src.utils.app_data.load_asset_metadata", return_value={"ticker": "VOO"})
    def test_ticker_exists_uses_storage_query_layer(self, _mock_metadata) -> None:
        self.assertTrue(app_data.ticker_exists("voo"))


if __name__ == "__main__":
    unittest.main()
