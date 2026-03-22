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

    @patch("src.utils.app_data._gnews_api_key_configured", return_value=False)
    @patch("src.utils.app_data.load_recent_news_articles", return_value=[])
    def test_ensure_sentiment_handles_missing_gnews_key_gracefully(self, _mock_rows, _mock_key) -> None:
        result = app_data.ensure_sentiment_for_ticker("SCHO")

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "sentiment_unavailable")
        self.assertEqual(result["sentiment_unavailable_reason"], "GNEWS_API_KEY not configured")
        self.assertIn("not configured", result["message"])

    @patch(
        "src.utils.app_data.load_latest_ml_forecast",
        side_effect=[
            {
                "ticker": "SCHO",
                "predicted_return_20d": 0.0056,
                "regression_model_name": "random_forest_regressor",
            },
            {
                "ticker": "SCHO",
                "predicted_return_20d": 0.0056,
                "probability_positive_20d": 0.57,
                "downside_probability_20d": 0.43,
                "composite_ml_score": 18.5,
                "confidence_score": 0.64,
                "history_score": 0.31,
                "risk_score": 0.12,
                "sentiment_score": 0.0,
                "directional_signal": "Neutral",
                "prediction_horizon_days": 20,
                "selected_model_name": "ridge_regression",
                "classification_model_name": "random_forest_classifier",
                "target_name": "forward_return_20d",
                "article_count_7d": 0,
                "source_count_7d": 0,
                "pillar_weights_json": [],
                "feature_importance_json": [],
                "top_features_json": [],
            },
        ],
    )
    @patch("src.utils.app_data.ensure_ml_forecast_for_ticker", return_value={"success": True, "status": "generated"})
    @patch("src.utils.app_data.load_ml_prediction_history", return_value=[])
    def test_build_ml_forecast_summary_rebuilds_incomplete_snapshot(
        self,
        _mock_history,
        mock_ensure,
        _mock_snapshot,
    ) -> None:
        summary = app_data.build_ml_forecast_summary("SCHO")

        self.assertTrue(summary["available"])
        self.assertEqual(summary["snapshot"]["composite_ml_score"], 18.5)
        self.assertEqual(summary["snapshot"]["sentiment_score_reason"], "sentiment provider unavailable or no cached sentiment; neutral fallback was used")
        mock_ensure.assert_called_once_with("SCHO")


if __name__ == "__main__":
    unittest.main()
