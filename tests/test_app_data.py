import unittest
from unittest.mock import patch

from src.data.ingestion.service import IngestionResult
from src.utils import app_data


class AppDataTests(unittest.TestCase):
    def test_normalize_app_ticker_handles_case_whitespace_and_separator_variants(self) -> None:
        self.assertEqual(app_data.normalize_app_ticker("  brk/b "), "BRK-B")
        self.assertEqual(app_data.normalize_app_ticker(" msft "), "MSFT")
        self.assertEqual(app_data.normalize_app_ticker("btc-usd"), "BTC-USD")

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

    @patch(
        "src.utils.app_data._sentiment_provider_diagnostics",
        return_value={
            "availability": {"gnews": False, "finnhub": False, "newsapi": False},
            "selected_provider": None,
            "fallback_provider_used": False,
            "live_provider_available": False,
        },
    )
    @patch("src.utils.app_data.load_recent_news_articles", return_value=[])
    def test_ensure_sentiment_handles_missing_provider_gracefully(self, _mock_rows, _mock_diagnostics) -> None:
        result = app_data.ensure_sentiment_for_ticker("SCHO")

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "sentiment_unavailable_provider_not_configured")
        self.assertEqual(result["sentiment_unavailable_reason"], "GNEWS_API_KEY not configured")
        self.assertEqual(
            result["ui_message"],
            "News sentiment is currently unavailable because the configured news provider is not set up and no cached sentiment is available.",
        )
        self.assertFalse(result["used_cache"])
        self.assertEqual(result["sentiment_records_count"], 0)

    @patch("src.utils.app_data._has_stored_price_history", return_value=True)
    @patch("src.utils.app_data.ticker_exists", return_value=True)
    def test_ingest_single_ticker_prefers_stored_asset_for_existing_manual_ticker(
        self,
        _mock_exists,
        _mock_prices,
    ) -> None:
        result = app_data.ingest_single_ticker(" voo ")

        self.assertTrue(result["success"])
        self.assertEqual(result["ticker"], "VOO")
        self.assertEqual(result["status"], "database")
        self.assertTrue(result["used_cached_asset"])

    @patch("src.utils.app_data.load_price_history", return_value=[])
    @patch("src.utils.app_data.load_asset_metadata", return_value={"ticker": "VOO", "asset_name": "Vanguard 500"})
    def test_load_asset_dataset_for_app_rejects_missing_price_history(
        self,
        _mock_metadata,
        _mock_prices,
    ) -> None:
        result = app_data.load_asset_dataset_for_app("VOO")

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "missing_price_history")

    @patch(
        "src.utils.app_data.load_price_history",
        return_value=[
            {
                "price_date": "2026-03-20",
                "close_price": 100.0,
                "adjusted_close": 100.0,
                "open_price": 99.0,
                "high_price": 101.0,
                "low_price": 98.0,
                "volume": 1000,
            },
            {
                "price_date": "2026-03-21",
                "close_price": 101.0,
                "adjusted_close": 101.0,
                "open_price": 100.0,
                "high_price": 102.0,
                "low_price": 99.0,
                "volume": 1200,
            },
        ],
    )
    @patch("src.utils.app_data.load_asset_metadata", return_value={"ticker": "VOO", "asset_name": "Vanguard 500"})
    def test_load_asset_dataset_for_app_returns_validated_dataset(
        self,
        _mock_metadata,
        _mock_prices,
    ) -> None:
        result = app_data.load_asset_dataset_for_app("VOO")

        self.assertTrue(result["success"])
        self.assertEqual(result["dataset"].ticker, "VOO")
        self.assertEqual(result["dataset"].metadata["ticker"], "VOO")
        self.assertEqual(result["dataset"].price_frame["analysis_price"].dropna().shape[0], 2)

    @patch(
        "src.utils.app_data._sentiment_provider_diagnostics",
        return_value={
            "availability": {"gnews": False, "finnhub": False, "newsapi": False},
            "selected_provider": None,
            "fallback_provider_used": False,
            "live_provider_available": False,
        },
    )
    @patch("src.utils.app_data.sentiment_is_fresh", return_value=False)
    @patch("src.utils.app_data.load_recent_news_articles", return_value=[{"headline": "cached"}])
    def test_ensure_sentiment_uses_cached_status_when_provider_missing(
        self,
        _mock_rows,
        _mock_freshness,
        _mock_diagnostics,
    ) -> None:
        result = app_data.ensure_sentiment_for_ticker("NVDA")

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "cached_sentiment_loaded")
        self.assertEqual(result["ui_message"], "Live news sentiment unavailable; showing cached sentiment.")
        self.assertTrue(result["used_cache"])
        self.assertEqual(result["provider_used"], "cache")
        self.assertEqual(result["sentiment_unavailable_reason"], "news sentiment provider not configured")

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


    @patch(
        "src.utils.app_data.load_latest_ml_forecast",
        return_value={
            "ticker": "VOO",
            "predicted_return_20d": -19.3097,
            "probability_positive_20d": 0.8511,
            "downside_probability_20d": 0.1489,
            "composite_ml_score": -25.3,
            "confidence_score": 0.61,
            "history_score": -0.18,
            "risk_score": -0.41,
            "sentiment_score": -0.14,
            "directional_signal": "Unfavorable",
            "prediction_horizon_days": 20,
            "selected_model_name": "ridge_regression",
            "classification_model_name": "random_forest_classifier",
            "target_name": "forward_return_20d",
            "article_count_7d": 4,
            "source_count_7d": 2,
            "pillar_weights_json": [],
            "feature_importance_json": [],
            "top_features_json": [],
        },
    )
    @patch("src.utils.app_data.load_ml_prediction_history", return_value=[])
    def test_build_ml_forecast_summary_clamps_extreme_stored_return(
        self,
        _mock_history,
        _mock_snapshot,
    ) -> None:
        summary = app_data.build_ml_forecast_summary("VOO")

        self.assertTrue(summary["available"])
        self.assertEqual(summary["snapshot"]["predicted_return_20d"], -0.60)
        self.assertIn("-60.00%", summary["interpretation"])
        self.assertNotIn("-1930.97%", summary["interpretation"])



if __name__ == "__main__":
    unittest.main()
