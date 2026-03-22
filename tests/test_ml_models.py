"""
Tests for the SQL-backed ML forecasting pipeline.
"""

from __future__ import annotations

import importlib
import os
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parent / ".tmp"


def _reload_modules(sqlite_path: str):
    os.environ.pop("DATABASE_URL", None)
    os.environ["SQLITE_DB_PATH"] = sqlite_path

    import src.utils.config as config_module
    import src.database.connection as connection_module
    import src.database.loaders as loaders_module
    import src.database.queries as queries_module
    import src.features.feature_store as feature_store_module
    import src.ml.train as train_module
    import src.ml.predict as predict_module

    importlib.reload(config_module)
    connection_module = importlib.reload(connection_module)
    loaders_module = importlib.reload(loaders_module)
    queries_module = importlib.reload(queries_module)
    feature_store_module = importlib.reload(feature_store_module)
    train_module = importlib.reload(train_module)
    predict_module = importlib.reload(predict_module)
    return (
        connection_module,
        loaders_module,
        queries_module,
        feature_store_module,
        train_module,
        predict_module,
    )


class ForecastingPipelineTests(unittest.TestCase):
    """Validate feature persistence, model training, and prediction writes."""

    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = TEST_ROOT / f"ml_pipeline_{uuid.uuid4().hex}.db"
        (
            self.connection_module,
            self.loaders_module,
            self.queries_module,
            self.feature_store_module,
            self.train_module,
            self.predict_module,
        ) = _reload_modules(
            sqlite_path=str(self.sqlite_path),
        )
        schema_path = Path(__file__).resolve().parents[1] / 'sql' / 'schema.sql'
        self.connection_module.initialize_database(schema_path=schema_path)

    def tearDown(self) -> None:
        self.connection_module.ENGINE.dispose()
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()

    def test_end_to_end_feature_store_training_and_prediction_flow(self) -> None:
        with self.connection_module.session_scope() as session:
            self.loaders_module.upsert_asset_metadata(
                session=session,
                assets=[
                    {"ticker": "AAPL", "asset_name": "Apple Inc.", "currency": "USD"},
                    {"ticker": "MSFT", "asset_name": "Microsoft Corporation", "currency": "USD"},
                ],
                source_name="unit_test_feed",
                source_type="market_data",
            )

            trading_days = pd.bdate_range("2024-01-02", periods=180)
            for ticker, phase_shift in [("AAPL", 0.0), ("MSFT", 0.7)]:
                prices = []
                for index, trading_day in enumerate(trading_days):
                    base_price = 100.0 + (0.12 * index) + (2.5 * np.sin((index / 9.0) + phase_shift))
                    price = round(base_price + (0.7 * np.cos(index / 5.0)), 4)
                    prices.append(
                        {
                            "price_date": trading_day.date().isoformat(),
                            "open_price": price - 0.6,
                            "high_price": price + 1.0,
                            "low_price": price - 1.0,
                            "close_price": price,
                            "adjusted_close": price,
                            "volume": int(1_000_000 + index * 2_500 + (15_000 * abs(np.sin(index / 11.0)))),
                        }
                    )
                self.loaders_module.load_historical_prices(
                    session=session,
                    ticker=ticker,
                    source_name="unit_test_feed",
                    source_type="market_data",
                    price_rows=prices,
                )

                articles = []
                for index, trading_day in enumerate(trading_days[::5]):
                    signal = np.sin((index / 4.0) + phase_shift)
                    sentiment_score = round(float(signal), 4)
                    sentiment_label = (
                        "positive" if sentiment_score > 0.15 else "negative" if sentiment_score < -0.15 else "neutral"
                    )
                    articles.append(
                        {
                            "headline": f"{ticker} sentiment {index}",
                            "url": f"https://example.com/{ticker.lower()}/{index}",
                            "published_at": pd.Timestamp(trading_day).to_pydatetime(),
                            "sentiment_score": sentiment_score,
                            "sentiment_label": sentiment_label,
                        }
                    )
                self.loaders_module.load_news_articles(
                    session=session,
                    ticker=ticker,
                    articles=articles,
                    source_name="unit_test_news",
                    source_type="news_api",
                )

        with self.connection_module.session_scope() as session:
            refresh_result = self.feature_store_module.refresh_feature_store(session=session)
            self.assertGreater(refresh_result["technical_features_loaded"], 0)

            training_frame = self.feature_store_module.load_training_frame_from_store(session=session)
            self.assertIn("target_forward_return_20d", training_frame.columns)
            self.assertIn("sentiment_mean_7d", training_frame.columns)

            training_result = self.train_module.train_forecasting_models(training_frame)
            self.assertIn("holdout", training_result["evaluation"])
            self.assertIn("ridge_regression", training_result["evaluation"]["holdout"]["regression"])
            self.assertIn("random_forest_classifier", training_result["evaluation"]["holdout"]["classification"])

            model_run_id = "unit_test_run"
            self.loaders_module.load_ml_model_run(
                session=session,
                run_record={
                    "run_id": model_run_id,
                    "run_timestamp": pd.Timestamp("2026-01-01 00:00:00").to_pydatetime(),
                    "regression_model_name": training_result["selected_models"]["regression"],
                    "classification_model_name": training_result["selected_models"]["classification"],
                    "training_start_date": str(training_frame["feature_date"].min()),
                    "training_end_date": str(training_frame["feature_date"].max()),
                    "evaluation_summary": training_result["evaluation"],
                    "feature_version": "v1",
                    "notes": "unit test",
                },
            )

            prediction_frame = self.predict_module.predict_from_feature_store(
                session=session,
                training_result=training_result,
                model_run_id=model_run_id,
                write_to_sql=True,
            )
            self.assertEqual(len(prediction_frame), 2)
            self.assertIn("predicted_return_20d", prediction_frame.columns)
            self.assertIn("downside_probability_20d", prediction_frame.columns)

            stored_predictions = self.queries_module.get_ml_predictions(session=session)
            self.assertEqual(len(stored_predictions), 2)
            self.assertTrue(
                all(0.0 <= float(row["downside_probability_20d"]) <= 1.0 for row in stored_predictions)
            )


if __name__ == "__main__":
    unittest.main()


