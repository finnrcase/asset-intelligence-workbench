"""
Integration tests for product-surface ML forecast wiring.
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
    import src.utils.app_data as app_data_module
    import src.reporting.report_data as report_data_module
    import src.reporting.pdf_report as pdf_report_module

    importlib.reload(config_module)
    connection_module = importlib.reload(connection_module)
    loaders_module = importlib.reload(loaders_module)
    queries_module = importlib.reload(queries_module)
    feature_store_module = importlib.reload(feature_store_module)
    train_module = importlib.reload(train_module)
    predict_module = importlib.reload(predict_module)
    app_data_module = importlib.reload(app_data_module)
    report_data_module = importlib.reload(report_data_module)
    pdf_report_module = importlib.reload(pdf_report_module)
    return (
        connection_module,
        loaders_module,
        queries_module,
        feature_store_module,
        train_module,
        predict_module,
        app_data_module,
        report_data_module,
        pdf_report_module,
    )


class MlIntegrationTests(unittest.TestCase):
    """Validate ML outputs through SQL, app-data shaping, simulation, and reporting."""

    def setUp(self) -> None:
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = TEST_ROOT / f"ml_integration_{uuid.uuid4().hex}.db"
        (
            self.connection_module,
            self.loaders_module,
            self.queries_module,
            self.feature_store_module,
            self.train_module,
            self.predict_module,
            self.app_data_module,
            self.report_data_module,
            self.pdf_report_module,
        ) = _reload_modules(
            sqlite_path=str(self.sqlite_path),
        )
        schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
        self.connection_module.initialize_database(schema_path=schema_path)

    def tearDown(self) -> None:
        self.connection_module.ENGINE.dispose()
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()

    def test_ml_outputs_flow_into_app_data_and_reporting(self) -> None:
        report_path: Path | None = None
        with self.connection_module.session_scope() as session:
            self.loaders_module.upsert_asset_metadata(
                session=session,
                assets=[{"ticker": "AAPL", "asset_name": "Apple Inc.", "currency": "USD"}],
                source_name="unit_test_feed",
                source_type="market_data",
            )

            trading_days = pd.bdate_range("2024-01-02", periods=220)
            price_rows = []
            for index, trading_day in enumerate(trading_days):
                base_price = 110.0 + (0.10 * index) + (2.1 * np.sin(index / 8.0))
                price = round(base_price + (0.8 * np.cos(index / 6.0)), 4)
                price_rows.append(
                    {
                        "price_date": trading_day.date().isoformat(),
                        "open_price": price - 0.5,
                        "high_price": price + 1.1,
                        "low_price": price - 1.0,
                        "close_price": price,
                        "adjusted_close": price,
                        "volume": int(1_250_000 + index * 3000 + (12500 * abs(np.sin(index / 10.0)))),
                    }
                )
            self.loaders_module.load_historical_prices(
                session=session,
                ticker="AAPL",
                source_name="unit_test_feed",
                source_type="market_data",
                price_rows=price_rows,
            )

            articles = []
            for index, trading_day in enumerate(trading_days[::4]):
                sentiment_signal = np.sin(index / 3.5)
                sentiment_score = round(float(sentiment_signal), 4)
                sentiment_label = (
                    "positive" if sentiment_score > 0.10 else "negative" if sentiment_score < -0.10 else "neutral"
                )
                articles.append(
                    {
                        "headline": f"AAPL article {index}",
                        "url": f"https://example.com/aapl/{index}",
                        "published_at": pd.Timestamp(trading_day).to_pydatetime(),
                        "sentiment_score": sentiment_score,
                        "sentiment_label": sentiment_label,
                    }
                )
            self.loaders_module.load_news_articles(
                session=session,
                ticker="AAPL",
                articles=articles,
                source_name="unit_test_news",
                source_type="news_api",
            )

        with self.connection_module.session_scope() as session:
            self.feature_store_module.refresh_feature_store(session=session, ticker="AAPL")
            training_frame = self.feature_store_module.load_training_frame_from_store(session=session, ticker="AAPL")
            training_result = self.train_module.train_forecasting_models(training_frame)
            self.loaders_module.load_ml_model_run(
                session=session,
                run_record={
                    "run_id": "integration_run",
                    "run_timestamp": pd.Timestamp("2026-01-01 00:00:00").to_pydatetime(),
                    "regression_model_name": training_result["selected_models"]["regression"],
                    "classification_model_name": training_result["selected_models"]["classification"],
                    "training_start_date": str(training_frame["feature_date"].min()),
                    "training_end_date": str(training_frame["feature_date"].max()),
                    "evaluation_summary": training_result["evaluation"],
                    "feature_version": "v1",
                    "notes": "integration test",
                },
            )
            prediction_frame = self.predict_module.predict_from_feature_store(
                session=session,
                training_result=training_result,
                ticker="AAPL",
                model_run_id="integration_run",
                write_to_sql=True,
            )
            self.assertEqual(len(prediction_frame), 1)

            latest_prediction = self.queries_module.get_latest_ml_prediction(session=session, ticker="AAPL")
            prediction_history = self.queries_module.get_ml_prediction_history(session=session, ticker="AAPL", limit=10)
            driver_frame = self.queries_module.get_feature_driver_frame(session=session, ticker="AAPL")
            self.assertIsNotNone(latest_prediction)
            self.assertGreaterEqual(len(prediction_history), 1)
            self.assertFalse(driver_frame.empty)

        ml_summary = self.app_data_module.build_ml_forecast_summary("AAPL")
        self.assertTrue(ml_summary["available"])
        self.assertIn("probability", ml_summary["interpretation"].lower())
        self.assertIn("regime_label", ml_summary["snapshot"])
        self.assertIn("composite_ml_score", ml_summary["snapshot"])
        self.assertTrue(ml_summary["pillar_contributions"])
        self.assertTrue(ml_summary["feature_importance"])

        report_context = self.report_data_module.build_asset_report_context(
            ticker="AAPL",
            forecast_horizon=63,
            simulation_count=120,
        )
        self.assertIn("ml_forecast", report_context)
        self.assertTrue(report_context["ml_forecast"]["available"])
        self.assertIn("historical", report_context["simulation"])
        self.assertIn("ml_informed", report_context["simulation"])
        self.assertIn("ml_commentary", report_context["narrative"])
        self.assertIn("comparative_simulation_commentary", report_context["narrative"])

        report_path = Path(
            self.pdf_report_module.generate_asset_pdf_report(
                ticker="AAPL",
                forecast_horizon=63,
                simulation_count=100,
            )
        )
        self.assertTrue(report_path.exists())
        self.assertGreater(report_path.stat().st_size, 0)

        try:
            report_path.unlink(missing_ok=True)
            report_path.with_suffix(".html").unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
