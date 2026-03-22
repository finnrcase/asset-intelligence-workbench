"""
Tests for finance-oriented feature engineering and target construction.
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.features.engineering import build_sentiment_feature_frame
from src.features.engineering import build_technical_feature_frame
from src.features.targets import build_forward_return_targets


class FeatureEngineeringTests(unittest.TestCase):
    """Validate technical features and forward targets on deterministic inputs."""

    def setUp(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=50)
        prices = pd.Series(range(100, 150), index=dates, dtype=float)
        self.market_frame = pd.DataFrame(
            {
                "asset_id": 1,
                "ticker": "AAPL",
                "price_date": dates,
                "open_price": prices - 1.0,
                "high_price": prices + 1.5,
                "low_price": prices - 1.5,
                "close_price": prices,
                "adjusted_close": prices,
                "analysis_price": prices,
                "volume": [1_000_000 + (index * 10_000) for index in range(len(dates))],
            }
        )
        self.news_frame = pd.DataFrame(
            {
                "asset_id": [1, 1, 1, 1],
                "ticker": ["AAPL"] * 4,
                "published_at": pd.to_datetime(
                    [
                        "2025-01-05 09:00:00",
                        "2025-01-05 14:00:00",
                        "2025-01-06 10:00:00",
                        "2025-01-08 08:30:00",
                    ]
                ),
                "sentiment_score": [0.4, -0.2, 0.1, -0.5],
                "sentiment_label": ["positive", "negative", "neutral", "negative"],
                "source_name": ["wire", "wire", "blog", "terminal"],
            }
        )

    def test_build_technical_feature_frame_includes_finance_features(self) -> None:
        features = build_technical_feature_frame(self.market_frame)

        expected_columns = {
            "return_lag_1d",
            "rolling_mean_return_20d",
            "rolling_volatility_20d",
            "momentum_20d",
            "ma_distance_20d",
            "drawdown_from_peak",
            "downside_volatility_20d",
            "recent_realized_volatility_5d",
            "volume_ratio_20d",
        }
        self.assertTrue(expected_columns.issubset(features.columns))
        last_row = features.iloc[-1]
        self.assertAlmostEqual(last_row["momentum_20d"], 20.0 / 129.0, places=10)
        self.assertLessEqual(last_row["drawdown_from_peak"], 0.0)

    def test_build_forward_return_targets_uses_forward_window(self) -> None:
        targets = build_forward_return_targets(self.market_frame, horizon_days=20)
        row = targets.iloc[0]
        expected_return = 120.0 / 100.0 - 1.0
        self.assertAlmostEqual(row["target_forward_return_20d"], expected_return, places=10)
        self.assertEqual(row["target_negative_return_20d"], 0.0)
        self.assertTrue(pd.isna(targets.iloc[-1]["target_forward_return_20d"]))

    def test_build_sentiment_feature_frame_aggregates_daily_and_trailing_signals(self) -> None:
        features = build_sentiment_feature_frame(self.news_frame)
        jan_5 = features.loc[features["feature_date"] == pd.Timestamp("2025-01-05").date()].iloc[0]
        self.assertEqual(jan_5["article_count_1d"], 2)
        self.assertAlmostEqual(jan_5["sentiment_mean_1d"], 0.1, places=10)
        self.assertAlmostEqual(jan_5["negative_article_share_7d"], 0.5, places=10)
        self.assertEqual(jan_5["source_count_7d"], 1)
        jan_8 = features.loc[features["feature_date"] == pd.Timestamp("2025-01-08").date()].iloc[0]
        self.assertGreaterEqual(jan_8["source_count_7d"], 1)
        self.assertTrue(pd.notna(jan_8["source_sentiment_dispersion_7d"]) or pd.isna(jan_8["source_sentiment_dispersion_7d"]))


if __name__ == "__main__":
    unittest.main()
