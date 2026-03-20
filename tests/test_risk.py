"""
Tests for risk analytics functions.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analytics.risk import build_risk_summary
from src.analytics.risk import compute_annualized_volatility
from src.analytics.risk import compute_expected_shortfall
from src.analytics.risk import compute_historical_var
from src.analytics.risk import compute_max_drawdown
from src.analytics.risk import compute_rolling_volatility


class RiskAnalyticsTests(unittest.TestCase):
    """Validate first-pass risk calculations on deterministic input data."""

    def setUp(self) -> None:
        self.return_series = pd.Series(
            [-0.02, 0.01, -0.03, 0.02, -0.01],
            index=pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
            ),
            name="daily_return",
        )
        self.price_series = pd.Series(
            [100.0, 120.0, 90.0, 95.0, 80.0],
            index=pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
            ),
            name="close_price",
        )

    def test_compute_annualized_volatility(self) -> None:
        """Annualized volatility should scale sample standard deviation."""

        expected = self.return_series.std(ddof=1) * np.sqrt(252)
        actual = compute_annualized_volatility(self.return_series)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_compute_rolling_volatility(self) -> None:
        """Rolling volatility should return NaN until the window is full."""

        rolling = compute_rolling_volatility(self.return_series, window=3, periods_per_year=252)
        self.assertTrue(np.isnan(rolling.iloc[0]))
        self.assertTrue(np.isnan(rolling.iloc[1]))
        expected_last = self.return_series.iloc[-3:].std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(rolling.iloc[-1], expected_last, places=10)

    def test_compute_max_drawdown(self) -> None:
        """Max drawdown should capture the largest peak-to-trough decline."""

        max_drawdown = compute_max_drawdown(self.price_series)
        self.assertAlmostEqual(max_drawdown, -1.0 / 3.0, places=10)

    def test_compute_historical_var(self) -> None:
        """Historical VaR should report the positive magnitude of the left-tail cutoff."""

        var_80 = compute_historical_var(self.return_series, confidence_level=0.80)
        self.assertAlmostEqual(var_80, 0.022, places=10)

    def test_compute_expected_shortfall(self) -> None:
        """Expected shortfall should average returns at or below the VaR threshold."""

        es_80 = compute_expected_shortfall(self.return_series, confidence_level=0.80)
        self.assertAlmostEqual(es_80, 0.03, places=10)

    def test_build_risk_summary(self) -> None:
        """Risk summary should combine the main first-pass risk outputs."""

        frame = pd.DataFrame({"close_price": [100.0, 102.0, 99.0, 101.0, 98.0, 103.0]})
        summary = build_risk_summary(
            frame,
            price_column="close_price",
            confidence_level=0.80,
            volatility_window=3,
            periods_per_year=252,
        )

        self.assertIn("annualized_volatility", summary)
        self.assertIn("max_drawdown", summary)
        self.assertIn("historical_var", summary)
        self.assertIn("expected_shortfall", summary)
        self.assertIn("latest_rolling_volatility", summary)
        self.assertLessEqual(summary["max_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()

