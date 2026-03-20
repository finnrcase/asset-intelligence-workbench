"""
Tests for return analytics functions.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from src.analytics.returns import build_return_frame
from src.analytics.returns import compute_annualized_return
from src.analytics.returns import compute_cumulative_returns
from src.analytics.returns import compute_daily_returns
from src.analytics.returns import compute_total_return


class ReturnAnalyticsTests(unittest.TestCase):
    """Validate first-pass return analytics on controlled price inputs."""

    def setUp(self) -> None:
        self.price_series = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
            name="close_price",
        )

    def test_compute_daily_returns(self) -> None:
        """Daily returns should be simple percentage changes between prices."""

        returns = compute_daily_returns(self.price_series)
        expected = pd.Series(
            [0.10, 0.10],
            index=self.price_series.index[1:],
            name="close_price",
        )
        pd.testing.assert_series_equal(returns, expected)

    def test_compute_cumulative_returns(self) -> None:
        """Cumulative returns should compound sequential daily returns."""

        cumulative = compute_cumulative_returns(self.price_series)
        expected = pd.Series(
            [0.10, 0.21],
            index=self.price_series.index[1:],
            name="close_price",
        )
        pd.testing.assert_series_equal(cumulative.round(10), expected.round(10))

    def test_compute_total_return(self) -> None:
        """Total return should measure ending price versus starting price."""

        total_return = compute_total_return(self.price_series)
        self.assertAlmostEqual(total_return, 0.21, places=10)

    def test_compute_annualized_return(self) -> None:
        """Annualized return should use geometric scaling across observed periods."""

        annualized_return = compute_annualized_return(self.price_series, periods_per_year=2)
        self.assertAlmostEqual(annualized_return, 0.21, places=10)

    def test_build_return_frame_from_dataframe(self) -> None:
        """Return frame helper should work cleanly from a queried price DataFrame."""

        frame = pd.DataFrame({"close_price": self.price_series})
        return_frame = build_return_frame(frame, price_column="close_price")

        self.assertEqual(list(return_frame.columns), ["price", "daily_return", "cumulative_return"])
        self.assertTrue(math.isnan(return_frame["daily_return"].iloc[0]))
        self.assertAlmostEqual(return_frame["cumulative_return"].iloc[-1], 0.21, places=10)


if __name__ == "__main__":
    unittest.main()

