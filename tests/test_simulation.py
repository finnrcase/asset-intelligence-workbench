"""
Tests for forward simulation helpers.
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.analytics.simulation import compute_percentile_bands
from src.analytics.simulation import estimate_simulation_inputs
from src.analytics.simulation import run_monte_carlo_simulation
from src.analytics.simulation import simulate_price_paths
from src.analytics.simulation import summarize_terminal_outcomes


class SimulationAnalyticsTests(unittest.TestCase):
    """Validate first-pass Monte Carlo simulation helpers."""

    def setUp(self) -> None:
        self.return_series = pd.Series(
            [0.01, -0.005, 0.012, 0.004, -0.003, 0.006, 0.002],
            name="daily_return",
        )
        self.price_frame = pd.DataFrame(
            {
                "analysis_price": [100.0, 101.0, 100.5, 101.7, 102.1, 101.8, 102.4, 103.0]
            }
        )

    def test_estimate_simulation_inputs(self) -> None:
        """Simulation input estimation should return the expected fields."""

        inputs = estimate_simulation_inputs(self.return_series)
        self.assertIn("daily_drift", inputs)
        self.assertIn("daily_volatility", inputs)
        self.assertIn("annualized_drift", inputs)
        self.assertIn("annualized_volatility", inputs)
        self.assertEqual(inputs["observations"], 7.0)

    def test_simulate_price_paths_shape(self) -> None:
        """Path simulation should return one starting row plus the forecast horizon."""

        paths = simulate_price_paths(
            starting_price=100.0,
            drift=0.0005,
            volatility=0.01,
            horizon_days=10,
            simulation_count=25,
            random_seed=7,
        )
        self.assertEqual(paths.shape, (11, 25))
        self.assertTrue((paths.iloc[0] == 100.0).all())

    def test_terminal_summary_fields(self) -> None:
        """Terminal summary should expose core analyst-facing outcome fields."""

        paths = simulate_price_paths(
            starting_price=100.0,
            drift=0.0005,
            volatility=0.01,
            horizon_days=10,
            simulation_count=50,
            random_seed=11,
        )
        summary = summarize_terminal_outcomes(paths)
        self.assertIn("median_terminal_price", summary)
        self.assertIn("p05_terminal_price", summary)
        self.assertIn("p95_terminal_price", summary)
        self.assertIn("probability_above_start", summary)
        self.assertGreaterEqual(summary["probability_above_start"], 0.0)
        self.assertLessEqual(summary["probability_above_start"], 1.0)

    def test_percentile_bands_shape(self) -> None:
        """Percentile bands should align to simulation steps and expected columns."""

        paths = simulate_price_paths(
            starting_price=100.0,
            drift=0.0005,
            volatility=0.01,
            horizon_days=8,
            simulation_count=40,
            random_seed=3,
        )
        bands = compute_percentile_bands(paths)
        self.assertEqual(bands.shape, (9, 5))
        self.assertEqual(list(bands.columns), ["p05", "p25", "p50", "p75", "p95"])

    def test_run_monte_carlo_simulation(self) -> None:
        """End-to-end simulation should return paths, bands, inputs, and terminal summary."""

        result = run_monte_carlo_simulation(
            self.price_frame,
            price_column="analysis_price",
            horizon_days=15,
            simulation_count=30,
            random_seed=5,
        )
        self.assertEqual(set(result.keys()), {"inputs", "paths", "bands", "terminal_summary"})
        self.assertEqual(result["paths"].shape, (16, 30))
        self.assertEqual(result["bands"].shape[0], 16)
        self.assertGreater(result["terminal_summary"]["max_terminal_price"], 0.0)


if __name__ == "__main__":
    unittest.main()
