"""
Forward-looking price simulation utilities for single-asset analysis.

This module uses historical daily simple returns to estimate drift and
volatility, then simulates future price paths with a geometric Brownian motion
style process. The goal is a practical, readable forecasting layer for an
analyst-facing risk workbench rather than a highly specialized quant library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.returns import DEFAULT_TRADING_DAYS
from src.analytics.returns import compute_daily_returns


def estimate_simulation_inputs(
    historical_returns: pd.Series | pd.DataFrame,
    return_column: str | None = None,
    periods_per_year: int = DEFAULT_TRADING_DAYS,
) -> dict[str, float]:
    """
    Estimate drift and volatility inputs from historical daily returns.

    Assumptions:
    - historical daily simple returns are used as the empirical input series
    - the arithmetic mean of daily returns is used as the daily drift estimate
    - sample standard deviation is used as the daily volatility estimate
    - annualized drift and volatility are provided for display and diagnostics
    """

    if isinstance(historical_returns, pd.DataFrame):
        if return_column is None:
            raise ValueError(
                "`return_column` is required when historical returns are provided as a DataFrame."
            )
        if return_column not in historical_returns.columns:
            raise KeyError(f"Return column '{return_column}' was not found in the DataFrame.")
        series = historical_returns[return_column]
    else:
        series = historical_returns

    clean_returns = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(clean_returns) < 2:
        raise ValueError("At least two daily returns are required to estimate simulation inputs.")

    daily_drift = float(clean_returns.mean())
    daily_volatility = float(clean_returns.std(ddof=1))

    return {
        "daily_drift": daily_drift,
        "daily_volatility": daily_volatility,
        "annualized_drift": float(daily_drift * periods_per_year),
        "annualized_volatility": float(daily_volatility * np.sqrt(periods_per_year)),
        "observations": float(len(clean_returns)),
    }


def simulate_price_paths(
    starting_price: float,
    drift: float,
    volatility: float,
    horizon_days: int = 252,
    simulation_count: int = 1000,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """
    Simulate future price paths with a GBM-style daily process.

    Formula:
    `S(t+1) = S(t) * exp((mu - 0.5 * sigma^2) + sigma * z_t)`

    Inputs `drift` and `volatility` are expected to be daily values.
    """

    if starting_price <= 0:
        raise ValueError("`starting_price` must be positive.")
    if horizon_days <= 0:
        raise ValueError("`horizon_days` must be positive.")
    if simulation_count <= 0:
        raise ValueError("`simulation_count` must be positive.")
    if volatility < 0:
        raise ValueError("`volatility` cannot be negative.")

    rng = np.random.default_rng(random_seed)
    shocks = rng.standard_normal((horizon_days, simulation_count))
    log_returns = (drift - 0.5 * (volatility ** 2)) + (volatility * shocks)
    cumulative_log_returns = np.cumsum(log_returns, axis=0)

    simulated_prices = starting_price * np.exp(cumulative_log_returns)
    starting_row = np.full((1, simulation_count), float(starting_price))
    full_paths = np.vstack([starting_row, simulated_prices])

    path_index = pd.RangeIndex(start=0, stop=horizon_days + 1, step=1, name="step")
    columns = [f"path_{index + 1}" for index in range(simulation_count)]
    return pd.DataFrame(full_paths, index=path_index, columns=columns)


def summarize_terminal_outcomes(simulated_paths: pd.DataFrame) -> dict[str, float]:
    """
    Summarize the terminal values of simulated price paths.

    Returns core analyst-facing summary statistics for the ending price
    distribution.
    """

    if simulated_paths.empty:
        raise ValueError("`simulated_paths` cannot be empty.")

    terminal_values = simulated_paths.iloc[-1].astype(float)
    starting_price = float(simulated_paths.iloc[0, 0])

    return {
        "starting_price": starting_price,
        "mean_terminal_price": float(terminal_values.mean()),
        "median_terminal_price": float(terminal_values.median()),
        "min_terminal_price": float(terminal_values.min()),
        "max_terminal_price": float(terminal_values.max()),
        "p05_terminal_price": float(terminal_values.quantile(0.05)),
        "p25_terminal_price": float(terminal_values.quantile(0.25)),
        "p75_terminal_price": float(terminal_values.quantile(0.75)),
        "p95_terminal_price": float(terminal_values.quantile(0.95)),
        "probability_above_start": float((terminal_values > starting_price).mean()),
        "probability_below_start": float((terminal_values < starting_price).mean()),
    }


def compute_percentile_bands(
    simulated_paths: pd.DataFrame,
    percentiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> pd.DataFrame:
    """
    Compute percentile bands across simulated paths at each future step.

    The output is indexed by simulation step and is suitable for plotting a
    fan-chart style percentile summary in the app layer.
    """

    if simulated_paths.empty:
        raise ValueError("`simulated_paths` cannot be empty.")

    band_data = {
        f"p{int(percentile * 100):02d}": simulated_paths.quantile(percentile, axis=1)
        for percentile in percentiles
    }
    return pd.DataFrame(band_data, index=simulated_paths.index)


def run_monte_carlo_simulation(
    price_data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
    horizon_days: int = 252,
    simulation_count: int = 1000,
    periods_per_year: int = DEFAULT_TRADING_DAYS,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """
    Run the end-to-end simulation workflow from stored historical price data.

    Returns the estimated inputs, simulated price paths, percentile bands, and
    terminal summary in one app-friendly structure.
    """

    daily_returns = compute_daily_returns(price_data, price_column=price_column)
    inputs = estimate_simulation_inputs(
        daily_returns,
        periods_per_year=periods_per_year,
    )

    if isinstance(price_data, pd.DataFrame):
        if price_column is None:
            raise ValueError("`price_column` is required when price data is a DataFrame.")
        starting_price = float(pd.to_numeric(price_data[price_column], errors="coerce").dropna().iloc[-1])
    else:
        starting_price = float(pd.to_numeric(price_data, errors="coerce").dropna().iloc[-1])

    paths = simulate_price_paths(
        starting_price=starting_price,
        drift=inputs["daily_drift"],
        volatility=inputs["daily_volatility"],
        horizon_days=horizon_days,
        simulation_count=simulation_count,
        random_seed=random_seed,
    )
    bands = compute_percentile_bands(paths)
    terminal_summary = summarize_terminal_outcomes(paths)

    return {
        "inputs": inputs,
        "paths": paths,
        "bands": bands,
        "terminal_summary": terminal_summary,
    }

