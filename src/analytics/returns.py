"""
Return analytics utilities for stored market price data.

These functions are designed for use on pandas Series or on DataFrames that
contain a price column such as `close_price` or `adjusted_close`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRADING_DAYS = 252


def _coerce_price_series(
    data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
) -> pd.Series:
    """
    Normalize price input into a clean float Series ordered by the existing index.

    If a DataFrame is provided, `price_column` must identify the price field to
    use for the calculation.
    """

    if isinstance(data, pd.DataFrame):
        if price_column is None:
            raise ValueError("`price_column` is required when input data is a DataFrame.")
        if price_column not in data.columns:
            raise KeyError(f"Price column '{price_column}' was not found in the DataFrame.")
        series = data[price_column]
    else:
        series = data

    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def compute_daily_returns(
    data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
) -> pd.Series:
    """
    Compute simple daily returns from a price series.

    Formula:
    `return_t = price_t / price_(t-1) - 1`
    """

    prices = _coerce_price_series(data, price_column=price_column)
    returns = prices.pct_change()
    return returns.dropna()


def compute_cumulative_returns(
    data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
) -> pd.Series:
    """
    Compute cumulative returns from a price series.

    The first valid cumulative return starts after the first daily return is
    available.
    """

    daily_returns = compute_daily_returns(data, price_column=price_column)
    cumulative_returns = (1.0 + daily_returns).cumprod() - 1.0
    return cumulative_returns


def compute_total_return(
    data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
) -> float:
    """
    Compute total holding-period return across the full available price window.

    Formula:
    `ending_price / starting_price - 1`
    """

    prices = _coerce_price_series(data, price_column=price_column)
    if len(prices) < 2:
        return float("nan")
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def compute_annualized_return(
    data: pd.Series | pd.DataFrame,
    price_column: str | None = None,
    periods_per_year: int = DEFAULT_TRADING_DAYS,
) -> float:
    """
    Compute annualized geometric return from a price series.

    Convention:
    the number of observed return periods is treated as the number of trading
    periods elapsed.
    """

    prices = _coerce_price_series(data, price_column=price_column)
    if len(prices) < 2:
        return float("nan")

    total_return = prices.iloc[-1] / prices.iloc[0]
    periods_observed = len(prices) - 1
    if periods_observed <= 0 or total_return <= 0:
        return float("nan")

    annualized_return = total_return ** (periods_per_year / periods_observed) - 1.0
    return float(annualized_return)


def build_return_frame(
    data: pd.DataFrame,
    price_column: str = "close_price",
) -> pd.DataFrame:
    """
    Build a simple return analysis frame from a queried price-history DataFrame.

    This is a convenience wrapper that keeps the raw price series alongside the
    daily and cumulative return outputs.
    """

    prices = _coerce_price_series(data, price_column=price_column)
    daily_returns = prices.pct_change()
    cumulative_returns = (1.0 + daily_returns.fillna(0.0)).cumprod() - 1.0

    return pd.DataFrame(
        {
            "price": prices,
            "daily_return": daily_returns,
            "cumulative_return": cumulative_returns,
        }
    )

